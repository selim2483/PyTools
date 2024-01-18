import collections
from dataclasses import dataclass
from math import floor
import os
from rich.progress import MofNCompleteColumn, Progress
from rich.console import Console
import torch
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from torchvision.utils import save_image
from typing import Any, Union

from ..options.options import OptimizerOptions
from ..options.head_options import CoachOptions
from ..logging import mean_loss_dict, console_print, profiled_function 
from ..logging import TrainingLogs
from ..utils.misc import get_device


@dataclass
class Phase :
    """Dataclass embedding every necessary objects and information on a
    training phase : name, modules, the corresponding optimizer and its
    options.

    Args:
        name (str): Name of the training phase.
        modules (list[torch.nn.Module]): list of the modules concerned by the
            training phases.
        opt (torch.optim.Optimizer): The phase's optimizer.
        opt_options (OptimizerOptions): The optimizer options.
    """
    def __init__(
            self, 
            name        :str,
            modules     :list[torch.nn.Module], 
            opt         :torch.optim.Optimizer, 
            opt_options :OptimizerOptions
        ) :
        self.name        = name
        self.modules     = modules
        self.opt         = opt
        self.opt_options = opt_options

class Coach(CoachOptions, TrainingLogs) :
    """Base class embedding training loop.
    It embeds basic training, validation, logging and model saving 
    functionnalities.
    
    Args:
        options (Union[str, dict]): training loop options : path of the config
            file or config dict. 
        job_name (str): job name used for logging.
    """
    def __init__(
            self, 
            options  :Union[str, dict], 
            job_name :str, 
        ) :
        self.job_name = job_name
        self.global_step = 0
        self.nimg = 0
        self.console = Console()
        self.device = get_device()

        self._initialize_options(options)
        self._initialize_logger()
        self._initialize_dataset()
        self._initialize_loss()
        self._initialize_model()
        self._initialize_optimizers()

        if self.checkpoint_path is not None :
            self.load_from_train_checkpoint()

    model_name: str
    phases: list[Phase]
    train_loader: DataLoader
    valid_loader: Union[DataLoader, None]

    @property
    def max_steps(self) :
        return self.training_options.max_steps

    @property
    def max_img(self) :
        return self.training_options.max_kimg * 1000
    
    @property
    def kimg(self):
        return floor(self.nimg / 1000)
    
    @property
    def progress_key(self):
        if self.logging_options.progress_key=="step" :
            return self.global_step
        elif self.logging_options.progress_key=="kimg" :
            return self.kimg
    
    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~ Initializers ~~~~~~~~~~~~~~~~~~~~~~~~~~~~ #
    
    def _initialize_dataset(self) :
        raise NotImplementedError
    
    def _initialize_loss(self) :
        raise NotImplementedError
    
    def _initialize_model(self) :
        raise NotImplementedError
    
    def _initialize_optimizers(self) :
        raise NotImplementedError
    
    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ Training ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ #

    @profiled_function
    def data_fetch(self) -> Any:
        x, c = next(self.iter_train_loader)
        self.global_step += 1
        self.nimg += x.shape[0]
        return x, c

    def calc_loss(self, phase:str, data:Any) -> dict:
        raise NotImplementedError
    
    def train(self) :
        self.console.log(
            f"[green]Started training {self.model_name}[/green]")
        print(self.training_options)
        print("")

        progress = Progress(
            *Progress.get_default_columns(), MofNCompleteColumn())

        with progress as self.progress :
            self.task_kimg = self.progress.add_task(
                "kimg", total=self.training_options.max_kimg)
            self.task_steps = self.progress.add_task(
                "Steps", total=self.max_steps)

            self.iter_train_loader = iter(self.train_loader)
            agg_loss_dict = collections.deque(
                maxlen=self.logging_options.max_queue_train)
            while not self._reach_max():
                self._train()

                # Forward pass and loss
                data = self.data_fetch()
                loss_dict = dict()
                for phase in self.phases :
                    phase.opt.zero_grad()
                    for module in phase.modules :
                        module.requires_grad_(True)
                    loss_dict = {
                        **loss_dict, 
                        **self.calc_loss(phase.name, data)
                    }
                    for module in phase.modules :
                        module.requires_grad_(False)
                    with torch.autograd.profiler.record_function(
                        phase.name + '_opt'):
                        phase.opt.step()
                agg_loss_dict.append(loss_dict)
                self._update_hyparams()
                self.progress.update(self.task_steps, completed=self.global_step)
                self.progress.update(self.task_kimg, completed=self.kimg)
                
                train_loss_dict = mean_loss_dict(list(agg_loss_dict))

                with torch.no_grad() :
                    # Validation
                    self._eval()
                    valid_loss_dict = self.validation()

                    # Logging
                    self.log_metrics(train_loss_dict, valid_loss_dict)
                    if self.check_image_interval("both") :
                        self.parse_and_log_images()

                    # Save models
                    if self.check_for_interval_checkpoint() :
                        self.checkpoint(valid_loss_dict, is_best=False)

                    if self.is_best(valid_loss_dict) :
                        self.checkpoint(valid_loss_dict, is_best=True)
    
    def _update_hyparams(self) :
        raise NotImplementedError
    
    @profiled_function
    def validation(self) :
        if self.valid_loader is None :
            return None
        
        agg_loss_dict = []
        for _, x in enumerate(self.valid_loader) :
            _, loss_dict = self.calc_loss(x)
            agg_loss_dict.append(loss_dict)

        return mean_loss_dict(agg_loss_dict)
    
    def _train(self) :
        raise NotImplementedError

    def _eval(self) :
        raise NotImplementedError
    
    def _reach_max(self) :
        if self.max_img is None and self.max_steps is None :
            raise ValueError(
                "Either 'max_steps' or 'max_img' should be provided")
        elif self.max_steps is None :
            return self.nimg > self.max_img
        elif self.max_img is None :
            return self.global_step > self.max_steps
        else :
            return (self.nimg > self.max_img 
                    or self.global_step > self.max_steps)
    
    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ Logging ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ #

    @profiled_function
    def log_metrics(self, metrics_dict_train:dict, metrics_dict_valid:dict) :  
        # TensorBoard logging
        if self.logging_options.logger == "tensorboard" :
            for key, value in metrics_dict_train.items():
                self.logger.add_scalar(
                    f'train/{key}', value, self.progress_key)
            self.logger.add_scalar("train/kimg", self.kimg, self.progress_key)
            self.logger.add_scalar(
                "train/steps", self.global_step, self.progress_key)
            if metrics_dict_valid is not None :
                for key, value in metrics_dict_valid.items():
                    self.logger.add_scalar(
                        f'valid/{key}', value, self.progress_key)
        # WandB logging
        elif self.logging_options.logger == "wandb" :
            prefix_metrics_dict = {}
            for key, value in metrics_dict_train.items() :
                prefix_metrics_dict[f'train/{key}'] = value
            if metrics_dict_valid is not None :
                for key, value in metrics_dict_valid.items() :
                    prefix_metrics_dict[f'valid/{key}'] = value

            # wandb.log(prefix_metrics_dict, step=self.epoch, commit=True)

    def check_image_interval(self, mode:str="local") :
        # Initialization counters
        if not hasattr(self, "image_local_count") :
            self.image_local_count = 0
        if not hasattr(self, "image_logger_count") :
            self.image_logger_count = 0

        # Checks
        if mode=="local" :
            res = (self.progress_key 
                   // self.logging_options.image_interval_local)
            return res > self.image_local_count
        elif mode=="logger" :
            res = (self.progress_key 
                   // self.logging_options.image_interval_logger)
            return res > self.image_logger_count
        elif mode=="both" :
            res = max(
                (self.progress_key 
                 // self.logging_options.image_interval_local 
                 - self.image_local_count), 
                (self.progress_key 
                 // self.logging_options.image_interval_logger 
                 - self.image_logger_count))
            return res > 0
    
    @profiled_function
    def parse_and_log_images(self) :
        images_dicts = self.parse_images()
        for mode, image_dict in images_dicts.items() :
            for key, img in image_dict.items() :
                if self.check_image_interval("local") :
                    name = f"{self.job_name}_{key}_{self._save_suffix()}.png"
                    save_image(img, os.path.join(self.figdir, mode, name))

                if self.check_image_interval("logger") :
                    if self.logging_options.logger == "tensorboard" :
                        self.logger.add_image(
                            f"{mode}/{key}", 
                            img, 
                            self.progress_key)
                    if self.logging_options.logger == "wandb" :
                        # TODO
                        pass

        self.image_local_count  = (self.progress_key 
                                   // self.logging_options.image_interval_local)
        self.image_logger_count = (self.progress_key 
                                   // self.logging_options.image_interval_logger) 

    def parse_images(self) -> dict[str, dict[str, torch.Tensor]]:
        raise NotImplementedError

    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ Checkpoint ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ #

    @profiled_function
    def checkpoint(self, loss_dict, is_best="False") :
        """
        Save the wanted model in self.chkdir.

        Arguments:
            loss_dict -- dictionnary containing current loss values.
            is_best   -- Allow to chose between regular interval model saving
                         and best model saving. (default: {"False"})
        """
        if is_best :
            save_name = f'{self.job_name}_best_model.pt'
        else :
            save_name = f'{self.job_name}_{self._save_suffix()}.pt'

        save_dict = self.get_save_dict()
        checkpoint_path = os.path.join(self.chkdir, save_name)
        torch.save(save_dict, checkpoint_path)

        with open(os.path.join(self.chkdir, 'timestamp.txt'), 'a') as f:
            if is_best:
                f.write(
                    f"**Best**: Step - {self.progress_key},\n{loss_dict}")
            else:
                f.write(f"Step - {self.progress_key},\n{loss_dict}\n")

    def get_save_dict(self) :
        """Converts every usefull informations (models parameters and options)
        in a dictionnary. 

        Returns:
            (dict): save dict
        """
        save_dict = {
            **self.cfg,
            **self.get_state_dict(),
        }

        # Save necessary information to enable training continuation from 
        # checkpoint
        if self.logging_options.save_training_data:  
            save_dict['global_step'] = self.global_step
            save_dict['nimg']        = self.nimg
            for phase in self.phases :
                save_dict[f"{phase.name}_optimizer"] = phase.opt.state_dict()

        return save_dict

    def get_state_dict(self) :
        raise NotImplementedError
    
    def load_model(self) :
        raise NotImplementedError
    
    @console_print("Loading previous training data...")
    def load_from_train_checkpoint(self):
        print("Checkpoint path :")
        print(self.checkpoint_path)

        self.load_model()
        self.global_step = self.checkpoint_dict['global_step']

        for phase in self.phases:
            if phase.opt_options.from_checkpoint :
                phase.opt.load_state_dict(self.checkpoint_dict['optimizer'])
            
        del self.checkpoint_dict
        
        print(
            f'Resuming training from step {self.global_step}'
            + f' and {self.kimg} kimg\n')
    
    def check_for_interval_checkpoint(self) :
        if self.logging_options.checkpoint_interval is None :
            return False
        else :
            if not hasattr(self, "checkpoint_count") :
                self.checkpoint_count = 0
            res = (self.progress_key 
                   // self.logging_options.checkpoint_interval)
            if res > self.checkpoint_count :
                self.checkpoint_count = res
                return True
            else :
                return False
    
    def is_best(self, valid_loss_dict:Union[dict, None]=None) :
        #return (self.best_val_loss is None) or (val_loss > self.best_val_loss)
        return True
    
    def _save_suffix(self) :
        if self.logging_options.progress_key=="step":
            return f'it{self.global_step}'
        elif self.logging_options.progress_key=="kimg":
            return f'{self.kimg}k'
        else :
            raise NotImplementedError("'progress_key' parameter from\
'logging_options' should be either 'step' or 'kimg'." )