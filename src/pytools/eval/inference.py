import os
from typing import Any, Union

from rich.console import Console
from rich.progress import Progress, MofNCompleteColumn
import torch
from torch.utils.data.dataloader import _SingleProcessDataLoaderIter

from ..logging.misc import console_print, mean_loss_dict, std_loss_dict
from ..options.head_options import InferenceOptions
from ..utils.misc import get_device

class Inference(InferenceOptions):
    """Base class for Inference processes

    Args:
        options (Union[str, dict]): training loop options : path of the config
            file or config dict. 
        init_model (bool, optional): Whether to initialize the model or not.
            Defaults to ``True``.
    """

    model :Any
    loader :_SingleProcessDataLoaderIter

    def __init__(self, options:Union[str, dict], init_model:bool=True):
        super().__init__()
        self.device = get_device()
        self.console = Console()
        
        self._initialize_logger()
        self._initialize_dataset()
        self._initialize_metrics()
        if init_model:
            self._initialize_model()

    @property
    def dirnames(self):
        return [self.logdir]
    
    @property
    def _log_images(self):
        raise True
    
    def _initialize_dataset(self):
        raise NotImplementedError
    
    def _initialize_metrics(self):
        raise NotImplementedError
    
    @console_print("Initializing Logger")
    def _initialize_logger(self) :
        """Initializes logger : job_name and logging directories."""
        self.job_name = self.__class__.__name__.lower()
        self.logdir = os.path.join(self.logging_options.logdir, self.job_name)

        for dirname in self.dirnames :
            os.makedirs(dirname, exist_ok=True)

    def _initialize_model(self):
        self.model.initialize_model()
        self.load_from_train_checkpoint()

    @console_print("Loading model...")
    def load_from_train_checkpoint(self):        

        self.model.load_model()
        self.global_step = self.checkpoint_dict['global_step']

        del self.checkpoint_dict
        
        print(f'Loaded model trained for {self.global_step} steps\n')

    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ Inference ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ #

    @torch.no_grad()
    def run(self) :
        if self.metric_options.compute :
            self.log_metrics(*self.calc_metrics(), save=True, init=True)
        
        if self._log_images :
            self.parse_and_log_images()

    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ Metrics ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ #

    def fn_metrics(self, *args) -> dict:
        raise NotImplementedError
    
    @torch.no_grad()
    def calc_metrics(self) :
        self.console.log("[green]Computing metrics over loader...[/green]")
        progress = Progress(
            *Progress.get_default_columns(), MofNCompleteColumn())

        with progress as self.progress :
            self.task = self.progress.add_task(
                "Images", total=len(self.loader))
            
            agg_metric_dict = []

            for _ in range(len(self.loader)) :
                metric_dict = self.fn_metrics()
                agg_metric_dict.append(metric_dict)
                self.progress.update(self.task, advance=1)

        self.console.log(
            "[green]Finish computing metrics over loader[/green]")
        return (mean_loss_dict(agg_metric_dict, list_output=True), 
                std_loss_dict(agg_metric_dict, list_output=True))
    
    def log_metrics(
            self, mean_metrics:dict, std_metrics:dict, save=True, init=False, **kwargs) :
        
        prompt = f'''
{self.job_name}
Mean : {mean_metrics}
Std : {std_metrics}
'''
        print(prompt)

        if save :
            filename = kwargs.get(
                "filename", 
                os.path.join(self.logdir, f"{self.job_name}_metrics.txt"))
            with open(filename, 'w' if init else 'a') as f :
                f.write(prompt)

    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ Images ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ #

    @console_print("Plotting images")
    def parse_and_log_images(self) :
        raise NotImplementedError