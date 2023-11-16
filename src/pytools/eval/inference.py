import os
from typing import Any, Iterable, Union

from rich.console import Console
from rich.progress import Progress, MofNCompleteColumn
import torch
from torch.utils.data import DataLoader

from .utils import LoggingTools
from ..logging.misc import concatenate_loss_dict, console_print, mean_loss_dict, std_loss_dict
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
    loader :DataLoader

    def __init__(self, options:Union[str, dict]):
        super().__init__(options)
        self.device = get_device()
        self.console = Console()
        
        self._initialize_logger()
        self._initialize_dataset()
        self._initialize_metrics()
        if hasattr(self, "checkpoint_path"):
            self._initialize_model()

    @property
    def dirnames(self):
        dirnames = [self.logdir]
        if self.logging_options.reconstruction:
            dirnames.append(os.path.join(self.logdir, "reconstruction"))
        if self.logging_options.fft2D:
            dirnames.append(os.path.join(self.logdir, "fft2D"))
        if self.logging_options.fft_rad:
            dirnames.append(os.path.join(self.logdir, "fft_rad"))

        return dirnames
    
    @property
    def _log_images(self):
        return len(self.dirnames) > 1
    
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
        self.global_step = self.checkpoint_dict['global_step']

        del self.checkpoint_dict
        
        print(f'Loaded model trained for {self.global_step} steps\n')

    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ Inference ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ #

    @torch.no_grad()
    def infer(self) :
        if self.metric_options.compute :
            self.log_metrics(*self.calc_metrics(), save=True, init=True)
        
        if self._log_images :
            self.parse_and_log_images()

    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ Comparison ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ #

    @torch.no.grad()
    def compare(self):
        self.init_images_compare()
        names, mean_agg_dict, std_agg_dict = [], [], []
        while True:
            try :
                self._initialize_model()
            except StopIteration :
                break

            # Metrics
            if self.metric_options.compute :
                mean_metrics, std_metrics = self.calc_metrics()
                mean_agg_dict.append(mean_metrics)
                std_agg_dict.append(std_metrics)
                self.log_metrics(
                    mean_metrics, std_metrics,
                    save=True, 
                    init=(len(names)==0),
                    filename=os.path.join(self.logdir, "metrics.txt"))
            
            self.parse_images_compare()
            names.append(self.name)

        # Graphs
        if self.metric_options.compute :
            if self.logging_options.bar_graphs :
                self.plot_graphs(mean_agg_dict, names, type="bar")
            if self.logging_options.line_graphs :
                self.plot_graphs(mean_agg_dict, names, type="line")

        # Images
        self.log_images_compare()

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

            self.iter_loader = iter(self.loader)
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

    def plot_graphs(
            self, 
            mean_agg_dict :Iterable[dict], 
            names         :Iterable[str], 
            **kwargs
    ) :
        mean_dict = concatenate_loss_dict(mean_agg_dict)
        logtools = LoggingTools(
            self.logging_options.groups, self.logging_options.infos)
        for group_name in logtools.groups.keys() :
            logtools.plot_comparison_graph(
                mean_dict    = mean_dict, 
                group_name   = group_name, 
                names        = names,
                logdir       = self.logdir,
                group_scales = (group_name!="main"), 
                **kwargs)  

    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ Images ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ #

    @console_print("Plotting images")
    def parse_and_log_images(self) :
        raise NotImplementedError

    def init_images_compare(self):
        raise NotImplementedError
    
    def parse_images_compare(self):
        raise NotImplementedError
    
    def log_images_compare(self):
        raise NotImplementedError
    