import os
from typing import Any

import torch

from ..logging.misc import console_print
from ..options.head_options import InferenceOptions

class Inference(InferenceOptions):

    model :Any

    def __init__(self) -> None:
        ...

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
    def initialize_logger(self) :
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
