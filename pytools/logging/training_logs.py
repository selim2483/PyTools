import os

from torch.utils.tensorboard import SummaryWriter
from torchvision.utils import save_image

from . import console_print, profiled_function, map_dict
from .images import scale_tensor
from ..options import LoggingOptions
from ..utils.path import generate_unique_path

class TrainingLogs:

    job_name: str
    logging_options: LoggingOptions
    global_step: int

    @profiled_function
    @console_print("Initializing Logger")
    def _initialize_logger(self):
        """Initializes logger : tensorboard, wandb or local.
        """
        print(self.logging_options)
        valid_logger = ["tensorboard", "wandb", "local"]
        if self.logging_options.logger in valid_logger :
            self.mkdirs(self.logging_options.toplogdir, self.job_name)
        
        if self.logging_options.logger == "tensorboard" :
            self.logger = SummaryWriter(log_dir=self.logdir)
        elif self.logging_options.logger == "wandb" :
            # TODO
            # self.logger = wandb.init(
            #     project=self.logging_options.project,
            #     entity=self.logging_options.entity,
            #     name=self.job_name
            # )
            self.logger = None
        elif self.logging_options.logger == "local":
            self.logger = None
        else :
            self.logger = None
            
    def mkdir(self, dir) :
        if not os.path.isdir(dir):
            os.mkdir(dir)

    def mkdirs(self, top_logdir: str, job_name: str) :
        
        self.toplogdir = generate_unique_path(
            os.path.join(top_logdir, job_name))
        self.logdir = os.path.join(self.toplogdir, "logs")
        self.figdir = os.path.join(self.toplogdir, "images")
        self.chkdir = os.path.join(self.toplogdir, "models")

        print("Logging to {}".format(self.toplogdir))

        self.mkdir(self.toplogdir)
        self.mkdir(self.logdir)
        self.mkdir(self.figdir)
        self.mkdir(self.chkdir)

        trainfigdir = os.path.join(self.figdir, "train")
        validfigdir = os.path.join(self.figdir, "valid")

        self.mkdir(trainfigdir)
        self.mkdir(validfigdir)

    def log_metrics_tensorboard(
            self, metric_dict: dict, dir: str = "metrics"):
        return map_dict(
            lambda key, value: self.logger.add_scalar(
                f"{dir}/{key}", value, self.global_step),
            metric_dict
        )
    
    def log_images_tensorboard(self, img_dict: dict):
        map_dict(
            lambda key, value: self.logger.add_image(
                f"images/{key}", 
                scale_tensor(value, in_range=(-1., 1.), out_range=(0., 1.)), 
                self.global_step
            ), 
            img_dict
        )
    
    def log_images_local(self, img_dict: dict):
        def fn(key, value):
            name = f"{self.job_name}_it{self.global_step}_{key}.png"
            img = scale_tensor(
                value, in_range=(-1., 1.), out_range=(0., 1.))
            save_image(img, os.path.join(self.figdir, name))
        map_dict(fn, img_dict)

    def check_image_interval(self, mode:str="local") :
        # Initialization counters
        if not hasattr(self, "image_local_count") :
            self.image_local_count = 0
        if not hasattr(self, "image_logger_count") :
            self.image_logger_count = 0

        # Checks
        if mode=="local" :
            res = (self.global_step 
                   // self.logging_options.image_interval_local)
            return res > self.image_local_count
        elif mode=="logger" :
            res = (self.global_step 
                   // self.logging_options.image_interval_logger)
            return res > self.image_logger_count
        elif mode=="both" :
            res = max(
                (self.global_step 
                 // self.logging_options.image_interval_local 
                 - self.image_local_count), 
                (self.global_step 
                 // self.logging_options.image_interval_logger 
                 - self.image_logger_count))
            return res > 0
    