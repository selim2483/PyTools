import os
import torch

from torch.utils.tensorboard import SummaryWriter
from torchvision.utils import save_image
from torchvision.transforms.functional import to_tensor
from PIL import Image

from . import console_print, profiled_function, map_dict
from .images import scale_tensor
from ..options import LoggingOptions
from ..utils.path import generate_unique_path


def load_image_from_png(path: str):
    return 2 * to_tensor(Image.open(path))[:3] - 1

class TrainingLogs:

    job_name: str
    logging_options: LoggingOptions
    global_step: int
    metric_dict: dict
    data_count: int | None

    @property
    def progress_key(self):
        return self.global_step

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
                f"{dir}/{key}", value, self.progress_key),
            metric_dict
        )
    
    def log_images_tensorboard(self, img_dict: dict, grid: bool = True):
        def fn(key: str, value: torch.Tensor):
            img = scale_tensor(
                value, in_range=(-1., 1.), out_range=(0., 1.))
            if value.ndim == 4:
                if grid:
                    self.logger.add_images(
                        f"images/{key}", img, self.progress_key)
                else:
                    for i, val in enumerate(img):
                        self.logger.add_image(
                            f"images/{key}_{self.data_count + i}", 
                            val, 
                            self.progress_key
                        )
            else:
                self.logger.add_image(f"images/{key}", img, self.progress_key)
        
        map_dict(fn, img_dict)
    
    def log_images_local(self, img_dict: dict, grid: bool = True):
        def fn(key: str, value: torch.Tensor):
            if value.ndim == 4 and not grid:
                for i, val in enumerate(value):
                    name = (f"{self.job_name}_it{self.progress_key}_{key}" 
                            + f"_{self.data_count + i}")
                    torch.save(val, os.path.join(self.figdir, f"{name}.pt"))
                    save_image(
                        scale_tensor(
                            val, in_range=(-1., 1.), out_range=(0., 1.)), 
                        os.path.join(self.figdir, f"{name}.png"), 
                        value_range=(-1., 1.)
                    )
            else:
                name = f"{self.job_name}_it{self.progress_key}_{key}"
                torch.save(value, os.path.join(self.figdir, f"{name}.pt"))
                reload_pt = torch.load(
                    os.path.join(self.figdir, f"{name}.pt"))
                print(key, float(torch.dist(value, reload_pt)))
                save_image(
                    scale_tensor(
                        value, in_range=(-1., 1.), out_range=(0., 1.)), 
                    os.path.join(self.figdir, f"{name}.png"), 
                    value_range=(-1., 1.)
                )
        map_dict(fn, img_dict)
            

    def check_image_interval(self, mode:str="local") :
        if self.progress_key == 0:
            return True  
        if mode=="local" :
            if not hasattr(self, "image_local_count") :
                self.image_local_count = 0
            res = (self.progress_key 
                   // self.logging_options.image_interval_local)
            return res > self.image_local_count
        elif mode=="logger" :
            if not hasattr(self, "image_logger_count") :
                self.image_logger_count = 0
            res = (self.progress_key 
                   // self.logging_options.image_interval_logger)
            return res > self.image_logger_count
        elif mode=="both" :
            return (self.check_image_interval("local") 
                    or self.check_image_interval("logger"))
        
    def update_logger_counters(self):
        self.image_local_count  = (
            self.progress_key // self.logging_options.image_interval_local)
        self.image_logger_count = (
            self.progress_key // self.logging_options.image_interval_logger)
        
    def parse_images(self) -> dict:
        raise NotImplementedError
    
    def parse_and_log_images(self, grid: bool = True):
        if self.check_image_interval("both") :
            img_dict = self.parse_images()
            
            if self.check_image_interval("local") :
                self.log_images_local(img_dict, grid=grid)
            if self.check_image_interval("logger") :
                self.log_images_tensorboard(img_dict, grid=grid)

            self.update_logger_counters()

    def add_metric(self, name: str, value: torch.Tensor):
        if value.numel() == 1:
            self.metric_dict[name] = float(value)
        elif self.data_count is not None:
            for i, val in enumerate(value):
                self.add_metric(f"{name}_{self.data_count + i}", val)
        else: 
            raise ValueError()
    