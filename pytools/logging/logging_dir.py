import os

from ..utils.path import generate_unique_path

class LoggingDir :
    """
    Class used in order to create unique logging and figures directories.
    """
    def __init__(self, top_logdir:str) -> None:
        self.top_logdir = top_logdir
        self.mkdir(self.top_logdir)

    def mkdir(self, dir) :
        if not os.path.isdir(dir):
            os.mkdir(dir)

    def mkdirs(self, job_name:str) :
        
        toplogdir = generate_unique_path(
            os.path.join(self.top_logdir, job_name))
        logdir = os.path.join(toplogdir, "logs")
        figdir = os.path.join(toplogdir, "images")
        chkdir = os.path.join(toplogdir, "models")

        print("Logging to {}".format(toplogdir))

        self.mkdir(toplogdir)
        self.mkdir(logdir)
        self.mkdir(figdir)
        self.mkdir(chkdir)

        trainfigdir = os.path.join(figdir, "train")
        validfigdir = os.path.join(figdir, "valid")

        self.mkdir(trainfigdir)
        self.mkdir(validfigdir)

        return logdir, figdir, chkdir