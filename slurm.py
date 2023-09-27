import os
import subprocess
import argparse
import yaml


class BatchJobLauncher:
    def __init__(
        self, pythonfile: str, configfile: str, sbatchfile: str) :
        """
        Args:
            pythonfile (str): path to the python file to launch (.py)
            configfile (str): path to the config file to use (.yml)
            sbatchfile (str): path to the sbatch configuration file to use 
                (.yml)
        """

        self.pythonfile = pythonfile
        self.configfile = configfile

        self.get_sbatch_options(sbatchfile=sbatchfile)

        # Unsure that the log directory exists
        os.system(f"mkdir -p {self.logdir}")

        self.topdir = os.path.dirname(os.path.abspath(__file__))
        self.pydir = os.path.dirname(os.path.abspath(self.pythonfile))

    def get_sbatch_options(self, sbatchfile: str):

        if not os.path.exists(sbatchfile):
            raise Exception(
                "Sbatch file not found please try again with a good path")

        with open(sbatchfile, "r") as ymlfile:
            cfg:dict = yaml.load(ymlfile, Loader=yaml.CFullLoader)

        self.job_name      = cfg.get("JOB_NAME", "sbatch")
        self.ntasks        = cfg.get("NTASKS", 1)
        self.cpus_per_task = cfg.get("CPUS_PER_TASK", 1)
        self.qos_name      = cfg.get("QOS", "co_long_gpu")
        self.time          = cfg.get("TIME", "96:00:00")
        self.logdir        = cfg.get("LOGDIR", "../logslurm")
        self.exclude       = cfg.get("EXCLUDE", "")

    def launch_job(self):
        # Perform verifications
        self.commit_id = self.git_verif()
        self.files_verif()
        self.verif()

        job_script = self.make_job()
        self.submit_job(job_script)

    def git_verif(self):
        """
        Perform git verifications : raises RuntimeError exception if there are
        modifications not staged or not commited.

        Raises:
            RuntimeError:
            You must stage and commit every modification before submission

        Returns:
            the latest commit_id.
        """
        result = int(
            subprocess.run(
                ("expr $(git diff --name-only | wc -l)"
                 + " + $(git diff --name-only --cached | wc -l)"),
                shell=True,
                stdout=subprocess.PIPE,
            ).stdout.decode()
        )

        if result > 0:
            print(
                f"We found {result} modifications either not staged or not \
                    commited")
            raise RuntimeError(
                "You must stage and commit every modification before \
                    submission")

        commit_id = subprocess.check_output(
            "git log --pretty=format:'%H' -n 1", shell=True).decode()

        return commit_id

    def files_verif(self):
        # Check if the files exist
        if not os.path.exists(self.pythonfile):
            raise Exception(
                "Python file not found please try again with a good path")
        if not os.path.exists(self.configfile):
            raise Exception(
                "Config file not found please try again with a good path")

        # Check if the code needs a package installation
        if os.path.exists(os.path.join(self.pydir, "setup.py")):
            self.package = f"pip install -e {self.pydir}"
        else:
            self.package = ""

    def verif(self):
        """
        Other verifications to perform before submit the job to slurm.
        These are specific to your application, therefore, this method needs
        to be overrided in the child class dedicated to your application.
        """
        return

    def add_bash(self):
        """
        Other bash instruction to perform before launching training.
        These are specific to your application, therefore, this method needs
        to be overrided in the child class dedicated to your application.
        """
        return ""

    def add_args(self):
        """
        Other arguments required for your application.
        These are specific to your application, therefore, this method needs
        to be overrided in the child class dedicated to your application.
        """
        return ""
    
    def environnement(self) :
        return ""

    def make_job(self):
        return f"""#!/bin/bash 

#SBATCH --job-name={self.job_name}
#SBATCH --ntasks={self.ntasks}
#SBATCH --cpus-per-task={self.cpus_per_task}
#SBATCH --time={self.time}
#SBATCH --qos={self.qos_name}
#SBATCH --output={self.logdir}/{self.job_name}-%A_%a.out
#SBATCH --error={self.logdir}/{self.job_name}-%A_%a.err
#SBATCH --exclude={self.exclude}

current_dir=`pwd`

echo "Session " {self.job_name}_${{SLURM_ARRAY_JOB_ID}}_${{SLURM_ARRAY_TASK_ID}}

job_name={self.job_name}_${{SLURM_ARRAY_JOB_ID}}_${{SLURM_ARRAY_TASK_ID}}

echo "\nCopying the source directory and data"
date
mkdir $TMPDIR/{self.job_name}
mkdir $TMPDIR/{self.job_name}/code
rsync -r code/ $TMPDIR/{self.job_name}/code 
rsync -r {self.configfile} $TMPDIR/{self.job_name}/

echo "\nChecking out the correct version of the code commit_id {self.commit_id}"
cd $TMPDIR/{self.job_name}/
git checkout {self.commit_id}

echo "\nSetting up environnment and dependencies"
{self.environnement()}
python --version

{self.package}

{self.add_bash()}

echo "\nStarting computation..."
nvidia-smi -q | grep "CUDA Version"

python {self.pythonfile} --config {self.configfile} --job_name $job_name 

if [[ $? != 0 ]]; then
    exit -1
fi
"""

    def submit_job(self, job_script: str):
        with open("job.sbatch", "w") as file:
            file.write(job_script)
        os.system("sbatch job.sbatch")

class PipBatchJobLauncher(BatchJobLauncher) :

    def __init__(
            self, 
            pythonfile: str, 
            configfile: str, 
            sbatchfile: str, 
            requirements_file: str
        ):
        """
        Args:
            pythonfile (str): path to the python file to launch (.py)
            configfile (str): path to the config file to use (.yml)
            sbatchfile (str): path to the sbatch configuration file to use 
                (.yml)
            requirements_file (str): path to the requirement file to use 
                (.txt)
        """
        super().__init__(pythonfile, configfile, sbatchfile)
        self.requirements_file = requirements_file

    def environnement(self) :
        if os.path.exists(self.requirements_file) :
            return f"""echo "\\nSetting up the virtual environment"
python3 -m pip install virtualenv --user
virtualenv -p python3 venv
source venv/bin/activate
pip3 install torch torchvision --extra-index-url https://download.pytorch.org/whl/cu116
pip3 install -r {self.requirements_file}"""
        else :
            raise Exception(
                "Requirement file not found please try again with a good path"
            )
        
class CondaBatchJobLauncher(BatchJobLauncher) :
    """Batch launcher using conda/miniconda package management.
    The conda environnement is supposed to already exist and be available with
    all the required packages.
    The environnement variable $CONDA_DIR is supposed to have been exported
    earlier. (ex : export CONDA_DIR=/stck/sollivie/miniconda3)
    """

    def __init__(
            self, 
            pythonfile: str, 
            configfile: str, 
            sbatchfile: str, 
            conda_env: str,
        ):
        """
        Args:
            pythonfile (str): path to the python file to launch (.py)
            configfile (str): path to the config file to use (.yml)
            sbatchfile (str): path to the sbatch configuration file to use 
                (.yml)
            conda_env (str): name of the conda environnement to use
        """
        super().__init__(pythonfile, configfile, sbatchfile)
        self.conda_env = conda_env

    def environnement(self) :
        return f"""
conda activate {self.conda_env}
export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:$CONDA_PREFIX/lib/"""

def parse_args() :
    parser = argparse.ArgumentParser()
    parser.add_argument("python", type=str)
    parser.add_argument("config", type=str)
    parser.add_argument("sbatch", type=str)

    subparsers = parser.add_subparsers(dest="packagemanager")

    # Pip
    pip = subparsers.add_parser("pip")
    pip.add_argument("requirements", type=str)

    # Conda
    conda = subparsers.add_parser("conda")
    conda.add_argument("--env", type=str)

    return parser.parse_args()

def parse_and_launch() :
    args = parse_args()

    if args.packagemanager == "pip":
        launcher = PipBatchJobLauncher(
            pythonfile        = args.python,
            configfile        = args.config,
            sbatchfile        = args.sbatch,
            requirements_file = args.requirements,
        )
    elif args.packagemanager == "conda":
        launcher = CondaBatchJobLauncher(
            pythonfile = args.python,
            configfile = args.config,
            sbatchfile = args.sbatch,
            conda_env  = args.env,
        )

    launcher.launch_job()