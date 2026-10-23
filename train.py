from scripts.diff_model_setting import setup_logging
import copy
import os
import json
import numpy as np
import nibabel as nib
import subprocess
import sys 

import select
from monai.config import print_config

from IPython.display import display
import warnings

warnings.filterwarnings("ignore")

print_config()

logger = setup_logging("notebook")


env_config_path = "./configs/flow/environment_flow_matching_model.json"
model_config_path = "./configs/flow/config_flow_matching_train.json"
model_def_path = "./configs/flow/config_flow.json"


with open(env_config_path, "r") as f:
    env_config = json.load(f)

with open(model_config_path, "r") as f:
    model_config = json.load(f)

with open(model_def_path, "r") as f:
    model_def = json.load(f)

env_config_out = copy.deepcopy(env_config)
model_config_out = copy.deepcopy(model_config)
model_def_out = copy.deepcopy(model_def)





os.makedirs(env_config_out["model_dir"], exist_ok=True)
os.makedirs(env_config_out["output_dir"], exist_ok=True)



env_config_filepath = env_config_path

model_config_filepath = model_config_path

model_def_filepath = model_def_path




def run_torchrun(module, module_args, num_gpus=1):
    # Define the arguments for torchrun
    num_nodes = 1

    # Build the torchrun command
    torchrun_command = [
        "torchrun",
        "--nproc_per_node",
        str(num_gpus),
        "--nnodes",
        str(num_nodes),
        "-m",
        module,
    ] + module_args

    # Set the OMP_NUM_THREADS environment variable
    env = os.environ.copy()
    env["OMP_NUM_THREADS"] = "1"

    # Execute the command
    process = subprocess.Popen(torchrun_command, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True, env=env)

    try:
        while True:
            reads = [process.stdout.fileno(), process.stderr.fileno()]
            ret = select.select(reads, [], [], 1.0)
            for fd in ret[0]:
                if fd == process.stdout.fileno():
                    output = process.stdout.readline()
                    if output:
                        print(output.strip())
                elif fd == process.stderr.fileno():
                    error = process.stderr.readline()
                    if error:
                        print(error.strip(), file=sys.stderr)
            if process.poll() is not None:
                break
    except Exception as e:
        print(f"An error occurred: {e}")
    finally:
        # Capture and print any remaining output
        stdout, stderr = process.communicate()
        print(stdout)
        if stderr:
            print(stderr)
    return







num_gpus = 8
logger.info(f"number of GPUs: {num_gpus}.")

logger.info("Training the model...")



module = "scripts.train_model"
module_args = [
    "--env_config",
    env_config_filepath,
    "--model_config",
    model_config_filepath,
    "--model_def",
    model_def_filepath,
    "--num_gpus",
    str(num_gpus),
]

run_torchrun(module, module_args, num_gpus=num_gpus)