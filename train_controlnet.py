import copy
import json
import os
import subprocess
import sys 
import select

import nibabel as nib
import numpy as np
from monai.config import print_config
from monai.data import create_test_image_3d
from scripts.diff_model_setting import setup_logging
from IPython.display import Image, display

print_config()

logger = setup_logging("notebook")


env_config_path = "./configs/environment_maisi_controlnet_train.json"
train_config_path = "./configs/config_maisi_controlnet_train.json"
model_def_path = "./configs/config_maisi.json"


with open(env_config_path, "r") as f:
    env_config = json.load(f)

with open(train_config_path, "r") as f:
    train_config = json.load(f)

with open(model_def_path, "r") as f:
    model_def = json.load(f)

env_config_out = copy.deepcopy(env_config)
train_config_out = copy.deepcopy(train_config)
model_def_out = copy.deepcopy(model_def)



os.makedirs(env_config_out["model_dir"], exist_ok=True)
os.makedirs(env_config_out["output_dir"], exist_ok=True)
os.makedirs(env_config_out["tfevent_path"], exist_ok=True)


env_config_filepath = env_config_path
train_config_filepath = train_config_path
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

    # Print the output in real-time
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
    finally:
        # Capture and print any remaining output
        stdout, stderr = process.communicate()
        print(stdout)
        if stderr:
            print(stderr)
    return



### 1. train
num_gpus = 8
logger.info(f"number of GPUs: {num_gpus}.")
logger.info("Training the model...")

# Define the arguments for torchrun
module = "scripts.train_controlnet_self"
module_args = [
    "--environment-file",
    env_config_filepath,
    "--config-file",
    model_def_filepath,
    "--training-config",
    train_config_filepath,
    "--gpus",
    str(num_gpus),
]

run_torchrun(module, module_args, num_gpus=num_gpus)



### 2. infer
# logger.info("Inference...")

# module = "scripts.infer_controlnet"
# module_args = [
#     "--environment-file",
#     env_config_filepath,
#     "--config-file",
#     model_def_filepath,
#     "--training-config",
#     train_config_filepath,
# ]

# run_torchrun(module, module_args, num_gpus=num_gpus)