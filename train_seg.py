import copy
import json
import os
import subprocess
import sys 
import select



env_config_path = "./configs/seg/environment_maisi_seg_train.json"
train_config_path = "./configs/seg/config_maisi_seg_train.json"
model_def_path = "./configs/seg/config_maisi.json"

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
os.makedirs(env_config_out["cache_path"], exist_ok=True)
os.makedirs(env_config_out["valid_path"], exist_ok=True)
os.makedirs(env_config_out["log_path"], exist_ok=True)


env_config_filepath = env_config_path
train_config_filepath = train_config_path
model_def_filepath = model_def_path




def run_torchrun(module, module_args, num_gpus=1,master_port="29500"):
    num_nodes = 1

    torchrun_command = [
        "torchrun",
        "--nproc_per_node",
        str(num_gpus),
        "--nnodes",
        str(num_nodes),
        "--master_port", master_port,
        "-m",
        module,
    ] + module_args

    env = os.environ.copy()
    env["OMP_NUM_THREADS"] = "1"

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
    finally:
        stdout, stderr = process.communicate()
        print(stdout)
        if stderr:
            print(stderr)
    return




num_gpus = 4

module = "scripts.seg_train_maisi"
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

run_torchrun(module, module_args, num_gpus=num_gpus,master_port="29500")



### infer
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