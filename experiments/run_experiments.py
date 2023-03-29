from distutils.util import execute
import json
import subprocess

PROMPT_PATH = "experiments/resources/prompts.json"
PYTHON_FILE_PATH = "src\main.py"
EXPERIMENT_NAME = "mm-cot input fine-tuning" 
DATA_RANGE = ",500"

with open(PROMPT_PATH, "r") as f:
    prompts = json.loads(f.read())["prompts"]

args = [
    "--user_msg",
    "answer",
    "--output_len",
    "64",
    "--final_eval",
    "--prompt_format",
    "QCMG-A",
    "--evaluate_dir",
    "models/MM-CoT-UnifiedQA-base-Answer",
    "--task",
    "EVALUATE",
    "--dataset",
    "FAKEDDIT",
    "--data_range",
    DATA_RANGE,
    "--experiment_name",
    EXPERIMENT_NAME
]

image_run = ["--img_type", "cooelf_detr"]
image_rationale_run = image_run + ["--test_le", "data/fakeddit/partial/rationales/rationales.json"]

for prompt in prompts:
    for run in ( [], image_run, image_rationale_run):
        script_args = args + run
        script_args += ["--prompt", prompt]
        cmd = ["python", PYTHON_FILE_PATH] + script_args
        subprocess.run(cmd)