
import os
import tomli as tomlib
from datasets import load_dataset
from dotenv import load_dotenv

with open("pyproject.toml", "rb") as f:
    data = tomlib.load(f)

orca_db_ = data["dataset-orca"]["orca_math_dataset_name"]
orca_db_local_path_ = data["dataset-orca"]["orca_math_dataset_path"]
load_dotenv()

dataset = load_dataset(orca_db_, token= os.getenv("HF_TOKEN"), download_mode="force_redownload")
dataset.save_to_disk(orca_db_local_path_)

print('dataset has been downloaded to local')

print(f"Train dataset size: {len(dataset['train'])}")

