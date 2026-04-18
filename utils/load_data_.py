import os
import numpy as np
import tomli as tomlib
from datasets import load_from_disk
from dotenv import load_dotenv
import pandas as pd
import pyarrow as pa
from pathlib import Path 
from sklearn.model_selection import train_test_split


# ── Optional: PyTorch (only needed for image/folder datasets) ──
try:
    import torch
    from torch.utils.data import DataLoader, Subset, random_split
    from torchvision import datasets, transforms
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False
 


def get_data_columns_():
    the_cwd_ = Path(os.getcwd()).as_posix()
    the_cwd_ = the_cwd_.split('utils')[0]
    project_file_ = os.path.join(the_cwd_, 'pyproject.toml')
    print(project_file_)
    
    with open(project_file_, "rb") as f:
        data = tomlib.load(f)
    orca_db_local_path_ = data["dataset-orca"]["orca_math_dataset_path"]
    orca_data_path_ = os.path.join(the_cwd_, orca_db_local_path_,'train/')
    orca_data_path_ = orca_data_path_.replace('\\', '/')
    #------------##----------#----
    dataset_orca = load_from_disk(orca_data_path_)
    # Access the underlying pyarrow table
    arrow_table = dataset_orca.data.table
    dframe = arrow_table.to_pandas()
    col_names_ = dframe.columns
    ques_col_ = dframe[col_names_[0]]
    ans_col_ = dframe[col_names_[1]]
    return ques_col_, ans_col_, dataset_orca


"""
data_partitioning.py
====================
Train / Validation / Test splitting for custom datasets.
""" 
# =============================================================
# CONFIG — adjust these to your needs
# =============================================================
RANDOM_SEED  = 42
TRAIN_RATIO  = 0.70   # 70% train
VAL_RATIO    = 0.15   # 15% validation
TEST_RATIO   = 0.15   # 15% test
# Ratios must sum to 1.0
assert abs(TRAIN_RATIO + VAL_RATIO + TEST_RATIO - 1.0) < 1e-6, "Ratios must sum to 1.0"
 
 
def split_numpy_or_dataframe(X, y=None, stratify=False, verbose=True):
    strat_col = y if (stratify and y is not None) else None
 
    # First split: train vs (val + test)
    X_train, X_temp, y_train, y_temp = train_test_split(
        X, y if y is not None else np.zeros(len(X)),
        test_size=(VAL_RATIO + TEST_RATIO),
        random_state=RANDOM_SEED,
        stratify=strat_col
    )
 
    # Second split: val vs test (from the temp portion)
    val_relative = VAL_RATIO / (VAL_RATIO + TEST_RATIO)
    strat_temp   = y_temp if (stratify and y is not None) else None
 
    X_val, X_test, y_val, y_test = train_test_split(
        X_temp, y_temp,
        test_size=(1 - val_relative),
        random_state=RANDOM_SEED,
        stratify=strat_temp
    )
 
    if verbose:
        total = len(X)
        print(f"Total samples : {total}")
        print(f"  Train       : {len(X_train):>6}  ({len(X_train)/total*100:.1f}%)")
        print(f"  Validation  : {len(X_val):>6}  ({len(X_val)/total*100:.1f}%)")
        print(f"  Test        : {len(X_test):>6}  ({len(X_test)/total*100:.1f}%)")
 
    result = {"X_train": X_train, "X_val": X_val, "X_test": X_test}
    if y is not None:
        result.update({"y_train": y_train, "y_val": y_val, "y_test": y_test})
    return result
 

# ________------------_________# 
# TEST RUN -- >> 
# data_ques_, data_ans_ = get_data_columns_()
# res_ = split_numpy_or_dataframe(X=data_ques_, y=data_ans_, stratify=False, verbose=True)
# print(len(res_['X_train']), len(res_['X_val']))
# print( len(res_['X_train']), len(res_['y_train']) )