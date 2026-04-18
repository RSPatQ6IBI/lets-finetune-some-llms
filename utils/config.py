# config.py
from dataclasses import dataclass

@dataclass
class Config:
    # Model
    model_id: str        = "Qwen/Qwen2.5-0.5B-Instruct"
    output_dir: str      = "./qwen_orca_finetuned"

    # Data
    dataset_name: str    = "microsoft/orca-math-word-problems-200k"
    max_samples: int     = 5000      # start small on CPU; increase later
    max_seq_len: int     = 512       # keep short for CPU memory

    # LoRA
    lora_r: int          = 8
    lora_alpha: int      = 16
    lora_dropout: float  = 0.05
    target_modules: list = None      # set in __post_init__

    # Training
    num_epochs: int      = 3
    batch_size: int      = 2         # small batch for CPU
    grad_accum_steps: int = 8        # effective batch = 2 * 8 = 16
    learning_rate: float = 2e-4
    warmup_ratio: float  = 0.03
    weight_decay: float  = 0.01
    logging_steps: int   = 10
    save_steps: int      = 100
    seed: int            = 42

    def __post_init__(self):
        if self.target_modules is None:
            self.target_modules = ["q_proj", "k_proj", "v_proj", "o_proj"]

cfg = Config()