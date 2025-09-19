from mlx_lm import load, generate
from pathlib import Path
import json
from mlx_lm.tuner import TrainingArgs

# SETUP ADAPTER CONFIGURATION
adapter_path = Path("adapters")
adapter_path.mkdir(parents=True, exist_ok=True)

lora_config = {
    "num_layers" : 8,
    "lora_parameters" : {
        "rank" : 8,
        "scale" : 20.0,
        "dropout" : 0.05,
    }
}

with open(adapter_path / "adapter_config.json", "w") as fid:
    json.dump(lora_config, fid, indent=4)

training_args = TrainingArgs(
    adapter_file = adapter_path / "adapters.safetensors",
    iters = 200,
    steps_per_eval = 50
)

# FINE-TUNE BASE MODEL WITH ADAPTER - WORK IN PROGRESS
# - need to create training dataset, select an accessable base model
# model, tokenizer = load("mlx-community/Meta-Llama-3-8B-Instruct", adapter_path = "my-lora-adapters")

# GENERATE TEXT ON BASE MODEL
model, tokenizer = load("mlx-community/Meta-Llama-3-8B-Instruct")
prompt = "Write python code to compute the fibonacci sequence."

messages = [{"role": "user", "content": prompt}]
prompt = tokenizer.apply_chat_template(
    messages, add_generation_prompt=True
)

text = generate(model, tokenizer, prompt=prompt, verbose=True)