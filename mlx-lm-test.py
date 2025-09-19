from mlx_lm import load, generate
from pathlib import Path
import json

model, tokenizer = load("mlx-community/Meta-Llama-3-8B-Instruct")
prompt = "Write python code to compute the fibonacci sequence."

messages = [{"role": "user", "content": prompt}]
prompt = tokenizer.apply_chat_template(
    messages, add_generation_prompt=True
)

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

text = generate(model, tokenizer, prompt=prompt, verbose=True)