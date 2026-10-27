"""Small actual TRL GRPO compatibility probe, separate from task training."""

import argparse
import hashlib
from pathlib import Path
import time

import torch
from datasets import Dataset
from peft import LoraConfig, PeftModel, get_peft_model
from transformers import AutoModelForCausalLM, AutoTokenizer, set_seed
from trl import GRPOConfig, GRPOTrainer

from ml_training.common import ROOT, environment, read_json, write_json

parser = argparse.ArgumentParser()
parser.add_argument("--device", choices=["cpu", "mps"], default="cpu")
args = parser.parse_args()
set_seed(41)
torch.set_num_threads(4)
record = read_json(ROOT / "configs/models.json")["language_models"]["student"]
source = (
    Path.home()
    / ".cache/huggingface/hub"
    / ("models--" + record["model_id"].replace("/", "--"))
    / "snapshots"
    / record["revision"]
)
output = ROOT / "artifacts/probes" / f"grpo-{args.device}"
output.mkdir(parents=True, exist_ok=True)
start = time.perf_counter()
tokenizer = AutoTokenizer.from_pretrained(source, local_files_only=True)
tokenizer.pad_token = tokenizer.eos_token
tokenizer.padding_side = "left"
base = AutoModelForCausalLM.from_pretrained(
    source, local_files_only=True, dtype=torch.float32, attn_implementation="eager"
)
model = get_peft_model(
    base,
    LoraConfig(
        r=8,
        lora_alpha=16,
        lora_dropout=0.0,
        target_modules=["q_proj", "v_proj"],
        layers_to_transform=list(range(20, 24)),
        task_type="CAUSAL_LM",
    ),
)


def parameter_hash(net):
    result = hashlib.sha256()
    for name, value in net.named_parameters():
        if value.requires_grad:
            result.update(name.encode())
            result.update(value.detach().float().cpu().numpy().tobytes())
    return result.hexdigest()


groups = []


def reward(completions, completion_ids, trainer_state, **kwargs):
    text = [c[0]["content"] if isinstance(c, list) else c for c in completions]
    # Compatibility task only: grade a sampled letter, not query semantics.
    rewards = [float(t.strip().startswith("A")) for t in text]
    groups.append(
        {
            "step": trainer_state.global_step,
            "texts": text,
            "rewards": rewards,
            "completion_tokens": [len(tokens) for tokens in completion_ids],
        }
    )
    return rewards


dataset = Dataset.from_list(
    [
        {"prompt": [{"role": "user", "content": f"Choose either A or B. Respond with just one letter. Trial {i}."}]}
        for i in range(8)
    ]
)
config = GRPOConfig(
    output_dir=str(output / "trainer"),
    per_device_train_batch_size=4,
    num_generations=4,
    max_steps=2,
    max_completion_length=4,
    learning_rate=1e-4,
    temperature=1.5,
    top_p=1.0,
    top_k=0,
    bf16=False,
    fp16=False,
    gradient_checkpointing=False,
    optim="adamw_torch",
    beta=0.0,
    weight_decay=0.0,
    loss_type="grpo",
    report_to="none",
    save_strategy="no",
    logging_steps=1,
    use_cpu=args.device == "cpu",
    dataloader_pin_memory=False,
    seed=41,
    disable_tqdm=True,
)
before = parameter_hash(model)
trainer = GRPOTrainer(model=model, args=config, reward_funcs=reward, train_dataset=dataset, processing_class=tokenizer)
trainer.train()
after = parameter_hash(model)
model.save_pretrained(output / "adapter")
restored_base = AutoModelForCausalLM.from_pretrained(
    source, local_files_only=True, dtype=torch.float32, attn_implementation="eager"
)
restored = PeftModel.from_pretrained(restored_base, output / "adapter", is_trainable=True)
reloaded_hash = parameter_hash(restored)
if after != reloaded_hash:
    raise AssertionError("GRPO probe adapter changed on reload")
write_json(
    output / "probe.json",
    {
        "purpose": "Actual GRPO compatibility probe; not D03 task training",
        "model": record,
        "environment": environment(),
        "device": str(trainer.accelerator.device),
        "optimizer_steps": trainer.state.global_step,
        "before": before,
        "after": after,
        "changed": before != after,
        "reloaded_hash": reloaded_hash,
        "sampled_groups": groups,
        "training_log": trainer.state.log_history,
        "elapsed_seconds": time.perf_counter() - start,
    },
)
if before == after:
    raise AssertionError("No parameter change; inspect constant reward groups before claiming feasibility")
print(output / "probe.json")
