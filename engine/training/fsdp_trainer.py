from __future__ import annotations

import argparse
import os
from dataclasses import dataclass

import torch
import torch.distributed as dist
from accelerate import Accelerator
from datasets import load_dataset
from peft import LoraConfig, get_peft_model
from transformers import AutoModelForCausalLM, AutoTokenizer, DataCollatorForLanguageModeling


@dataclass(frozen=True)
class TrainingConfig:
    model: str
    dataset: str
    output: str
    max_length: int = 2048
    batch_size: int = 1
    gradient_accumulation: int = 8
    learning_rate: float = 2e-4
    epochs: int = 1
    qlora: bool = False


def build_model(config: TrainingConfig) -> tuple[AutoModelForCausalLM, AutoTokenizer]:
    tokenizer = AutoTokenizer.from_pretrained(config.model, token=os.getenv("HF_TOKEN"))
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    model_kwargs: dict[str, object] = {"torch_dtype": torch.bfloat16}
    if config.qlora:
        model_kwargs["load_in_4bit"] = True
        model_kwargs["bnb_4bit_quant_type"] = "nf4"
        model_kwargs["bnb_4bit_compute_dtype"] = torch.bfloat16
        model_kwargs["bnb_4bit_use_double_quant"] = True
    model = AutoModelForCausalLM.from_pretrained(config.model, **model_kwargs)
    lora = LoraConfig(
        r=16, lora_alpha=32, lora_dropout=0.05, bias="none",
        task_type="CAUSAL_LM", target_modules=["q_proj", "k_proj", "v_proj", "o_proj"],
    )
    return get_peft_model(model, lora), tokenizer


def train(config: TrainingConfig) -> None:
    accelerator = Accelerator(
        gradient_accumulation_steps=config.gradient_accumulation,
        mixed_precision="bf16",
    )
    model, tokenizer = build_model(config)
    dataset = load_dataset(config.dataset, split="train")
    text_column = "text" if "text" in dataset.column_names else dataset.column_names[0]

    def tokenize(batch: dict[str, list[str]]) -> dict[str, list[list[int]]]:
        return tokenizer(batch[text_column], truncation=True, max_length=config.max_length)

    tokenized = dataset.map(tokenize, batched=True, remove_columns=dataset.column_names)
    collator = DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=False)
    loader = torch.utils.data.DataLoader(tokenized, batch_size=config.batch_size, collate_fn=collator)
    optimizer = torch.optim.AdamW((p for p in model.parameters() if p.requires_grad), lr=config.learning_rate)
    model, optimizer, loader = accelerator.prepare(model, optimizer, loader)
    model.train()
    for _ in range(config.epochs):
        for batch in loader:
            with accelerator.accumulate(model):
                output = model(**batch)
                accelerator.backward(output.loss)
                optimizer.step()
                optimizer.zero_grad(set_to_none=True)
    accelerator.wait_for_everyone()
    if accelerator.is_main_process:
        unwrapped = accelerator.unwrap_model(model)
        unwrapped.save_pretrained(config.output)
        tokenizer.save_pretrained(config.output)
    if dist.is_available() and dist.is_initialized():
        dist.destroy_process_group()


def main() -> None:
    parser = argparse.ArgumentParser(description="LoRA/QLoRA distributed fine-tuning")
    parser.add_argument("--model", required=True)
    parser.add_argument("--dataset", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--qlora", action="store_true")
    args = parser.parse_args()
    train(TrainingConfig(args.model, args.dataset, args.output, qlora=args.qlora))


if __name__ == "__main__":
    main()
