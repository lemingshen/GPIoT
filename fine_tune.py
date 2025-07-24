import os
import torch
import wandb
from datasets import load_from_disk
from trl import SFTTrainer
from peft import LoraConfig, prepare_model_for_kbit_training
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig,
    TrainingArguments,
    pipeline,
    logging,
)

wandb.login(key="your key")

base_model = "meta-llama/Llama-2-13b-chat-hf"
# new_model = "GPIoT_Code_Generation"
new_model = "GPIoT_Task_Decomposition"
# dataset = load_from_disk("dataset/Code_Generation_dataset")
dataset = load_from_disk("dataset/Task_Decomposition_dataset")

compute_dtype = getattr(torch, "float16")

quantization_config = BitsAndBytesConfig(
    load_in_8bit=True,
    bnb_8bit_quant_type="nf8",
    bnb_8bit_compute_dtype=compute_dtype,
    bnb_8bit_use_double_quant=False,
)

model = AutoModelForCausalLM.from_pretrained(
    base_model, quantization_config=quantization_config
)
model.config.use_cache = False
model = prepare_model_for_kbit_training(model)

tokenizer = AutoTokenizer.from_pretrained(base_model)
tokenizer.pad_token = tokenizer.eos_token
tokenizer.padding_side = "right"

peft_parameters = LoraConfig(
    r=64, lora_alpha=16, lora_dropout=0.001, bias="lora_only", task_type="CAUSAL_LM"
)

training_params = TrainingArguments(
    # output_dir="GPIoT_Code_Generation",
    output_dir="GPIoT_Task_Decomposition",
    num_train_epochs=3,
    per_device_train_batch_size=2,
    per_device_eval_batch_size=2,
    gradient_accumulation_steps=4,
    optim="paged_adamw_32bit",
    save_steps=200,
    logging_steps=25,
    learning_rate=2e-4,
    weight_decay=0.001,
    fp16=True,
    bf16=False,
    max_grad_norm=0.3,
    max_steps=-1,
    warmup_ratio=0.03,
    group_by_length=True,
    lr_scheduler_type="cosine",
    report_to="wandb",
)
trainer = SFTTrainer(
    model=model,
    train_dataset=dataset,
    # eval_dataset=dataset["test"],
    peft_config=peft_parameters,
    dataset_text_field="data",
    max_seq_length=None,
    tokenizer=tokenizer,
    args=training_params,
    packing=False,
)

trainer.train()
trainer.model.save_pretrained(new_model)
trainer.tokenizer.save_pretrained(new_model)
