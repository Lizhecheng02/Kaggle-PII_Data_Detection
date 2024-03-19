import os
import torch
import warnings
import pandas as pd
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig,
    TrainingArguments,
    pipeline,
    TextStreamer
)
from peft import LoraConfig, PeftModel, prepare_model_for_kbit_training, get_peft_model
from datasets import Dataset, load_dataset
from trl import SFTTrainer
from sklearn.model_selection import train_test_split

df = pd.read_csv("llm_train.csv")
train_data, test_data = train_test_split(df, test_size=0.15, random_state=1)
train_dataset = Dataset.from_pandas(train_data)
test_dataset = Dataset.from_pandas(test_data)
print(train_dataset)
print(test_dataset)

base_model = "mistralai/Mistral-7B-Instruct-v0.2"
new_model = "Finetuned-PIIDD-Mistral-7B"

bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype=torch.bfloat16,
    bnb_4bit_use_double_quant=False
)

model = AutoModelForCausalLM.from_pretrained(
    base_model,
    quantization_config=bnb_config,
    load_in_4bit=True,
    torch_dtype=torch.bfloat16,
    device_map="auto",
    trust_remote_code=True
)
model.config.use_cache = False
model.config.pretraining_tp = 1
model.gradient_checkpointing_enable()

tokenizer = AutoTokenizer.from_pretrained(base_model, trust_remote_code=True)
tokenizer.pad_token = tokenizer.eos_token
tokenizer.add_eos_token = True

model = prepare_model_for_kbit_training(model=model)
peft_config = LoraConfig(
    r=64,
    lora_alpha=16,
    lora_dropout=0.05,
    bias="none",
    task_type="CAUSAL_LM",
    target_modules=[
        "q_proj", "k_proj", "v_proj",
        "o_proj", "gate_proj"
    ]
)
model = get_peft_model(model=model, peft_config=peft_config)
print(model)

training_arguments = TrainingArguments(
    output_dir="output",
    num_train_epochs=2,
    per_device_train_batch_size=1,
    gradient_accumulation_steps=8,
    per_device_eval_batch_size=1,
    optim="paged_adamw_8bit",
    save_steps=50,
    logging_steps=25,
    eval_steps=50,
    learning_rate=2e-4,
    weight_decay=0.001,
    fp16=False,
    bf16=False,
    max_grad_norm=1.0,
    max_steps=-1,
    warmup_ratio=0.05,
    group_by_length=True,
    lr_scheduler_type="cosine",
    report_to="none"
)

trainer = SFTTrainer(
    model=model,
    train_dataset=train_dataset,
    eval_dataset=test_dataset,
    peft_config=peft_config,
    max_seq_length=None,
    dataset_text_field="text",
    tokenizer=tokenizer,
    args=training_arguments,
    packing=False
)

trainer.train()
trainer.model.save_pretrained(new_model)
model.config.use_cache = True
model.eval()
