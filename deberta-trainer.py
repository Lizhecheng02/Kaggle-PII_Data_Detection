import json
import torch
from transformers import (
    AutoTokenizer,
    Trainer,
    TrainingArguments,
    AutoModelForTokenClassification,
    DataCollatorForTokenClassification
)
import evaluate
import numpy as np
from datasets import Dataset, features
from itertools import chain
from functools import partial
from seqeval.metrics import recall_score, f1_score, precision_score


class CFG:
    MAX_LENGTH = 512
    MODEL_NAME = "microsoft/deberta-v3-base"
    FREEZE_EMBEDDINGS = False
    FREEZE_LAYERS = 0
    VER = 1
    OUTPUT_DIR = f"Model-{VER}"
    TRAIN_FILE_PATH = ""


print("... Load Dataset ...")
data = json.load(open(CFG.TRAIN_FILE_PATH))
print(len(data))
print(data[0].keys())

print("... Construct Label Id Dict ...")
all_labels = sorted(list(set(chain(*[x["labels"] for x in data]))))
label2id = {l: i for i, l in enumerate(all_labels)}
id2label = {v: k for k, v in label2id.items()}
print(id2label)

target = [
    "B-EMAIL", "B-ID_NUM", "B-NAME_STUDENT", "B-PHONE_NUM",
    "B-STREET_ADDRESS", "B-URL_PERSONAL", "B-USERNAME", "I-ID_NUM",
    "I-NAME_STUDENT", "I-PHONE_NUM", "I-STREET_ADDRESS", "I-URL_PERSONAL"
]


def tokenize(example, tokenizer, label2id):
    text = []
    labels = []
    targets = []

    for t, l, ws in zip(example["tokens"], example["provided_labels"], example["trailing_whitespace"]):
        text.append(t)
        labels.extend([l] * len(t))

        if l in target:
            targets.append(1)
        else:
            targets.append(0)
        if ws:
            text.append(" ")
            labels.append("O")

    tokenized = tokenizer(
        "".join(text),
        return_offsets_mapping=True,
        truncation=True,
        max_length=CFG.MAX_LENGTH
    )

    target_num = sum(targets)
    labels = np.array(labels)

    text = "".join(text)
    token_labels = []

    for start_idx, end_idx in tokenized.offset_mapping:
        if start_idx == 0 and end_idx == 0:
            token_labels.append(label2id["O"])
            continue

        if text[start_idx].isspace():
            start_idx += 1

        token_labels.append(label2id[labels[start_idx]])

    length = len(tokenized.input_ids)

    return {
        **tokenized,
        "labels": token_labels,
        "length": length,
        "target_num": target_num,
        "group": 1 if target_num > 0 else 0
    }


print("... Load LLM Tokenizer ...")
tokenizer = AutoTokenizer.from_pretrained(CFG.MODEL_NAME)

print("... Reconstruct Data ...")
ds = Dataset.from_dict({
    "full_text": [x["full_text"] for x in data],
    "document": [x["document"] for x in data],
    "tokens": [x["tokens"] for x in data],
    "trailing_whitespace": [x["trailing_whitespace"] for x in data],
    "provided_labels": [x["labels"] for x in data],
})

ds = ds.map(
    tokenize,
    fn_kwargs={
        "tokenizer": tokenizer,
        "label2id": label2id
    },
    num_proc=2
)
ds = ds.class_encode_column("group")

print("... Print Dataset ...")
x = ds[0]
for t, l in zip(x["tokens"], x["provided_labels"]):
    if l != "O":
        print((t, l))

print("*" * 20)
for t, l in zip(tokenizer.convert_ids_to_tokens(x["input_ids"]), x["labels"]):
    if id2label[l] != "O":
        print((t, id2label[l]))


def compute_metrics(p, all_labels):
    predictions, labels = p
    predictions = np.argmax(predictions, axis=2)

    true_predictions = [
        [all_labels[p] for (p, l) in zip(prediction, label) if l != -100]
        for prediction, label in zip(predictions, labels)
    ]
    true_labels = [
        [all_labels[l] for (p, l) in zip(prediction, label) if l != -100]
        for prediction, label in zip(predictions, labels)
    ]

    recall = recall_score(true_labels, true_predictions)
    precision = precision_score(true_labels, true_predictions)
    f1_score = (1 + 5 * 5) * recall * precision / (5 * 5 * precision + recall)

    results = {
        "recall": recall,
        "precision": precision,
        "f1": f1_score
    }
    return results


print("... Load LLM Model ...")
model = AutoModelForTokenClassification.from_pretrained(
    CFG.MODEL_NAME,
    num_labels=len(all_labels),
    id2label=id2label,
    label2id=label2id,
    ignore_mismatched_sizes=True
)
collator = DataCollatorForTokenClassification(tokenizer, pad_to_multiple_of=16)

print("... Freeze ...")
if CFG.FREEZE_EMBEDDINGS:
    print("Freezing embeddings.")
    for param in model.deberta.embeddings.parameters():
        param.requires_grad = False

if CFG.FREEZE_LAYERS > 0:
    print(f"Freezing {CFG.FREEZE_LAYERS} layers.")
    for layer in model.deberta.encoder.layer[:CFG.FREEZE_LAYERS]:
        for param in layer.parameters():
            param.requires_grad = False

final_ds = ds.train_test_split(test_size=0.2, seed=42)
final_ds

print("... Training ...")
args = TrainingArguments(
    output_dir=CFG.OUTPUT_DIR,
    fp16=True,
    learning_rate=2e-5,
    num_train_epochs=1,
    per_device_train_batch_size=4,
    per_device_eval_batch_size=4,
    report_to="none",
    evaluation_strategy="epoch",
    save_strategy="epoch",
    save_total_limit=3,
    overwrite_output_dir=True,
    load_best_model_at_end=True,
    lr_scheduler_type="cosine",
    metric_for_best_model="f1",
    greater_is_better=True,
    weight_decay=0.001
)
trainer = Trainer(
    model=model,
    args=args,
    train_dataset=final_ds["train"],
    eval_dataset=final_ds["test"],
    data_collator=collator,
    tokenizer=tokenizer,
    compute_metrics=partial(compute_metrics, all_labels=all_labels),
)
trainer.train()

print("... Save Model ...")
trainer.save_model(CFG.OUTPUT_DIR)
torch.cuda.empty_cache()
