import json
import pandas as pd
import numpy as np
from itertools import chain
from datasets import Dataset
from pathlib import Path
from transformers import (
    AutoTokenizer,
    Trainer,
    TrainingArguments,
    AutoModelForTokenClassification,
    DataCollatorForTokenClassification
)


class CFG:
    INFERENCE_MAX_LENGTH = 2048
    MODEL_PATH = ""


def tokenize(example, tokenizer):
    text = []
    token_map = []

    idx = 0
    for t, ws in zip(example["tokens"], example["trailing_whitespace"]):
        text.append(t)
        token_map.extend([idx] * len(t))
        if ws:
            text.append(" ")
            token_map.append(-1)
        idx += 1

    tokenized = tokenizer(
        "".join(text),
        return_offsets_mapping=True,
        truncation=True,
        max_length=CFG.INFERENCE_MAX_LENGTH
    )

    return {
        **tokenized,
        "token_map": token_map
    }


data = json.load(open("./kaggle_dataset/competition/test.json"))

ds = Dataset.from_dict({
    "full_text": [x["full_text"] for x in data],
    "document": [x["document"] for x in data],
    "tokens": [x["tokens"] for x in data],
    "trailing_whitespace": [x["trailing_whitespace"] for x in data],
})

tokenizer = AutoTokenizer.from_pretrained(CFG.MODEL_PATH)
model = AutoModelForTokenClassification.from_pretrained(CFG.MODEL_PATH)

ds = ds.map(tokenize, fn_kwargs={"tokenizer": tokenizer}, num_proc=2)

collator = DataCollatorForTokenClassification(tokenizer, pad_to_multiple_of=16)
args = TrainingArguments(
    ".",
    per_device_eval_batch_size=4,
    report_to="none",
)
trainer = Trainer(
    model=model,
    args=args,
    data_collator=collator,
    tokenizer=tokenizer,
)

predictions = trainer.predict(ds).predictions
config = json.load(open(Path(CFG.MODEL_PATH) / "config.json"))
id2label = config["id2label"]
preds = predictions.argmax(-1)

triplets = []
document, token, label, token_str = [], [], [], []
for p, token_map, offsets, tokens, doc in zip(preds, ds["token_map"], ds["offset_mapping"], ds["tokens"], ds["document"]):

    for token_pred, (start_idx, end_idx) in zip(p, offsets):
        label_pred = id2label[str(token_pred)]

        if start_idx + end_idx == 0:
            continue

        if token_map[start_idx] == -1:
            start_idx += 1

        while start_idx < len(token_map) and tokens[token_map[start_idx]].isspace():
            start_idx += 1

        if start_idx >= len(token_map):
            break

        token_id = token_map[start_idx]

        if label_pred != "O" and token_id != -1:
            triplet = (label_pred, token_id, tokens[token_id])

            if triplet not in triplets:
                document.append(doc)
                token.append(token_id)
                label.append(label_pred)
                token_str.append(tokens[token_id])
                triplets.append(triplet)

df = pd.DataFrame({
    "document": document,
    "token": token,
    "label": label,
    "token_str": token_str
})
df["row_id"] = list(range(len(df)))
print(df.head(10))

df[["row_id", "document", "token", "label"]].to_csv(
    "./kaggle_dataset/competition/submission.csv", index=False
)
