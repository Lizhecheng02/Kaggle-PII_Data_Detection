from seqeval.metrics import recall_score, f1_score, precision_score
from functools import partial
from itertools import chain
from datasets import Dataset, features
import os
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
import argparse
import warnings
warnings.filterwarnings("ignore")


def train(args):

    print("... Load Dataset ...")
    data = json.load(open(args.train_file_path))
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
            max_length=args.max_length
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
    tokenizer = AutoTokenizer.from_pretrained(args.model_name)

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
        f1_score = (1 + 5 * 5) * recall * precision / \
            (5 * 5 * precision + recall)

        results = {
            "recall": recall,
            "precision": precision,
            "f1": f1_score
        }
        return results

    print("... Load LLM Model ...")
    model = AutoModelForTokenClassification.from_pretrained(
        args.model_name,
        num_labels=len(all_labels),
        id2label=id2label,
        label2id=label2id,
        ignore_mismatched_sizes=True
    )
    collator = DataCollatorForTokenClassification(
        tokenizer, pad_to_multiple_of=16
    )

    print("... Freeze ...")
    if args.freeze_embeddings:
        print("Freezing embeddings.")
        for param in model.deberta.embeddings.parameters():
            param.requires_grad = False

    if args.freeze_layers > 0:
        print(f"Freezing {args.freeze_layers} layers.")
        for layer in model.deberta.encoder.layer[:args.freeze_layers]:
            for param in layer.parameters():
                param.requires_grad = False

    final_ds = ds.train_test_split(
        test_size=args.test_size,
        seed=args.random_seed
    )
    print(final_ds)

    print("... Training ...")
    training_args = TrainingArguments(
        output_dir=args.output_dir,
        fp16=True,
        warmup_steps=50,
        learning_rate=args.learning_rate,
        num_train_epochs=args.num_train_epochs,
        per_device_train_batch_size=args.per_device_train_batch_size,
        per_device_eval_batch_size=args.per_device_eval_batch_size,
        report_to="none",
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        logging_steps=args.steps,
        evaluation_strategy="steps",
        eval_steps=args.steps,
        save_strategy="steps",
        save_steps=args.steps,
        save_total_limit=args.save_total_limit,
        overwrite_output_dir=True,
        load_best_model_at_end=True,
        lr_scheduler_type=args.lr_scheduler_type,
        metric_for_best_model=args.metric_for_best_model,
        greater_is_better=True,
        weight_decay=args.weight_decay
    )
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=final_ds["train"],
        eval_dataset=final_ds["test"],
        data_collator=collator,
        tokenizer=tokenizer,
        compute_metrics=partial(compute_metrics, all_labels=all_labels),
    )
    trainer.train()

    print("... Save Model ...")
    trainer.save_model(f"{args.output_dir}/best")
    torch.cuda.empty_cache()

    def delete_optimizer_files(directory):
        for root, dirs, files in os.walk(directory):
            for file in files:
                if file == "optimizer.pt":
                    os.remove(os.path.join(root, file))
                    print(f"Deleted: {os.path.join(root, file)}")

    delete_optimizer_files(args.output_dir)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Finetune a Transformers Model on a Token Classification Task"
    )
    parser.add_argument(
        "--train_file_path", default="./combined_dataset/origin_moth_0125.json", type=str
    )
    parser.add_argument(
        "--test_size", default=0.15, type=float
    )
    parser.add_argument(
        "--random_seed", default=42, type=int
    )
    parser.add_argument(
        "--model_name", default="microsoft/deberta-v3-base", type=str
    )
    parser.add_argument(
        "--max_length", default=512, type=int
    )
    parser.add_argument(
        "--freeze_embeddings", default=False, type=bool
    )
    parser.add_argument(
        "--freeze_layers", default=0, type=int
    )
    parser.add_argument(
        "--output_dir", default="output", type=str
    )
    parser.add_argument(
        "--learning_rate", default=2e-5, type=float
    )
    parser.add_argument(
        "--num_train_epochs", default=2, type=int
    )
    parser.add_argument(
        "--per_device_train_batch_size", default=1, type=int
    )
    parser.add_argument(
        "--per_device_eval_batch_size", default=1, type=int
    )
    parser.add_argument(
        "--gradient_accumulation_steps", default=16, type=int
    )
    parser.add_argument(
        "--steps", default=100, type=int
    )
    parser.add_argument(
        "--save_total_limit", default=3, type=int
    )
    parser.add_argument(
        "--lr_scheduler_type", default="cosine", type=str
    )
    parser.add_argument(
        "--weight_decay", default=0.001, type=float
    )
    parser.add_argument(
        "--metric_for_best_model", default="f1", type=str
    )
    args = parser.parse_args()
    print(args)
    train(args)
