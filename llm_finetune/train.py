import os
import gc
import json
import numpy as np
import pandas as pd
import codecs
import torch
import torch.nn as nn
import pickle
import re
import sys
import torch.nn.functional as F
import pytorch_lightning as pl
from metric import compute_metrics
from pytorch_lightning import Trainer, seed_everything
from pytorch_lightning.callbacks import ModelCheckpoint, EarlyStopping
from torch.utils.data import Dataset, DataLoader
from itertools import chain
from text_unidecode import unidecode
from typing import Dict, List, Tuple
from tqdm.auto import tqdm
from datasets import concatenate_datasets, load_dataset, load_from_disk
from sklearn.metrics import log_loss
from peft import get_peft_model, LoraConfig, TaskType
from transformers.models.llama.modeling_llama import *
from transformers.modeling_outputs import TokenClassifierOutput
from transformers import (
    AutoModel,
    AutoTokenizer,
    AdamW,
    DataCollatorWithPadding,
    get_polynomial_decay_schedule_with_warmup,
    get_cosine_schedule_with_warmup,
    get_linear_schedule_with_warmup,
    DataCollatorForTokenClassification,
    TrainingArguments,
    AutoConfig,
    AutoModelForTokenClassification
)
from transformers.tokenization_utils_base import BatchEncoding, PreTrainedTokenizerBase


class Config:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    seed = 69

    train_dataset_path_1 = "../kaggle_dataset/train_split.json"
    train_dataset_path_2 = "../kaggle_dataset/nb_mixtral-8x7b-v1.json"
    test_dataset_path = "../kaggle_dataset/test_split.json"
    save_dir = "exp1"

    downsample = 0.75
    truncation = True
    padding = False
    max_length = 1024
    freeze_layers = 0
    model_name = "h2oai/h2o-danube-1.8b-base"

    target_cols = [
        "B-EMAIL", "B-ID_NUM", "B-NAME_STUDENT", "B-PHONE_NUM",
        "B-STREET_ADDRESS", "B-URL_PERSONAL", "B-USERNAME", "I-ID_NUM",
        "I-NAME_STUDENT", "I-PHONE_NUM", "I-STREET_ADDRESS", "I-URL_PERSONAL", "O"
    ]

    load_from_disk = None

    learning_rate = 1e-4
    batch_size = 1
    epochs = 3
    NFOLDS = 4
    trn_fold = -2


seed_everything(Config.seed)
if not os.path.exists(Config.save_dir):
    os.makedirs(Config.save_dir)

data1 = json.load(open(Config.train_dataset_path_1))
data2 = json.load(open(Config.train_dataset_path_2))
data = data1 + data2

test_data = json.load(open(Config.test_dataset_path))
print("num_samples:", len(data))
print(data[0].keys())

all_labels = sorted(list(set(chain(*[x["labels"] for x in test_data]))))
label2id = {l: i for i, l in enumerate(all_labels)}
id2label = {v: k for k, v in label2id.items()}
print(id2label)

tokenizer = AutoTokenizer.from_pretrained(Config.model_name)
tokenizer.pad_token = tokenizer.eos_token
tokenizer.save_pretrained(f"{Config.save_dir}")

df_train = pd.DataFrame(data)
df_train = df_train.sample(frac=1, random_state=Config.seed)
df_train.reset_index(drop=True, inplace=True)
print(df_train.head(5))
df_train["document"] = range(len(df_train))
df_train["fold"] = df_train["document"] % 4


def tokenize_row(example):
    text = []
    token_map = []
    labels = []
    targets = []
    idx = 0
    for t, l, ws in zip(example["tokens"], example["labels"], example["trailing_whitespace"]):
        text.append(t)
        labels.extend([l] * len(t))
        token_map.extend([idx] * len(t))

        if l in Config.target_cols:
            targets.append(1)
        else:
            targets.append(0)

        if ws:
            text.append(" ")
            labels.append("O")
            token_map.append(-1)
        idx += 1

    if Config.valid_stride:
        tokenized = tokenizer(
            "".join(text),
            return_offsets_mapping=True,
            padding="longest",
            truncation=True,
            max_length=2048
        )
    else:
        tokenized = tokenizer(
            "".join(text),
            return_offsets_mapping=True,
            padding="longest",
            truncation=True,
            max_length=Config.max_length
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
        try:
            token_labels.append(label2id[labels[start_idx]])
        except:
            continue
    length = len(tokenized.input_ids)

    return {
        "input_ids": tokenized.input_ids,
        "attention_mask": tokenized.attention_mask,
        "offset_mapping": tokenized.offset_mapping,
        "labels": token_labels,
        "length": length,
        "target_num": target_num,
        "group": 1 if target_num > 0 else 0,
        "token_map": token_map,
    }


def downsample_df(train_df, percent):

    train_df["is_labels"] = train_df["labels"].apply(
        lambda labels: any(label != "O" for label in labels)
    )

    true_samples = train_df[train_df["is_labels"] == True]
    false_samples = train_df[train_df["is_labels"] == False]

    n_false_samples = int(len(false_samples) * percent)
    downsampled_false_samples = false_samples.sample(
        n=n_false_samples,
        random_state=42
    )

    downsampled_df = pd.concat([true_samples, downsampled_false_samples])
    return downsampled_df


def add_token_indices(doc_tokens):
    token_indices = list(range(len(doc_tokens)))
    return token_indices


df_train["token_indices"] = df_train["tokens"].apply(add_token_indices)

if Config.load_from_disk is None:
    for i in range(-1, Config.NFOLDS):
        train_df = df_train[df_train["fold"] == i].reset_index(drop=True)
        if i == Config.trn_fold:
            Config.valid_stride = True
        if i != Config.trn_fold and Config.downsample > 0:
            train_df = downsample_df(train_df, Config.downsample)
            Config.valid_stride = False

        print(len(train_df))
        ds = Dataset.from_pandas(train_df)

        ds = ds.map(
            tokenize_row,
            batched=False,
            num_proc=2,
            desc="Tokenizing",
        )

        ds.save_to_disk(f"{Config.save_dir}_fold_{i}.dataset")
        with open(f"{Config.save_dir}_pkl", "wb") as fp:
            pickle.dump(train_df, fp)
        print("Saving dataset to disk:", Config.save_dir)


def predictions_to_df(preds, ds, id2label=id2label):
    triplets = []
    pairs = set()
    document, token, label, token_str = [], [], [], []
    for p, token_map, offsets, tokens, doc in zip(preds, ds["token_map"], ds["offset_mapping"], ds["tokens"], ds["document"]):
        p = p.cpu().detach().numpy()

        for token_pred, (start_idx, end_idx) in zip(p, offsets):
            label_pred = id2label[(token_pred)]

            if start_idx + end_idx == 0:
                continue

            if token_map[start_idx] == -1:
                start_idx += 1

            # ignore "\n\n"
            while start_idx < len(token_map) and tokens[token_map[start_idx]].isspace():
                start_idx += 1

            if start_idx >= len(token_map):
                break

            token_id = token_map[start_idx]

            if label_pred == "O" or token_id == -1:
                continue

            pair = (doc, token_id)

            if pair in pairs:
                continue

            document.append(doc)
            token.append(token_id)
            label.append(label_pred)
            token_str.append(tokens[token_id])
            pairs.add(pair)

    df = pd.DataFrame({
        "document": document,
        "token": token,
        "label": label,
        "token_str": token_str
    })
    df["row_id"] = list(range(len(df)))
    return df


def process_predictions(flattened_preds, threshold=0.9):

    preds_final = []
    for predictions in flattened_preds:

        predictions_softmax = torch.softmax(predictions, dim=-1)
        predictions_argmax = predictions.argmax(-1)
        predictions_without_O = predictions_softmax[:, :12].argmax(-1)

        O_predictions = predictions_softmax[:, 12]
        pred_final = torch.where(
            O_predictions < threshold,
            predictions_without_O,
            predictions_argmax
        )
        preds_final.append(pred_final)

    return preds_final


class LlamaForTokenClassification(LlamaPreTrainedModel):
    def __init__(self, config):
        super().__init__(config)
        self.num_labels = config.num_labels
        self.model = LlamaModel(config)
        self.post_init()

    def get_input_embeddings(self):
        return self.model.embed_tokens

    def set_input_embeddings(self, value):
        self.model.embed_tokens = value

    def forward(
        self,
        input_ids: torch.LongTensor = None,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_values: Optional[List[torch.FloatTensor]] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        labels: Optional[torch.LongTensor] = None,
        use_cache: Optional[bool] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
    ) -> Union[Tuple, SequenceClassifierOutputWithPast]:
        r"""
        labels (`torch.LongTensor` of shape `(batch_size,)`, *optional*):
            Labels for computing the sequence classification/regression loss. Indices should be in `[0, ...,
            config.num_labels - 1]`. If `config.num_labels == 1` a regression loss is computed (Mean-Square loss), If
            `config.num_labels > 1` a classification loss is computed (Cross-Entropy).
        """
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        outputs = self.model(
            input_ids,
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_values=past_key_values,
            inputs_embeds=inputs_embeds,
            use_cache=use_cache,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )
        sequence_output = outputs[0]

        return sequence_output


class LSTMHead(nn.Module):
    def __init__(self, in_features, hidden_dim, n_layers):
        super().__init__()
        self.lstm = nn.LSTM(
            in_features,
            hidden_dim,
            n_layers,
            batch_first=True,
            bidirectional=True,
            dropout=0.1
        )
        self.out_features = hidden_dim

    def forward(self, x):
        self.lstm.flatten_parameters()
        hidden, (_, _) = self.lstm(x)
        out = hidden
        return out


class PIIModel(pl.LightningModule):
    def __init__(self, config, val_ds, true_val_df):
        super().__init__()
        self.cfg = config
        self.val_ds = val_ds
        self.true_val_df = true_val_df
        self.model_config = AutoConfig.from_pretrained(
            config.model_name
        )

        hidden_dropout_prob: float = 0.1
        layer_norm_eps: float = 1e-7
        self.model_config.update(
            {
                "output_hidden_states": True,
                "hidden_dropout_prob": hidden_dropout_prob,
                "layer_norm_eps": layer_norm_eps,
                "add_pooling_layer": False,
            }
        )

        self.transformers_model = LlamaForTokenClassification.from_pretrained(
            config.model_name,
            num_labels=len(self.cfg.target_cols),
            id2label=id2label,
            label2id=label2id
        )

        peft_config = LoraConfig(
            task_type=TaskType.TOKEN_CLS,
            inference_mode=False,
            r=16,
            lora_alpha=64,
            lora_dropout=0.0
        )

        self.transformers_model = get_peft_model(
            self.transformers_model,
            peft_config
        )
        self.transformers_model.gradient_checkpointing_enable()
        self.transformers_model.print_trainable_parameters()
        self.head = LSTMHead(
            in_features=self.model_config.hidden_size,
            hidden_dim=self.model_config.hidden_size // 2,
            n_layers=1
        )
        self.output = nn.Linear(
            self.model_config.hidden_size,
            len(self.cfg.target_cols)
        )

        self.loss_function = nn.CrossEntropyLoss(
            reduction="mean",
            ignore_index=-100
        )
        self.validation_step_outputs = []

    def forward(self, input_ids, attention_mask, train):

        transformer_out = self.transformers_model(
            input_ids,
            attention_mask=attention_mask
        )
        sequence_output = self.head(transformer_out)
        logits = self.output(sequence_output)

        return (logits, _)

    def training_step(self, batch, batch_idx):
        input_ids = batch["input_ids"]
        attention_mask = batch["attention_mask"]
        target = batch["labels"]

        outputs = self(input_ids, attention_mask, train=True)
        output = outputs[0]
        loss = self.loss_function(
            output.view(-1, len(self.cfg.target_cols)),
            target.view(-1)
        )

        self.log("train_loss", loss, prog_bar=True)
        return {"loss": loss}

    def train_epoch_end(self, outputs):
        avg_loss = torch.stack([x["loss"] for x in outputs]).mean()
        print(f"epoch {trainer.current_epoch} training loss {avg_loss}")
        return {"train_loss": avg_loss}

    def validation_step(self, batch, batch_idx):
        input_ids = batch["input_ids"]
        attention_mask = batch["attention_mask"]
        target = batch["labels"]

        outputs = self(input_ids, attention_mask, train=False)
        output = outputs[0]

        loss = self.loss_function(
            output.view(-1, len(self.cfg.target_cols)),
            target.view(-1)
        )

        self.log("val_loss", loss, prog_bar=True)
        self.validation_step_outputs.append(
            {
                "val_loss": loss,
                "logits": output,
                "targets": target
            }
        )
        return {"val_loss": loss, "logits": output, "targets": target}

    def on_validation_epoch_end(self):
        outputs = self.validation_step_outputs
        avg_loss = torch.stack([x["val_loss"] for x in outputs]).mean()

        flattened_preds = [
            logit for batch in outputs for logit in batch["logits"]
        ]

        flattened_preds = process_predictions(flattened_preds)
        pred_df = predictions_to_df(flattened_preds, self.val_ds)

        print(pred_df.shape)
        print(pred_df)

        self.validation_step_outputs = []

        avg_score = compute_metrics(pred_df, self.true_val_df)
        f5_score = avg_score["ents_f5"]
        print(f"epoch {trainer.current_epoch} validation loss {avg_loss}")
        print(f"epoch {trainer.current_epoch} validation scores {avg_score}")

        return {"val_loss": avg_loss, "val_f5": f5_score}

    def train_dataloader(self):
        return self._train_dataloader

    def validation_dataloader(self):
        return self._validation_dataloader

    def get_optimizer_params(self, encoder_lr, decoder_lr, weight_decay=0.0):
        param_optimizer = list(model.named_parameters())
        no_decay = ["bias", "LayerNorm.bias", "LayerNorm.weight"]
        optimizer_parameters = [
            {
                "params": [p for n, p in self.transformers_model.named_parameters() if not any(nd in n for nd in no_decay)],
                "lr": encoder_lr,
                "weight_decay": weight_decay
            },
            {
                "params": [p for n, p in self.transformers_model.named_parameters() if any(nd in n for nd in no_decay)],
                "lr": encoder_lr,
                "weight_decay": 0.0
            },
            {
                "params": [p for n, p in self.named_parameters() if "transformers_model" not in n],
                "lr": decoder_lr,
                "weight_decay": 0.0
            }
        ]
        return optimizer_parameters

    def configure_optimizers(self):
        optimizer = AdamW(self.parameters(), lr=Config.learning_rate)

        epoch_steps = self.cfg.data_length
        batch_size = self.cfg.batch_size

        warmup_steps = 0.0 * epoch_steps // batch_size
        training_steps = self.cfg.epochs * epoch_steps // batch_size
        # scheduler = get_linear_schedule_with_warmup(optimizer, warmup_steps, training_steps, -1)
        # scheduler = get_polynomial_decay_schedule_with_warmup(optimizer, warmup_steps, training_steps, lr_end=1e-6, power=3.0)
        scheduler = get_cosine_schedule_with_warmup(
            optimizer,
            warmup_steps,
            training_steps,
            num_cycles=1
        )

        lr_scheduler_config = {
            "scheduler": scheduler,
            "interval": "step",
            "frequency": 1,
        }

        return {"optimizer": optimizer, "lr_scheduler": lr_scheduler_config}


collator = DataCollatorForTokenClassification(
    tokenizer, pad_to_multiple_of=512)


def create_val_df(df, fold):
    val_df = df[df["fold"] == fold].reset_index(drop=True).copy()

    val_df = val_df[["document", "tokens", "labels"]].copy()
    val_df = val_df.explode(["tokens", "labels"]).reset_index(drop=True).rename(columns={"tokens": "token", "labels": "label"})
    val_df["token"] = val_df.groupby("document").cumcount()

    label_list = val_df["label"].unique().tolist()

    reference_df = val_df[val_df["label"] != "O"].copy()
    reference_df = reference_df.reset_index().rename(
        columns={"index": "row_id"}
    )
    reference_df = reference_df[
        ["row_id", "document", "token", "label"]
    ].copy()
    return reference_df


for fold in range(-1, Config.NFOLDS):
    if fold != Config.trn_fold:
        continue
    train_ds_list = []

    print(f"====== FOLD RUNNING {fold} ======")

    for i in range(-1, Config.NFOLDS):
        if i == fold:
            continue
        if len(train_ds_list) >= 0:
            print(len(train_ds_list))
            train_ds_list.append(load_from_disk(f"{Config.save_dir}_fold_{i}.dataset"))

    keep_cols = {"input_ids", "attention_mask", "labels"}
    train_ds = concatenate_datasets(train_ds_list).sort("length")

    train_ds = train_ds.remove_columns(
        [c for c in train_ds.column_names if c not in keep_cols]
    )
    valid_ds = load_from_disk(f"{Config.save_dir}_fold_{fold}.dataset").sort("length")
    valid_ds = valid_ds.remove_columns(
        [c for c in valid_ds.column_names if c not in keep_cols]
    )
    val_ds = load_from_disk(f"{Config.save_dir}_fold_{fold}.dataset").sort("length")

    true_val_df = create_val_df(df_train, fold)

    Config.data_length = len(train_ds)
    Config.len_token = len(tokenizer)
    print("Dataset Loaded....")
    print(train_ds[0].keys())
    print(valid_ds[0].keys())
    print("Generating Train DataLoader")
    train_dataloader = DataLoader(
        train_ds,
        batch_size=Config.batch_size,
        shuffle=True,
        num_workers=4,
        pin_memory=False,
        collate_fn=collator
    )

    print("Generating Validation DataLoader")
    validation_dataloader = DataLoader(
        valid_ds,
        batch_size=Config.batch_size,
        shuffle=False,
        num_workers=4,
        pin_memory=False,
        collate_fn=collator
    )

    early_stop_callback = EarlyStopping(
        monitor="val_loss",
        min_delta=0.00,
        patience=8,
        verbose=True,
        mode="min"
    )
    checkpoint_callback = ModelCheckpoint(
        monitor="val_loss",
        dirpath=Config.save_dir,
        save_top_k=1,
        save_last=True,
        save_weights_only=True,
        filename=f"ckeckpoint_{fold}",
        verbose=True,
        mode="min"
    )

    print("Model Creation")

    model = PIIModel(Config, val_ds, true_val_df)
    trainer = Trainer(
        max_epochs=Config.epochs,
        deterministic=True,
        val_check_interval=0.25,
        accumulate_grad_batches=1,
        devices=[0],
        precision=16,
        accelerator="gpu",
        callbacks=[checkpoint_callback, early_stop_callback]
    )
    trainer.fit(model, train_dataloader, validation_dataloader)

    print("prediction on validation data")

    del model, train_dataloader, validation_dataloader, train_ds, valid_ds
    gc.collect()
    torch.cuda.empty_cache()
