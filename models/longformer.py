# wandb sweep --project PII config_v4.yaml
from collections import defaultdict
from typing import Dict
from seqeval.metrics import f1_score
from seqeval.metrics import classification_report
from seqeval.metrics import recall_score, precision_score
from transformers.trainer_callback import (
    CallbackHandler,
    DefaultFlowCallback,
    PrinterCallback,
    ProgressCallback,
    TrainerCallback,
    TrainerControl,
    TrainerState
)

from focal_loss.focal_loss import FocalLoss
from copy import deepcopy
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm
from transformers import get_polynomial_decay_schedule_with_warmup
from transformers import AutoModelForTokenClassification, DataCollatorForTokenClassification
from transformers import AutoTokenizer, Trainer, TrainingArguments, TrainerCallback, LongformerTokenizerFast
from functools import partial
from itertools import chain
import yaml
import os
import torch.nn as nn
import pandas as pd
import numpy as np
import torch
import wandb
import argparse
import json
import random


def seed_everything(seed=None):
    '''

    固定seed
    :param seed: int, 随机种子
    '''
    max_seed_value = np.iinfo(np.uint32).max
    min_seed_value = np.iinfo(np.uint32).min

    if (seed is None) or not (min_seed_value <= seed <= max_seed_value):
        seed = random.randint(np.iinfo(np.uint32).min, np.iinfo(np.uint32).max)
    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    return seed


seed_everything(42)

OUTPUT_DIR = "output"  # your output path

# load dataset
valid_df = pd.read_json("../kaggle_dataset/test_split.json")

train1 = pd.read_json("../kaggle_dataset/train_split.json")
train2 = pd.read_json("../kaggle_dataset/nb_mixtral-8x7b-v1.json")
train3 = pd.read_json("../kaggle_dataset/znf_nb_general_topic_1k_0220_mixtral.json")
train4 = pd.read_json("../kaggle_dataset/znf_nb_general_topic_1k_0220_mixtral_2.json")
train5 = pd.read_json("../kaggle_dataset/znf_nb_general_topic_1k_0221_mixtral.json")
train6 = pd.read_json("../kaggle_dataset/znf_nb_lzc_general_topic_2k_0219_gpt.json")
train7 = pd.read_json("../kaggle_dataset/znf_nb_same_2k_0223_mixtral.json")
train8 = pd.read_json("../kaggle_dataset/no_label_1k_0222.json")
train9 = pd.read_json("../kaggle_dataset/no_label_1k_0223.json")
train10 = pd.read_json("../kaggle_dataset/znf_no_label_2k_0223.json")


train_df = pd.concat([
    train1, train2, train3, train4, train5, 
    train6, train7, train8, train9, train10
])
train_df = train_df.sample(frac=1, random_state=666)
train_df.reset_index(drop=True, inplace=True)

df = valid_df[["document", "tokens", "labels"]].copy()
df = df.explode(["tokens", "labels"]).reset_index(drop=True).rename(
    columns={"tokens": "token", "labels": "label"}
)
df["token_str"] = df["token"]
df["token"] = df.groupby("document").cumcount()
label_list = df["label"].unique().tolist()
reference_df = df[df["label"] != "O"].copy()
reference_df = reference_df.reset_index().rename(columns={"index": "row_id"})
reference_df = reference_df[["row_id", "document", "token", "label", "token_str"]].copy()


class EMA:
    def __init__(self, model, decay=0.9):
        self.module = deepcopy(model)
        self.module.eval()
        self.decay = decay
        self.device = device  # perform ema on different device from model if set
        if self.device is not None:
            self.module.to(device=device)

    def _update(self, model, update_fn):
        with torch.no_grad():
            for ema_v, model_v in zip(self.module.state_dict().values(), model.state_dict().values()):
                if self.device is not None:
                    model_v = model_v.to(device=self.device)
                ema_v.copy_(update_fn(ema_v, model_v))

    def update(self, model):
        self._update(
            model,
            update_fn=lambda e, m: self.decay * e + (1. - self.decay) * m
        )

    def set(self, model):
        self._update(model, update_fn=lambda e, m: m)


class EMACallback(TrainerCallback):
    def __init__(self, trainer, decay=0.99, use_ema_weights=True) -> None:
        super().__init__()
        self._trainer = trainer
        self.decay = decay
        self.use_ema_weights = use_ema_weights
        self.ema = None

    def on_init_end(self, args: TrainingArguments, state: TrainerState, control: TrainerControl, **kwargs):
        """
        Event called at the end of the initialization of the [`Trainer`].
        """
        self.ema = EMA(self._trainer.model, decay=self.decay, device=None)
        return control

    def store(self, parameters):
        "Save the current parameters for restoring later."
        self.collected_params = [param.clone() for param in parameters]

    def restore(self, parameters):
        """
        Restore the parameters stored with the `store` method.
        Useful to validate the model with EMA parameters without affecting the
        original optimization process.
        """
        for c_param, param in zip(self.collected_params, parameters):
            param.data.copy_(c_param.data)

    def copy_to(self, shadow_parameters, parameters):
        "Copy current parameters into given collection of parameters."
        for s_param, param in zip(shadow_parameters, parameters):
            if param.requires_grad:
                param.data.copy_(s_param.data)

    def on_step_end(self, args: TrainingArguments, state: TrainerState, control: TrainerControl, **kwargs):
        """
        Event called at the end of a training step. If using gradient accumulation, one training step might take
        several inputs.
        """
        self.ema.update(self._trainer.model)
        self.store(self._trainer.model.parameters())
        self.copy_to(self.ema.module.parameters(),
                     self._trainer.model.parameters())
        return control

    def on_evaluate(self, args: TrainingArguments, state: TrainerState, control: TrainerControl, **kwargs):
        """
        Event called after an evaluation phase.
        """
        self.restore(self._trainer.model.parameters())
        return control

    def on_train_end(self, args: TrainingArguments, state: TrainerState, control: TrainerControl, **kwargs):
        """
        Event called at the end of training.
        """
        if self.use_ema_weights:
            self.copy_to(self.ema.module.parameters(),
                         self._trainer.model.parameters())
            # msg = "Model weights replaced with the EMA version."
            # log_main_process(_logger, logging.INFO, msg)
        return control

    def on_save(self, args: TrainingArguments, state: TrainerState, control: TrainerControl, **kwargs):
        """
        Event called after a checkpoint save.
        """
        checkpoint_folder = f"ema-checkpoint-{self._trainer.state.global_step}"
        run_dir = self.args.output_dir
        output_dir = os.path.join(run_dir, checkpoint_folder)
        self.copy_to(self.ema.module.parameters(),
                     self._trainer.model.parameters())
        self._trainer.save_model(output_dir, _internal_call=True)
        self.restore(self._trainer.model.parameters())
        return control


class AWP:
    def __init__(self, model, adv_param="weight", adv_lr=0.1, adv_eps=1e-4):
        self.model = model
        self.adv_param = adv_param
        self.adv_lr = adv_lr
        self.adv_eps = adv_eps
        self.backup = {}
        self.backup_eps = {}

    def _attack_step(self):
        e = 1e-6
        for name, param in self.model.named_parameters():
            if param.requires_grad and param.grad is not None and self.adv_param in name:
                norm1 = torch.norm(param.grad)
                norm2 = torch.norm(param.data.detach())
                if norm1 != 0 and not torch.isnan(norm1):
                    # 在损失函数之前获得梯度
                    r_at = self.adv_lr * param.grad / (norm1 + e) * (norm2 + e)
                    param.data.add_(r_at)
                    param.data = torch.min(
                        torch.max(param.data, self.backup_eps[name][0]),
                        self.backup_eps[name][1]
                    )

    def _save(self):
        for name, param in self.model.named_parameters():
            if param.requires_grad and param.grad is not None and self.adv_param in name:
                if name not in self.backup:
                    self.backup[name] = param.data.clone()
                    grad_eps = self.adv_eps * param.abs().detach()
                    self.backup_eps[name] = (
                        self.backup[name] - grad_eps,
                        self.backup[name] + grad_eps
                    )

    def _restore(self,):
        for name, param in self.model.named_parameters():
            if name in self.backup:
                param.data = self.backup[name]
        self.backup = {}
        self.backup_eps = {}


class CustomTrainer(Trainer):
    def __init__(
        self,
        model=None,
        args=None,
        data_collator=None,
        train_dataset=None,
        eval_dataset=None,
        tokenizer=None,
        model_init=None,
        compute_metrics=None,
        callbacks=None,
        optimizers=(None, None),
        preprocess_logits_for_metrics=None,
        awp_lr=0.1,
        awp_eps=1e-4,
        awp_start_epoch=0.5,
        weights=None
    ):

        super().__init__(
            model=model,
            args=args,
            data_collator=data_collator,
            train_dataset=train_dataset,
            eval_dataset=eval_dataset,
            tokenizer=tokenizer,
            model_init=model_init,
            compute_metrics=compute_metrics,
            callbacks=callbacks,
            optimizers=optimizers,
            preprocess_logits_for_metrics=preprocess_logits_for_metrics
        )

        self.awp_lr = awp_lr
        self.awp_eps = awp_eps
        self.awp_start_epoch = awp_start_epoch
        self.weights = weights

    def compute_loss(self, model, inputs, return_outputs=False):
        """
        focal loss
        """
        labels = inputs.pop("labels")
        # forward pass
        overflow_to_sample_mapping = inputs.pop("overflow_to_sample_mapping")
        wids = inputs.pop("wids")
        outputs = model(**inputs)
        logits = outputs.get("logits")

        weights = torch.tensor(self.weights, device=model.device)
        loss_fct = nn.CrossEntropyLoss(
            weight=weights
        )
        loss = loss_fct(
            logits.view(-1, self.model.config.num_labels),
            labels.view(-1)
        )
        return (loss, outputs) if return_outputs else loss

    def training_step(self, model, inputs):
        """
        Perform a training step on a batch of inputs.

        Subclass and override to inject custom behavior.

        Args:
            model (`nn.Module`):
                The model to train.
            inputs (`Dict[str, Union[torch.Tensor, Any]]`):
                The inputs and targets of the model.

                The dictionary will be unpacked before being fed to the model. Most models expect the targets under the
                The inputs and targets of the model.

                The dictionary will be unpacked before being fed to the model. Most models expect the targets under the
                The dictionary will be unpacked before being fed to the model. Most models expect the targets under the
                argument `labels`. Check your model"s documentation for all accepted arguments.

        Return:
            `torch.Tensor`: The tensor with training loss on this batch.
        """
        model.train()
        o_inputs = inputs.copy()
        inputs = self._prepare_inputs(inputs)

        with self.compute_loss_context_manager():
            loss = self.compute_loss(model, inputs)

        if self.args.n_gpu > 1:
            loss = loss.mean()  # mean() to average on multi-gpu parallel training

        if self.use_apex:
            with amp.scale_loss(loss, self.optimizer) as scaled_loss:
                scaled_loss.backward()
        else:
            self.accelerator.backward(loss)

        ########################
        # AWP
        if self.awp_lr != 0 and self.state.epoch >= self.awp_start_epoch:
           # print(inputs)
           # print("Start amp")
            self.awp = AWP(model, adv_lr=self.awp_lr, adv_eps=self.awp_eps)
            self.awp._save()
            self.awp._attack_step()
            with self.compute_loss_context_manager():
                awp_loss = self.compute_loss(self.awp.model, o_inputs)

            if self.args.n_gpu > 1:
                awp_loss = awp_loss.mean()  # mean() to average on multi-gpu parallel training

            if self.use_apex:
                with amp.scale_loss(awp_loss, self.optimizer) as awp_scaled_loss:
                    awp_scaled_loss.backward()
            else:
                self.accelerator.backward(awp_loss)
            self.awp._restore()
        ########################

        return loss.detach() / self.args.gradient_accumulation_steps


data = json.load(open("../kaggle_dataset/test_split.json"))

all_labels = sorted(list(set(chain(*[x["labels"] for x in data]))))
label2id = {l: i for i, l in enumerate(all_labels)}
id2label = {v: k for k, v in label2id.items()}
print(f"predict lables : {all_labels}")
print(id2label)

# 算权重
t = train_df.explode(["tokens", "labels"]).reset_index(drop=True).rename(columns={"tokens": "token", "labels": "label"})
p = (1 / (t.label.value_counts() / len(t.label))).to_dict()
weights = []
for i in id2label.values():
    if i in p.keys():
        if p[i] > 100:
            weights.append(p[i] / 10)
        else:
            weights.append(p[i])
    else:
        weights.append(max(p.values()) / 10)
weights[2] = weights[2] * 10
weights[-2] = weights[-2] / 200
print({i: weights[idx] for idx, i in enumerate(all_labels)})

'''
['B-EMAIL',
 'B-ID_NUM',
 'B-NAME_STUDENT',
 'B-PHONE_NUM',
 'B-STREET_ADDRESS',
 'B-URL_PERSONAL',
 'B-USERNAME',
 'I-ID_NUM',
 'I-NAME_STUDENT',
 'I-PHONE_NUM',
 'I-STREET_ADDRESS',
 'I-URL_PERSONAL',
 'O']
'''

# This function is a simple map between text_split and entities
# We have verified that we have a 1:1 mapping above
# See above: (df_texts["text_split"].str.len() == df_texts["entities"].str.len()).all() == True


def get_labels(word_ids, word_labels):
    label_ids = []
    for word_idx in word_ids:
        if word_idx is None:
            label_ids.append(-100)
        else:
            label_ids.append(label2id[word_labels[word_idx]])
    return label_ids


class PIIDataset(Dataset):
    def __init__(self, tokenized_ds):
        self.data = tokenized_ds

    def __getitem__(self, index):
        item = {k: self.data[k][index] for k in self.data.keys()}
        return item

    def __len__(self):
        return len(self.data["input_ids"])


def compute_metrics(p, all_labels):
    predictions, labels = p
    predictions = np.argmax(predictions, axis=2)

    # Remove ignored index (special tokens)
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
        "f5": f1_score
    }
    return results


class PRFScore:
    """A precision / recall / F score."""

    def __init__(
        self,
        *,
        tp: int = 0,
        fp: int = 0,
        fn: int = 0,
    ) -> None:
        self.tp = tp
        self.fp = fp
        self.fn = fn

    def __len__(self) -> int:
        return self.tp + self.fp + self.fn

    def __iadd__(self, other):  # in-place add
        self.tp += other.tp
        self.fp += other.fp
        self.fn += other.fn
        return self

    def __add__(self, other):
        return PRFScore(
            tp=self.tp + other.tp, fp=self.fp + other.fp, fn=self.fn + other.fn
        )

    def score_set(self, cand: set, gold: set) -> None:
        self.tp += len(cand.intersection(gold))
        self.fp += len(cand - gold)
        self.fn += len(gold - cand)

    @property
    def precision(self) -> float:
        return self.tp / (self.tp + self.fp + 1e-100)

    @property
    def recall(self) -> float:
        return self.tp / (self.tp + self.fn + 1e-100)

    @property
    def f1(self) -> float:
        p = self.precision
        r = self.recall
        return 2 * ((p * r) / (p + r + 1e-100))

    @property
    def f5(self) -> float:
        beta = 5
        p = self.precision
        r = self.recall

        fbeta = (1 + (beta ** 2)) * p * r / ((beta ** 2) * p + r + 1e-100)
        return fbeta

    def to_dict(self) -> Dict[str, float]:
        return {"p": self.precision, "r": self.recall, "f5": self.f5}


def compute_lb_metrics(pred_df, gt_df):
    """
    Compute the LB metric (lb) and other auxiliary metrics
    """

    references = {(row.document, row.token, row.label)
                  for row in gt_df.itertuples()}
    predictions = {(row.document, row.token, row.label)
                   for row in pred_df.itertuples()}

    score_per_type = defaultdict(PRFScore)
    references = set(references)

    for ex in predictions:
        pred_type = ex[-1]  # (document, token, label)
        if pred_type != "O":
            pred_type = pred_type[2:]  # avoid B- and I- prefix

        if pred_type not in score_per_type:
            score_per_type[pred_type] = PRFScore()

        if ex in references:
            score_per_type[pred_type].tp += 1
            references.remove(ex)
        else:
            score_per_type[pred_type].fp += 1

    for doc, tok, ref_type in references:
        if ref_type != "O":
            ref_type = ref_type[2:]  # avoid B- and I- prefix

        if ref_type not in score_per_type:
            score_per_type[ref_type] = PRFScore()
        score_per_type[ref_type].fn += 1

    totals = PRFScore()

    for prf in score_per_type.values():
        totals += prf

    results = {
        "precision": totals.precision,
        "recall": totals.recall,
        "f5": totals.f5
    }

    return results


def compute_metrics_v2(p, valid_df, reference_df, valid_dataset, id2label):
    token_pred = defaultdict(lambda: defaultdict(int))
    token_cnt = defaultdict(lambda: defaultdict(int))

    preds, labels = p
    assert preds.shape[0] == len(valid_dataset)
    preds_softmax = np.exp(preds) / np.sum(np.exp(preds),
                                           axis=2).reshape(preds.shape[0], preds.shape[1], 1)

    for preds, batch in zip(preds_softmax, valid_dataset):
        word_ids = batch["wids"].numpy()
        text_id = batch["overflow_to_sample_mapping"].item()
        for idx, word_idx in enumerate(word_ids):
            if word_idx != -1:
                token_pred[text_id][word_idx] += preds[idx]
                token_cnt[text_id][word_idx] += 1

    for text_id in token_pred:
        for word_idx in token_pred[text_id]:
            token_pred[text_id][word_idx] /= token_cnt[text_id][word_idx]

    document, token, label, token_str = [], [], [], []
    for text_id in token_pred:
        for word_idx in token_pred[text_id]:
            pred = token_pred[text_id][word_idx].argmax(-1)
            if id2label[pred] != "O":
                document.append(valid_df.loc[text_id, "document"])
                token.append(word_idx)
                label.append(id2label[pred])
                token_str.append(valid_df.loc[text_id, "tokens"][word_idx])

    df = pd.DataFrame({
        "document": document,
        "token": token,
        "label": label,
        "token_str": token_str
    })

    results = compute_lb_metrics(df, reference_df)

    return results


def main():
    # Set up your default hyperparameters
    with open("./longformer.yaml") as file:
        config = yaml.load(file, Loader=yaml.FullLoader)

    wandb.init(config=config)

    # hyper-parameters
    lr = wandb.config.lr
    max_length = wandb.config.max_length
    ga_steps = wandb.config.gradient_accumulation_steps
    batch_size = wandb.config.batch_size
    stride = wandb.config.stride
    awp_lr = wandb.config.awp_lr
    awp_eps = wandb.config.awp_eps
    awp_start_epoch = wandb.config.awp_start_epoch
    nna = wandb.config.neftune_noise_alpha
    TRAINING_MODEL_PATH = wandb.config.training_model_path

    # Tokenize texts, possibly generating more than one tokenized sample for each text

    def tokenize(df, to_tensor=True, with_labels=True):

        # This is what"s different from a longformer
        # Read the parameters with attention
        encoded = tokenizer(
            df["tokens"].tolist(),
            is_split_into_words=True,
            return_overflowing_tokens=True,
            stride=stride,
            max_length=max_length,
            padding="max_length",
            truncation=True
        )

        if with_labels:
            encoded["labels"] = []

        encoded["wids"] = []
        n = len(encoded["overflow_to_sample_mapping"])
        for i in range(n):

            # Map back to original row
            text_idx = encoded["overflow_to_sample_mapping"][i]

            # Get word indexes (this is a global index that takes into consideration the chunking :D )
            word_ids = encoded.word_ids(i)

            if with_labels:
                # Get word labels of the full un-chunked text
                word_labels = df["labels"].iloc[text_idx]

                # Get the labels associated with the word indexes
                label_ids = get_labels(word_ids, word_labels)
                encoded["labels"].append(label_ids)

            encoded["wids"].append(
                [w if w is not None else -1 for w in word_ids]
            )

        if to_tensor:
            encoded = {key: torch.as_tensor(val)
                       for key, val in encoded.items()}

        return encoded

    tokenizer = LongformerTokenizerFast.from_pretrained(TRAINING_MODEL_PATH, add_prefix_space=True) #这里是和deberta不一样的地方
    tokenized_train = tokenize(train_df)
    tokenized_valid = tokenize(valid_df)
    train_dataset = PIIDataset(tokenized_train)
    valid_dataset = PIIDataset(tokenized_valid)

    collator = DataCollatorForTokenClassification(
        tokenizer, pad_to_multiple_of=16
    )

    args = TrainingArguments(
        output_dir=f"output/{wandb.run.name}",
        fp16=True,
        gradient_accumulation_steps=ga_steps,
        logging_steps=100,
        warmup_ratio=0.05,
        learning_rate=lr,          # tune
        num_train_epochs=3,
        per_device_train_batch_size=batch_size,
        per_device_eval_batch_size=batch_size,
        report_to="wandb",
        evaluation_strategy="steps",
        eval_steps=100,
        save_strategy="steps",
        save_steps=100,
        save_total_limit=6,
        overwrite_output_dir=True,
        load_best_model_at_end=True,
        lr_scheduler_type="linear",
        metric_for_best_model="f5",
        greater_is_better=True,
        weight_decay=0.01,
        save_only_model=True,
        neftune_noise_alpha=nna,
        remove_unused_columns=False
    )

    model = AutoModelForTokenClassification.from_pretrained(
        TRAINING_MODEL_PATH,
        num_labels=len(all_labels),
        id2label=id2label,
        label2id=label2id,
        ignore_mismatched_sizes=True
    )
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.learning_rate)

    gpu_count = torch.cuda.device_count()
    print(f"Number of GPUs: {gpu_count}")

    total_steps = args.num_train_epochs * \
        int(len(train_dataset) * 1.0 / gpu_count /
            args.per_device_train_batch_size / args.gradient_accumulation_steps)
    scheduler = get_polynomial_decay_schedule_with_warmup(
        optimizer,
        num_warmup_steps=int(total_steps) * 0.1,
        num_training_steps=total_steps,
        power=1.5,
        lr_end=3e-6
    )

    trainer = CustomTrainer(
        model=model,
        args=args,
        train_dataset=train_dataset,
        eval_dataset=valid_dataset,
        data_collator=collator,
        tokenizer=tokenizer,
        compute_metrics=partial(
            compute_metrics_v2, valid_df=valid_df,
            reference_df=reference_df, valid_dataset=valid_dataset,
            id2label=id2label
        ),
        awp_lr=awp_lr,
        awp_eps=awp_eps,
        awp_start_epoch=awp_start_epoch,
        optimizers=(optimizer, scheduler),
        weights=weights
    )

    trainer.train()
    wandb.finish()


if __name__ == "__main__":
    main()
