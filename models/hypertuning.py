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
from copy import deepcopy
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm
from transformers import AutoModelForTokenClassification, DataCollatorForTokenClassification
from transformers import AutoTokenizer, Trainer, TrainingArguments, TrainerCallback
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

OUTPUT_DIR = "output"  # your output path
TRAINING_MODEL_PATH = "microsoft/deberta-v3-large"

# load dataset
valid_df = pd.read_json("../kaggle_dataset/test_split.json")

train1 = pd.read_json("../kaggle_dataset/train_split.json")
train2 = pd.read_json("../kaggle_dataset/pjm_gpt_2k_0126_fixed.json")
train3 = pd.read_json("../kaggle_dataset/nb_mixtral-8x7b-v1.json")
train4 = pd.read_json("../kaggle_dataset/darek_persuade_train_version3.json")
train_df = pd.concat([train1, train2, train3, train4])
train_df.reset_index(drop=True, inplace=True)
train_df = train_df.sample(frac=1, random_state=777)
train_df.reset_index(drop=True, inplace=True)


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
        ce_weight=1
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
        self.ce_weight = ce_weight

    def compute_loss(self, model, inputs, return_outputs=False):
        labels = inputs.pop("labels")
        # forward pass
        outputs = model(**inputs)
        logits = outputs.get("logits")
        # compute custom loss (suppose one has 3 labels with different weights)
        loss_fct = nn.CrossEntropyLoss(
            weight=torch.tensor(
                [self.ce_weight] * 12 + [1.0],
                device=model.device
            )
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
        # inputs = self._prepare_inputs(inputs)
        inputs = self._prepare_inputs(inputs)
      #  print("---" * 60)
      #  print(inputs)

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
print(id2label)


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


def main():
    # Set up your default hyperparameters
    with open("./config.yaml") as file:
        config = yaml.load(file, Loader=yaml.FullLoader)

    run = wandb.init(config=config)

    # hyper-parameters
    lr = wandb.config.lr
    max_length = wandb.config.max_length
    stride = wandb.config.stride
    awp_lr = wandb.config.awp_lr
    awp_eps = wandb.config.awp_eps
    awp_start_epoch = wandb.config.awp_start_epoch
    ce_weight = wandb.config.ce_weight

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

    tokenizer = AutoTokenizer.from_pretrained(TRAINING_MODEL_PATH)
    tokenized_train = tokenize(train_df)
    tokenized_valid = tokenize(valid_df)
    train_dataset = PIIDataset(tokenized_train)
    valid_dataset = PIIDataset(tokenized_valid)

    collator = DataCollatorForTokenClassification(
        tokenizer, pad_to_multiple_of=16
    )

    args = TrainingArguments(
        output_dir=OUTPUT_DIR,
        fp16=True,
        gradient_accumulation_steps=16,
        logging_steps=100,
        warmup_ratio=0.05,
        learning_rate=lr,          # tune
        num_train_epochs=3,
        per_device_train_batch_size=1,
        per_device_eval_batch_size=1,
        report_to="wandb",
        evaluation_strategy="steps",
        eval_steps=100,
        save_strategy="steps",
        save_steps=100,
        save_total_limit=1,
        overwrite_output_dir=True,
        load_best_model_at_end=True,
        lr_scheduler_type="linear",
        metric_for_best_model="f5",
        greater_is_better=True,
        weight_decay=0.01,
        save_only_model=True,
        neftune_noise_alpha=0.1
    )

    model = AutoModelForTokenClassification.from_pretrained(
        TRAINING_MODEL_PATH,
        num_labels=len(all_labels),
        id2label=id2label,
        label2id=label2id,
        ignore_mismatched_sizes=True
    )

    trainer = CustomTrainer(
        model=model,
        args=args,
        train_dataset=train_dataset,
        eval_dataset=valid_dataset,
        data_collator=collator,
        tokenizer=tokenizer,
        compute_metrics=partial(compute_metrics, all_labels=all_labels),
        awp_lr=awp_lr,
        awp_eps=awp_eps,
        awp_start_epoch=awp_start_epoch,
        ce_weight=ce_weight
    )

    trainer.train()


if __name__ == "__main__":
    main()
