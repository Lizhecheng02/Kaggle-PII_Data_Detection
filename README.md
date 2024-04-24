## This Repo is for [Kaggle - PII Data Detection](https://www.kaggle.com/competitions/pii-detection-removal-from-educational-data)

### Python Environment

#### 1. Install Packages

```b
pip install -r requirements.txt
```

### Prepare Data

#### 1. Set Kaggle Api

```bash
export KAGGLE_USERNAME="your_kaggle_username"
export KAGGLE_KEY="your_api_key"
```

#### 2. Download Dataset

```bash
cd kaggle_dataset
kaggle datasets download -d lizhecheng/pii-data-detection-dataset
unzip pii-data-detection-dataset.zip
kaggle datasets download -d lizhecheng/piidd-reliable-cv
unzip piidd-reliable-cv.zip
```

```bash
cd kaggle_dataset
cd competition
kaggle competitions download -c pii-detection-removal-from-educational-data
unzip pii-detection-removal-from-educational-data.zip
```

#### 3. Simple Deberta

```bash
cd kaggle_notebook
run train-0.3-validation.ipynb
```


#### 4. Run Wandb Sweep
```bash
cd models
wandb sweep --project PII config.yaml
```

```bash
wandb agent xxx/PII/xxxxxxxx
```

#### 5. Four-Fold Cross Validation

```
cd kfold
wandb login --relogin
(input your wandb api key)
wandb init -e (input your wandb username)

export KFOLD=0/1/2/3
wandb sweep --project PII hypertuning_kfold.yaml
wandb agent xxx/PII/xxxxxxxx
```



## This is [Public 9th Private 25th Solution] For [The Learning Agency Lab - PII Data Detection](https://www.kaggle.com/competitions/pii-detection-removal-from-educational-data)

First of all, thank you to ``Kaggle`` and ``THE LEARNING AGENCY LAB`` for hosting this competition, and thanks to everyone in the team for their efforts. Although the result was not perfect, we still learned a lot, and we will continue to move forward. Congratulations to all the winners!

### Full Code

Here is the ``GitHub Repo`` for this competition, where you can find almost all of the codes: [https://github.com/Lizhecheng02/Kaggle-PII_Data_Detection](https://github.com/Lizhecheng02/Kaggle-PII_Data_Detection)

### Fine-tuning

- ``AWP (Adversarial Weight Perturbation)``

​	Enhance the model's robustness by using a custom ``AWP`` class and write own ``CustomTrainer``. This is a method our team often uses in NLP competitions, and it does have some good results. (The corresponding code can be found on my GitHub under ``models`` directory).

- ``Wandb Sweep``

​	With this tool, we can try various combinations of different hyperparameters to select the ones that produce the best fine-tuning results. (The corresponding code can be found on my GitHub under ``models`` directory).

- ``Replace \n\n with | in all documents``

​	In this case, we trained a set of models with both 4-fold cross-validation and LB scores of 0.977. Although there was some improvement on the LB, the results showed no improvement on the PB.

### Post-processing

- Correct the incorrect labels for the order of student names. (**B, B -> B, I**)
- Pay special attention to ``\n`` appearing in addresses, which should be labeled with the **I** label.
- Filter out email addresses and phone numbers, removing results that are clearly not part of these two categories. (**No significant improvement**)
- Handle the cases where titles like ``Dr.`` are predicted with the **B** label. (**No significant improvement**)

```
def pp(new_pred_df):
    df = new_pred_df.copy()
    i = 0
    while i < len(df):
        st = i
        doc = df.loc[st, "document"]
        tok = df.loc[st, "token"]
        pred_tok = df.loc[st, "label"]
        if pred_tok == 'O':
            i += 1
            continue
        lab = pred_tok.split('-')[1]
        cur_doc = doc
        cur_lab = lab
        last_tok = tok
        cur_tok = last_tok

        while i < len(df) and cur_doc == doc and cur_lab == lab and last_tok == cur_tok:
            last_tok = cur_tok + 1
            i += 1
            cur_doc = df.loc[i, "document"]
            cur_tok = df.loc[i, "token"]
            if i >= len(df) or df.loc[i, "label"] == 'O':
                break
            cur_lab = df.loc[i, "label"].split('-')[1]

        if st - 2 >= 0 and df.loc[st - 2, "document"] == df.loc[st, "document"] and df.loc[st - 1, "token_str"] == '\n' and df.loc[st - 2, "label"] != 'O' and df.loc[st - 2, "label"].split('-')[1] == lab:
            df.loc[st - 1, "label"] = 'I-' + lab
            df.loc[st - 1, "score"] = 1
            for j in range(st, i):
                if df.loc[j, "label"] != 'I-' + lab:
                    df.loc[j, "score"] = 1
                    df.loc[j, "label"] = 'I-' + lab
            continue

        for j in range(st, i):
            if j == st:
                if df.loc[j, "label"] != 'B-' + lab:
                    df.loc[j, "score"] = 1
                    df.loc[j, "label"] = 'B-' + lab
            else:
                if df.loc[j, "label"] != 'I-' + lab:
                    df.loc[j, "score"] = 1
                    df.loc[j, "label"] = 'I-' + lab

        if lab == 'NAME_STUDENT' and any(len(item) == 2 and item[0].isupper() and item[1] == "." for item in df.loc[st:i-1, 'token_str']):
            for j in range(st, i):
                df.loc[j, "score"] = 0
                df.loc[j, "label"] = 'O'

    return df
```

### Ensemble

- ``Average Ensemble``

​	Use the method of taking the average of probabilities to obtain the final result. Since recall is more important than precision in this competition, I set the threshold to 0.0 in order to avoid missing any potential correct recall.

```
for text_id in final_token_pred:
    for word_idx in final_token_pred[text_id]:
        pred = final_token_pred[text_id][word_idx].argmax(-1)
        pred_without_O = final_token_pred[text_id][word_idx][:12].argmax(-1)
        if final_token_pred[text_id][word_idx][12] < 0.0:
            final_pred = pred_without_O
            tmp_score = final_token_pred[text_id][word_idx][final_pred]
        else:
            final_pred = pred
            tmp_score = final_token_pred[text_id][word_idx][final_pred]
```

- ``Vote Ensemble``

​	In our final submission, we ensembled 7 models, and we accepted a label as the correct prediction if at least two of the models predicted that same label.

```
for tmp_pred in single_pred:
    for text_id in tmp_pred:
        max_id = 0
        for word_idx in tmp_pred[text_id]:
            max_id = tmp_pred[text_id][word_idx].argmax(-1)
            tmp_pred[text_id][word_idx] = np.zeros(tmp_pred[text_id][word_idx].shape)
            tmp_pred[text_id][word_idx][max_id] = 1.0
        for word_idx in tmp_pred[text_id]:
            final_token_pred[text_id][word_idx] += tmp_pred[text_id][word_idx]
```

```
for text_id in final_token_pred:
    for word_idx in final_token_pred[text_id]:
        pred = final_token_pred[text_id][word_idx].argmax(-1)
        pred_without_O = final_token_pred[text_id][word_idx][:12].argmax(-1)
        if final_token_pred[text_id][word_idx][pred] >= 2:
            final_pred = pred
            tmp_score = final_token_pred[text_id][word_idx][final_pred]
        else:
            final_pred = 12
            tmp_score = final_token_pred[text_id][word_idx][final_pred]
```

### Inference

- ``Two GPU Inference``

​	Using T4*2 GPUs doubles the inference speed compared to a single GPU. To ensemble 8 models, the maximum max_length is ``896``; if ensemble 7 models, the max_length can be set to ``1024``, which is a more ideal value. (The corresponding code can be found on my GitHub under ``submissions`` directory).

- ``Convert Non-English Characters`` (Make the LB lower)

```
def replace_non_english_chars(text):
    mapping = {
        'à': 'a', 'á': 'a', 'â': 'a', 'ã': 'a', 'ä': 'a', 'å': 'a',
        'è': 'e', 'é': 'e', 'ê': 'e', 'ë': 'e',
        'ì': 'i', 'í': 'i', 'î': 'i', 'ï': 'i',
        'ò': 'o', 'ó': 'o', 'ô': 'o', 'õ': 'o', 'ö': 'o', 'ø': 'o',
        'ù': 'u', 'ú': 'u', 'û': 'u', 'ü': 'u',
        'ÿ': 'y',
        'ç': 'c',
        'ñ': 'n',
        'ß': 'ss'
    }

    result = []
    for char in text:
        if char not in string.ascii_letters:
            replacement = mapping.get(char.lower())
            if replacement:
                result.append(replacement)
            else:
                result.append(char)
        else:
            result.append(char)

    return ''.join(result)
```



### Two-stage LLM (Unsuccessful)

We have annotated approximately 10,000 non-student names from the dataset using the ``GPT-4 ``API, since student names are the most common label type. We hope to enhance the model's accuracy in predicting this particular type of label.

I tried fine-tuning the ``Mistral-7b`` model on name-related labels, but the scores on the LB showed a significant decrease.

Therefore, I tried using ``Mistral-7b`` for few-shot learning to determine whether the content predicted to be the label ``name student`` is actually a name. (Here we cannot expect the model to distinguish whether it is a student's name or not, but only to exclude predictions that are clearly not names).

The prompt is in the below, doing this produced a very slight improvement on the LB, less than 0.001.

```
f"I'll give you a name, and you need to tell me if it's a normal person name, cited name or even not a name. Do not consider other factors.\nExample:\n- Is Matt Johnson a normal person name? Answer: Yes\n- Is Johnson. T a normal person name? Answer: No, this is likely a cited name.\n- Is Andsgjdu a normal person name? Answer: No, it is even not a name.\nNow the question is:\n- Is {name} a normal person name? Answer:"
```

### Submission

| Models                                                       | LB        | PB        | Choose  |
| ------------------------------------------------------------ | --------- | --------- | ------- |
| ``Seven single models that exceed 0.974 on the LB``          | ``0.978`` | ``0.964`` | **Yes** |
| ``Two 4-fold cross-validation models, with LB scores of 0.977 and 0.974 respectively.`` | ``0.978`` | ``0.961`` | **Yes** |
| ``Three single models with ensemble LB score of 0.979, plus one set of 4-fold cross-validation models with an LB score of 0.977. (Use vote ensemble)`` | ``0.979`` | ``0.963`` | **Yes** |
| ``Two single models ensemble``                               | ``0.972`` | ``0.967`` | **No**  |
| ``Four single models ensemble``                              | ``0.979`` | ``0.967`` | **No**  |

### Code

[LB 0.978 PB 0.964](https://www.kaggle.com/code/lizhecheng/ensemble-replace-and-no-replace)

[LB 0.978 PB 0.961](https://www.kaggle.com/code/lizhecheng/ensemble-replace-and-no-replace/notebook)

[LB 0.979 PB 0.963](https://www.kaggle.com/code/lizhecheng/vote-ensemble-replace-and-no-replace)

### Conclusion

Thanks to my teammates, we have known each other through Kaggle for over half a year. I feel fortunate to be able to learn and progress together with you all. [@rdxsun](https://www.kaggle.com/rdxsun), [@bianshengtao](https://www.kaggle.com/bianshengtao), [@xuanmingzhang777](https://www.kaggle.com/xuanmingzhang777), [@tonyarobertson](https://www.kaggle.com/tonyarobertson).

Going beyond is just as wrong as not going far enough.   ——Confucius 
