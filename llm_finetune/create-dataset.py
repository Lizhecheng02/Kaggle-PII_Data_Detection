import pandas as pd
import os
from tqdm import tqdm

df1 = pd.read_json("labeled_Fake_data_same_nb_1000_0.json")
df2 = pd.read_json("labeled_nb_mixtral-8x7b-v1_no-email.json")
df3 = pd.read_json("test_other_name.json")
df4 = pd.read_json("train_other_name.json")

df = pd.concat([df1, df2, df3, df4])
df = df.reset_index(drop=True)
print(df.shape)

train_df = pd.DataFrame(
    columns=["full_text", "entity", "label output", "label"]
)

student_name_count = 0
other_name_count = 0
other_label_count = 0

non_name_b_labels = [
    "B-EMAIL", "B-USERNAME", "B-ID_NUM",
    "B-PHONE_NUM", "B-URL_PERSONAL", "B-STREET_ADDRESS"
]

non_name_i_labels = [
    "I-EMAIL", "I-USERNAME", "I-ID_NUM",
    "I-PHONE_NUM", "I-URL_PERSONAL", "I-STREET_ADDRESS"
]

for idx, row in tqdm(df.iterrows(), total=len(df)):
    full_text = row["full_text"]
    student_name_dict = {}
    other_name_dict = {}
    other_label_dict = {}

    labels = row["labels"]
    # print(type(labels), len(labels))
    tokens = row["tokens"]
    # print(type(tokens), len(tokens))
    trailing_whitespace = row["trailing_whitespace"]
    # print(type(trailing_whitespace), len(trailing_whitespace))

    i = 0
    while i < len(labels):
        if labels[i] == "B-NAME_STUDENT":
            # print(1)
            start_idx = i
            end_idx = i
            while i < len(labels) - 1:
                i = i + 1
                if labels[i] == "I-NAME_STUDENT":
                    end_idx = i
                else:
                    break
            new_tokens = tokens[start_idx:end_idx + 1]
            new_trailing_whitespace = trailing_whitespace[start_idx:end_idx + 1]
            new_student_name = "".join([token + " " * space for token, space in zip(new_tokens, new_trailing_whitespace)]).strip()
            if new_student_name in student_name_dict:
                pass
            else:
                student_name_dict[new_student_name] = 1
                train_df_row = pd.DataFrame({
                    "full_text": [full_text],
                    "entity": [new_student_name],
                    "label output": ["Yes, this is a student name"],
                    "label": ["student_name"]
                })
                train_df = pd.concat(
                    [train_df, train_df_row],
                    ignore_index=True
                )
                train_df.to_csv("llm_train.csv", index=False)
                student_name_count += 1
        else:
            i = i + 1

    i = 0
    while i < len(labels):
        if labels[i] == "B-OTHER_NAME":
            start_idx = i
            end_idx = i
            while i < len(labels) - 1:
                i = i + 1
                if labels[i] == "I-OTHER_NAME":
                    end_idx = i
                else:
                    break
            new_tokens = tokens[start_idx:end_idx + 1]
            new_trailing_whitespace = trailing_whitespace[start_idx:end_idx + 1]
            new_other_name = "".join([token + " " * space for token, space in zip(new_tokens, new_trailing_whitespace)]).strip()
            if new_other_name in other_name_dict:
                pass
            else:
                other_name_dict[new_other_name] = 1
                train_df_row = pd.DataFrame({
                    "full_text": [full_text],
                    "entity": [new_other_name],
                    "label output": ["No, it's a name, but it's not a student name."],
                    "label": ["other_name"]
                })
                train_df = pd.concat(
                    [train_df, train_df_row],
                    ignore_index=True
                )
                train_df.to_csv("llm_train.csv", index=False)
                other_name_count += 1
        else:
            i = i + 1

    i = 0
    while i < len(labels):
        if labels[i] in non_name_b_labels:
            start_label = labels[i]
            start_idx = i
            end_idx = i
            while i < len(labels) - 1:
                i = i + 1
                if labels[i] in non_name_i_labels and labels[1:] == start_label[1:]:
                    end_idx = i
                else:
                    break
            new_tokens = tokens[start_idx:end_idx + 1]
            new_trailing_whitespace = trailing_whitespace[start_idx:end_idx + 1]
            new_other_label = "".join([token + " " * space for token, space in zip(new_tokens, new_trailing_whitespace)]).strip()
            if new_other_label in other_label_dict:
                pass
            else:
                other_label_dict[new_other_label] = 1
                train_df_row = pd.DataFrame({
                    "full_text": [full_text],
                    "entity": [new_other_label],
                    "label output": ["No, it's even not a name."],
                    "label": ["not_name"]
                })
                train_df = pd.concat(
                    [train_df, train_df_row],
                    ignore_index=True
                )
                train_df.to_csv("llm_train.csv", index=False)
                other_label_count += 1
        else:
            i = i + 1

print("The total number of rows for student name is:", student_name_count)
print("The total number of rows for other name is:", other_name_count)
print("The total number of rows for other label is:", other_label_count)
