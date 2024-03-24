import os
import pandas as pd
from tqdm import tqdm

folder_path = "all_files"
combined_csv = pd.DataFrame(
    columns=["full_text", "entity", "label output", "label"]
)


for file in os.listdir(folder_path):
    if file.endswith(".csv"):
        file_path = os.path.join(folder_path, file)
        combined_csv = pd.concat(
            [combined_csv, pd.read_csv(file_path)],
            ignore_index=True
        )

combined_csv.to_csv("llm_train.csv", index=False)

df = pd.read_csv("llm_train.csv")
print(df.shape)
print(df["label"].value_counts())

df_student_name = df[df["label"] == "student_name"]
df_other_name = df[df["label"] == "other_name"]
df_student_name = df_student_name.sample(len(df_other_name))
df_not_name = df[df["label"] == "not_name"]
df = pd.concat([df_student_name, df_other_name], ignore_index=True)
df.reset_index(drop=True, inplace=True)
print(df.shape)
print(df["label"].value_counts())

texts = []
system_prompt = """
Suppose you are an expert at recogonizing student names in the long texts. Here is a long text written by a student:
"""
question_prompt = """
You can easily find {} in the above text, so is {} a student name in the text? Think step by step, only return 'Yes' or 'No', do not explain the reason.
"""
for idx, row in tqdm(df.iterrows(), total=len(df)):
    full_text = row["full_text"]
    entity = row["entity"]
    label_output = row["label output"]
    label = row["label"]

    if label == "student_name":
        text = "<s>[INST]" + system_prompt + "\n" + full_text + "\n" + question_prompt.format(entity, entity) + "[/INST]" + "Yes" + "</s>"
    elif label == "other_name":
        text = "<s>[INST]" + system_prompt + "\n" + full_text + "\n" + question_prompt.format(entity, entity) + "[/INST]" + "No" + "</s>"
    
    if idx == 0:
        print(text)
    texts.append(text)

df["text"] = texts
df.to_csv("llm_train_single_word_output.csv", index=False)
