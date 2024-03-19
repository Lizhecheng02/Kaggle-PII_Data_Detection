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
df = pd.concat([df_student_name, df_other_name], ignore_index=True)
print(df.shape)
print(df["label"].value_counts())

texts = []
system_prompt = """
You are an expert at recogonizing student names from long texts. Now here is a long text written by a student:
"""
question_prompt = """
The question is that you can easily find {} in the above text, so is {} a student name in the text?
"""
for idx, row in tqdm(df.iterrows(), total=len(df)):
    full_text = row["full_text"]
    entity = row["entity"]
    label_output = row["label output"]

    text = system_prompt + "\n" + full_text + "\n" + \
        question_prompt.format(entity, entity) + "[/INST] " + label_output
    if idx == 0:
        print(text)
    texts.append(text)

df["text"] = texts
df.to_csv("llm_train.csv", index=False)
