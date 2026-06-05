import pandas as pd

WORK_DIR = "."

df = pd.read_csv(f"{WORK_DIR}/train_cleaned.csv")

wrong_index = [500291056445845504, 498305825341845504, 498277678706077696, 544396587192311808]
df = df[~df["id"].isin(wrong_index)]

df.to_csv(f"{WORK_DIR}/train_cleaned.csv", index=False)
print(f"清洗完成，剩余数据: {len(df)}")