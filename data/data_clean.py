import pandas as pd
df = pd.read_csv("train.csv")

# 根据data_conflict_detect.py的结果，id=500396175753633792的数据错误，被删除
df = df[df["id"] != 500396175753633792]

df.to_csv("train_cleaned.csv", index=False)
print(f"清洗后样本数: {len(df)}")