import re
import pandas as pd
from collections import defaultdict

df = pd.read_csv("train.csv")

# ============ 1. 清除所有 URL ============
url_pattern = re.compile(r"https?://\S+")
df["text"] = df["text"].apply(lambda t: url_pattern.sub("", t).strip())

# ============ 2. 去重：标签冲突全删，同标签只留一条 ============
before = len(df)
conflict_texts = df.groupby("text")["label"].apply(lambda x: set(x)).loc[lambda s: s.apply(len) > 1]

for text, labels in conflict_texts.items():
    rows = df[df["text"] == text]
    print(f"[标签冲突]: {text[:60]}")
    for _, r in rows.iterrows():
        print(f"  id={r['id']}, label={r['label']}, event={r['event']}")
    df = df[df["text"] != text]

if len(conflict_texts) == 0:
    print("URL清除后, 未发现新的标签冲突。")

df = df.drop_duplicates(subset="text", keep="first")
after = len(df)

df.to_csv("train_cleaned.csv", index=False)
print(f"\n清洗完成。保存至 train_cleaned.csv, 共 {after} 条 (删除 {before - after} 条)")
print("说明: 所有文本中的 URL 已被移除.")