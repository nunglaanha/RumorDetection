import re
import pandas as pd
from collections import defaultdict

df = pd.read_csv("train.csv")

# ============ 1. 清除所有 URL ============
url_pattern = re.compile(r"https?://\S+")
df["text"] = df["text"].apply(lambda t: url_pattern.sub("", t).strip())

# ============ 2. 重新检测标签冲突 ============
rows = df.to_dict("records")
groups = defaultdict(list)
for r in rows:
    groups[r["text"]].append(r)

conflict_found = False
for text, group in groups.items():
    labels = set(r["label"] for r in group)
    if len(labels) > 1:
        conflict_found = True
        print(f"[冲突文本]: {text[:60]}")
        for r in group:
            print(f"  id={r['id']}, label={r['label']}, event={r['event']}")

if not conflict_found:
    print("URL清除后, 未发现新的标签冲突。")

# ============ 4. 保存清洗结果 ============
df.to_csv("train_cleaned.csv", index=False)
print(f"\n清洗完成。保存至 train_cleaned.csv, 共 {len(df)} 条")
print("说明: 所有文本中的 URL 已被移除.")