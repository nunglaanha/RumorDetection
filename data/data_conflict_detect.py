import csv
from collections import defaultdict

with open("train.csv", encoding="utf-8") as f:
    rows = list(csv.DictReader(f))

# 按文本内容分组
groups = defaultdict(list)
for r in rows:
    groups[r["text"].strip()].append(r)

# 找出标签冲突
for text, group in groups.items():
    labels = set(r["label"] for r in group)
    if len(labels) > 1:
        print(f"[冲突] 文本: {text[:60]}")
        for r in group:
            print(f"  id={r['id']}, label={r['label']}, event={r['event']}")