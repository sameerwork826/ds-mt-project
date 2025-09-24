import json
import random

input_file = "data/train_en_hi.jsonl"
train_file = "data/train.jsonl"
valid_file = "data/valid.jsonl"
test_file = "data/test.jsonl"

# Set your split ratios
train_ratio = 0.8
valid_ratio = 0.1
test_ratio = 0.1

with open(input_file, "r", encoding="utf-8") as f:
    lines = f.readlines()

random.shuffle(lines)
total = len(lines)
train_end = int(total * train_ratio)
valid_end = train_end + int(total * valid_ratio)

train_lines = lines[:train_end]
valid_lines = lines[train_end:valid_end]
test_lines = lines[valid_end:]

with open(train_file, "w", encoding="utf-8") as f:
    f.writelines(train_lines)
with open(valid_file, "w", encoding="utf-8") as f:
    f.writelines(valid_lines)
with open(test_file, "w", encoding="utf-8") as f:
    f.writelines(test_lines)

print(f"Split complete: {len(train_lines)} train, {len(valid_lines)} valid, {len(test_lines)} test")