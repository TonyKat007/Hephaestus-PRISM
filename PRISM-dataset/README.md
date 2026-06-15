---
license: apache-2.0
task_categories:
- text-generation
- conversational
language:
- en
size_categories:
- n>1M
dataset_info:
  features:
  - name: messages
    list:
    - name: role
      dtype: string
    - name: content
      dtype: string
  splits:
  - name: train
    num_bytes: 550377845
    num_examples: 2035791
  - name: val
    num_bytes: 129059978
    num_examples: 476634
  download_size: 40308864
  dataset_size: 679437823
configs:
- config_name: default
  data_files:
  - split: train
    path: data/train-*
  - split: val
    path: data/val-*
---

# aLLoyM Training Dataset

This dataset was used to fine-tune the aLLoyM model (Mistral-based).

## Dataset Statistics


## Format

The dataset is in JSONL format where each line contains:

```json
{
  "messages": [
    {"role": "system", "content": "System instruction"},
    {"role": "user", "content": "User question"},
    {"role": "assistant", "content": "Assistant response"}
  ]
}
```

## Usage

```python
import json
from datasets import Dataset

# Load the dataset
data = []
with open("train.jsonl", "r") as f:
    for line in f:
        data.append(json.loads(line))

# Convert to Hugging Face Dataset
dataset = Dataset.from_list(data)
print(f"Dataset size: {len(dataset)}")
```

## License

Apache 2.0
