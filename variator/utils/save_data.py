import datasets
import os

os.makedirs("glue", exist_ok=True)
for dataname in ["mnli", "mrpc", "qnli", "qqp", "rte", "sst2", "qnli"]:
    # for split in ["train", "validation"]:
    data = datasets.load_dataset("glue", dataname)
    data.save_to_disk(f"glue/{dataname}")
