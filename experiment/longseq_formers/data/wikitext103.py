import re

import datasets


def load_wikitext103_data() -> datasets.Dataset:
    dataset = datasets.load_dataset(
        "wikitext",
        "wikitext-103-raw-v1",
        split={"train": f"train", "dev": "validation", "test": "test"},
    )

    def _join_segment_text(example):
        whole_text = "".join(example["text"])
        start_idxs = [m.start() - 1 for m in re.finditer(r"\n\s*= [^=]+ =\s*\n", whole_text)]
        all_idxs = [0] + start_idxs + [len(whole_text)]
        segments = [whole_text[all_idxs[i] : all_idxs[i + 1]].strip() for i in range(len(all_idxs) - 1)]
        return {"text": segments}

    dataset = dataset.map(
        _join_segment_text,
        load_from_cache_file=True,
        batched=True,
        batch_size=len(dataset["train"]),
        drop_last_batch=False,
        remove_columns=["text"],
    )
    return dataset
