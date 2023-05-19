import datasets


def load_pg19_data(train_dataset_percent: int = 7) -> datasets.Dataset:
    dataset = datasets.load_dataset(
        "pg19",
        split={
            "train": f"train[:{train_dataset_percent}%]",
            "dev": "validation",
            "test": "test[:1]",
        },
    )
    return dataset
