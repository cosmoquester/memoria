import datasets


def load_pg19_data(train_dataset_percent: int = 7) -> datasets.Dataset:
    dataset = datasets.load_dataset(
        "pg19",
        revision="dd75f494ab94328d0ce92c05390ab91a96920a9d",
        split={
            "train": f"train[:{train_dataset_percent}%]",
            "dev": "validation",
            "test": "test",
        },
    )
    return dataset
