from .enwik8 import enwik8_tokenize, load_enwik8_data
from .hyperpartisan import load_hyperpartisan_data
from .pg19 import load_pg19_data
from .wikitext103 import load_wikitext103_data

CLASSIFICATION_DATASETS = ["hyperpartisan"]
LANGUAGE_MODELING_DATASETS = ["wikitext103", "pg19", "enwik8"]
DATASETS = CLASSIFICATION_DATASETS + LANGUAGE_MODELING_DATASETS

__all__ = [
    "enwik8_tokenize",
    "load_enwik8_data",
    "load_hyperpartisan_data",
    "load_pg19_data",
    "load_wikitext103_data",
    "DATASETS",
    "CLASSIFICATION_DATASETS",
    "LANGUAGE_MODELING_DATASETS",
]
