"""
Refered https://github.com/salesforce/awd-lstm-lm/blob/master/data/enwik8/prep_enwik8.py
"""
from typing import Dict

from datasets import Dataset, DatasetDict, load_dataset

# fmt: off
CHAR_INDICES = ['9', '32', '33', '34', '35', '36', '37', '38', '39', '40', '41', '42', '43', '44', '45', '46', '47', '48', '49', '50', '51', '52', '53', '54', '55', '56', '57', '58', '59', '60', '61', '62', '63', '64', '65', '66', '67', '68', '69', '70', '71', '72', '73', '74', '75', '76', '77', '78', '79', '80', '81', '82', '83', '84', '85', '86', '87', '88', '89', '90', '91', '92', '93', '94', '95', '96', '97', '98', '99', '100', '101', '102', '103', '104', '105', '106', '107', '108', '109', '110', '111', '112', '113', '114', '115', '116', '117', '118', '119', '120', '121', '122', '123', '124', '125', '126', '128', '129', '130', '131', '132', '133', '134', '135', '136', '137', '138', '139', '140', '141', '142', '143', '144', '145', '146', '147', '148', '149', '150', '151', '152', '153', '154', '155', '156', '157', '158', '159', '160', '161', '162', '163', '164', '165', '166', '167', '168', '169', '170', '171', '172', '173', '174', '175', '176', '177', '178', '179', '180', '181', '182', '183', '184', '185', '186', '187', '188', '189', '190', '191', '194', '195', '196', '197', '198', '199', '200', '201', '202', '203', '204', '205', '206', '207', '208', '209', '210', '211', '212', '213', '214', '215', '216', '217', '218', '219', '220', '222', '224', '225', '226', '227', '228', '229', '230', '231', '232', '233', '234', '235', '236', '237', '239', '240']
CHAR_TO_INDEX = {c: i for i, c in enumerate(CHAR_INDICES)}
VOCAB_SIZE = len(CHAR_INDICES)
assert VOCAB_SIZE == 204
# fmt: on


def enwik8_tokenize(text: str) -> Dict:
    input_ids = [CHAR_TO_INDEX[c] for c in text.split()]
    return {"input_ids": input_ids, "attention_mask": [1.0] * len(input_ids)}


def load_enwik8_data() -> Dataset:
    dataset = load_dataset("enwik8", "enwik8-raw")["train"]

    num_test_chars = 5000000

    def _preprocess(data):
        whole_text = data["text"]
        whole_bytes = whole_text.encode()

        train_data = whole_bytes[: -2 * num_test_chars]
        valid_data = whole_bytes[-2 * num_test_chars : -num_test_chars]
        test_data = whole_bytes[-num_test_chars:]

        train, dev, test = (
            " ".join([str(c) if c != ord("\n") else "\n" for c in part]) for part in (train_data, valid_data, test_data)
        )

        return {"train": train, "dev": dev, "test": test}

    dataset = dataset.map(_preprocess, remove_columns=dataset.column_names, load_from_cache_file=True)

    train = dataset["train"][0]
    dev = dataset["dev"][0]
    test = dataset["test"][0]

    def _gen(source):
        yield {"text": source}

    train_dataset = Dataset.from_generator(_gen, gen_kwargs={"source": train})
    dev_dataset = Dataset.from_generator(_gen, gen_kwargs={"source": dev})
    test_dataset = Dataset.from_generator(_gen, gen_kwargs={"source": test})
    dataset = DatasetDict({"train": train_dataset, "dev": dev_dataset, "test": test_dataset})
    return dataset
