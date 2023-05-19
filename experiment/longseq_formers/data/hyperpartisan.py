import re
from collections import defaultdict
from typing import Dict, List

import datasets
from bs4 import BeautifulSoup

from ..dataset.classification import ClassificationDatum

# fmt: off
# hp-splits from longformer (https://github.com/allenai/longformer/blob/master/scripts/hp-splits.json)
HYPERPARTISAN_SPLITS = {
    "train": [239, 342, 401, 424, 518, 374, 457, 81, 208, 216, 112, 77, 448, 596, 388, 505, 362, 180, 587, 398, 636, 297, 363, 389, 148, 567, 163, 549, 472, 26, 427, 227, 213, 470, 346, 383, 585, 352, 22, 20, 390, 3, 97, 439, 637, 197, 392, 480, 225, 414, 333, 561, 615, 359, 598, 107, 12, 195, 54, 459, 23, 455, 624, 233, 17, 499, 307, 416, 578, 568, 220, 334, 65, 73, 170, 215, 447, 446, 606, 276, 502, 534, 582, 241, 425, 356, 192, 301, 514, 589, 466, 207, 82, 201, 391, 366, 476, 594, 477, 126, 393, 508, 158, 483, 604, 206, 15, 353, 372, 512, 543, 330, 290, 539, 444, 399, 410, 169, 125, 487, 74, 381, 479, 556, 292, 576, 224, 173, 441, 205, 29, 559, 509, 552, 317, 231, 296, 643, 524, 209, 433, 397, 488, 18, 553, 149, 380, 168, 484, 234, 586, 486, 555, 232, 246, 373, 139, 458, 157, 644, 257, 91, 53, 59, 341, 159, 36, 109, 2, 106, 485, 258, 422, 404, 313, 402, 183, 419, 283, 87, 351, 75, 187, 310, 320, 19, 304, 38, 471, 129, 66, 151, 266, 268, 548, 328, 405, 371, 580, 51, 492, 474, 510, 468, 396, 308, 408, 526, 622, 511, 63, 274, 531, 128, 368, 599, 426, 43, 360, 541, 454, 263, 407, 138, 76, 530, 517, 165, 641, 436, 493, 326, 194, 202, 546, 238, 382, 92, 52, 120, 437, 71, 504, 532, 237, 314, 625, 617, 605, 171, 331, 456, 607, 542, 55, 475, 584, 251, 611, 40, 122, 100, 570, 338, 137, 597, 101, 324, 95, 577, 31, 116, 176, 145, 211, 236, 627, 143, 638, 620, 219, 10, 60, 198, 7, 293, 452, 590, 579, 141, 558, 160, 214, 166, 593, 538, 33, 364, 635, 119, 250, 223, 319, 619, 339, 616, 618, 284, 533, 603, 302, 49, 588, 572, 575, 515, 21, 1, 103, 150, 529, 506, 69, 343, 323, 482, 222, 535, 188, 14, 299, 489, 108, 140, 39, 420, 285, 86, 554, 259, 564, 400, 269, 281, 248, 272, 24, 629, 130, 226, 525, 80, 117, 115, 305, 370, 465, 186, 93, 113, 46, 461, 378, 184, 336, 50, 309, 48, 72, 495, 131, 507, 325, 298, 412, 406, 240, 278, 212, 279, 5, 90, 181, 8, 288, 61, 300, 174, 608, 58, 520, 449, 218, 294, 354, 494, 417, 99, 154, 89, 527, 273, 11, 162, 610, 179, 56, 613, 329, 377, 335, 253, 501, 442, 252, 614, 327, 98, 88, 631, 609, 547, 376, 581, 621, 152, 228, 4, 565, 540, 132, 110, 191, 30, 6, 189, 303, 270, 255, 415, 172, 64, 267, 503, 78, 118, 235, 435, 167, 453, 282, 573, 291, 642, 123, 395, 551, 94, 450, 478, 311, 289, 153, 102, 421, 277, 583, 164, 244, 229, 178, 217, 523, 96, 280, 68, 497, 430, 190, 516, 445, 428, 633, 536, 434, 387, 355, 528, 287, 144, 210, 295, 385, 185, 467, 256, 44, 83, 67, 175, 204, 602, 42, 358, 384, 28, 45, 569, 127, 47, 491, 265, 463, 121, 135, 460],
    "dev": [182, 438, 545, 286, 142, 27, 394, 261, 411, 0, 79, 550, 640, 254, 560, 386, 62, 440, 104, 473, 155, 432, 124, 133, 136, 519, 322, 318, 245, 249, 612, 349, 623, 591, 429, 306, 592, 375, 203, 544, 312, 114, 41, 344, 571, 134, 462, 347, 464, 566, 350, 199, 562, 357, 361, 521, 574, 315, 243, 601, 260, 409, 337, 177],
    "test": [537, 517, 23, 459, 593, 258, 227, 16, 204, 367, 159, 142, 214, 82, 182, 564, 411, 600, 610, 306, 21, 434, 625, 197, 202, 489, 404, 400, 551, 320, 36, 435, 344, 183, 134, 19, 253, 231, 383, 572, 201, 528, 15, 116, 265, 221, 462, 342, 465, 436, 490, 442, 547, 282, 535, 256, 160, 140, 555, 51, 540, 165, 504, 181, 147],
}
# fmt: on


def load_hyperpartisan_data() -> Dict[str, List[ClassificationDatum]]:
    """Load Hyperpartisan dataset

    Returns:
        datasets like below
        {
            "train": [{
                "text": "...",
                "label": 1
            }, {...}, ...],
            "dev": ...,
            "test": ...
        }
    """
    data = datasets.load_dataset("hyperpartisan_news_detection", "byarticle")["train"]

    split_datasets = defaultdict(list)
    for split, indices in HYPERPARTISAN_SPLITS.items():
        for index in indices:
            datum = data[index]
            normalized_text = BeautifulSoup(datum["text"], "html.parser").get_text()
            text = re.sub(r"\s+", " ", normalized_text)
            label = int(datum["hyperpartisan"])
            split_datasets[split].append({"text": text, "label": label})
    return split_datasets