import os
import sys
import json
import random
from typing import Dict, Union, Any
from string import punctuation

sys.path.append("model")
sys.path.append("data")
sys.path.append("caption_evaluation_tools")

import numpy as np
import pandas as pd
from transformers.modeling_utils import PreTrainedModel
from transformers.tokenization_utils import PreTrainedTokenizer

from data.clotho_captioning_dataset import ClothoCaptioningDataset
from caption_evaluation_tools.eval_metrics import evaluate_metrics_from_lists

inference_csv = sys.argv[1]
test_split = sys.argv[2]

strip_punct_table = str.maketrans("", "", punctuation)


def read_inference_csv(csv_path):
    df = pd.read_csv(csv_path)

    filenames = []
    captions = []

    for i, row in df.iterrows():
        filenames.append(row["file_name"])
        captions.append(row["caption_predicted"])

    return filenames, captions


if __name__ == "__main__":
    test_dset = ClothoCaptioningDataset(
        f"clotho/{test_split}",
        "facebook/bart-base",
        f"clotho/clotho_captions_{test_split}.csv",
    )

    gen_files, gen_captions = read_inference_csv(inference_csv)
    assert len(gen_files) == len(test_dset)

    # print(gen_captions[:10])

    # exit()

    true_captions = []
    for i in range(len(test_dset)):
        assert test_dset.idx_to_sample[i] == gen_files[i]
        samp_name = gen_files[i]

        true_captions.append(
            [
                s.strip().lower().translate(strip_punct_table)
                for s in test_dset.captions[samp_name]
            ]
        )

    eval_res = evaluate_metrics_from_lists(gen_captions, true_captions)
