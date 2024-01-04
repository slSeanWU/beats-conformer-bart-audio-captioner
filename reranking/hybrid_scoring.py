import json
import os
import sys
from collections import defaultdict

sys.path.append(
    os.path.dirname(__file__)
    + "/../caption_evaluation_tools/coco_caption/pycocoevalcap/"
)
sys.path.append(
    os.path.dirname(__file__)
    + "/../caption_evaluation_tools/coco_caption/pycocoevalcap/fense"
)

import numpy as np
import torch
import pandas as pd

from fense.evaluator import Evaluator

fense = Evaluator(
    device="cuda" if torch.cuda.is_available() else "cpu", sbert_model=None
)

INFERENCE_DIR = sys.argv[1]
DO_NORMALIZE = False

if len(sys.argv) > 2:
    OUT_DIR = sys.argv[2]
    if not os.path.exists(OUT_DIR):
        os.makedirs(OUT_DIR)
else:
    OUT_DIR = INFERENCE_DIR

if len(sys.argv) > 3:
    DO_NORMALIZE = sys.argv[3] == "True"

ingredients = [
    "decoder",
    "encoder",
]
ing_multiplier = [
    -1,
    1,
]
weights = [
    0.3,
    0.7,
]

score_names = [
    "normed_nll",
    "audio_text_similarity",
]


def patch_spider_score(text, spider_fl=None):
    # print(text)
    spiders = []

    try:
        fl_err = fense.detect_error_sents(text, batch_size=len(text) // 2 + 1)
    except:
        print(text, file=open("tmp/error_fense_text.log", "a"))
        # exit()
        fl_err = np.array([0 for _ in range(len(text))])

    # print(fl_err)
    if spider_fl is not None:
        assert len(text) == len(spider_fl)
        for i in range(len(fl_err)):
            if fl_err[i]:
                spiders.append(10 * spider_fl[i])
            else:
                spiders.append(spider_fl[i])

        assert len(fl_err) == len(spiders)
    else:
        spiders = None

    return spiders, list(fl_err)


if __name__ == "__main__":
    all_samples = None

    beam_diff_indices = set()

    for s, ing in enumerate(ingredients):
        reranked_gens = json.load(
            open(os.path.join(INFERENCE_DIR, f"gen_captions_{ing}_reranked.json"))
        )

        if all_samples is None:
            all_samples = [dict() for _ in range(len(reranked_gens))]

        for i in range(len(reranked_gens)):
            samples = reranked_gens[i]["generated_captions"]
            seen_samples = set()
            for j in range(len(samples)):
                if samples[j]["text"] not in seen_samples:
                    seen_samples.add(samples[j]["text"])

                    if samples[j]["text"] not in all_samples[i]:
                        all_samples[i][samples[j]["text"]] = {
                            "text": samples[j]["text"],
                            "scores": [ing_multiplier[s] * samples[j][score_names[s]]],
                        }
                    else:
                        all_samples[i][samples[j]["text"]]["scores"].append(
                            ing_multiplier[s] * samples[j][score_names[s]]
                        )

    if DO_NORMALIZE:
        for i in range(len(reranked_gens)):
            for s in range(len(score_names)):
                unnormed_scores = np.array(
                    [
                        all_samples[i][text]["scores"][s]
                        for text in sorted(list(all_samples[i].keys()))
                    ]
                )
                texts = [
                    all_samples[i][text]["text"]
                    for text in sorted(list(all_samples[i].keys()))
                ]

                normed_scores = unnormed_scores - np.mean(unnormed_scores)
                normed_scores /= np.std(unnormed_scores)

                for j in range(len(normed_scores)):
                    all_samples[i][texts[j]]["scores"][s] = normed_scores[j]

                # print(len(all_samples[i]))

    all_gens = [list(v.values()) for v in all_samples]
    print(len(all_gens), len(all_gens[0]))

    for i in range(len(all_gens)):
        for j in range(len(all_gens[i])):
            all_gens[i][j]["weighted_score"] = float(
                np.dot(weights, all_gens[i][j]["scores"])
            )
            # print(all_gens[i][j]["weighted_score"])

        all_gens[i] = sorted(
            all_gens[i], key=lambda x: x["weighted_score"], reverse=True
        )

    print(json.dumps(all_gens[0], indent=2))

    reranked_spider_fl = [0.0 for _ in range(len(all_gens))]
    reranked_spider = []

    out_dict = dict()

    for i in range(len(all_gens)):
        print(i)

        spider_scores, is_disfluent = patch_spider_score(
            [x["text"] for x in all_gens[i]],
        )

        select_idx = None
        # assert len(is_disfluent) == len(oracle_scores[i]["generated_captions"])
        # print(is_disfluent)

        for j in range(len(is_disfluent)):
            if not is_disfluent[j]:
                select_idx = j
                break

        out_dict[reranked_gens[i]["audio_file"]] = all_gens[i][select_idx]["text"]

        assert len(out_dict)

        true_out_dict = {"file_name": [], "caption_predicted": []}
        for fname in out_dict.keys():
            true_out_dict["file_name"].append(fname)
            true_out_dict["caption_predicted"].append(out_dict[fname])

    print("FENSE used")
    print(f"{'do normalize':20}: {DO_NORMALIZE}\n")
    print(f"{'score components':20}: {ingredients}\n")
    print(f"{'weights':20}: {weights}\n")

    with open(os.path.join(OUT_DIR, "source_model.log"), "a") as f:
        f.write(f"{INFERENCE_DIR}\n")

    with open(os.path.join(OUT_DIR, "reranking_config.log"), "a") as f:
        if DO_NORMALIZE:
            f.write(f"{'do normalize':20}: {DO_NORMALIZE}\n")

        f.write(f"FENSE used\n")
        f.write(f"{'score components':20}: {ingredients}\n")
        f.write(f"{'weights':20}: {weights}\n")
        f.write("\n")

    df_out = pd.DataFrame.from_dict(true_out_dict)
    df_out.to_csv(
        os.path.join(
            OUT_DIR,
            f"hybrid_score_{weights}_output.csv",
        ),
        index=False,
    )
