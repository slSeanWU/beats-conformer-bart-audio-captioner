import os
import sys
import json
from string import punctuation

sys.path.append(os.path.dirname(__file__) + "/../")
sys.path.append(os.path.dirname(__file__) + "/../model")
sys.path.append(os.path.dirname(__file__) + "/../data")

import torch
import numpy as np
from InstructorEmbedding import INSTRUCTOR
from sklearn.metrics.pairwise import cosine_similarity
from transformers.models.bart.configuration_bart import BartConfig

from data.clotho_captioning_dataset import ClothoCaptioningDataset
from data.data_collator import ClothoCaptioningDataCollator
from model.modeling_beats_conformer_bart import (
    Wav2Vec2ConformerConfig,
    BeatsConformerBartSeq2SeqForCaptioning,
)

ckpt_dir = sys.argv[1]
inference_dir = sys.argv[2]
conformer_config_json = sys.argv[3]
test_split = sys.argv[4]


device = "cuda" if torch.cuda.is_available() else "cpu"
strip_punct_table = str.maketrans("", "", punctuation)
data_collator = ClothoCaptioningDataCollator(decoder_only=True)


@torch.no_grad()
def collect_audio_text_cosine_sims(
    model,
    instructor_model,
    samp_data,
    gen_caption,
):
    instructor_inputs = [["Represent the audio caption: ", cap] for cap in gen_caption]
    instructor_embeds = instructor_model.encode(
        instructor_inputs,
        show_progress_bar=False,
        batch_size=128,
    )

    # clone encoder inputs for all captions
    samp_enc_input = torch.tensor(samp_data["encoder_input"]).to(device).unsqueeze(0)
    samp_enc_mask = torch.tensor(samp_data["attention_mask"]).to(device).unsqueeze(0)

    audio_embed = (
        model.get_audio_embed_from_audio(
            encoder_input=samp_enc_input, attention_mask=samp_enc_mask
        )
        .cpu()
        .numpy()
    )

    cosine_sims = cosine_similarity(audio_embed, instructor_embeds)[0]

    return 0, cosine_sims


def rerank_captions_with_cosine_sim(
    cosine_sims,
    samp_idx,
    samp_name,
    samp_true_captions,
    samp_generated_captions,
    samp_sources=None,
):
    samp_cap_ranked_idx = np.argsort(-cosine_sims)
    samp_json_obj = {
        "idx": samp_idx,
        "audio_file": samp_name,
        "true_captions": samp_true_captions,
        "generated_captions": [],
    }

    for j in samp_cap_ranked_idx:
        samp_json_obj["generated_captions"].append(
            {
                "text": samp_generated_captions[j],
                "audio_text_similarity": float(cosine_sims[j]),
                "source": None if samp_sources is None else samp_sources[j],
            }
        )

    return samp_json_obj


if __name__ == "__main__":
    model = BeatsConformerBartSeq2SeqForCaptioning.from_pretrained(
        ckpt_dir,
        Wav2Vec2ConformerConfig.from_json_file(conformer_config_json),
        for_inference=True,
    ).to(device)

    model.eval()

    instructor_model = INSTRUCTOR(
        f"hkunlp/instructor-xl", cache_folder="./instructor_pretrained_weights"
    )
    instructor_model.eval()

    inference_dset = ClothoCaptioningDataset(
        f"clotho/{test_split}",
        "tokenizer/clotho_bpe1000"
        if getattr(model.config, "tokenizer_dir", None) is None
        else model.config.tokenizer_dir,
        f"clotho/clotho_captions_{test_split}.csv",
        do_audio_preload=True,
    )

    gen_captions = json.load(open(os.path.join(inference_dir, "gen_captions.json")))
    print(json.dumps(gen_captions[0], indent=4))

    reranked_json = []

    for i in range(len(gen_captions)):
        # for i in range(10):
        samp = inference_dset[i]

        samp_name = samp["sample_name"]
        gen_name = gen_captions[i]["audio_file"]
        assert samp_name == gen_name

        samp_gen_caps = gen_captions[i]["generated_captions"]

        _, gen_cosine_sim = collect_audio_text_cosine_sims(
            model, instructor_model, samp, samp_gen_caps
        )

        if "true_captions" not in gen_captions[i]:
            gen_captions[i]["true_captions"] = []

        if test_split not in ["clotho_analysis", "test"]:
            reranked_json_obj = rerank_captions_with_cosine_sim(
                gen_cosine_sim,
                gen_captions[i]["idx"],
                gen_captions[i]["audio_file"],
                gen_captions[i]["true_captions"],
                samp_gen_caps,
                samp_sources=None
                if "sources" not in gen_captions[i]
                else gen_captions[i]["sources"],
            )
        else:
            reranked_json_obj = rerank_captions_with_cosine_sim(
                gen_cosine_sim,
                gen_captions[i]["idx"],
                gen_captions[i]["audio_file"],
                [],
                samp_gen_caps,
                samp_sources=None
                if "sources" not in gen_captions[i]
                else gen_captions[i]["sources"],
            )

        reranked_json.append(reranked_json_obj)

        if not (i + 1) % 10:
            print(f"[info] processed {i + 1} / {len(inference_dset)} samples")

    with open(
        os.path.join(inference_dir, "gen_captions_encoder_reranked.json"), "w"
    ) as f:
        f.write(json.dumps(reranked_json, indent=4))
        f.write("\n")
