import os
import sys
import json
from string import punctuation

sys.path.append(os.path.dirname(__file__) + "/../")
sys.path.append(os.path.dirname(__file__) + "/../model")
sys.path.append(os.path.dirname(__file__) + "/../data")

import torch
import numpy as np
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
use_ensemble = False

device = "cuda" if torch.cuda.is_available() else "cpu"
strip_punct_table = str.maketrans("", "", punctuation)
data_collator = ClothoCaptioningDataCollator(decoder_only=True)


@torch.no_grad()
def collect_length_normalized_nll(model, dset, samp_data, gen_caption):
    # append generated caption
    samp_captions = [dset.tokenize(gcap) for gcap in gen_caption]

    assert all([type(s) == np.ndarray for s in samp_captions])

    # clone encoder inputs for all captions
    samp_enc_input = torch.tensor(samp_data["encoder_input"]).to(device).unsqueeze(0)
    samp_enc_mask = torch.tensor(samp_data["attention_mask"]).to(device).unsqueeze(0)

    encoder_outputs = model.encode_audio(
        encoder_input=samp_enc_input, attention_mask=samp_enc_mask
    )

    if "ens" in inference_dir:
        for i in ["last_hidden_state", "attention_mask"]:
            for j in range(len(encoder_outputs[i])):
                keep_dim = [-1] * (int(encoder_outputs[i][j].dim()) - 1)
                encoder_outputs[i][j] = encoder_outputs[i][j].expand(
                    len(samp_captions), *keep_dim
                )
    else:
        for i in ["last_hidden_state", "attention_mask"]:
            if encoder_outputs[i] is not None:
                keep_dim = [-1] * (int(encoder_outputs[i].dim()) - 1)
                encoder_outputs[i] = encoder_outputs[i].expand(
                    len(samp_captions), *keep_dim
                )

    data_batch = [{"labels": samp_captions[i]} for i in range(len(samp_captions))]
    data_batch = data_collator(data_batch)
    data_batch = {k: v.to(device) for k, v in data_batch.items()}

    nlls = model(
        encoder_outputs=encoder_outputs,
        attention_mask=encoder_outputs.attention_mask,
        **data_batch,
    ).loss
    nlls = nlls.detach().cpu().numpy()
    nlls = nlls.tolist()

    gen_caption_nlls = nlls

    return 0, gen_caption_nlls


def rerank_captions_with_nll(
    nlls,
    samp_idx,
    samp_name,
    samp_true_captions,
    samp_generated_captions,
    samp_sources=None,
):
    samp_cap_ranked_idx = np.argsort(nlls)
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
                "normed_nll": nlls[j],
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

    inference_dset = ClothoCaptioningDataset(
        f"clotho/{test_split}",
        "tokenizer/clotho_bpe1000"
        if getattr(model.config, "tokenizer_dir", None) is None
        else model.config.tokenizer_dir,
        f"clotho/clotho_captions_{test_split}.csv",
        do_audio_preload=True,
    )

    audio_secs = []
    textlen_ratios = []
    text_lens = []
    gen_spiders = []

    true_nlls = []
    gen_nlls = []

    gen_captions = json.load(open(os.path.join(inference_dir, "gen_captions.json")))
    print(json.dumps(gen_captions[0], indent=4))

    reranked_json = []

    for i in range(len(gen_captions)):
        samp = inference_dset[i]

        samp_name = samp["sample_name"]
        gen_name = gen_captions[i]["audio_file"]
        assert samp_name == gen_name

        samp_gen_caps = gen_captions[i]["generated_captions"]

        _, gen_nll = collect_length_normalized_nll(
            model, inference_dset, samp, samp_gen_caps
        )

        if "true_captions" not in gen_captions[i] and test_split not in [
            "clotho_analysis",
            "test",
        ]:
            gen_captions[i]["true_captions"] = inference_dset.captions[samp_name]

        if test_split not in ["clotho_analysis", "test"]:
            reranked_json_obj = rerank_captions_with_nll(
                gen_nll,
                gen_captions[i]["idx"],
                gen_captions[i]["audio_file"],
                gen_captions[i]["true_captions"],
                samp_gen_caps,
                samp_sources=None
                if "sources" not in gen_captions[i]
                else gen_captions[i]["sources"],
            )
        else:
            reranked_json_obj = rerank_captions_with_nll(
                gen_nll,
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
        os.path.join(inference_dir, "gen_captions_decoder_reranked.json"), "w"
    ) as f:
        f.write(json.dumps(reranked_json, indent=4))
        f.write("\n")
