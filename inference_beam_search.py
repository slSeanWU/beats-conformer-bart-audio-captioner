import os
import sys
import json
import time
from typing import Dict, Union, Any
from string import punctuation

sys.path.append("model")
sys.path.append("data")
sys.path.append("caption_evaluation_tools")

import torch
import numpy as np
import pandas as pd
from transformers.modeling_utils import PreTrainedModel
from transformers.tokenization_utils import PreTrainedTokenizer

from data.clotho_captioning_dataset import ClothoCaptioningDataset
from model.modeling_beats_conformer_bart import (
    BeatsConformerBartSeq2SeqForCaptioning,
    Wav2Vec2ConformerConfig,
)

ckpt_dir = sys.argv[1]
inference_config_path = sys.argv[2]
test_split = sys.argv[3]
is_conformer_encoder = sys.argv[4] == "True" if len(sys.argv) > 4 else False
conformer_config_json = sys.argv[5] if len(sys.argv) > 5 else None

device = "cuda" if torch.cuda.is_available() else "cpu"
strip_punct_table = str.maketrans("", "", punctuation)


@torch.no_grad()
def generate_caption_for_audio(
    model: PreTrainedModel,
    tokenizer: PreTrainedTokenizer,
    inputs: Dict[str, Union[list, np.ndarray]],
    inference_config: Dict[str, Any],
    strip_punct: bool = True,
):
    wav_input = torch.tensor(inputs["encoder_input"]).to(device).unsqueeze(0)
    wav_mask = torch.tensor(inputs["attention_mask"]).to(device).unsqueeze(0)
    audio_len = len(inputs["encoder_input"])

    if not "clap" in ckpt_dir:
        encoder_outputs = model.encode_audio(
            encoder_input=wav_input, attention_mask=wav_mask
        )
    else:
        encoder_input = model.preprocess_audio(wav_input, wav_mask)
        encoder_outputs = model.encode_audio(encoder_input, wav_mask)

    if inference_config["num_beams"] > 1:
        wav_mask = wav_mask.expand(inference_config["num_beams"], -1)

    caption_seqs = model.generate(
        encoder_outputs=encoder_outputs,
        attention_mask=encoder_outputs.attention_mask,
        **inference_config,
    )
    caption_seqs = caption_seqs.cpu().numpy()

    for i in range(len(caption_seqs)):
        caption_text = tokenizer.decode(caption_seqs[i], skip_special_tokens=True)

    # to conform to DCASE evaluation standards
    if strip_punct:
        caption_text = caption_text.translate(strip_punct_table)

    return inputs["sample_name"], caption_text, audio_len


if __name__ == "__main__":
    inference_dir = os.path.join(
        ckpt_dir,
        f"inference_{test_split}_"
        f"{os.path.basename(inference_config_path).split('.')[0]}",
    )
    if not os.path.exists(inference_dir):
        os.makedirs(inference_dir)

    model = BeatsConformerBartSeq2SeqForCaptioning.from_pretrained(
        ckpt_dir,
        Wav2Vec2ConformerConfig.from_json_file(conformer_config_json),
        for_inference=True,
    ).to(device)

    print("[info] model loaded from ckpt:", ckpt_dir)
    print(
        "[info] # trainable parameters: ",
        sum(p.numel() for p in model.parameters() if p.requires_grad),
    )
    print("[info] encoder repr layer weights:", model.get_encoder_repr_layer_weights())
    model.eval()

    test_dset = ClothoCaptioningDataset(
        f"clotho/{test_split}",
        "tokenizer/clotho_bpe1000"
        if getattr(model.config, "tokenizer_dir", None) is None
        else model.config.tokenizer_dir,
        f"clotho/clotho_captions_{test_split}.csv",
        do_audio_normalize="pyln" in ckpt_dir,
    )

    sample_names = []
    generated_captions = []
    true_captions = []
    rtfs = []

    for i in range(len(test_dset)):
        st_time = time.time()
        samp_name, gen_caption, audio_len = generate_caption_for_audio(
            model,
            test_dset.tokenizer,
            test_dset[i],
            json.load(open(inference_config_path)),
        )
        ed_time = time.time()

        if i > 0:  # omit initial one
            time_gen = ed_time - st_time
            time_audio = audio_len / test_dset.sample_rate
            rtf = time_gen / time_audio
            print(f"gen = {time_gen:.2f} | audio = {time_audio:.2f} | rtf = {rtf:.4f}")
            rtfs.append(rtf)

        sample_names.append(samp_name)
        generated_captions.append(gen_caption)

        if test_split not in ["clotho_analysis", "test"]:
            true_captions.append(
                [
                    s.strip().lower().translate(strip_punct_table)
                    for s in test_dset.captions[samp_name]
                ]
            )
            print(gen_caption)
            print(true_captions[-1], "\n")
        else:
            print(gen_caption, "\n")

    out_dict = {"file_name": [], "caption_predicted": []}
    for i in range(len(sample_names)):
        out_dict["file_name"].append(sample_names[i])
        out_dict["caption_predicted"].append(generated_captions[i])

    df_out = pd.DataFrame.from_dict(out_dict)
    df_out.to_csv(
        os.path.join(
            inference_dir,
            f"beam_search_output.csv",
        ),
        index=False,
    )
