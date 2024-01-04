import os
import sys
import json
from typing import Dict, Union, Any
from string import punctuation

sys.path.append("model")
sys.path.append("data")
sys.path.append("caption_evaluation_tools")

import torch
import numpy as np
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
remove_duplicate = False


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

    encoder_outputs = model.encode_audio(
        encoder_input=wav_input, attention_mask=wav_mask
    )

    if "num_beams" in inference_config:
        if inference_config["num_beams"] > 1:
            wav_mask = wav_mask.expand(inference_config["num_beams"], -1)
    elif "num_return_sequences" in inference_config:
        if inference_config["num_return_sequences"] > 1:
            wav_mask = wav_mask.expand(inference_config["num_return_sequences"], -1)

    caption_seqs = model.generate(
        encoder_outputs=encoder_outputs,
        attention_mask=encoder_outputs.attention_mask,
        **inference_config,
    )
    caption_seqs = caption_seqs.cpu().numpy()

    caption_texts = []

    if remove_duplicate:
        seen_text = set()

    for i in range(len(caption_seqs)):
        caption_text = tokenizer.decode(caption_seqs[i], skip_special_tokens=True)
        # to conform to DCASE evaluation standards
        if strip_punct:
            caption_text = caption_text.translate(strip_punct_table)

        if remove_duplicate:
            if caption_text in seen_text:
                continue
            else:
                caption_texts.append(caption_text)
                seen_text.add(caption_text)
        else:
            caption_texts.append(caption_text)

    print(f"[info] got {len(caption_texts)} unique gen captions")

    return inputs["sample_name"], caption_texts


if __name__ == "__main__":
    inference_dir = os.path.join(
        ckpt_dir,
        f"inference_{test_split}_"
        f"{os.path.basename(inference_config_path).split('.json')[0]}",
    )

    if remove_duplicate:
        inference_dir = str(inference_dir) + "_deduped"

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

    inference_config = json.load(open(inference_config_path))

    for i in range(len(test_dset)):
        samp_name, gen_caption = generate_caption_for_audio(
            model, test_dset.tokenizer, test_dset[i], inference_config
        )

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

    out_json = []

    if test_split not in ["clotho_analysis", "test"]:
        for i in range(len(sample_names)):
            out_json.append(
                {
                    "idx": i,
                    "audio_file": sample_names[i],
                    "true_captions": true_captions[i],
                    "generated_captions": generated_captions[i],
                }
            )
    else:
        for i in range(len(sample_names)):
            out_json.append(
                {
                    "idx": i,
                    "audio_file": sample_names[i],
                    "generated_captions": generated_captions[i],
                }
            )

    with open(os.path.join(inference_dir, "gen_captions.json"), "w") as f:
        f.write(json.dumps(out_json, indent=4))
        f.write("\n")
