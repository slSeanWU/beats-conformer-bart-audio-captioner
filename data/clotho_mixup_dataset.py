import os
import json
import random
from copy import deepcopy
from collections import defaultdict
from typing import Optional, List
from string import punctuation
from multiprocessing.pool import Pool

import librosa
import soundfile as sf
import pandas as pd
import pyloudnorm as pyln
import numpy as np
from torch.utils.data import Dataset
from transformers import AutoTokenizer


SAMPLE_RATE = 16000
strip_punct_table = str.maketrans("", "", punctuation)


class ClothoMixupDataset(Dataset):
    def __init__(
        self,
        audio_dir: str,
        tokenizer_dir: str,
        chatgpt_augmented_data_path: str,
        chatgpt_rejected_data_path: str,
        split_name: Optional[str] = None,
        bypass_resample: Optional[bool] = False,
        do_audio_preload: Optional[bool] = False,
        lemma_file: Optional[str] = None,
        n_lemmas: Optional[int] = 2609,
        caption_embed_dir: Optional[str] = None,
        clap_embed_dir: Optional[str] = None,
        do_speed_augment: Optional[bool] = False,
        do_pitch_augment: Optional[bool] = False,
        speed_augment_range: Optional[List[float]] = [0.8, 1.2],
        pitch_augment_range: Optional[List[int]] = [-3, 4],
        max_audio_len: Optional[int] = 480000,
        do_mixup_normalize: Optional[bool] = False,
        sample_rate: Optional[int] = SAMPLE_RATE,
    ):
        super().__init__()

        self.sample_rate = sample_rate
        self.bypass_resample = bypass_resample
        self.audio_dir = audio_dir
        self.split_name = split_name if split_name is not None \
                                     else os.path.basename(audio_dir)

        (
            self.idx_to_sample,
            self.idx_to_audio,
            self.captions
        ) = \
            self.get_captions(
                chatgpt_augmented_data_path,
                chatgpt_rejected_data_path,
            )

        self.tokenizer = self.get_tokenizer(tokenizer_dir)

        self.loaded_audios = dict()
        if do_audio_preload:
            self.preload_audios()

        # lemma labels for encoder loss
        if lemma_file is not None:
            self.lemma_data = json.load(
                open(lemma_file)
            )
            self.n_lemmas = n_lemmas
        else:
            self.lemma_data = None

        # text embedding for encoder loss
        self.caption_embed_dir = caption_embed_dir
        self.clap_embed_dir = clap_embed_dir

        # parameters for audio speed/pitch augmentation
        self.do_speed_augment = do_speed_augment
        self.do_pitch_augment = do_pitch_augment
        self.speed_augment_range = speed_augment_range
        self.pitch_augment_range = list(range(*pitch_augment_range))
        if self.do_pitch_augment:
            print(
                "[info] perform pitch shifting:",
                self.pitch_augment_range,
                "half notes"
            )
        self.max_audio_len = max_audio_len

        self.do_mixup_normalize = do_mixup_normalize

    
    def tokenize(self, text):
        text_id = self.tokenizer.encode(
                                    text,
                                    add_special_tokens=False
                                )
        text_id += [self.tokenizer.eos_token_id]
        text_id = np.array(text_id, dtype=int)

        return text_id


    def get_captions(self, caption_json, rejected_json):
        rejected_df = json.load(open(rejected_json))
        caption_df = json.load(open(caption_json))["dataset"]

        all_rejects = set()
        seen_combs = set()
        for rej in rejected_df:
            all_rejects.add(tuple(rej["idx"]))
        
        idx_to_sample = []
        idx_to_audio = []
        captions = defaultdict(list)

        for i in range(len(caption_df)):
            # if i >= 100:
            #     break
            samp_a, samp_b = caption_df[i]["selected_pair"]
            audio_a, audio_b = caption_df[i]["audio_files"]

            assert (samp_a, samp_b) not in seen_combs
            seen_combs.add((samp_a, samp_b))

            for j in range(len(caption_df[i]["chatgpt_mixups"])):
                if (i, j) in all_rejects:
                    continue

                idx_to_sample.append((samp_a, samp_b))
                idx_to_audio.append((audio_a, audio_b))
            
                captions[(samp_a, samp_b)].append(
                    caption_df[i]["chatgpt_mixups"][j]
                        .strip()
                        .lower() 
                        .translate(strip_punct_table)
                )

        return idx_to_sample, idx_to_audio, captions

    
    def get_tokenizer(self, tokenizer_dir):
        return AutoTokenizer.from_pretrained(tokenizer_dir, use_fast=True)


    def preload_audios(self):
        mp_pool = Pool(processes=32)
        mp_args = []

        for i in range(len(self)):
            sample = self.idx_to_sample[i]
            mp_args.append(
                (os.path.join(
                    self.audio_dir,
                    sample
                ),)
            )
            # self.loaded_audios[sample] = self.load_audio(
            #     os.path.join(
            #         self.audio_dir,
            #         sample
            #     )
            # )

        mp_return = mp_pool.starmap(self.load_audio, mp_args)

        for i in range(len(self)):
            self.loaded_audios[ self.idx_to_sample[i] ] = mp_return[i]

        return


    def load_audio(self, audio_file):
        if not self.bypass_resample:
            wav, sr = librosa.load(
                audio_file,
                sr=self.sample_rate,
            )
            assert sr == self.sample_rate
        else:
            wav, sr = sf.read(audio_file)
            assert sr == 44100

        return wav, sr
    

    def mixup_audio(self, audio_a, audio_b):
        audio_a = deepcopy(audio_a)
        audio_b = deepcopy(audio_b)

        if not self.do_mixup_normalize:
            energy_a = np.mean(
                librosa.feature.rms(y=audio_a)[0] ** 2
            )
            energy_b = np.mean(
                librosa.feature.rms(y=audio_b)[0] ** 2
            )

            mix_db_ratio = np.random.uniform(-5, 5)
            mix_scale = np.sqrt(
                energy_a /
                (10 ** (mix_db_ratio / 10) * energy_b)
            )
        else:
            mix_scale = 1
            base_db = np.random.uniform(-12, -5)
            mix_db_ratio = np.random.uniform(-5, 5)
            audio_a = pyln.normalize.peak(
                audio_a,
                base_db
            )
            audio_b = pyln.normalize.peak(
                audio_b,
                base_db + mix_db_ratio
            )

        # sf.write(
        #     os.path.join("tmp", f"audio_{base_db:.2f}_{mix_db_ratio:.2f}_A_prenorm.wav"),
        #     audio_a,
        #     SAMPLE_RATE
        # )
        # sf.write(
        #     os.path.join("tmp", f"audio_{base_db:.2f}_{mix_db_ratio:.2f}_B_prenorm.wav"),
        #     audio_b,
        #     SAMPLE_RATE
        # )

        # sf.write(
        #     os.path.join("tmp", f"audio_{base_db:.2f}_{mix_db_ratio:.2f}_A.wav"),
        #     audio_a,
        #     SAMPLE_RATE
        # )
        # sf.write(
        #     os.path.join("tmp", f"audio_{base_db:.2f}_{mix_db_ratio:.2f}_B.wav"),
        #     audio_b,
        #     SAMPLE_RATE
        # )

        if len(audio_a) > len(audio_b):
            st_idx = random.choice(
                range(len(audio_a) - len(audio_b) + 1)
            )
            audio_a[st_idx : st_idx + len(audio_b)] += \
                mix_scale * audio_b
            mixed_audio = audio_a
        else:
            st_idx = random.choice(
                range(len(audio_b) - len(audio_a) + 1)
            )
            audio_b[st_idx : st_idx + len(audio_a)] += \
                mix_scale * audio_a
            mixed_audio = audio_b

        # sf.write(
        #     os.path.join("tmp", f"{base_db:.2f}_{mix_db_ratio:.2f}_mixed.wav"),
        #     mixed_audio,
        #     SAMPLE_RATE
        # )

        return mixed_audio
    

    def augment_audio(self, audio):
        if self.do_pitch_augment and random.random() < 0.5:
            audio = librosa.effects.pitch_shift(
                audio,
                sr=SAMPLE_RATE,
                n_steps=random.choice(self.pitch_augment_range)
            )

        if self.do_speed_augment and random.random() < 0.5:
            if audio.shape[0] / self.speed_augment_range[0] > self.max_audio_len:
                _min_speed = audio.shape[0] / self.max_audio_len
            else:
                _min_speed = self.speed_augment_range[0]

            assert audio.shape[0] / _min_speed <= self.max_audio_len + 1
            
            audio = librosa.effects.time_stretch(
                audio,
                rate=np.random.uniform(
                    _min_speed,
                    self.speed_augment_range[1]
                )
            )

        return audio
    

    def read_lemma_labels(self, sample_name, sample_idx):
        lemmas = self.lemma_data[sample_name][sample_idx]["caption_kw_labels"]
        labels = np.zeros((self.n_lemmas,))

        for l in lemmas:
            labels[l] = 1

        return labels


    def read_caption_embed(self, sample_pair, sample_idx):
        sample_embed_dir = os.path.join(
            self.caption_embed_dir,
            str(sample_pair[0]) + "_" + str(sample_pair[1])
        )
        embed_choices = [
            f for f in os.listdir(sample_embed_dir) \
            if f"chatgpt{sample_idx + 1:>02d}" in f
        ]
        # print(len(embed_choices))

        return np.load(
            os.path.join(
                sample_embed_dir,
                random.choice(embed_choices)
            )
        )


    def read_clap_embed(self, sample_pair, sample_idx):
        sample_embed_dir = os.path.join(
            self.clap_embed_dir,
            str(sample_pair[0]) + "_" + str(sample_pair[1])
        )
        embed_choices = [
            f for f in os.listdir(sample_embed_dir) \
            if f"chatgpt{sample_idx + 1:>02d}" in f
        ]
        # print(len(embed_choices))

        return np.load(
            os.path.join(
                sample_embed_dir,
                random.choice(embed_choices)
            )
        )


    def __len__(self):
        return len(self.idx_to_sample)
        # return 10

    
    def __getitem__(self, index):
        samp_a, samp_b = self.idx_to_sample[index]
        audio_a, audio_b = self.idx_to_audio[index]

        audios_to_mix = []
        for audfile in [audio_a, audio_b]:
            if audfile in self.loaded_audios:
                audio, sr = self.loaded_audios[audfile]
            else:
                audio, sr = self.load_audio(
                                    os.path.join(
                                        self.audio_dir,
                                        audfile
                                    )
                                )
                self.loaded_audios[audfile] = (audio, sr)

            audios_to_mix.append(audio)

        audio = self.mixup_audio(
                        audios_to_mix[0],
                        audios_to_mix[1]
                    )

        if (self.do_pitch_augment or self.do_speed_augment) and \
           random.random() < 0.5:
            audio = self.augment_audio(audio)

        caption_idx = random.choice(
                        range(len(self.captions[(samp_a, samp_b)]))
                    )

        caption = self.captions[(samp_a, samp_b)][caption_idx]
        caption = self.tokenize(caption)

        bundle = {
            "sample_name": str(samp_a) + "_" + str(samp_b),
            "sr": sr,
            "encoder_input": audio,
            "labels": caption,
            "attention_mask": np.full_like(audio, False, dtype=bool)
        }

        if self.lemma_data is not None:
            encoder_labels = self.read_lemma_labels(
                                    (samp_a, samp_b),
                                    caption_idx
                                )
            bundle["encoder_labels"] = encoder_labels
        elif self.caption_embed_dir is not None:
            encoder_embed = self.read_caption_embed(
                                    (samp_a, samp_b),
                                    caption_idx
                                )

            if self.clap_embed_dir is not None:
                clap_embed = self.read_clap_embed(
                    (samp_a, samp_b),
                    caption_idx,
                )
                encoder_embed = np.concatenate(
                    [encoder_embed, clap_embed],
                    axis=-1
                )

            bundle["encoder_labels"] = encoder_embed
        else:
            bundle["encoder_labels"] = np.array([0])
        
        
        return bundle


if __name__ == "__main__":
    dset = ClothoMixupDataset(
        "clotho/development",
        # "tokenizer/clotho_bpe1000",
        "facebook/bart-base",
        "/scratch/bbjs/slseanwu/dcase23_aac/beats_baseline/chatgpt/clotho_development_chatgpt_mixups.json",
        "/scratch/bbjs/slseanwu/dcase23_aac/beats_baseline/chatgpt/clotho_development_chatgpt_mixups_err.json",
        caption_embed_dir="sent_embedding/datasets/instructor-xl/clotho/development_chatgpt_mixup",
        clap_embed_dir="sent_embedding/datasets/clap/clotho/development_chatgpt_mixup",
    )
    print(len(dset))
    # exit()

    for i in random.sample(range(len(dset)), 200):
    # for i in range(len(dset)):
        samp = dset[i]
        print(samp["labels"])

        for k in ["encoder_input", "labels", "attention_mask", "encoder_labels"]:
            print(
                f"{k:16}, "
                f"{str(type(samp[k])):32}, "
                f"{str(samp[k].dtype):12}, "
                f"{samp[k].shape}"
            )