import os
import json
import random
from collections import defaultdict
from typing import Optional, List
from string import punctuation
from multiprocessing.pool import Pool
from copy import deepcopy

import librosa
import soundfile as sf
import pandas as pd
import pyloudnorm as pyln
import numpy as np
from torch.utils.data import Dataset
from transformers import AutoTokenizer


SAMPLE_RATE = 16000
strip_punct_table = str.maketrans("", "", punctuation)


class ClothoCaptioningDataset(Dataset):
    def __init__(
        self,
        audio_dir: str,
        tokenizer_dir: str,
        caption_csv: str,
        split_name: Optional[str] = None,
        bypass_resample: Optional[bool] = False,
        do_audio_preload: Optional[bool] = False,
        extra_dataset: Optional[Dataset] = None,
        p_extra_data_samp: Optional[float] = 0.0,
        lemma_file: Optional[str] = None,
        n_lemmas: Optional[int] = 2609,
        caption_embed_dir: Optional[str] = None,
        clap_embed_dir: Optional[str] = None,
        do_speed_augment: Optional[bool] = False,
        do_pitch_augment: Optional[bool] = False,
        speed_augment_range: Optional[List[float]] = [0.8, 1.2],
        pitch_augment_range: Optional[List[int]] = [-3, 4],
        max_audio_len: Optional[int] = 480000,
        chatgpt_augmented_data_path: Optional[str] = None,
        chatgpt_caption_embed_dir: Optional[str] = None,
        prepend_sos: Optional[bool] = False,
        skip_audio: Optional[bool] = False,
        include_all_captions: Optional[bool] = False,
        do_audio_normalize: Optional[bool] = False,
        preserve_audio: Optional[bool] = True,
        sample_rate: Optional[bool] = SAMPLE_RATE
    ):
        super().__init__()

        self.sample_rate = sample_rate
        self.preserve_audio = preserve_audio
        self.bypass_resample = bypass_resample
        self.audio_dir = audio_dir
        self.split_name = split_name if split_name is not None \
                                     else os.path.basename(audio_dir)

        self.include_all_captions = include_all_captions

        if os.path.exists(caption_csv):
            self.idx_to_sample, self.captions = \
                self.get_captions(caption_csv)
        else:
            self.idx_to_sample = self.grab_all_audios()
            self.captions = None

        self.tokenizer = self.get_tokenizer(tokenizer_dir)

        self.loaded_audios = dict()
        if do_audio_preload:
            self.preload_audios()

        # below are used to incorporate AudioCaps dataset during training
        self.extra_dataset = extra_dataset
        self.p_extra_data_samp = p_extra_data_samp
        if self.extra_dataset is not None:
            self.extra_data_indices = range(len(extra_dataset))
        else:
            self.extra_data_indices = None

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

        # chatgpt augmentations
        if chatgpt_augmented_data_path is not None:
            self.chatgpt_captions = json.load(
                open(chatgpt_augmented_data_path)
            )["dataset"]
        else:
            self.chatgpt_captions = None

        self.chatgpt_caption_embed_dir = chatgpt_caption_embed_dir

        self.prepend_sos = prepend_sos
        self.skip_audio = skip_audio
        self.do_audio_normalize = do_audio_normalize


    def grab_all_audios(self):
        audios = os.listdir(self.audio_dir)

        idx_to_sample = []
        for i, a in enumerate(audios):
            idx_to_sample.append(a)

        return idx_to_sample

    
    def tokenize(self, text):
        text_id = self.tokenizer.encode(
                                    text,
                                    add_special_tokens=False
                                )
        text_id += [self.tokenizer.eos_token_id]

        if self.prepend_sos:
            text_id = [self.tokenizer.bos_token_id] + text_id

        text_id = np.array(text_id, dtype=int)

        return text_id


    def get_captions(self, caption_csv):
        caption_df = pd.read_csv(caption_csv)
        
        idx_to_sample = []
        captions = defaultdict(list)

        for i, row in caption_df.iterrows():
            # if i >= 100:
            #     break

            filename = row["file_name"]

            if self.include_all_captions:
                idx_to_sample.extend([filename] * 5)
            else:
                idx_to_sample.append(filename)

            assert os.path.exists(
                os.path.join(self.audio_dir, filename)
            )
            
            for j in range(5):
                captions[filename].append(
                    row[f"caption_{j + 1}"]
                        .strip()
                        .lower() 
                        .translate(strip_punct_table)
                )

        return idx_to_sample, captions

    
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


    def peak_normalize_audio(self, audio):
        audio = deepcopy(audio)

        if "development" not in self.audio_dir:
            rel_db = -3.0
        else:
            rel_db = np.random.uniform(-6.0, 0.0)

        # print(f"[info] normalizing, {rel_db:.2f} dB")

        return pyln.normalize.peak(
            audio,
            rel_db
        )
    

    def read_lemma_labels(self, sample_name, sample_idx):
        lemmas = self.lemma_data[sample_name][sample_idx]["caption_kw_labels"]
        labels = np.zeros((self.n_lemmas,))

        for l in lemmas:
            labels[l] = 1

        return labels


    def read_caption_embed(self, sample_name, sample_idx):
        sample_embed_dir = os.path.join(
            self.caption_embed_dir,
            sample_name
        )
        embed_choices = [
            f for f in os.listdir(sample_embed_dir) \
            if f"caption{sample_idx + 1:>02d}" in f
        ]
        # print(len(embed_choices))

        return np.load(
            os.path.join(
                sample_embed_dir,
                random.choice(embed_choices)
            )
        )

    def read_clap_embed(self, sample_name, sample_idx):
        sample_embed_dir = os.path.join(
            self.clap_embed_dir,
            sample_name
        )
        embed_choices = [
            f for f in os.listdir(sample_embed_dir) \
            if f"caption{sample_idx + 1:>02d}" in f
        ]
        # print(len(embed_choices))

        return np.load(
            os.path.join(
                sample_embed_dir,
                random.choice(embed_choices)
            )
        )
    

    def read_chatgpt_caption_embed(
            self,
            sample_name,
            sample_idx,
            aug_idx
        ):
        sample_embed_dir = os.path.join(
            self.chatgpt_caption_embed_dir,
            sample_name
        )
        embed_choices = [
            f for f in os.listdir(sample_embed_dir) \
            if f"caption{sample_idx + 1:>02d}_chatgpt{aug_idx + 1:>02d}" in f
        ]
        # print(len(embed_choices))

        # print("[info] chatgpt text embed loaded")

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
        if self.extra_dataset is not None and \
           random.random() < self.p_extra_data_samp:
            extra_data_idx = random.choice(self.extra_data_indices)
            return self.extra_dataset[extra_data_idx]

        sample = self.idx_to_sample[index]

        if not self.skip_audio:
            if sample in self.loaded_audios:
                audio, sr = self.loaded_audios[sample]
            else:
                audio, sr = self.load_audio(
                                    os.path.join(
                                        self.audio_dir,
                                        sample
                                    )
                                )

                if self.preserve_audio:
                    self.loaded_audios[sample] = (audio, sr)

            if (self.do_pitch_augment or self.do_speed_augment) and \
            random.random() < 0.5:
                audio = self.augment_audio(audio)
        else:
            audio = np.array([0])
            sr = 16000

        if self.do_audio_normalize:
            audio = self.peak_normalize_audio(audio)

        if self.include_all_captions:
            caption_idx = index % 5
        else:
            if self.captions is not None:
                caption_idx = random.choice(range(len(self.captions[sample])))
            else:
                caption_idx = 0

        aug_idx = None
        if self.captions is None:
            caption = "</s>"
        elif self.chatgpt_captions is None:
            caption = self.captions[sample][caption_idx]
        else:
            if random.random() < 0.5:
                aug_caps = self.chatgpt_captions[index]["chatgpt_captions"][caption_idx]
                aug_idx = random.choice(range(len(aug_caps)))
                caption = aug_caps[aug_idx]
                # print("[info] got augmented caption:", caption)
            else:
                caption = self.captions[sample][caption_idx]
                # print("[info] got orig caption:", caption)

        caption = self.tokenize(caption)

        bundle = {
            "sample_name": sample,
            "sr": sr,
            "encoder_input": audio,
            "labels": caption,
            "attention_mask": np.full_like(audio, False, dtype=bool)
        }

        if self.lemma_data is not None:
            encoder_labels = self.read_lemma_labels(
                                    sample,
                                    caption_idx
                                )
            bundle["encoder_labels"] = encoder_labels
        elif self.caption_embed_dir is not None:
            if aug_idx is not None:
                assert self.chatgpt_caption_embed_dir is not None
                encoder_embed = self.read_chatgpt_caption_embed(
                    sample,
                    caption_idx,
                    aug_idx
                )
            else:
                encoder_embed = self.read_caption_embed(
                                        sample,
                                        caption_idx
                                    )

                if self.clap_embed_dir is not None:
                    clap_embed = self.read_clap_embed(
                        sample,
                        caption_idx
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
    dset = ClothoCaptioningDataset(
        "clotho/validation",
        # "tokenizer/clotho_bpe1000",
        "facebook/bart-base",
        "clotho/clotho_captions_validation.csv",
        # lemma_file="lemma_analysis/derived_datasets/clotho_validation.json"
        caption_embed_dir="sent_embedding/datasets/instructor-xl/clotho/validation",
        clap_embed_dir="sent_embedding/datasets/clap/clotho/validation",
        # do_speed_augment=True,
        # chatgpt_augmented_data_path="chatgpt/clotho_validation_chatgpt_captions_all.json",
        # chatgpt_caption_embed_dir="sent_embedding/datasets/instructor-xl/clotho/validation_chatgpt",
        # include_all_captions=True,
    )
    print(len(dset))
    # exit()

    # for i in random.sample(range(len(dset)), 100):
    for i in range(len(dset)):
        samp = dset[i]
        # print(samp["labels"])

        for k in ["encoder_input", "labels", "attention_mask", "encoder_labels"]:
            print(
                f"{k:16}, "
                f"{str(type(samp[k])):32}, "
                f"{str(samp[k].dtype):12}, "
                f"{samp[k].shape}"
            )