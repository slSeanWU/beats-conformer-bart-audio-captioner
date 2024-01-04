import numpy as np
from torch.utils.data import default_collate

CAPTION_PADDING_IDX = -100


class ClothoCaptioningDataCollator(object):
    def __init__(self, decoder_only=False):
        self.decoder_only = decoder_only
        return

    def __call__(self, data):
        if not self.decoder_only:
            encoder_maxlen = max(
                [len(d["encoder_input"]) for d in data]
            )
            decoder_maxlen = max(
                [len(d["labels"]) for d in data]
            )

            for i in range(len(data)):
                enc_inp = data[i]["encoder_input"]
                dec_inp = data[i]["labels"]
                attn_mask = data[i]["attention_mask"]

                data[i]["encoder_input"] = np.concatenate(
                    (
                        enc_inp,
                        np.zeros(
                            (encoder_maxlen - len(enc_inp),),
                            dtype=enc_inp.dtype
                        )
                    ),
                    axis=-1
                )

                data[i]["labels"] = np.concatenate(
                    (
                        dec_inp,
                        np.full(
                            (decoder_maxlen - len(dec_inp),),
                            CAPTION_PADDING_IDX,
                            dtype=dec_inp.dtype
                        )
                    ),
                    axis=-1
                )
                assert len(data[i]["labels"]) == decoder_maxlen

                data[i]["attention_mask"] = np.concatenate(
                    (
                        attn_mask,
                        np.ones(
                            (encoder_maxlen - len(enc_inp),),
                            dtype=attn_mask.dtype
                        )
                    )
                )

                assert (
                    len(data[i]["attention_mask"]) == 
                    len(data[i]["encoder_input"])
                )

        else:
            decoder_maxlen = max(
                [len(d["labels"]) for d in data]
            )
            for i in range(len(data)):
                dec_inp = data[i]["labels"]

                data[i]["labels"] = np.concatenate(
                    (
                        dec_inp,
                        np.full(
                            (decoder_maxlen - len(dec_inp),),
                            CAPTION_PADDING_IDX,
                            dtype=dec_inp.dtype
                        )
                    ),
                    axis=-1
                )

      
        return default_collate(data)
    


RERANKING_PADDING_INDEX = 0

class ClothoRerankingDataCollator(object):
    def __init__(self):
        return

    def __call__(self, data):
        encoder_maxlen = max(
            [len(d["audio_input"]) for d in data]
        )
        decoder_maxlen = max(
            [len(d["input_ids"]) for d in data]
        )

        for i in range(len(data)):
            enc_inp = data[i]["audio_input"]
            dec_inp = data[i]["input_ids"]
            attn_mask = data[i]["audio_attention_mask"]
            dec_attn_mask = data[i]["text_attention_mask"]

            data[i]["audio_input"] = np.concatenate(
                (
                    enc_inp,
                    np.zeros(
                        (encoder_maxlen - len(enc_inp),),
                        dtype=enc_inp.dtype
                    )
                ),
                axis=-1
            )

            data[i]["input_ids"] = np.concatenate(
                (
                    dec_inp,
                    np.full(
                        (decoder_maxlen - len(dec_inp),),
                        RERANKING_PADDING_INDEX,
                        dtype=dec_inp.dtype
                    )
                ),
                axis=-1
            )
            assert len(data[i]["input_ids"]) == decoder_maxlen

            data[i]["audio_attention_mask"] = np.concatenate(
                (
                    attn_mask,
                    np.ones(
                        (encoder_maxlen - len(enc_inp),),
                        dtype=attn_mask.dtype
                    )
                )
            )

            assert (
                len(data[i]["audio_attention_mask"]) == 
                len(data[i]["audio_input"])
            )

            data[i]["text_attention_mask"] = np.concatenate(
                (
                    dec_attn_mask,
                    np.zeros(
                        (decoder_maxlen - len(dec_inp),),
                        dtype=dec_attn_mask.dtype
                    )
                )
            )

            assert (
                len(data[i]["text_attention_mask"]) == 
                len(data[i]["input_ids"])
            )
      
        return default_collate(data)