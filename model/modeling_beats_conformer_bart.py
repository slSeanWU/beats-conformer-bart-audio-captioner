import logging
import random
from typing import Optional, List, Tuple, Union

import torch
import torch.nn as nn
from transformers.modeling_outputs import ModelOutput
from transformers.models.bart.modeling_bart import (
    BartConfig,
    BartDecoder,
    BartPretrainedModel,
    BartLearnedPositionalEmbedding,
    Seq2SeqLMOutput,
    shift_tokens_right,
    logger,
)
from transformers.models.wav2vec2_conformer.modeling_wav2vec2_conformer import (
    Wav2Vec2ConformerConfig,
    Wav2Vec2ConformerEncoder,
)
from info_nce import InfoNCE

from BEATs import BEATs, BEATsConfig
from modules import GradMultiply
from specaug import SpecAug


class BidirectionalInfoNCELoss(nn.Module):
    def __init__(self, temperature=0.07) -> None:
        super().__init__()
        self.loss_func = InfoNCE(temperature=temperature, negative_mode="paired")
        # print(
        #     "[info] set InfoNCE temp to",
        #     self.loss_func.temperature
        # )

    def forward(self, audio_embeds, text_embeds):
        assert audio_embeds.size() == text_embeds.size()
        batch_size = audio_embeds.size(0)

        text_negatives = []
        audio_negatives = []

        for i in range(batch_size):
            text_negatives.append(
                torch.cat([text_embeds[:i], text_embeds[i + 1 :]], dim=0)
            )

            audio_negatives.append(
                torch.cat([audio_embeds[:i], audio_embeds[i + 1 :]], dim=0)
            )

        text_negatives = torch.stack(text_negatives, dim=0)
        audio_negatives = torch.stack(audio_negatives, dim=0)

        total_loss = 0.5 * (
            self.loss_func(audio_embeds, text_embeds, text_negatives)
            + self.loss_func(text_embeds, audio_embeds, audio_negatives)
        )

        return total_loss


class RelationalDistilCosineMSELoss(nn.Module):
    def __init__(self, eps=1e-8) -> None:
        super().__init__()
        self.eps = eps
        self.loss_func = nn.MSELoss()

    def forward(self, audio_embeds, text_embeds):
        print("[debug] relational distil used !!")
        assert audio_embeds.size() == text_embeds.size()

        audio_norms = audio_embeds.norm(dim=1)[:, None]
        text_norms = text_embeds.norm(dim=1)[:, None]

        audio_embeds = audio_embeds / torch.clamp(audio_norms, min=self.eps)
        text_embeds = text_embeds / torch.clamp(text_norms, min=self.eps)

        audio_ssm = torch.mm(audio_embeds, audio_embeds.transpose(0, 1))

        text_ssm = torch.mm(text_embeds, text_embeds.transpose(0, 1))

        return self.loss_func(audio_ssm, text_ssm)


class TextEmbeddingPredictor(nn.Module):
    def __init__(self, config: BartConfig):
        super().__init__()

        self.fc1 = nn.Linear(config.d_model, config.embed_predictor_ffn_dim, bias=False)
        self.bn = nn.BatchNorm1d(config.embed_predictor_ffn_dim)
        self.act_fn = nn.GELU()
        self.fc2 = nn.Linear(
            config.embed_predictor_ffn_dim, config.embed_predictor_out_dim, bias=False
        )

    def forward(self, x):
        x = self.fc1(x)
        x = self.bn(x)
        x = self.act_fn(x)
        x = self.fc2(x)

        return x


class BeatsConformerBartSeq2SeqForCaptioning(BartPretrainedModel):
    def __init__(
        self,
        config: BartConfig,
        conformer_config: Wav2Vec2ConformerConfig,
        beats_config: BEATsConfig = None,
        for_inference: bool = False,
    ):
        super().__init__(config)
        if beats_config is not None:
            self.encoder = BEATs(beats_config)
        elif getattr(config, "pretrained_beats_path", None) is not None:
            print("[info] loading BEATs config from ckpt")
            self.encoder = BEATs(
                BEATsConfig(torch.load(config.pretrained_beats_path)["cfg"])
            )
        else:
            raise ValueError(
                "provide either `beats_config` " "or `config.pretrained_beats_path`"
            )

        self.postencoder = Wav2Vec2ConformerEncoder(conformer_config)

        self.main_input_name = "decoder_input_ids"
        self.for_inference = for_inference

        self.decoder = BartDecoder(config)
        self.register_parameter(
            "final_logits_bias", nn.Parameter(torch.zeros((1, config.vocab_size)))
        )
        self.lm_head = nn.Linear(config.d_model, config.vocab_size, bias=False)
        self.cross_embed_positions = BartLearnedPositionalEmbedding(
            config.max_cross_position_embeddings, config.d_model
        )

        self.lsm_weight = config.lsm_weight
        self.use_weighted_encoder_repr = config.use_weighted_encoder_repr

        if self.use_weighted_encoder_repr:
            self.register_parameter(
                "encoder_repr_layer_weights",
                nn.Parameter(torch.ones((config.encoder_layers + 1, 1))),
            )
            self.encoder_repr_layer_idx = None
        else:
            self.encoder_repr_layer_weights = None
            self.encoder_repr_layer_idx = getattr(
                config, "encoder_repr_layer_idx", config.encoder_layers
            )

        if config.d_model != conformer_config.hidden_size:
            self.enc_dec_proj = nn.Linear(
                conformer_config.hidden_size, config.d_model, bias=False
            )
        else:
            self.enc_dec_proj = None

        # freeze beats encoder
        if getattr(config, "freeze_encoder", True):
            for param in self.encoder.parameters():
                param.requires_grad_(False)

        # downsample encoder representations
        self.encoder_downsample_rate = 1
        self.downsample_conv = None

        if getattr(config, "encoder_downsample_rate", None) is not None:
            ds_rate = int(config.encoder_downsample_rate)
            if ds_rate < 1:
                raise ValueError(
                    "`encoder_downsample_rate` should be >= 1, " f"got {ds_rate}"
                )
            self.encoder_downsample_rate = ds_rate

        if getattr(config, "use_conv_downsample", False):
            ds_rate = getattr(config, "encoder_downsample_rate", 1)
            kernel_multiplier = getattr(config, "downsample_kernel_size", 1.5)
            self.downsample_conv = nn.Conv1d(
                in_channels=conformer_config.hidden_size,
                out_channels=conformer_config.hidden_size,
                kernel_size=int(round(ds_rate * kernel_multiplier)),
                stride=ds_rate,
            )
            self.encoder_downsample_rate = ds_rate

        # For encoder multilabel keyword recognition
        self.encoder_clf_proj = None
        if getattr(config, "use_encoder_clf", False):
            n_enc_clf_classes = getattr(config, "n_enc_clf_classes", 2609)
            self.encoder_clf_proj = nn.Linear(config.d_model, n_enc_clf_classes)

        self.encoder_clf_loss_weight = 1.0
        if hasattr(config, "encoder_clf_loss_weight"):
            self.encoder_clf_loss_weight = config.encoder_clf_loss_weight
        print("[info] setting kw classifier weight to", self.encoder_clf_loss_weight)

        # For encoder text embedding prediction
        self.encoder_embed_mlp = None
        if getattr(config, "use_encoder_embed_mlp", False):
            assert (
                self.encoder_clf_proj is None,
                "keyword clf & text embed predictor should not " "be used together",
            )
            self.encoder_embed_mlp = TextEmbeddingPredictor(config)

        # Use InfoNCE on text embedding prediction, instead of cosine sim
        self.use_contrastive_embed_loss = getattr(
            config, "use_contrastive_embed_loss", False
        )
        self.contrastive_temperature = getattr(config, "contrastive_temperature", 0.07)

        # Use Relational distillation, instead of cosine sim or InfoNCE
        self.use_relational_embed_distil = getattr(
            config, "use_relational_embed_distil", False
        )

        if self.use_relational_embed_distil:
            assert not self.use_contrastive_embed_loss

        if self.use_contrastive_embed_loss or self.use_relational_embed_distil:
            assert self.encoder_embed_mlp is not None

        self.embed_predictor_loss_weight = (
            5.0 if not self.use_contrastive_embed_loss else 1.0
        )
        if hasattr(config, "embed_predictor_loss_weight"):
            self.embed_predictor_loss_weight = config.embed_predictor_loss_weight
        print("[info] setting embed prediction to", self.embed_predictor_loss_weight)

        # Initialize weights and apply final processing
        self.post_init()

        if (
            getattr(config, "pretrained_beats_path", None) is not None
            and not for_inference
        ):
            self.load_encoder_pretrained_weights(config.pretrained_beats_path)

        self.spec_aug = None
        if getattr(config, "spec_aug", None) is not None and not for_inference:
            self.add_encoder_spec_aug(config.spec_aug)
            self.add_spec_aug(config.spec_aug)

        if getattr(config, "encoder_grad_multiplier", None) is not None:
            self.encoder_grad_multiplier = config.encoder_grad_multiplier
        else:
            self.encoder_grad_multiplier = None

        self.randomize_audio_feats = getattr(config, "randomize_audio_feats", False)
        if self.randomize_audio_feats:
            print("[info] audio will be randomized as noise")

    def forward_padding_mask(
        self,
        features: torch.Tensor,
        padding_mask: torch.Tensor,
    ) -> torch.Tensor:
        extra = padding_mask.size(1) % features.size(1)
        if extra > 0:
            padding_mask = padding_mask[:, :-extra]
        padding_mask = padding_mask.view(padding_mask.size(0), features.size(1), -1)
        padding_mask = padding_mask.all(-1)

        return padding_mask

    def load_encoder_pretrained_weights(self, ckpt_path):
        self.encoder.load_state_dict(
            torch.load(ckpt_path)["model"],
        )
        print("[info] pretrained BEATs weights loaded")

    def add_spec_aug(self, spec_aug_config):
        self.spec_aug = SpecAug(**spec_aug_config)

    def get_encoder(self):
        return self.encoder

    def get_decoder(self):
        return self.decoder

    def get_encoder_repr_layer_weights(self):
        if self.use_weighted_encoder_repr:
            return nn.functional.softmax(self.encoder_repr_layer_weights.squeeze())
        else:
            return None

    def add_encoder_spec_aug(self, spec_aug_config):
        self.encoder.add_spec_aug(spec_aug_config)
        print("[info] SpecAug added to BEATs encoder")

    def get_input_embeddings(self):
        return self.decoder.embed_tokens

    def get_output_embeddings(self):
        return self.lm_head

    def set_output_embeddings(self, new_embeddings):
        self.lm_head = new_embeddings

    def run_encoder_clf(self, encoder_hidden_states, encoder_attention_mask):
        encoder_hidden_states_mean = encoder_hidden_states.sum(dim=-2)
        encoder_hidden_states_mean /= encoder_attention_mask.sum(dim=-1).unsqueeze(1)

        clf_logits = self.encoder_clf_proj(encoder_hidden_states_mean)

        return clf_logits

    def run_encoder_mlp(self, encoder_hidden_states, encoder_attention_mask):
        encoder_hidden_states_mean = encoder_hidden_states.sum(dim=-2)
        encoder_hidden_states_mean /= encoder_attention_mask.sum(dim=-1).unsqueeze(1)

        mlp_logits = self.encoder_embed_mlp(encoder_hidden_states_mean)

        return mlp_logits

    def get_audio_embed_from_audio(self, encoder_input, attention_mask):
        encoder_outputs = self.encode_audio(encoder_input, attention_mask)

        pooled_hidden_states = encoder_outputs.last_hidden_state
        attention_mask = encoder_outputs.attention_mask

        attention_mask = (~attention_mask).float()

        if self.enc_dec_proj is not None:
            pooled_hidden_states = self.enc_dec_proj(pooled_hidden_states)

        audio_embed = self.run_encoder_mlp(pooled_hidden_states, attention_mask)

        return audio_embed

    def encode_audio(self, encoder_input, attention_mask):
        encoder_outputs = self.encoder(
            source=encoder_input,
            padding_mask=attention_mask,
            max_layer=self.encoder_repr_layer_idx,
        )
        encoder_hidden_states = encoder_outputs.hidden_states

        if self.use_weighted_encoder_repr:
            repr_layer_weights = nn.functional.softmax(
                self.encoder_repr_layer_weights, dim=-2
            )
            pooled_encoder_hidden_states = (
                torch.stack(encoder_hidden_states, dim=-2) * repr_layer_weights
            )
            pooled_encoder_hidden_states = pooled_encoder_hidden_states.sum(dim=-2)
        else:
            pooled_encoder_hidden_states = encoder_hidden_states[
                self.encoder_repr_layer_idx
            ]

        if self.encoder_downsample_rate > 1 and self.downsample_conv is None:
            offset = random.choice(range(self.encoder_downsample_rate))
            pooled_encoder_hidden_states = pooled_encoder_hidden_states[
                :, offset :: self.encoder_downsample_rate
            ]
            encoder_outputs.attention_mask = encoder_outputs.attention_mask[
                :, offset :: self.encoder_downsample_rate
            ]
        elif self.downsample_conv is not None:
            # print("[debug] using conv downsampling")
            pooled_encoder_hidden_states = self.downsample_conv(
                pooled_encoder_hidden_states.transpose(1, 2)
            ).transpose(1, 2)

            encoder_outputs.attention_mask = self.forward_padding_mask(
                pooled_encoder_hidden_states, encoder_outputs.attention_mask
            )

        if self.spec_aug is not None and self.training:
            encoder_hidden_states = self.spec_aug(pooled_encoder_hidden_states)[0]

        # to handle incompatibility btw torch & huggingface
        if encoder_outputs.attention_mask is not None:
            conformer_attn_mask = ~encoder_outputs.attention_mask
        else:
            conformer_attn_mask = None

        # run through conformer
        pooled_encoder_hidden_states = self.postencoder(
            pooled_encoder_hidden_states,
            attention_mask=conformer_attn_mask,
        ).last_hidden_state

        if encoder_outputs.attention_mask is not None:
            assert pooled_encoder_hidden_states.size(
                1
            ) == encoder_outputs.attention_mask.size(1)

        return ModelOutput(
            last_hidden_state=pooled_encoder_hidden_states,
            hidden_states=None,
            attentions=None,
            attention_mask=encoder_outputs.attention_mask,
        )

    def forward(
        self,
        encoder_input: torch.FloatTensor = None,
        attention_mask: Optional[torch.Tensor] = None,
        decoder_input_ids: Optional[torch.LongTensor] = None,
        decoder_attention_mask: Optional[torch.LongTensor] = None,
        head_mask: Optional[torch.Tensor] = None,
        decoder_head_mask: Optional[torch.Tensor] = None,
        cross_attn_head_mask: Optional[torch.Tensor] = None,
        encoder_labels: Optional[torch.Tensor] = None,
        encoder_outputs: Optional[List[torch.FloatTensor]] = None,
        past_key_values: Optional[List[torch.FloatTensor]] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        decoder_inputs_embeds: Optional[torch.FloatTensor] = None,
        labels: Optional[torch.LongTensor] = None,
        use_cache: Optional[bool] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
    ) -> Union[Tuple, Seq2SeqLMOutput]:
        r"""
        labels (`torch.LongTensor` of shape `(batch_size, sequence_length)`, *optional*):
            Labels for computing the masked language modeling loss. Indices should either be in `[0, ...,
            config.vocab_size]` or -100 (see `input_ids` docstring). Tokens with indices set to `-100` are ignored
            (masked), the loss is only computed for the tokens with labels in `[0, ..., config.vocab_size]`.
        Returns:
        """
        return_dict = (
            return_dict if return_dict is not None else self.config.use_return_dict
        )

        if labels is not None:
            if use_cache:
                logger.warning(
                    "The `use_cache` argument is changed to `False` since `labels` is provided."
                )
            use_cache = False
            if decoder_input_ids is None and decoder_inputs_embeds is None:
                decoder_input_ids = shift_tokens_right(
                    labels, self.config.pad_token_id, self.config.decoder_start_token_id
                )

        if encoder_outputs is None:
            encoder_outputs = self.encode_audio(
                encoder_input,
                attention_mask,
            )
            # NOTE(Shih-Lun): attention_mask changes due to
            # patch processing in encoder
            attention_mask = encoder_outputs.attention_mask

        pooled_encoder_hidden_states = encoder_outputs.last_hidden_state

        if self.randomize_audio_feats:
            pooled_encoder_hidden_states = torch.randn_like(
                pooled_encoder_hidden_states
            )

        if self.encoder_grad_multiplier is not None:
            pooled_encoder_hidden_states = GradMultiply.apply(
                pooled_encoder_hidden_states, self.encoder_grad_multiplier
            )

        # to handle incompatibility btw torch & huggingface
        if attention_mask is not None:
            attention_mask = (~attention_mask).float()

        if self.enc_dec_proj is not None:
            pooled_encoder_hidden_states = self.enc_dec_proj(
                pooled_encoder_hidden_states
            )

        if self.encoder_clf_proj is not None:
            encoder_clf_logits = self.run_encoder_clf(
                pooled_encoder_hidden_states, attention_mask
            )

        if self.encoder_embed_mlp is not None:
            encoder_mlp_logits = self.run_encoder_mlp(
                pooled_encoder_hidden_states, attention_mask
            )

        pooled_encoder_hidden_states = (
            pooled_encoder_hidden_states
            + self.cross_embed_positions(pooled_encoder_hidden_states)
        )

        outputs = self.decoder(
            input_ids=decoder_input_ids,
            attention_mask=decoder_attention_mask,
            encoder_hidden_states=pooled_encoder_hidden_states,
            encoder_attention_mask=attention_mask,
            head_mask=decoder_head_mask,
            cross_attn_head_mask=cross_attn_head_mask,
            past_key_values=past_key_values,
            inputs_embeds=decoder_inputs_embeds,
            use_cache=use_cache,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )

        lm_logits = self.lm_head(outputs[0])
        lm_logits = lm_logits + self.final_logits_bias.to(lm_logits.device)

        masked_lm_loss = None
        total_loss = None
        if labels is not None:
            if not self.for_inference:
                loss_fct = nn.CrossEntropyLoss(label_smoothing=self.lsm_weight)
                masked_lm_loss = loss_fct(
                    lm_logits.view(-1, self.config.vocab_size), labels.view(-1)
                )
                total_loss = masked_lm_loss
            else:
                loss_fct = nn.CrossEntropyLoss(reduction="none")
                masked_lm_loss = loss_fct(lm_logits.permute(0, 2, 1), labels)
                masked_lm_loss = masked_lm_loss.sum(dim=-1)

                seqlens = torch.count_nonzero(labels != -100, dim=-1).float()

                masked_lm_loss = masked_lm_loss / seqlens
                total_loss = masked_lm_loss

        if (
            self.encoder_clf_proj is not None
            and encoder_labels is not None
            and encoder_labels.size() == encoder_clf_logits.size()
        ):
            print("[WARN] should not be here")
            encoder_clf_loss = nn.BCEWithLogitsLoss()(
                encoder_clf_logits, encoder_labels
            )
            total_loss += self.encoder_clf_loss_weight * encoder_clf_loss

        if (
            self.encoder_embed_mlp is not None
            and encoder_labels is not None
            and encoder_labels.size() == encoder_mlp_logits.size()
            and not self.use_contrastive_embed_loss
            and not self.use_relational_embed_distil
        ):
            print("[WARN] should not be here")
            encoder_mlp_loss = -nn.CosineSimilarity()(
                encoder_mlp_logits, encoder_labels
            ).mean()
            total_loss += self.embed_predictor_loss_weight * encoder_mlp_loss
        elif (
            self.encoder_embed_mlp is not None
            and encoder_labels is not None
            and encoder_labels.size() == encoder_mlp_logits.size()
            and self.use_contrastive_embed_loss
        ):
            print("[WARN] should not be here")
            encoder_mlp_loss = BidirectionalInfoNCELoss(self.contrastive_temperature)(
                encoder_mlp_logits, encoder_labels
            )
            total_loss += self.embed_predictor_loss_weight * encoder_mlp_loss
        elif (
            self.encoder_embed_mlp is not None
            and encoder_labels is not None
            and encoder_labels.size() == encoder_mlp_logits.size()
            and self.use_relational_embed_distil
        ):
            print("[WARN] should not be here")
            encoder_mlp_loss = RelationalDistilCosineMSELoss()(
                encoder_mlp_logits, encoder_labels
            )
            total_loss += self.embed_predictor_loss_weight * encoder_mlp_loss

        if not return_dict:
            output = (lm_logits,) + outputs[1:]
            return ((total_loss,) + output) if total_loss is not None else output

        return Seq2SeqLMOutput(
            loss=total_loss,
            logits=lm_logits,
            past_key_values=outputs.past_key_values,
            decoder_hidden_states=outputs.hidden_states,
            decoder_attentions=outputs.attentions,
            cross_attentions=outputs.cross_attentions,
            # encoder_last_hidden_state=pooled_encoder_hidden_states,
            # encoder_hidden_states=encoder_hidden_states,
        )

    def prepare_inputs_for_generation(
        self,
        decoder_input_ids,
        past_key_values=None,
        attention_mask=None,
        decoder_attention_mask=None,
        head_mask=None,
        decoder_head_mask=None,
        cross_attn_head_mask=None,
        use_cache=None,
        encoder_outputs=None,
        encoder_input=None,
        **kwargs,
    ):
        # cut decoder_input_ids if past_key_values is used
        if past_key_values is not None:
            decoder_input_ids = decoder_input_ids[:, -1:]

        return {
            "encoder_input": encoder_input,
            "encoder_outputs": encoder_outputs,
            "past_key_values": past_key_values,
            "decoder_input_ids": decoder_input_ids,
            "attention_mask": attention_mask,
            "decoder_attention_mask": decoder_attention_mask,
            "head_mask": head_mask,
            "decoder_head_mask": decoder_head_mask,
            "cross_attn_head_mask": cross_attn_head_mask,
            "use_cache": use_cache,  # change this to avoid caching (presumably for debugging)
        }

    def prepare_decoder_input_ids_from_labels(self, labels: torch.Tensor):
        return shift_tokens_right(
            labels, self.config.pad_token_id, self.config.decoder_start_token_id
        )

    @staticmethod
    def _reorder_cache(past, beam_idx):
        reordered_past = ()
        for layer_past in past:
            # cached cross_attention states don't have to be reordered -> they are always the same
            reordered_past += (
                tuple(
                    past_state.index_select(0, beam_idx)
                    for past_state in layer_past[:2]
                )
                + layer_past[2:],
            )
        return reordered_past


if __name__ == "__main__":
    import json

    criterion = RelationalDistilCosineMSELoss()
    bsize, embed_dim = 64, 1024
    info_nce = criterion(torch.randn(bsize, embed_dim), torch.randn(bsize, embed_dim))
    print(info_nce)
    exit()

    config = BartConfig.from_json_file(
        "config/acaps_model/beats_ft_frozen_bart_8l_baseline_mincap6.json"
    )
    config.use_encoder_clf = True

    # checkpoint = torch.load('/scratch/bbjs/slseanwu/dcase23_aac/beats_baseline/pretrained_weights/BEATs_iter3_plus_AS2M.pt')
    # beats_config = BEATsConfig(checkpoint["cfg"])
    # beats_config.spec_aug = {
    #                 "freq_mask_width_range": (0, 64),
    #                 "num_freq_mask": 2,
    #                 "time_mask_width_ratio_range": (0, 0.12),
    #                 "num_time_mask": 5
    #             }

    conformer_config = Wav2Vec2ConformerConfig.from_json_file(
        "config/conformer_postencoder/conformer_4l.json"
    )

    model = BartConformerBeatsSeq2SeqForCaptioning(
        config,
        conformer_config,
    ).to("cuda")

    # print(str(model.prepare_inputs_for_generation))
    # print(model.config.is_encoder_decoder)
    # print(model.config.bos_token_id, model.config.eos_token_id)
    # print(model.main_input_name)
    # print(model.config.decoder_start_token_id)
    # print(model.config.output_attentions, model.config.output_hidden_states)
    # print(model.config.return_dict)
    # exit()

    # model.load_encoder_pretrained_weights(checkpoint["model"])
    # model.encoder.add_spec_aug(beats_config.spec_aug)
    print(
        "[info] # trainable parameters: ",
        sum(p.numel() for p in model.parameters() if p.requires_grad),
    )
    model.train()

    bsize, enc_len, dec_len = 4, 320000, 40
    encoder_input = torch.randn(bsize, enc_len).to("cuda")
    labels = torch.randint(0, config.vocab_size, (bsize, dec_len)).to("cuda")
    encoder_labels = torch.randint(0, 2, (bsize, 2609)).to("cuda").float()

    attn_mask_lens = [random.randint(150000, 320000) for _ in range(bsize)]
    print(attn_mask_lens)
    attn_mask = torch.zeros_like(encoder_input).bool()
    for i, l in enumerate(attn_mask_lens):
        attn_mask[i][l:] = True

    out = model.forward(
        encoder_input=encoder_input,
        labels=labels,
        encoder_labels=encoder_labels,
        attention_mask=attn_mask,
    )

    # print(vars(beats_config))
    print(len(out))
    for i in range(len(out)):
        try:
            print(len(out[i]), type(out[i]), out[i].size())
        except:
            try:
                print(out[i][0].size())
            except:
                print("0-d tensor")
