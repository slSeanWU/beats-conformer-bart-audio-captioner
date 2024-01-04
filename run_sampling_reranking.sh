#!/bin/bash

python3 -u inference_sampling.py \
    ./exp \
    config/nucleus_t0.5_p95.json \
    evaluation \
    True \
    ./config/conformer_config.json

python3 -u reranking/decoder_rerank_sampling_outputs.py \
    ./exp \
    ./exp/inference_evaluation_nucleus_t0.5_p95 \
    ./config/conformer_config.json \
    evaluation

python3 -u reranking/encoder_rerank_sampling_outputs.py \
    ./exp \
    ./exp/inference_evaluation_nucleus_t0.5_p95 \
    ./config/conformer_config.json \
    evaluation

python3 -u reranking/hybrid_scoring.py \
    ./exp/inference_evaluation_nucleus_t0.5_p95

python3 -u evaluate.py \
    "./exp/inference_evaluation_nucleus_t0.5_p95/hybrid_score_[0.3, 0.7]_output.csv" \
    evaluation
