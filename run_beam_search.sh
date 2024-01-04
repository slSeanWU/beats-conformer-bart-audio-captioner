#!/bin/bash

python3 -u inference_beam_search.py \
    ./exp \
    ./config/beam_search.json \
    evaluation \
    True \
    ./config/conformer_config.json

python3 -u evaluate.py \
    ./exp/inference_evaluation_beam_search/beam_search_output.csv \
    evaluation
