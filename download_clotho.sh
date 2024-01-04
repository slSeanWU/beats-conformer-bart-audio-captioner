#!/bin/bash

mkdir clotho/
cd clotho/

# Get & extract audio files
wget https://zenodo.org/record/4783391/files/clotho_audio_development.7z?download=1 -O clotho_audio_development.7z
7z x clotho_audio_development.7z
wget https://zenodo.org/record/4783391/files/clotho_audio_evaluation.7z?download=1 -O clotho_audio_evaluation.7z
7z x clotho_audio_evaluation.7z
wget https://zenodo.org/record/4783391/files/clotho_audio_validation.7z?download=1 -O clotho_audio_validation.7z
7z x clotho_audio_validation.7z
rm *.7z

# Get ground-truth captions
wget https://zenodo.org/record/4783391/files/clotho_captions_development.csv?download=1 -O clotho_captions_development.csv
wget https://zenodo.org/record/4783391/files/clotho_captions_evaluation.csv?download=1 -O clotho_captions_evaluation.csv
wget https://zenodo.org/record/4783391/files/clotho_captions_validation.csv?download=1 -O clotho_captions_validation.csv

# Get additional metadata
wget https://zenodo.org/record/4783391/files/clotho_metadata_development.csv?download=1 -O clotho_metadata_development.csv
wget https://zenodo.org/record/4783391/files/clotho_metadata_evaluation.csv?download=1 -O clotho_metadata_evaluation.csv
wget https://zenodo.org/record/4783391/files/clotho_metadata_validation.csv?download=1 -O clotho_metadata_validation.csv

cd ../