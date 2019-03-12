#!/bin/bash
python get_char_vocab.py gap-development.json gap-validation.json test_stage_1.json

python filter_embeddings.py glove.840B.300d.txt gap-development.json gap-validation.json test_stage_1.json
#python cache_elmo.py train.english.jsonlines dev.english.jsonlines
python cache_elmo.py gap-development.json gap-validation.json test_stage_1.json
