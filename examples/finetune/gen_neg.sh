#!/bin/sh
INP="/u/vineeku6/data/embeddings-data/output.jsonl"
OUT="/u/vineeku6/data/embeddings-data/output_mined_hn.jsonl"

python -m FlagEmbedding.baai_general_embedding.finetune.hn_mine \
--model_name_or_path BAAI/bge-base-en-v1.5 \
--input_file  "${INP}" \
--output_file "${OUT}" \
--range_for_sampling 2-200 \
--use_gpu_for_searching