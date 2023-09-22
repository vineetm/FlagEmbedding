#!/bin/sh
NUM_GPUS=2
LOGS_DIR="logs"
mkdir -p ${LOGS_DIR}

TB_DIR="tb"
mkdir -p ${TB_DIR}

MODEL_CKPTS="model-ckpts"
mkdir -p "${MODEL_CKPTS}"

MODEL_NAME="synthetic-data-slate-74M"

CMD="""
torchrun --nproc_per_node ${NUM_GPUS} \
-m FlagEmbedding.baai_general_embedding.finetune.run \
--output_dir "${MODEL_CKPTS}/${MODEL_NAME}" \
--model_name_or_path /dccstor/retrieve-rerank2/models/slate_rtvr_74M_st/ \
--train_data /u/vineeku6/data/embeddings-data/output_mined_hn.jsonl \
--learning_rate 1e-5 \
--fp16 \
--num_train_epochs 5 \
--per_device_train_batch_size 8 \
--dataloader_drop_last True \
--normlized True \
--temperature 0.02 \
--query_max_len 64 \
--passage_max_len 256 \
--train_group_size 15 \
--negatives_cross_device \
--logging_steps 100 \
--save_steps 1000 \
--save_total_limit 5 \
--report_to "tensorboard" \
--logging_dir "${TB_DIR}/${MODEL_NAME}" \
"""

jbsub -q x86_24h -mem 32g -cores 1x1+2 -name femb -out "${LOGS_DIR}/${MODEL_NAME}.out" ${CMD}