import logging
import os
import torch
from pathlib import Path

from transformers import AutoConfig, AutoTokenizer
# from transformers import TrainerCallback
from transformers import (
    HfArgumentParser,
    set_seed,
)

from .arguments import ModelArguments, DataArguments, \
    RetrieverTrainingArguments as TrainingArguments
from .data import TrainDatasetForEmbedding, EmbedCollator
from .modeling import BiEncoderModel
from .trainer import BiTrainer

logger = logging.getLogger(__name__)

# class ProfCallback(TrainerCallback):
#     def __init__(self, prof):
#         self.prof = prof

#     def on_step_end(self, args, state, control, **kwargs):
#         self.prof.step()

def main():
    parser = HfArgumentParser((ModelArguments, DataArguments, TrainingArguments))
    model_args, data_args, training_args = parser.parse_args_into_dataclasses()
    model_args: ModelArguments
    data_args: DataArguments
    training_args: TrainingArguments

    if (
            os.path.exists(training_args.output_dir)
            and os.listdir(training_args.output_dir)
            and training_args.do_train
            and not training_args.overwrite_output_dir
    ):
        raise ValueError(
            f"Output directory ({training_args.output_dir}) already exists and is not empty. Use --overwrite_output_dir to overcome."
        )

    # Setup logging
    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s -   %(message)s",
        datefmt="%m/%d/%Y %H:%M:%S",
        level=logging.INFO if training_args.local_rank in [-1, 0] else logging.WARN,
    )
    logger.warning(
        "Process rank: %s, device: %s, n_gpu: %s, distributed training: %s, 16-bits training: %s",
        training_args.local_rank,
        training_args.device,
        training_args.n_gpu,
        bool(training_args.local_rank != -1),
        training_args.fp16,
    )
    logger.info("Training/evaluation parameters %s", training_args)
    logger.info("Model parameters %s", model_args)
    logger.info("Data parameters %s", data_args)

    # Set seed
    set_seed(training_args.seed)

    num_labels = 1
    tokenizer = AutoTokenizer.from_pretrained(
        model_args.tokenizer_name if model_args.tokenizer_name else model_args.model_name_or_path,
        cache_dir=model_args.cache_dir,
        use_fast=False,
    )
    config = AutoConfig.from_pretrained(
        model_args.config_name if model_args.config_name else model_args.model_name_or_path,
        num_labels=num_labels,
        cache_dir=model_args.cache_dir,
    )
    logger.info('Config: %s', config)

    model = BiEncoderModel(model_name=model_args.model_name_or_path,
                           normlized=model_args.normlized,
                           sentence_pooling_method=training_args.sentence_pooling_method,
                           negatives_cross_device=training_args.negatives_cross_device,
                           temperature=training_args.temperature,
                           num_pos_queries=data_args.num_pos_queries,
                           loss_type=training_args.loss_type)

    if training_args.fix_position_embedding:
        for k, v in model.named_parameters():
            if "position_embeddings" in k:
                logging.info(f"Freeze the parameters for {k}")
                v.requires_grad = False

    train_dataset = TrainDatasetForEmbedding(args=data_args, tokenizer=tokenizer)

    eval_dataset = None
    if data_args.eval_data is not None:
        eval_dataset = TrainDatasetForEmbedding(args=data_args, tokenizer=tokenizer, mode='eval')
    

    trainer = BiTrainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        data_collator=EmbedCollator(
            tokenizer,
            query_max_len=data_args.query_max_len,
            passage_max_len=data_args.passage_max_len
        ),
        tokenizer=tokenizer
        
    )

    # with torch.profiler.profile(activities=[torch.profiler.ProfilerActivity.CPU,
    #                                     torch.profiler.ProfilerActivity.CUDA], 
    #                         schedule=torch.profiler.schedule(skip_first=100, wait=1, warmup=1, active=2, repeat=20),
    #                         on_trace_ready=torch.profiler.tensorboard_trace_handler(training_args.logging_dir),
    #                         profile_memory=True,
    #                         with_stack=True,
    #                         record_shapes=True) as prof:
    #     trainer.add_callback(ProfCallback(prof=prof))

    Path(training_args.output_dir).mkdir(parents=True, exist_ok=True)

    # Training
    trainer.train()
    trainer.save_model()
    # For convenience, we also re-save the tokenizer to the same directory,
    # so that you can share your model easily on huggingface.co/models =)
    if trainer.is_world_process_zero():
        tokenizer.save_pretrained(training_args.output_dir)


if __name__ == "__main__":
    main()
