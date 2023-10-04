import argparse
from tqdm import tqdm
import logging
from FlagEmbedding.baai_general_embedding.finetune.data import TrainDatasetForEmbedding
from transformers import AutoTokenizer

logging.basicConfig(level=logging.INFO)

def setup_args():
  parser = argparse.ArgumentParser()
  parser.add_argument('--jsonl')
  return parser.parse_args()

class DataArgs:
    def __init__(self, train_data):
        self.train_data = train_data
        self.query_instruction_for_retrieval = ''
        self.passage_instruction_for_retrieval = ''
        self.train_group_size = 15
        self.query_max_len = 64 
        self.passage_max_len = 256

def bucket_distn(distn):
    buckets = {
        '<64': 0,
        '64-128': 0,
        '128-256': 0,
        '>256': 0
    }

    for key in distn:
        if key < 64:
            bucket_key = '<64'
        elif 64 <= key <=128:
            bucket_key = '64-128'
        elif 128 < key <= 256:
            bucket_key = '128-256'
        else:
            bucket_key = '>256'

        buckets[bucket_key] += distn[key]

    return buckets    

def main():
  args = setup_args()
  logging.info(f'{args=}')

  data_args = DataArgs(args.jsonl)
  model_id = 'bert-base-uncased'
  tokenizer = AutoTokenizer.from_pretrained(model_id)
  
  distn = {
    'query': {},
    'psg': {}
  }



  dataset = TrainDatasetForEmbedding(args=data_args, tokenizer=tokenizer)  
  for datum in tqdm(dataset):
    query, passages = datum
    
    num_tokens_query = len(tokenizer.encode(query[0]))
    distn['query'][num_tokens_query] = distn['query'].get(num_tokens_query, 0) + 1

    for psg in passages[:1]:
        num_tokens_psg = len(tokenizer.encode(psg))
        distn['psg'][num_tokens_psg] = distn['psg'].get(num_tokens_psg, 0) + 1

  for key in distn:
      buckets = bucket_distn(distn[key])
      logging.info(f'{key=}, {buckets=}')

if __name__ == '__main__':
  main()