import pickle
from transformers import GPT2Tokenizer, GPT2LMHeadModel, AdamW, get_linear_schedule_with_warmup, PreTrainedTokenizer

def load_tokenizer_from_cache():
    file = open('gpt_tokenizer.pkl', 'rb')
    tokenizer = pickle.load(file)
    file.close()
    return tokenizer

def load_gpt_from_cache():
    file = open('gpt_head.pkl', 'rb')
    gpt = pickle.load(file)
    file.close()
    return gpt

