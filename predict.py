import torch
import pickle
from utils import *
from train import *
torch.manual_seed(0)


def make_prediction(sen_str):
    '''
    sen_str: input
    word_dic: pass in the loaded word dictionary of the trained model
    '''
    result = predict(model=model,
                     sen_str=sen_str,
                     dic=word_dic,
                     mode="multi_char")
    return result

if __name__ == "main":
    model = GPT(vocab_size=vocab_size, n_heads=n_heads, n_blocks=n_blocks, d_embeddings=d_embeddings, max_len=max_len,
                dropout=dropout, d_hidden=d_hidden, activation_function=activation_function)
    # set / get model & dictionary path in train.py
    model.load_state_dict(torch.load(model_output_path, map_location=torch.device('cpu')))
    with open(dic_output_path, 'rb') as word_dict:
        word_dic = pickle.load(word_dict)
    sen_str = '雄鸡喜唱升平日'
    print(make_prediction(sen_str))

