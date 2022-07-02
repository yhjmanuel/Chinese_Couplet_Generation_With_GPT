import random
random.seed(42)
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import math
torch.cuda.is_available()
from utils import *
import pickle

# set the training config here
dic_output_path = 'dic.pickle'
vocab_size = 7305
n_heads = 8
n_blocks = 12
d_embeddings = 512
max_len = 14
dropout = 0.1
d_hidden = 1024
activation_function = "relu"
batch_size = 2048
n_epochs = 25
lr = 1e-4
model_output_path = 'dic.pickle'


def main():
    print("Loading data...")
    cps = Couplets(DatasetConfig)
    cps.read_data()
    train, dev, test = cps.get_indexed_data(dataset="train"), cps.get_indexed_data(dataset="dev"), cps.get_indexed_data(dataset="test")
    x_train, y_train = get_fea_lbl(train)
    x_dev, y_dev = get_fea_lbl(dev)
    x_dev, y_dev = torch.tensor(x_dev), torch.tensor(y_dev)
    print("Data successfully loaded")
    GPT_model = GPT(vocab_size=vocab_size, n_heads=n_heads, n_blocks=n_blocks, d_embeddings=d_embeddings, max_len=max_len,
                    dropout=dropout, d_hidden=d_hidden, activation_function=activation_function)
    x_dev = x_dev.to("cuda") if torch.cuda.is_available() else x_dev
    y_dev = y_dev.to("cuda") if torch.cuda.is_available() else y_dev
    GPT_model = GPT_model.to("cuda") if torch.cuda.is_available() else GPT_model

    # save the dictionary
    with open(dic_output_path, 'wb') as doc:
        pickle.dump(cps.dic, doc)
    print("Word dict saved to "+dic_output_path)
    print("Start training")
    run_model(x_train, y_train, x_dev, y_dev, GPT_model, lr = lr, batch_size=batch_size, n_epochs=n_epochs,
              output_path = model_output_path)


if __name__ == "__main__":
    main()