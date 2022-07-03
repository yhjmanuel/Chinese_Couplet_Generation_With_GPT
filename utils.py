import random
random.seed(42)
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import math
torch.cuda.is_available()

# useful activation functions
# relu: max(0, x)
# gelu: x∗Φ(x), where Φ(x) is the Cumulative Distribution Function for Gaussian Distribution
# swish: x * sigmoid(x)
ACT_FNS = {'relu': nn.ReLU(),
           'gelu': nn.GELU(),
           'swish': nn.SiLU()}


class DatasetConfig:
    '''
    Defines where to read the data, how long each couplet is
    '''
    train_in_dir = "Dataset/train_in.txt"
    train_out_dir = "Dataset/train_out.txt"
    test_in_dir = "Dataset/test_in.txt"
    test_out_dir = "Dataset/test_out.txt"
    max_len = 14

class Couplets:
    '''
    Dataset processor
    '''
    def __init__(self, config):
        self.config = config
        self.train_set = None
        self.dev_set = None
        self.test_set = None
        self.dic = {}

    def read_data(self):
        self.read_data_helper()

    def read_data_helper(self):
        # get train and dev set
        train_data_x = []
        train_data_y = []
        with open(self.config.train_in_dir) as doc:
            for line in doc:
                if len(line[:-1].split()) == self.config.max_len // 2:
                    train_data_x.append(line[:-1])
        with open(self.config.train_out_dir) as doc:
            for line in doc:
                if len(line[:-1].split()) == self.config.max_len // 2:
                    train_data_y.append(line[:-1])
        for i in range(len(train_data_x)):
            train_data_x[i] += train_data_y[i]
            train_data_x[i] = train_data_x[i].replace(" ", "")
        self.create_word_dict(train_data_x)
        # get dev set
        random.shuffle(train_data_x)
        split_point = int(len(train_data_x) * 0.9)
        self.train_set = train_data_x[: split_point]
        self.dev_set = train_data_x[split_point:]

        # get test set
        test_data_x = []
        test_data_y = []
        with open(self.config.test_in_dir) as doc:
            for line in doc:
                if len(line[:-1].split()) == self.config.max_len // 2:
                    test_data_x.append(line[:-1])
        with open(self.config.test_out_dir) as doc:
            for line in doc:
                if len(line[:-1].split()) == self.config.max_len // 2:
                    test_data_y.append(line[:-1])
        for i in range(len(test_data_x)):
            test_data_x[i] += test_data_y[i]
            test_data_x[i] = test_data_x[i].replace(" ", "")
        self.test_set = test_data_x

    def create_word_dict(self, data):
        word_list = []
        for i in range(len(data)):
            for j in data[i]:
                word_list.append(j)
        self.dic = {token: i + 2 for i, token in enumerate(list(set(word_list)))}
        self.dic['[PAD]'] = 0
        self.dic['[UNK]'] = 1

    # convert Chinese characters into their indices in the word_dict
    def get_indexed_data(self, dataset="train"):
        indexed_data = []
        if dataset == "train":
            for i in range(len(self.train_set)):
                j = len(self.train_set[i])
                indexed_data.append([self.dic[self.train_set[i][k]] for k in range(j)])
        elif dataset == "dev":
            for i in range(len(self.dev_set)):
                j = len(self.dev_set[i])
                indexed_data.append([self.dic[self.dev_set[i][k]] for k in range(j)])
        elif dataset == "test":
            for i in range(len(self.test_set)):
                j = len(self.test_set[i])
                temp = []
                for k in range(j):
                    if self.test_set[i][k] in self.dic:
                        temp.append(self.dic[self.test_set[i][k]])
                    else:
                        temp.append(self.dic['[UNK]'])
                indexed_data.append(temp)
        return indexed_data

class LayerNorm(nn.Module):
    '''
    take a "row" of data as a semantic unit and do normalization
    normalization method: every tensor subtract the mean of the row it's in, divided by the std of the same row
    '''

    def __init__(self, features, eps=1e-5):
        super(LayerNorm, self).__init__()
        self.w = nn.Parameter(torch.ones(features))
        self.b = nn.Parameter(torch.zeros(features))
        self.eps = eps

    def forward(self, x):
        mean = x.mean(-1, keepdim=True)
        std = x.std(-1, keepdim=True)
        return self.w * (x - mean) / (self.eps + std) + self.b

class PositionalEmbedding(nn.Module):
    '''
    Use sin/cos positional encoding
    '''
    def __init__(self, d_embeddings, max_len):
        super(PositionalEmbedding, self).__init__()
        pos = torch.arange(0.0, max_len).unsqueeze(1)
        denom = torch.exp(torch.arange(0.0, d_embeddings, 2) * (-math.log(10000) / d_embeddings))
        self.positional_embeddings = torch.zeros(max_len, d_embeddings)
        self.positional_embeddings[:, 0::2] = torch.sin(pos * denom)
        self.positional_embeddings[:, 1::2] = torch.cos(pos * denom)
        self.positional_embeddings = self.positional_embeddings.unsqueeze(0)

    def forward(self, x):
        pos = torch.autograd.Variable(self.positional_embeddings[:, :x.size(1)], requires_grad=False)
        pos = pos.to("cuda") if torch.cuda.is_available() else pos
        return x + pos


def attention(q, k, v, mask=None, dropout=None):
    # a single-head attention. Mapping q, k, v matrix to an output, which is the weighted sum of the v matrix
    # the weights are calculated using the scaled dot product attention
    # the shape of q, k, and v: (batch_size, max_len_seq, d_embedding)
    # if we use a 8-head attention, the shape will be (batch_size, 8, max_len_seq, d_embeeding / 8)
    assert q.shape == k.shape == v.shape
    k = k.transpose(-2, -1)

    # q * k
    if mask:
        scores = torch.matmul(q, k) / math.sqrt(q.shape[-1]).masked_fill(mask == 0, -1e9)
    scores = torch.matmul(q, k) / math.sqrt(q.shape[-1])
    # get attention score using softmax
    scores_attention = F.softmax(scores, dim=-1)
    # apply dropout if any
    if dropout:
        dropout_layer = nn.Dropout(dropout)
        scores_attention = dropout_layer(scores_attention)
    return torch.matmul(scores_attention, v), scores_attention


# implement multi-headed attention layer
class MultiHeadAttention(nn.Module):
    '''
    The multi-head attention mechanism allows model to learn from different subspaces
    We set the default number of heads to be 8
    In this module, the single-head attention method that we define above is used
    '''
    def __init__(self, n_heads, d_embeddings, dropout=0.1):
        super(MultiHeadAttention, self).__init__()
        self.n_heads = n_heads
        self.d_embeddings = d_embeddings
        self.dropout = dropout
        self.wq = nn.Linear(d_embeddings, d_embeddings)
        self.wk = nn.Linear(d_embeddings, d_embeddings)
        self.wv = nn.Linear(d_embeddings, d_embeddings)
        # wo: used to reproject the attention-weighted v matrix
        self.wo = nn.Linear(d_embeddings, d_embeddings)
        assert self.d_embeddings % self.n_heads == 0
        self.scores_attention = None

    def forward(self, q, k, v, mask=None):
        batch_size = q.shape[0]
        # reshape: (batch_size, max_len, d_embedding) -> (batch_size, n_heads, max_len, d_embeeding / n_heads)
        # matmul and split
        q = self.wq(q).view(batch_size, self.n_heads, -1, self.d_embeddings // self.n_heads)
        k = self.wk(k).view(batch_size, self.n_heads, -1, self.d_embeddings // self.n_heads)
        v = self.wv(v).view(batch_size, self.n_heads, -1, self.d_embeddings // self.n_heads)
        if mask:
            # we add a dimension to q, k, v, so we'll need to add a dimension to the original mask too
            mask = mask.unsqueeze(1)
        # the shape of the returned v: (batch_size, n_heads, max_len, d_embeeding / n_heads)
        v, self.scores_attention = attention(q, k, v, mask=mask, dropout=self.dropout)
        return self.wo(v.view(batch_size, -1, self.d_embeddings))


class FeedForwardNN(nn.Module):
    '''
    Feed forward neural network with activation function and dropout
    we store all the activation functions in a dict to facilitate different experiments
    (d_model, d_model) -> (d_model, d_hidden) -> (d_model, d_model)
    '''
    def __init__(self, d_embeddings, d_hidden, dropout=0.1, activation_function='relu'):
        super(FeedForwardNN, self).__init__()
        self.d_embeddings = d_embeddings
        self.d_hidden = d_hidden
        self.dropout = dropout
        self.linear1 = nn.Linear(d_embeddings, d_hidden)
        self.linear2 = nn.Linear(d_hidden, d_embeddings)
        self.act_fn = ACT_FNS[activation_function]

    def forward(self, x):
        x = self.linear2(self.linear1(x))
        dropout_layer = nn.Dropout(self.dropout)
        return dropout_layer(self.act_fn(x))


class Block(nn.Module):
    '''
    Assembles several modules to form a single block of GPT
    '''
    def __init__(self, n_heads, d_embeddings, dropout, d_hidden, activation_function="relu"):
        super(Block, self).__init__()
        self.attention = MultiHeadAttention(n_heads=n_heads, d_embeddings=d_embeddings, dropout=dropout)
        self.ln1 = LayerNorm(d_embeddings)
        self.ln2 = LayerNorm(d_embeddings)
        self.ffnn = FeedForwardNN(d_embeddings, d_hidden,
                                  dropout=dropout, activation_function=activation_function)

    def forward(self, x):
        v = self.attention(x, x, x)
        # residual connection + layer normalization
        y1 = self.ln1(x + v)
        y2 = self.ffnn(y1)

        # residual connection + layer normalization
        return self.ln2(y1 + y2)

class GPT(nn.Module):
    '''
    The final GPT model
    '''
    def __init__(self, vocab_size, n_heads, n_blocks, d_embeddings, max_len,
                 dropout, d_hidden, activation_function="relu"):
        super(GPT, self).__init__()
        self.vocab_size = vocab_size
        self.embed = nn.Embedding(vocab_size, d_embeddings)
        self.pos_embed = PositionalEmbedding(d_embeddings, max_len)
        self.d_embeddings = d_embeddings
        self.dropout = dropout
        self.max_len = max_len
        self.blocks = nn.ModuleList([Block(n_heads=n_heads, d_embeddings=d_embeddings, dropout=dropout,
                                           d_hidden=d_hidden, activation_function=activation_function) for _ in
                                     range(n_blocks)])
        self.linear = nn.Linear(max_len * d_embeddings, vocab_size)

    def forward(self, x):
        # token embedding: with dropout
        # positional embedding: without dropout

        # here, x is a batch with shape (batch_size, max_len)
        dropout_layer = nn.Dropout(self.dropout)
        # x1 shape: (batch_size, max_len, d_embeddings)
        x1 = dropout_layer(self.embed(x))

        x1 = self.pos_embed(x1)
        for block in self.blocks:
            x1 = block(x1)
        # out final logits
        return self.linear(x1.view(-1, self.max_len * self.d_embeddings))

# non auto-regressive training
def run_model(x_train, y_train, x_dev, y_dev, GPT_model,
              batch_size=1024, n_epochs=10, lr=1e-4, output_path="model.pt"):
    optimizer = torch.optim.Adam(GPT_model.parameters(), lr=lr)
    lr_scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer,
                                                              factor=0.9,
                                                              mode="min",
                                                              patience=10,
                                                              cooldown=10,
                                                              min_lr=5e-6,
                                                              verbose=True)
    loss_func = nn.CrossEntropyLoss()

    iter_num = 1
    for epoch in range(n_epochs):
        print("Epoch {:} out of {:}".format(epoch + 1, n_epochs))
        temp = list(zip(x_train, y_train))
        random.shuffle(temp)
        x_train_ts, y_train_ts = zip(*temp)
        x_train_ts = torch.tensor(list(x_train_ts)).to("cuda") if torch.cuda.is_available() else torch.tensor(x_train_ts)
        y_train_ts = torch.tensor(list(y_train_ts)).to("cuda") if torch.cuda.is_available() else torch.tensor(y_train_ts)
        train_for_epoch(iter_num, x_train_ts, y_train_ts, x_dev, y_dev, GPT_model, optimizer,
                        loss_func, batch_size, lr_scheduler)
        iter_num += 1
    torch.save(GPT_model.state_dict(), output_path)
    print("Model successfully saved to " + output_path)

# defines an epoch of training
def train_for_epoch(iter_num, x_train, y_train, x_dev, y_dev, GPT_model, optimizer,
                    loss_func, batch_size, lr_scheduler):
    GPT_model.train()
    n_minibatches = math.ceil(len(x_train) / batch_size)
    correct = 0
    n_tokens_to_predict = 0
    loss_sum = 0.
    for i in range(n_minibatches):
        optimizer.zero_grad()
        x_train_minibatch = x_train[i * batch_size: (i + 1) * batch_size]
        y_train_minibatch = y_train[i * batch_size: (i + 1) * batch_size]
        logits = GPT_model(x_train_minibatch)
        n_tokens_to_predict += len(y_train_minibatch)
        correctly_predicted = sum(logits.argmax(dim=1) == y_train_minibatch)
        correct += correctly_predicted
        loss = loss_func(logits, y_train_minibatch)
        if (i + 100) % 1 == 0:
            print("minibatch {} loss: {}".format(i + 1, loss.data.item()))
            print("Average Unigram Accuracy after batch {}: {}".format(i + 1, int(correctly_predicted) / int(n_tokens_to_predict)))
        loss.backward()
        optimizer.step()
        lr_scheduler.step(loss)
        loss_sum += loss
    
    GPT_model.eval()
    with torch.no_grad():
        eval_logits = GPT_model(x_dev)
        print ("Dev Loss: {}".format(loss_func(eval_logits, y_dev)))
        dev_acc = sum(eval_logits.argmax(dim=-1) == y_dev) / len(y_dev)
        print ("Dev Unigram Accuracy: {}".format(dev_acc))

# helper method, make masks
def get_fea_lbl(dataset, max_len=14):
    features, labels = [], []
    for data in dataset:
        for i in range(max_len // 2, max_len):
            features.append(data[: i] + [0] * (max_len - i))
            labels.append(data[i])
    return features, labels

# used for inference
def predict(model, sen_str, dic, mode='multi_char'):
    # set mode = "single_char" to predict only a single character
    reverse_dic = {dic[key]: key for key in dic}
    if mode == 'multi_char':
        pred_char = ''
        for i in range(7, 0, -1):
            test = [i for i in sen_str] + ['[PAD]'] * i
            test = torch.tensor([dic[i] for i in test]).reshape(1, 14).to("cuda")
            output = model(test)
            pred_char = reverse_dic[int(torch.argmax(output))]
            sen_str += pred_char
        return sen_str[:7] + "," + sen_str[7:]
    else:
        test = [i for i in sen_str] + ['[PAD]'] * (14 - len(sen_str))
        test = torch.tensor([dic[i] for i in test]).reshape(1, 14).to("cuda")
        output = model(test)
        pred_char = reverse_dic[int(torch.argmax(output))]
        return sen_str[:7] + "," + sen_str[7:] + pred_char
