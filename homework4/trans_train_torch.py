import torch
import random
import torch.nn as nn
from torch.nn import Transformer
from torch.utils.data import DataLoader, Dataset
import csv
import math
import time
from collections import Counter, defaultdict
import torch.nn.functional as F
import ast
import re
from functools import partial
from torch.nn.utils.rnn import pad_sequence

# 禁用 torchtext 弃用警告
import torchtext
torchtext.disable_torchtext_deprecation_warning()

# 全局变量声明
source_word_2_idx = None
target_word_2_idx = None
source_idx_2_word = None
target_idx_2_word = None


class CorpusDataset(Dataset):
    def __init__(self, source_data, target_data, source_word_2_idx, target_word_2_idx):
        self.source_data = source_data
        self.target_data = target_data
        self.source_word_2_idx = source_word_2_idx
        self.target_word_2_idx = target_word_2_idx

    def __getitem__(self, index):
        # create one-hot index for every word
        src = self.source_data[index]
        tgt = self.target_data[index]
        src_index = [self.source_word_2_idx[i] for i in src]
        tgt_index = [self.target_word_2_idx[i] for i in tgt]

        return src_index, tgt_index


    def batch_data_alignment(self, batch_datas):
        global device
        src_index, tgt_index = [], []
        src_len, tgt_len = [], []

        for src, tgt in batch_datas:
            src_index.append(src)
            tgt_index.append(tgt)
            src_len.append(len(src))
            tgt_len.append(len(tgt))

        max_src_len = max(src_len)
        max_tgt_len = max(tgt_len)
        src_index = [[self.source_word_2_idx["<BOS>"]] + tmp_src_index + [self.source_word_2_idx["<EOS>"]] + 
                     [self.source_word_2_idx["<PAD>"]] * (max_src_len - len(tmp_src_index)) for tmp_src_index in src_index]
        tgt_index = [[self.target_word_2_idx["<BOS>"]] + tmp_src_index + [self.target_word_2_idx["<EOS>"]] + 
                     [self.target_word_2_idx["<PAD>"]] * (max_tgt_len - len(tmp_src_index)) for tmp_src_index in tgt_index]
        src_index = torch.tensor(src_index, device=device)
        tgt_index = torch.tensor(tgt_index, device=device)

        return src_index, tgt_index


    def __len__(self):
        assert len(self.source_data) == len(self.target_data)
        return len(self.target_data)

def data_pro(data_pat,num_corpus,num_test_corpus,batch_size):

    char_to_be_replaced = "\n 0123456789qwertyuiopasdfghjklzxcvbnm[]{};':\",./<>?ａｎｔｉ－ｃｌｉｍａｘ＋．／０１２３４５６７８９＜＝＞＠Ａ" \
                          "ＢＣＤＥＦＧＨＩＪＫＬＭＮＯＰＱＲＳＴＶＷＸＹＺ［＼］ｂｄｅｆｇｈｊｋｏｐｒｓ" \
                          "ｕｖｗｙｚ￣\u3000\x1a"

    source_target_corpus_ori = []

    with open(data_path, "r", encoding="UTF-8", errors="ignore") as tmp_file:
        tmp_file_context = tmp_file.read()
        for tmp_char in char_to_be_replaced:
            tmp_file_context = tmp_file_context.replace(tmp_char, "")
        tmp_file_context = tmp_file_context.replace("本书来自免费小说下载站更多更新免费电子书请关注", "")

        tmp_file_sentences = tmp_file_context.split("。")
        tmp_idx = 0
        # print(len(tmp_file_sentences))
        for i in range(len(tmp_file_sentences)-1):
            # if ("李" in tmp_sentence) and (10 <= len(tmp_sentence) <= 40) and (10 <= len(tmp_file_sentences[tmp_idx + 1]) <= 40):
            source_target_corpus_ori.append((tmp_file_sentences[tmp_idx], tmp_file_sentences[tmp_idx + 1]))
            tmp_idx += 1
    
    sample_indexes = random.sample(list(range(len(source_target_corpus_ori))), num_corpus)
    source_corpus, target_corpus = [], []
    for idx in sample_indexes:
        source_corpus.append(source_target_corpus_ori[idx][0])
        target_corpus.append(source_target_corpus_ori[idx][1])

    test_corpus = []
    for idx in range(len(source_target_corpus_ori)):
        if idx not in sample_indexes:
            test_corpus.append((source_target_corpus_ori[idx][0], source_target_corpus_ori[idx][1]))
    test_corpus = random.sample(test_corpus, num_test_corpus)
    test_source_corpus, test_target_corpus = [], []
    for tmp_src, tmp_tgt in test_corpus:
        test_source_corpus.append(tmp_src)
        test_target_corpus.append(tmp_tgt)

    ### one-hot dict generation
    idx_cnt = 0
    word_2_idx_dict = dict()
    idx_2_word_list = list()
    for tmp_corpus in [source_corpus, target_corpus, test_source_corpus, test_target_corpus]:
        for tmp_sentence in tmp_corpus:
            for tmp_word in tmp_sentence:
                if tmp_word not in word_2_idx_dict.keys():
                    word_2_idx_dict[tmp_word] = idx_cnt
                    idx_2_word_list.append(tmp_word)
                    idx_cnt += 1

    one_hot_dict_len = len(word_2_idx_dict)
    word_2_idx_dict.update({"<PAD>": one_hot_dict_len, "<BOS>": one_hot_dict_len + 1, "<EOS>": one_hot_dict_len + 2})
    idx_2_word_list += ["<PAD>", "<BOS>", "<EOS>"]
    one_hot_dict_len += 3
    
    global source_word_2_idx, target_word_2_idx, source_idx_2_word, target_idx_2_word
    source_word_2_idx, target_word_2_idx = word_2_idx_dict, word_2_idx_dict
    source_idx_2_word, target_idx_2_word = idx_2_word_list, idx_2_word_list
    source_corpus_len, target_corpus_len = one_hot_dict_len, one_hot_dict_len

    ### dataloader
    dataset = CorpusDataset(source_corpus, target_corpus, source_word_2_idx, target_word_2_idx)
    dataloader = DataLoader(dataset, batch_size, shuffle=False, collate_fn=dataset.batch_data_alignment)

    # return dataloader
    return word_2_idx_dict, idx_2_word_list, dataloader, len(word_2_idx_dict), test_corpus

# 定义位置编码和Transformer模型
class PositionalEncoding(nn.Module):
    def __init__(self, d_model, dropout=0.1, max_len=5000):
        super(PositionalEncoding, self).__init__()
        self.dropout = nn.Dropout(p=dropout)

        position = torch.arange(max_len).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2) * -(math.log(10000.0) / d_model))
        pe = torch.zeros(max_len, 1, d_model)
        pe[:, 0, 0::2] = torch.sin(position * div_term)
        pe[:, 0, 1::2] = torch.cos(position * div_term)

        self.register_buffer('pe', pe)

    def forward(self, x):
        x = x + self.pe[:x.size(0), :]
        return self.dropout(x)


class TransformerModel(nn.Module):
    def __init__(self, vocab_size, embed_size, num_heads, num_layers, hidden_dim, dropout=0.1):
        super(TransformerModel, self).__init__()
        self.embed_size = embed_size
        self.dropout_layer = nn.Dropout(p=dropout)  # 正确初始化dropout层
        self.encoder = nn.Embedding(vocab_size, embed_size)
        self.pos_encoder = PositionalEncoding(embed_size, dropout)
        self.transformer = nn.Transformer(d_model=embed_size, nhead=num_heads, num_encoder_layers=num_layers, num_decoder_layers=num_layers, dim_feedforward=hidden_dim, dropout=dropout, batch_first=True)
        self.decoder = nn.Linear(embed_size, vocab_size)

    def generate_square_subsequent_mask(self, sz):
        mask = (torch.triu(torch.ones(sz, sz)) == 1).transpose(0, 1)
        mask = mask.float().masked_fill(mask == 0, float('-inf')).masked_fill(mask == 1, float(0.0))
        return mask

    def forward(self, src, tgt):
        src_mask = self.generate_square_subsequent_mask(src.size(1)).to(src.device)
        tgt_mask = self.generate_square_subsequent_mask(tgt.size(1)).to(tgt.device)
        # 在forward方法内
        src = self.encoder(src) * math.sqrt(self.embed_size)
        src = self.dropout_layer(src)  # 使用正确的dropout实例
        src = self.pos_encoder(src)
        tgt = self.encoder(tgt) * math.sqrt(self.embed_size)
        tgt = self.pos_encoder(tgt)
        output = self.transformer(src, tgt, src_mask, tgt_mask)
        output = self.decoder(output)
        return output


def generate_sentence_transformer(sentence, model, max_len=40):
    d_model =256
    src_index = torch.tensor([[source_word_2_idx[i] for i in sentence]], device=device)
    
    with torch.no_grad():
        # 编码源句子
        memory = model.encoder(src_index) * math.sqrt(d_model)  # 确保sqrt(d_model)正确使用
        memory = model.pos_encoder(memory)
        
        # 初始化目标序列
        outs = [target_word_2_idx["<BOS>"]]
        tgt_tensor = torch.tensor([outs], device=device)  # 初始化

        for i in range(max_len - 1):
            tgt_mask = model.generate_square_subsequent_mask(tgt_tensor.size(1)).to(device)
            
            # 位置编码并确保维度正确
            tgt_tensor_with_pe = model.pos_encoder(model.encoder(tgt_tensor) * math.sqrt(d_model))
            
            # 确保tgt_tensor_with_pe的维度与memory匹配
            out = model.transformer(memory, tgt_tensor_with_pe, tgt_mask=tgt_mask)
            next_word_probs = model.decoder(out[:, -1, :])  # 获取最后一个时间步的输出
            next_word_idx = next_word_probs.argmax(dim=-1).item()
            temperature = 0.5  # 调节温度值，较低的值使模型更倾向于高概率词，较高的值增加随机性
            next_word_probs = next_word_probs / temperature
            next_word_probs = next_word_probs.softmax(dim=-1)
            next_word_idx = torch.multinomial(next_word_probs, 1).item()
            # 打印新生成的单词及其概率
            #print("Generated word:", target_idx_2_word[next_word_idx], "Probability:", next_word_probs.max().item())
            if next_word_idx == target_word_2_idx["<EOS>"]:
                break
            
            outs.append(next_word_idx)
            new_token = torch.tensor([[next_word_idx]], device=device)
            tgt_tensor = torch.cat((tgt_tensor, new_token), dim=1)
            
    return "".join([target_idx_2_word[i] for i in outs if i != target_word_2_idx["<BOS>"]])


if __name__ == "__main__":
    ### data preparation
    num_corpus = 100
    num_test_corpus = 10
    batch_size = 128   
    # 从 CSV 文件中读取数据
    novels = '白马啸西风'
    data_path = "./data白马啸西风.csv"

    word_2_idx_dict, idx_2_word_list, dataloader, vocab_size, test_corpus = data_pro(data_path,num_corpus,num_test_corpus,batch_size)
    #vocab_dict, train_dataset, train_dataloader, len(vocab_dict)
    # 训练模型
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    ntokens = len(word_2_idx_dict)  # 词汇表大小
    emsize = 256  # 嵌入维度
    nhid = 512 # Transformer中feedforward网络模型的维度
    nlayers = 2  # Transformer的层数
    nhead = 8  # 多头注意力机制的头数
    dropout = 0.1
    # dropout概率
    num_epochs = 100

    criterion = nn.CrossEntropyLoss()
    lr = 0.001  # 学习率


    ### model
    transformer_model = TransformerModel(
        vocab_size=vocab_size,
        embed_size=emsize,
        num_heads=nhead,
        num_layers=nlayers,
        hidden_dim=nhid,
        dropout=dropout
    ).to(device)

    ### train model
    optimizer = torch.optim.Adam(transformer_model.parameters(), lr=lr)
    losses = []

    for epoch in range(num_epochs):
        for step, (src_index, tgt_index) in enumerate(dataloader):
            src_index = src_index.clone().detach().to(device)
            tgt_index = tgt_index.clone().detach().to(device)

            optimizer.zero_grad()
            output = transformer_model(src_index, tgt_index)
            output = output.permute(1, 0, 2)

            # 计算损失，注意这里需要过滤掉 <PAD> 标记
            loss = nn.CrossEntropyLoss(ignore_index=word_2_idx_dict["<PAD>"], reduction='mean')(output.reshape(-1, vocab_size), tgt_index.reshape(-1))
            loss.backward()
            optimizer.step()

        losses.append(loss.item())
        print("Epoch: {}, Loss: {:.5f}".format(epoch + 1, loss.item()))

    
    for idx, (tmp_src_sentence, tmp_gt_sentence) in enumerate(test_corpus):
        print("第"+str(idx+1)+"个测试数据:")
        # 示例
        generated_sentences = []
        start_text = tmp_src_sentence
        print(tmp_src_sentence)
        # start_text = src_sentence[0].split()  # 分割句子为单词列表
        # generated_sentences.extend(tmp_src_sentence)
        num_iterations = 1
        for _ in range(num_iterations):
            # 生成下一个句子
            next_sentence = generate_sentence_transformer(start_text,transformer_model)
            
            # 将新生成的句子添加到列表中
            generated_sentences.extend(next_sentence)
            
            # 更新输入句子为新生成的句子，以便下一次迭代
            start_text = next_sentence
            # start_text = src_sentence[0].split()  # 分割句子为单词列表
              
            print(next_sentence)
            
        generated_sentences = "".join(generated_sentences)    
        # 打印生成的句子
        print("Generated sentences:")
        print(generated_sentences)
        print("Real sentences:")
        print(tmp_gt_sentence)
