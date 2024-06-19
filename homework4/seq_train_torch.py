import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from collections import Counter
import random
import time
from torch.cuda.amp import GradScaler, autocast
import re
import torch.multiprocessing as mp
from functools import partial

# Clear CUDA cache if necessary
torch.cuda.empty_cache()

# Function to convert text to a sequence of indices based on vocabulary
def text_to_sequence(text, vocab):
    return [vocab.get(word, vocab['<pad>']) for word in text]

# Dataset class for handling text sequences
class TextDataset(Dataset):
    def __init__(self, sequences):
        self.sequences = sequences

    def __len__(self):
        return len(self.sequences)

    def __getitem__(self, idx):
        return self.sequences[idx]

# Custom collate function for DataLoader
def collate_fn(batch, vocab):
    batch.sort(key=lambda x: len(x[0]), reverse=True)
    src_seqs = [torch.tensor(x[0]) for x in batch]
    trg_seqs = [torch.tensor(x[1]) for x in batch]
    src_lengths = [len(x) for x in src_seqs]
    trg_lengths = [len(x) for x in trg_seqs]

    padded_src_seqs = torch.nn.utils.rnn.pad_sequence(src_seqs, batch_first=True, padding_value=vocab['<pad>'])
    padded_trg_seqs = torch.nn.utils.rnn.pad_sequence(trg_seqs, batch_first=True, padding_value=vocab['<pad>'])

    return padded_src_seqs, src_lengths, padded_trg_seqs, trg_lengths

# Data preprocessing function
def data_pre(data_path):
    with open(data_path, 'r', encoding='utf-8') as file:
        lines = file.readlines()

    cleaned_texts = []
    for line in lines:
        cleaned_text = re.findall(r'[\u4e00-\u9fff]+', line.strip())
        cleaned_texts.append(cleaned_text)

    all_words = [word for text in cleaned_texts for word in text]
    word_counts = Counter(all_words)
    vocab = sorted(word_counts, key=word_counts.get, reverse=True)
    vocab_dict = {'<pad>': 0, '<unk>': 1, '<sos>': 2, '<eos>': 3}
    vocab_dict.update({word: idx + 4 for idx, word in enumerate(vocab)})

    sequences = []
    for text in cleaned_texts:
        src_seq = text_to_sequence(text, vocab_dict)
        trg_seq = [vocab_dict['<sos>']] + src_seq + [vocab_dict['<eos>']]
        sequences.append((src_seq, trg_seq))

    dataset = TextDataset(sequences)
    collate_fn_with_vocab = partial(collate_fn, vocab=vocab_dict)
    dataloader = DataLoader(dataset, batch_size=4, shuffle=True, collate_fn=collate_fn_with_vocab, pin_memory=True, num_workers=4)
    
    return vocab_dict, dataset, dataloader, len(vocab_dict)

# Encoder class
class Encoder(nn.Module):
    def __init__(self, input_dim, emb_dim, hidden_dim, n_layers, dropout):
        super(Encoder, self).__init__()
        self.embedding = nn.Embedding(input_dim, emb_dim)
        self.rnn = nn.LSTM(emb_dim, hidden_dim, n_layers, dropout=dropout, batch_first=True)
        self.dropout = nn.Dropout(dropout)

    def forward(self, src, src_len):
        embedded = self.dropout(self.embedding(src))
        packed_embedded = nn.utils.rnn.pack_padded_sequence(embedded, src_len, batch_first=True, enforce_sorted=False)
        packed_outputs, (hidden, cell) = self.rnn(packed_embedded)
        outputs, _ = nn.utils.rnn.pad_packed_sequence(packed_outputs, batch_first=True)
        return outputs, hidden, cell

# Attention mechanism class
class SmallerAttention(nn.Module):
    def __init__(self, hidden_dim):
        super(SmallerAttention, self).__init__()
        self.attn = nn.Linear(hidden_dim * 2, hidden_dim // 2)
        self.v = nn.Parameter(torch.rand(hidden_dim // 2))

    def forward(self, hidden, encoder_outputs):
        src_len = encoder_outputs.shape[1]
        hidden = hidden[-1].unsqueeze(1).repeat(1, src_len, 1)
        energy = torch.tanh(self.attn(torch.cat((hidden, encoder_outputs), dim=2)))
        energy = energy.permute(0, 2, 1)
        v = self.v.repeat(encoder_outputs.shape[0], 1).unsqueeze(1)
        attention = torch.bmm(v, energy).squeeze(1)
        return torch.softmax(attention, dim=1)

# Decoder class
class Decoder(nn.Module):
    def __init__(self, output_dim, emb_dim, hidden_dim, n_layers, dropout, attention):
        super(Decoder, self).__init__()
        self.output_dim = output_dim
        self.attention = attention
        self.embedding = nn.Embedding(output_dim, emb_dim)
        self.rnn = nn.LSTM(hidden_dim + emb_dim, hidden_dim, n_layers, dropout=dropout, batch_first=True)
        self.fc_out = nn.Linear(hidden_dim * 2 + emb_dim, output_dim)
        self.dropout = nn.Dropout(dropout)

    def forward(self, input, hidden, cell, encoder_outputs):
        input = input.unsqueeze(1)
        embedded = self.dropout(self.embedding(input))
        a = self.attention(hidden, encoder_outputs).unsqueeze(1)
        weighted = torch.bmm(a, encoder_outputs)
        rnn_input = torch.cat((embedded, weighted), dim=2)
        output, (hidden, cell) = self.rnn(rnn_input, (hidden, cell))
        prediction = self.fc_out(torch.cat((output.squeeze(1), weighted.squeeze(1), embedded.squeeze(1)), dim=1))
        return prediction, hidden, cell

# Seq2Seq model class
class Seq2Seq(nn.Module):
    def __init__(self, encoder, decoder, device):
        super(Seq2Seq, self).__init__()
        self.encoder = encoder
        self.decoder = decoder
        self.device = device

    def forward(self, src, src_len, trg, teacher_forcing_ratio=0.5):
        trg_len = trg.shape[1]
        batch_size = trg.shape[0]
        trg_vocab_size = self.decoder.output_dim

        outputs = torch.zeros(batch_size, trg_len, trg_vocab_size).to(self.device)
        encoder_outputs, hidden, cell = self.encoder(src, src_len)

        input = trg[:, 0]
        for t in range(1, trg_len):
            output, hidden, cell = self.decoder(input, hidden, cell, encoder_outputs)
            outputs[:, t, :] = output
            top1 = output.argmax(1)
            input = trg[:, t] if random.random() < teacher_forcing_ratio else top1

        return outputs

# Training function
def train(model, dataloader, optimizer, criterion, clip, scaler, num_classes):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model.to(device)
    model.train()
    epoch_loss = 0

    for src, src_len, trg, trg_len in dataloader:
        src, trg = src.to(device), trg.to(device)

        optimizer.zero_grad()
        
        with autocast():
            outputs = model(src, src_len, trg)
            output_dim = outputs.shape[-1]
            outputs = outputs[:, 1:].reshape(-1, output_dim)
            trg = trg[:, 1:].reshape(-1)

            loss = criterion(outputs, trg)

        scaler.scale(loss).backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), clip)
        scaler.step(optimizer)
        scaler.update()

        epoch_loss += loss.item()

    return epoch_loss / len(dataloader)

# # Function to generate the next sentence given an input sentence
# def generate_next_sentence(model, input_sentence, max_len, vocab, device):
#     model.eval()
#     with torch.no_grad():
#         input_indexes = text_to_sequence(input_sentence, vocab)
#         src_tensor = torch.tensor(input_indexes, dtype=torch.long).unsqueeze(0).to(device)
#         src_len = torch.tensor([len(input_indexes)])
#         encoder_outputs, hidden, cell = model.encoder(src_tensor, src_len)

#         trg_indexes = [vocab['<sos>']]
#         for _ in range(max_len):
#             trg_tensor = torch.tensor([trg_indexes[-1]], dtype=torch.long).to(device)
#             output, hidden, cell = model.decoder(trg_tensor, hidden, cell, encoder_outputs)
#             pred_token = output.argmax(1).item()
#             trg_indexes.append(pred_token)
#             if pred_token == vocab['<eos>']:
#                 break

#         trg_tokens = [list(vocab.keys())[i] for i in trg_indexes]
#         generated_sentence = ' '.join(trg_tokens[1:-1])

#     return generated_sentence

# # Function to generate a sequence of sentences based on an input sentence
# def generate_text_seq2seq(model, input_sentence, max_len, vocab, device):
#     model.eval()
#     generated_sentences = []
#     current_input = input_sentence

#     generated_sentence = generate_next_sentence(model, current_input, max_len, vocab, device)
#     generated_sentences.append(generated_sentence)
#     current_input = generated_sentence.split()

#     return generated_sentences

# Function to generate the next sentence given an input sentence
def generate_next_sentence(model, input_sentence, max_len, vocab, device):
    model.eval()
    with torch.no_grad():
        input_indexes = text_to_sequence(input_sentence, vocab)
        src_tensor = torch.tensor(input_indexes, dtype=torch.long).unsqueeze(0).to(device)
        src_len = torch.tensor([len(input_indexes)])
        encoder_outputs, hidden, cell = model.encoder(src_tensor, src_len)

        trg_indexes = [vocab['<sos>']]
        for _ in range(max_len):
            trg_tensor = torch.tensor([trg_indexes[-1]], dtype=torch.long).to(device)
            output, hidden, cell = model.decoder(trg_tensor, hidden, cell, encoder_outputs)
            pred_token = output.argmax(1).item()
            trg_indexes.append(pred_token)
            if pred_token == vocab['<eos>']:
                break

        trg_tokens = [list(vocab.keys())[i] for i in trg_indexes]
        generated_sentence = ''.join(trg_tokens[1:-1])

    return generated_sentence

# Function to generate a sequence of sentences based on an input sentence
def generate_text_seq2seq(model, input_sentence, max_len, vocab, device):
    model.eval()
    generated_sentences = []
    current_input = input_sentence

    generated_sentence = generate_next_sentence(model, current_input, max_len, vocab, device)
    generated_sentences.append(generated_sentence)
    current_input = generated_sentence.split()

    return generated_sentences

# 生成句子的函数
def generate_sentence_for_test_data(model, test_data, max_len, vocab, device):
    
    for input_sentence in test_data:
        generated_sentences = []
        generated_sentence = generate_next_sentence(model, [vocab['<sos>']] + text_to_sequence(input_sentence, vocab), max_len, vocab, device)
        print(f"Input: {input_sentence}")
        print(f"Generated Sentence: {generated_sentence}\n")

if __name__ == "__main__":
    start_time = time.time()
    novels = '白马啸西风'
    data_path = "./data白马啸西风.csv"#"./filtered_sentences_"+ novels +".txt"

    # Preprocess the data and prepare the dataset
    vocab, dataset, dataloader, vocab_size = data_pre(data_path)
    print(f"Data loaded and preprocessed in {time.time() - start_time:.2f} seconds")

    # Model hyperparameters
    INPUT_DIM = vocab_size
    OUTPUT_DIM = vocab_size
    ENC_EMB_DIM = 256
    DEC_EMB_DIM = 256
    HID_DIM = 256
    N_LAYERS = 2
    ENC_DROPOUT = 0.1
    DEC_DROPOUT = 0.1
    LEARNING_RATE = 0.0002
    N_EPOCHS = 50
    CLIP = 1

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # Initialize encoder, attention, decoder, and Seq2Seq model
    enc = Encoder(INPUT_DIM, ENC_EMB_DIM, HID_DIM, N_LAYERS, ENC_DROPOUT)
    attn = SmallerAttention(HID_DIM)
    dec = Decoder(OUTPUT_DIM, DEC_EMB_DIM, HID_DIM, N_LAYERS, DEC_DROPOUT, attn)
    model = Seq2Seq(enc, dec, device).to(device)

    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)
    criterion = nn.CrossEntropyLoss(ignore_index=vocab['<pad>'])
    scaler = GradScaler()

    # # Train the model
    # for epoch in range(N_EPOCHS):
    #     start_time = time.time()
    #     train_loss = train(model, dataloader, optimizer, criterion, CLIP, scaler, vocab_size)
    #     end_time = time.time()
    #     print(f"Epoch: {epoch+1}/{N_EPOCHS}, Train Loss: {train_loss:.4f}, Time: {end_time - start_time:.2f} seconds")

    # # Save the trained model
    # torch.save(model.state_dict(), 'seq2seq_model.pth')
    # print("Model saved as 'seq2seq_model.pth'")

    # # Load the trained model
    # model.load_state_dict(torch.load('seq2seq_model.pth'))
    # model.eval()

    # 初始化一个空的句子列表
    generated_sentences = []
    input_sentence = ['在黄沙莽莽的回疆大漠之上']
    generated_sentences.extend(input_sentence)
    # 循环生成指定次数的句子
    num_iterations = 2
    # for _ in range(num_iterations):
    #     # 生成下一个句子
    #     next_sentence = generate_text_seq2seq(model, input_sentence, 50, vocab, device)
        
    #     # 将新生成的句子添加到列表中
    #     generated_sentences.extend(next_sentence)
        
    #     # 更新输入句子为新生成的句子，以便下一次迭代
    #     input_sentence = generated_sentences
        
    #     print(input_sentence)
    for _ in range(num_iterations):
        # 生成下一个句子
        next_sentence = []
        next_sentence = generate_text_seq2seq(model, input_sentence, 50, vocab, device)
        
        # 将新生成的句子添加到列表中
        generated_sentences.extend(next_sentence)
        
        # 更新输入句子为新生成的句子，以便下一次迭代
        input_sentence += next_sentence
        print("next_sentence:")
        print(next_sentence)
        
    # 打印生成的句子
    print("Generated sentences:")
    print(generated_sentences)
    
    # 测试数据列表
    # test_data = [
    #     "这一著变起仓卒霍元龙和陈达海一惊之下急忙翻身下马上前抢救",
    #     "番耻笑罢了",
    #     "如此又相持良久从後洞映进来的日光越来越亮似乎已是正午突然",
    #     "一条大缝那个美丽的姑娘就跳了进去後来这对情人变成了一双蝴蝶总",
    #     "坐骑渐渐追近",
    #     "自从晋威镖局一干豪客在这带草原上大施劫掠之後哈萨克人对汉人极",
    #     "四顾打量周遭情景只见西北角上血红的夕阳之旁升起一片黄蒙蒙的云",
    #     "身上强盗就死了李文秀吃了一惊适才早见到他手中持针当时也没",
    #     "陈达海短小精悍原是辽东马贼出身後来却在山西落脚和霍史二人意气",
    #     "然记得那怎麽会忘记李文秀道你怎麽不去瞧瞧她的坟墓苏普"]

    # # 确保模型已经加载了权重
    # model.load_state_dict(torch.load('seq2seq_model.pth'))
    # model.eval()


    # # 调用函数生成句子
    # generate_sentence_for_test_data(model, test_data, 50, vocab, device)