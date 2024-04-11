import math
import os
import re
from collections import Counter

import chardet
import jieba
import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import linregress


def calculate_entropy(word_freq):
    total_words = sum(word_freq.values())
    entropy = 0
    for count in word_freq.values():
        probability = count / total_words
        entropy += -probability * math.log(probability, 2)
    return entropy


def book_titles(file_path):
    with open(file_path, 'r', encoding='gbk') as file:
        file_titles = file.read()
    file_titles = file_titles.split(',')
    return file_titles



def read_articles(file_titles):
    all_content = ""
    corpus = []
    # 遍历文件标题，查找并读取对应的文件内容
    for title in file_titles:
        title = title.strip()
        title = "./jyxstxtqj_downcc.com/" + title + ".txt"
        if os.path.exists(title):
            with open(title, 'rb') as f:
                rawdata = f.read()
                result = chardet.detect(rawdata)
                encoding = result['encoding']

            with open(title, 'r', encoding=encoding, errors='ignore') as file:
                text = [line.strip("\n").replace("\u3000", "").replace("\t", "") for line in file][3:]
                corpus += text
            print(title + ' is finished reading ! ')

    # corpus 存储语料库，每一个自然段作为一个分割
    regex_str = ".*?([^\u4E00-\u9FA5]).*?"
    english = u'[a-zA-Z0-9’!"#$%&\'()*+,-./:：;「<=>?@，。?★、…【】《》？“”‘’！[\\]^_`{|}~]+'
    symbol = []
    for j in range(len(corpus)):
        corpus[j] = re.sub(english, "", corpus[j])
        symbol += re.findall(regex_str, corpus[j])
    count_ = Counter(symbol)
    count_symbol = count_.most_common()
    noise_symbol = []
    for eve_tuple in count_symbol:
        if eve_tuple[1] < 200:
            noise_symbol.append(eve_tuple[0])
    noise_number = 0
    for line in corpus:
        for noise in noise_symbol:
            line.replace(noise, "")
            noise_number += 1
    print("完成的噪声数据替换次数：", noise_number)
    print("替换的噪声符号：")
    for i in range(len(noise_symbol)):
        print(noise_symbol[i], end=" ")
        if i % 50 == 0:
            print()
    return corpus



def write_data(items, filename):
    with open(filename, 'w', encoding='utf-8') as file:
        for word, count in items:
            new_context = f"{word},{count}\n"
            file.write(new_context)


def get_ngram_tokens(tokens, n):
    ngram_tokens = []
    for i in range(len(tokens) - n + 1):
        ngram = "".join(tokens[i:i + n])
        ngram_tokens.append(ngram)
    return ngram_tokens


def calculate_char_freq(text, cn_punctuation):
    char_freq = Counter(text)
    # 删除标点符号
    for punctuation in cn_punctuation:
        if punctuation in char_freq:
            del char_freq[punctuation]
    return char_freq


def combine2gram(cutword_list):
    if len(cutword_list) == 1:
        return []
    res = []
    for i in range(len(cutword_list)-1):
        res.append(cutword_list[i] + "s" + cutword_list[i+1])
    return res


def combine3gram(cutword_list):
    if len(cutword_list) <= 2:
        return []
    res = []
    for i in range(len(cutword_list)-2):
        res.append(cutword_list[i] + cutword_list[i+1] + "s" + cutword_list[i+2])
    return res


def plit_func(text):
    characters_list = []
    for line in text:
        characters_list.extend(list(line.strip()))
    return characters_list

if __name__ == "__main__":

    file_path = './jyxstxtqj_downcc.com/inf.txt'
    file_titles = book_titles(file_path)
    all_content = read_articles(file_titles)
    all_content_word = plit_func(all_content)

    # # 加载停用词表
    # cn_punctuation_path = './DLNLP2023-main/DLNLP2023-main/cn_punctuation.txt'
    # cn_stopwords_path = './DLNLP2023-main/DLNLP2023-main/cn_stopwords.txt'
    # cn_punctuation = load_stopwords(cn_punctuation_path)
    # cn_stopwords = load_stopwords(cn_stopwords_path)
    #
    # filtered_tokens = [word for word in all_content if word not in cn_punctuation and word not in cn_stopwords]

    # # 筛选出两个字和三个字的词语
    # two_word_tokens = [word for word in filtered_tokens if len(word) == 2]
    # three_word_tokens = [word for word in filtered_tokens if len(word) == 3]

    # 1-gram
    token = []
    for para in all_content_word: 
        token += jieba.lcut(para)
    token_num = len(token)
    ct = Counter(token)
    vocab1 = ct.most_common()
    entropy_1gram = sum([-(eve[1]/token_num)*math.log((eve[1]/token_num), 2) for eve in vocab1])
    print("词库总词数：", token_num, " ", "不同词的个数：", len(vocab1))
    print("出现频率前20的1-gram词语：", vocab1[:20])
    print("entropy_1gram:", entropy_1gram)
    write_data(vocab1[:20], 'data1.txt')

    # 2-gram
    token_2gram = []
    for para in all_content_word: #all_content
        cutword_list = jieba.lcut(para)
        token_2gram += combine2gram(cutword_list)
    # 2-gram的频率统计
    token_2gram_num = len(token_2gram)
    ct2 = Counter(token_2gram)
    vocab2 = ct2.most_common()
    # print(vocab2[:20])
    # 2-gram相同句首的频率统计
    same_1st_word = [eve.split("s")[0] for eve in token_2gram]
    assert token_2gram_num == len(same_1st_word)
    ct_1st = Counter(same_1st_word)
    vocab_1st = dict(ct_1st.most_common())
    entropy_2gram = 0
    for eve in vocab2:
        p_xy = eve[1]/token_2gram_num
        first_word = eve[0].split("s")[0]
        # p_y = eve[1]/vocab_1st[first_word]
        entropy_2gram += -p_xy*math.log(eve[1]/vocab_1st[first_word], 2)
    print("词库总词数：", token_2gram_num, " ", "不同词的个数：", len(vocab2))
    # 去除每个二元语法中的's'
    vocab2_cleaned = [(word.replace('s', ''), freq) for word, freq in vocab2[:20]]
    print("出现频率前10的2-gram词语：", vocab2_cleaned)
    write_data(vocab2_cleaned, 'data4.txt')
    print("entropy_2gram:", entropy_2gram)

    # 3-gram
    token_3gram = []
    for para in all_content_word: #all_content
        cutword_list = jieba.lcut(para)
        token_3gram += combine3gram(cutword_list)
    # 3-gram的频率统计
    token_3gram_num = len(token_3gram)
    ct3 = Counter(token_3gram)
    vocab3 = ct3.most_common()
    # print(vocab3[:20])
    # 3-gram相同句首两个词语的频率统计
    same_2st_word = [eve.split("s")[0] for eve in token_3gram]
    assert token_3gram_num == len(same_2st_word)
    ct_2st = Counter(same_2st_word)
    vocab_2st = dict(ct_2st.most_common())
    entropy_3gram = 0
    for eve in vocab3:
        p_xyz = eve[1]/token_3gram_num
        first_2word = eve[0].split("s")[0]
        entropy_3gram += -p_xyz*math.log(eve[1]/vocab_2st[first_2word], 2)
    print("词库总词数：", token_3gram_num, " ", "不同词的个数：", len(vocab3))
    vocab3_cleaned = [(word.replace('s', ''), freq) for word, freq in vocab3[:20]]
    print("出现频率前10的3-gram词语：", vocab3_cleaned)
    write_data(vocab3_cleaned, 'data5.txt')
    print("entropy_3gram:", entropy_3gram)

