import os
from collections import Counter

import chardet
import jieba
import numpy as np
import math
from scipy.stats import linregress
import matplotlib.pyplot as plt


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


# 加载停用词表
def load_stopwords(file_path):
    with open(file_path, 'r', encoding='utf-8') as f:
        stopwords = [line.strip() for line in f]
    return stopwords


def read_articles(file_titles):
    all_content = ""
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
                content = file.read().replace('\n', '').replace('\u3000', '').replace(' ', '').replace('=', '')  # 移除换行符、全角空格、空格和等号
                all_content += content + " "  # 拼接所有文档内容
            print(title + ' is finished reading ! ')
        else:
            print(f"File {title} not found.")
    return all_content


def write_data(items):
    with open('./data.txt', 'w', encoding='utf-8') as file:
        for word, count in items:
            # print("{0:<10}{1:>5}".format(word, count))
            new_context = f"{word},{count}\n"
            file.write(new_context)
    file.close()


def linregress_draw(x, y):

    log_rank = np.log(x)
    log_frequency = np.log(y)
    # 使用线性回归进行拟合
    slope, intercept, r_value, p_value, std_err = linregress(log_rank, log_frequency)

    # 创建拟合曲线
    x_values = np.linspace(min(log_rank), max(log_rank), len(y))
    y_values = slope * x_values + intercept
    plt.plot(np.exp(x_values), np.exp(y_values), color='red', label='Fitted Line')

    # 绘制原始数据
    plt.plot(np.exp(log_rank), np.exp(log_frequency), label='Original Data')


if __name__ == "__main__":

    file_path = './jyxstxtqj_downcc.com/inf.txt'
    file_titles = book_titles(file_path)
    all_content = read_articles(file_titles)

    tokens = jieba.lcut(all_content)  # 分词
    print('word cut is finished!')
    # 加载停用词表
    cn_punctuation_path = './DLNLP2023-main/DLNLP2023-main/cn_punctuation.txt'
    cn_stopwords_path = './DLNLP2023-main/DLNLP2023-main/cn_stopwords.txt'
    cn_punctuation = load_stopwords(cn_punctuation_path)
    cn_stopwords = load_stopwords(cn_stopwords_path)

    filtered_tokens = [word for word in tokens if word not in cn_punctuation and word not in cn_stopwords]
    print('delete stop word is finished!')
    word_freq = Counter(filtered_tokens)  # 统计词频

    # 按照词频从高到低排序
    sorted_word_freq = sorted(word_freq.items(), key=lambda x: x[1], reverse=True)
    write_data(sorted_word_freq)


    # 提取词频和排名
    frequencies = [item[1] for item in sorted_word_freq]
    ranks = list(range(1, len(frequencies) + 1))
    plt.figure()
    linregress_draw(ranks, frequencies)

    # 绘制图表
    # plt.figure(figsize=(10, 6))
    # plt.plot(ranks, frequencies, marker='o')
    plt.xscale('log')  # 使用对数坐标
    plt.yscale('log')  # 使用对数坐标
    plt.xlabel('Rank')
    plt.ylabel('Frequency')
    plt.title('Zipf\'s Law Verification for Chinese Text')
    plt.grid(True)
    plt.savefig('pic')

    plt.figure()
    # 绘制图表
    plt.figure(figsize=(10, 6))
    plt.plot(ranks, frequencies, marker='o')
    plt.xlabel('Rank')
    plt.ylabel('Frequency')
    plt.title('Zipf\'s Law Verification for Chinese Text')
    plt.grid(True)
    plt.savefig('pic1')


