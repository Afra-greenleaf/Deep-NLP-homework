import csv
import os
import random
import re

import jieba
import numpy as np
import pandas as pd
# import tensorflow as tf
# import torch
from gensim.models import Word2Vec as W2V


def save_files(csv_file_path, dict_of_novels):
    # 打开文件并写入数据
    with open(csv_file_path, 'w', newline='', encoding='utf-8') as csvfile:
        # 创建一个写入器对象
        csv_writer = csv.writer(csvfile)
        # 写入标题行（如果有标题行的话）
        csv_writer.writerow(['novel_name', 'novel_data'])
        # 遍历字典并写入每一行
        for novel_name, data in dict_of_novels.items():
            csv_writer.writerow([novel_name, ' '.join(data)])


def read_file(filename):  # 用于读取小说文本
    # 如果未指定名称，则默认为类名
    target = "data/" + filename + ".txt"
    with open(target, "r", encoding='gbk', errors='ignore') as f:
        data = f.read()
        # 去除特定文本
        ad = ['本书来自www.cr173.com免费txt小说下载站\n更多更新免费电子书请关注www.cr173.com', '----〖新语丝电子文库(www.xys.org)〗', '新语丝电子文库',
              '\u3000', '\n', '。', '？', '！', '，', '；', '：', '、', '《', '》', '“', '”', '‘', '’', '［', '］', '....', '......',
              '『', '』', '（', '）', '…', '「', '」', '\ue41b', '＜', '＞', '+', '\x1a', '\ue42b', '她', '他', '你', '我', '它', '这']  # 去掉其中的一些无意义的词语
        for a in ad:
            data = data.replace(a, '')

    # 使用正则表达式匹配中文字符
    chinese_data = re.sub(r'[^\u4e00-\u9fff]', '', data)

    # 读取停用词
    with open("tools/stop_words.txt", "r", encoding='utf-8') as fp:  # cn_stopwords.txt
        stop_word = set(fp.read().split('\n'))
    print('finish load stop_words ！')

    # 分词并过滤停用词
    split_word = []
    for word in jieba.cut(chinese_data):
        if word not in stop_word and not word.isspace():
            split_word.append(word)

    return split_word


# 定义文档向量化函数
def document_vector(words, model):
    # 将分词后的文本转换为向量，这里只是一个示例
    word_vectors = [model.wv.get_vector(word) for word in words if word in model.wv]
    if word_vectors:
        return np.mean(word_vectors, axis=0)
    else:
        return np.zeros(model.vector_size)


def train_W2V_model(data_path):
    with open(data_path, 'r', encoding='utf-8') as file:
        text_data = file.readlines()

    # Tokenize the text data
    tokenized_data = [list(jieba.cut(line)) for line in text_data]

    model = W2V(sentences=tokenized_data, vector_size=200, window=5, min_count=5, sg=0)  # 设置模型参数
    model_path = "W2V-up-3.model"
    model.save(model_path)

    print('finish training W2V Model!')
    return model_path


def load_model(model_path, flag):
    model = W2V.load(model_path)

    if flag == 1:
        test_name = ['张无忌', '乔峰', '郭靖', '杨过', '令狐冲', '韦小宝']
        test_name1 = ['张无忌', '赵敏','周芷若','段誉','黄蓉','小龙女','岳不群','东方不败','鳌拜']
        # test_name1 = ['段誉','韦春花']
        #test_menpai = ['明教', '逍遥派', '少林', '全真教', '华山派', '少林']

        # for i in range(0, len(test_name)):
        #     for j in range(0,len(test_name1)):

        #         word1 = test_name[i]
        #         word2 = test_name1[j]

        #         if word1 in model.wv and word2 in model.wv:
        #             # 计算两个词之间的相似度
        #             similarity_score = model.wv.similarity(word1, word2)
        #             print(f'The similarity between "{word1}" and "{word2}" is: {similarity_score}')

        #             # 计算欧式距离
        #             # 获取两个词的词向量
        #             vector1 = model.wv[word1]
        #             vector2 = model.wv[word2]

        #             # 计算两个词向量之间的欧几里得距离
        #             euclidean_distance = np.linalg.norm(vector1 - vector2)
        #             print(f'The Euclidean distance between "{word1}" and "{word2}" is: {euclidean_distance}')

        #             # 计算余弦相似度
        #             # 计算两个词向量之间的余弦相似度
        #             cosine_similarity = np.dot(vector1, vector2) / (np.linalg.norm(vector1) * np.linalg.norm(vector2))
        #             print(f'The cosine similarity between "{word1}" and "{word2}" is: {cosine_similarity}')
        #         else:
        #             print(f'One or both words "{word1}" and "{word2}" are not in the vocabulary.')
        # 初始化三个CSV文件
        with open('similarity_scores.csv', 'w', newline='') as sim_file, \
            open('euclidean_distances.csv', 'w', newline='') as euclidean_file, \
            open('cosine_similarities.csv', 'w', newline='') as cosine_file:

            # 创建CSV写入器
            sim_writer = csv.writer(sim_file)
            euclidean_writer = csv.writer(euclidean_file)
            cosine_writer = csv.writer(cosine_file)

            # 写入表头
            sim_writer.writerow(['Word1', 'Word2', 'Similarity Score'])
            euclidean_writer.writerow(['Word1', 'Word2', 'Euclidean Distance'])
            cosine_writer.writerow(['Word1', 'Word2', 'Cosine Similarity'])

            # 初始化一个6x6的表格
            similarity_matrix = np.zeros((len(test_name), len(test_name1)))
            euclidean_matrix = np.zeros((len(test_name), len(test_name1)))
            cosine_matrix = np.zeros((len(test_name), len(test_name1)))

            # 遍历两个词的列表
            for i in range(len(test_name)):
                for j in range(len(test_name1)):
                    word1 = test_name[i]
                    word2 = test_name1[j]

                    # 确保两个词都在词嵌入模型的词汇表中
                    if word1 in model.wv and word2 in model.wv:
                        # 计算两个词之间的相似度
                        similarity_score = model.wv.similarity(word1, word2)

                        # 计算欧式距离
                        vector1 = model.wv[word1]
                        vector2 = model.wv[word2]
                        euclidean_distance = np.linalg.norm(vector1 - vector2)

                        # 计算余弦相似度
                        cosine_similarity = np.dot(vector1, vector2) / (np.linalg.norm(vector1) * np.linalg.norm(vector2))

                        # 写入CSV文件
                        sim_writer.writerow([word1, test_name1[j], similarity_score])
                        euclidean_writer.writerow([word1, test_name1[j], euclidean_distance])
                        cosine_writer.writerow([word1, test_name1[j], cosine_similarity])

                        # 填充6x6的表格
                        similarity_matrix[i, j] = similarity_score
                        euclidean_matrix[i, j] = euclidean_distance
                        cosine_matrix[i, j] = cosine_similarity
                    else:
                        print(f'One or both words "{word1}" and "{word2}" are not in the vocabulary.')

        # 可选：打印6x6的表格
        print("Similarity Matrix:\n", similarity_matrix)
        print("Euclidean Distance Matrix:\n", euclidean_matrix)
        print("Cosine Similarity Matrix:\n", cosine_matrix)


if __name__ == '__main__':
    data_path = './data_up-3.csv'
    model_path = train_W2V_model(data_path)
    print(model_path)
    load_model(model_path, 1)
