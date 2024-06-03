import csv
import os
import random
import re

import jieba
import numpy as np
import pandas as pd
import chardet
# import tensorflow as tf
# import torch
from gensim.models import Word2Vec as W2V
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
# 指定字体路径，对于中文通常使用 'SimHei' 字体
plt.rcParams['font.sans-serif'] = ['WenQuanYi Zen Hei'] 
plt.rcParams['axes.unicode_minus'] = False  # 正确显示负号



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

    if flag == 2:
        
        novels = ['白马啸西风','碧血剑','飞狐外传','连城诀','鹿鼎记','三十三剑客图','射雕英雄传','神雕侠侣','书剑恩仇录',
                  '天龙八部','侠客行','笑傲江湖','雪山飞狐','倚天屠龙记','鸳鸯刀','越女剑']
            
       # 使用新的函数获取颜色映射
        # 定义颜色映射
        cmap = plt.cm.hsv  # 获取颜色映射对象
        colors = [cmap(i / len(novels)) for i in range(len(novels))]  # 生成颜色列表

        # 初始化存储人名词向量的列表
        person_vectors = []

        # 读取每个人名文件并提取词向量
        for i, novel in enumerate(novels):
            filename = os.path.join(f'{novel}人名.txt')  # 假设人名文件在data目录下
            novel_vectors = []
            with open(filename, 'r', encoding='utf-8') as f:  # 确保使用正确的文件编码
                for line in f:
                    person = line.strip()
                    if person in model.wv:
                        novel_vectors.append(model.wv[person])
            person_vectors.append(np.array(novel_vectors))  # 将当前小说的人名词向量添加到列表

        # 将所有人名词向量合并为一个数组
        all_person_vectors = np.vstack(person_vectors)

        # 使用PCA降维
        pca = PCA(n_components=2)
        person_vectors_pca = pca.fit_transform(all_person_vectors)

        # 执行聚类
        kmeans = KMeans(n_clusters=len(novels), n_init='auto')
        clusters = kmeans.fit_predict(person_vectors_pca)

        # # 可视化聚类结果
        # plt.figure(figsize=(12, 10))
        # for i, novel in enumerate(novels):
        #     plt.scatter(person_vectors_pca[clusters == i, 0], person_vectors_pca[clusters == i, 1],
        #                 color=colors[i], label=novel)  # 为每个小说的人名添加图例标签
        plt.figure(figsize=(12, 10))
        for i, novel in enumerate(novels):
            # 初始化一个空列表，用于存储当前小说的所有人名词向量索引
            novel_person_indices = []
            for j, person in enumerate(model.wv.index_to_key):
                # 检查每个人名是否在模型的词汇表中，并且是否属于当前小说
                if person in model.wv and novel in person:
                    novel_person_indices.append(j)
                    # 获取该人名的词向量
                    word_vector = model.wv[person]
                    # 降维后的坐标
                    x, y = pca.transform(word_vector.reshape(1, -1)).flatten()
                    # 绘制散点图并添加文本标签
                    plt.scatter(x, y, color=colors[i], label=novel)
                    plt.text(x, y, person, color=colors[i], fontsize=8, ha='right')  # ha='right' 表示水平对齐到文本的右侧

        # 显示图例
        plt.legend()

        plt.title('Person Name Clustering by Novel')
        plt.xlabel('PCA Component 1')
        plt.ylabel('PCA Component 2')
        # plt.legend(loc='best')  # 显示图例
        plt.grid(True)

        # 保存图像
        plt.savefig('person_name_clustering1.png', format='png', dpi=300)

        # 关闭图形窗口
        plt.close()
       
if __name__ == '__main__':
    # data_path = './data_up-3.csv'
    # model_path = train_W2V_model(data_path)
    # print(model_path)
    load_model('W2V-up-3.model', 2)  #model_path  #'W2V.model'