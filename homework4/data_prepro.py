import csv
import math
import os
import re
from collections import Counter, OrderedDict
import jieba
import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import linregress
import random

def extract_words_update():
    # 将标点添加到停词表中
    stop_words = []
    with open('tools/cn_punctuation.txt', "r", encoding='gbk', errors='ignore') as punc_words_file:
        for line in punc_words_file:
            line = line.strip()   # 去掉每行末尾的换行符
            stop_words.append(line)

    # 将停词添加到停词表中
    with open('tools/cn_stopwords.txt', "r", encoding='gbk', errors='ignore') as stop_words_file:
        for line in stop_words_file:
            line = line.strip()   # 去掉每行末尾的换行符
            stop_words.append(line)

    # 将人名添加到词汇表中
    people_names = []
    with open('tools/金庸小说全人物.txt', "r", encoding='gbk', errors='ignore') as people_names_file:
        for line in people_names_file:
            line = line.strip()  # 去掉每行末尾的换行符
            jieba.add_word(line)
            people_names.append(line)

    # 将武功添加到词汇表中
    kungfu_names = []
    with open('tools/金庸小说全武功.txt', "r", encoding='gbk', errors='ignore') as people_kungfus_file:
        for line in people_kungfus_file:
            line = line.strip()  # 去掉每行末尾的换行符
            jieba.add_word(line)
            kungfu_names.append(line)

    # 将门派添加到词汇表中
    gang_names = []
    with open('tools/金庸小说全门派.txt', "r", encoding='gbk', errors='ignore') as people_gangs_file:
        for line in people_gangs_file:
            line = line.strip()  # 去掉每行末尾的换行符
            jieba.add_word(line)
            gang_names.append(line)

    # 读取小说名
    with open('data/inf-1.txt', "r", encoding='UTF-8', errors='ignore') as novel_names_file:
        novel_names = novel_names_file.read().split(',')

    paragraphs = []
    lines = []
    seg_novel = []
    novel_num = 0

    for novel_name in novel_names:
        with open('data/' + novel_name + '.txt', "r", encoding='gbk', errors='ignore') as novel:
            name_list = []
            para = []
            print("正在加载 {}...".format(novel_name))
            line = novel.readline()

            while line:
                line = line.replace('本书来自www.cr173.com免费txt小说下载站', '')
                line = line.replace('更多更新免费电子书请关注www.cr173.com', '')
                line = re.sub(r'[^\u4e00-\u9fa5]', '', line)
                line_1 = line.strip()
                seg_line = jieba.lcut(line_1.strip())  # 使用jieba进行分词，并去除首尾空白字符
                for word in seg_line:
                    if (word not in stop_words) and (not word.isspace()):  # 不是停词不是空格
                        para.append(word)
                if len(para) > 10:
                    lines.append(para)

                output = ''
                line_seg = jieba.cut(line_1, cut_all=False)
                for word in line_seg:
                    if (word not in stop_words) and (not word.isspace()):  # 不是停词不是空格
                        if (word in people_names) and (word not in name_list):
                            name_list.append(word)
                        output += word
                        output += " "
                if len(str(output.strip())) != 0:
                    seg_novel.append(str(output.strip()).split())
                line = novel.readline()

            # 从小说文本中��机选择两个段落
            if len(lines) >= 2:
                random_paragraphs = random.sample(lines, 2)
                paragraphs.extend(random_paragraphs)


        print("《{}》 加载完成".format(novel_name))
        novel_num += 1

    if novel_num == len(novel_names):
        print("-" * 40)
        print("全部小说加载完成！")

    # 保存路径
    save_path = './data白马啸西风.csv'
    with open(save_path, 'w', newline='', encoding='utf-8') as fp:
        csv_writer = csv.writer(fp)
        # 遍历seg_novel中的每个分词列表
        for seg_list in seg_novel:
            # 将分词列表合并成一个句子
            sentence = ''.join(seg_list)+ '。'
            # 将句子写入CSV文件
            csv_writer.writerow([sentence])
        for word in seg_novel:
            csv_writer.writerow([word])

if __name__ == "__main__":
    extract_words_update()
