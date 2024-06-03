import csv
import numpy as np

from gensim.models import Word2Vec as W2V

def save_files(csv_file_path,dict_of_novels):
    # 打开文件并写入数据
    with open(csv_file_path, 'w', newline='', encoding='utf-8') as csvfile:
        # 创建一个写入器对象
        csv_writer = csv.writer(csvfile)
        # 写入标题行（如果有标题行的话）
        csv_writer.writerow(['novel_name', 'novel_data'])
        # 遍历字典并写入每一行
        for novel_name, data in dict_of_novels.items():
            csv_writer.writerow([novel_name, ' '.join(data)])

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

    model = W2V(sentences=text_data, hs=1, min_count=10, window=5, vector_size=200, sg=0, epochs=200, workers=5)#设置模型参数
    model_path = "W2V-up-3-flag3.model"
    model.save(model_path)
    # sentences: 分词后的文本数据列表。
    # vector_size: 生成的词向量的维度。
    # window: 考虑的单词上下文大小。
    # min_count: 忽略频率低于此值的单词。
    # workers: 用于训练的线程数。

    #sentences：可以是一个list，对于大语料集，建议使用BrownCorpus,Text8Corpus或LineSentence构建；
    #hs: 如果为1则会采用hierarchical softmax技巧。如果设置为0（defaut），则negative sampling会被使用；
    #min_count: 可以对字典做截断，词频少于min_count次数的单词会被丢弃掉, 默认值为5，这里的值为10。
    #window：表示当前词与预测词在一个句子中的最大距离是多少，这里为5；
    #vector_size：是指特征向量的维度，默认为100；
    #sg： 用于设置训练算法，默认为0，对应CBOW算法；sg=1则采用skip-gram算法；
    #epochs：迭代次数，默认为5。这里的参数不是iter，官方在新的版本中将这个关键词改为了epochs，使用iter会报错。

    print('finish traing W2V Model!')
    return model_path

def top_k_sampling(similarity_list, k=10, temperature=1.0):
    # 将相似度列表按相似度排序
    similarity_list = sorted(similarity_list, key=lambda x: x[1], reverse=True)

    # 取前k个词
    top_k_words = similarity_list[:k]

    # 提取词和相似度
    words, similarities = zip(*top_k_words)

    # 调整相似度以控制生成的随机性
    similarities = np.array(similarities) / temperature

    # 计算概率分布
    exp_similarities = np.exp(similarities)
    probabilities = exp_similarities / np.sum(exp_similarities)

    # 根据概率分布选择下一个词
    next_word = np.random.choice(words, p=probabilities)

    return next_word

def avoid_repetition(generated_text, next_word, n=2):
    # 检查是否生成了重复的n-gram
    if len(generated_text) >= n:
        n_gram = generated_text[-(n-1):] + [next_word]
        if n_gram in [generated_text[i:i+n] for i in range(len(generated_text)-n+1)]:
            return True
    # 检查是否生成了重复的单词
    if next_word in generated_text:
        return True
    return False

def filter_unrelated_words(model,similarity_list, context_vectors, threshold=0.7):
    filtered_list = []
    for word, similarity in similarity_list:
        word_vector = model.wv[word]
        max_similarity = max(np.dot(word_vector, context_vector) / (np.linalg.norm(word_vector) * np.linalg.norm(context_vector)) for context_vector in context_vectors)
        if max_similarity >= threshold:
            filtered_list.append((word, similarity))
    return filtered_list

def load_model(model_path,flag):
    model = W2V.load(model_path)

    if flag == 3:
        # context = ['男孩', '没想到', '居然', '会肯', '接过', '玉镯']
        context = ['男孩', '不去理', '脸上', '正气凛然', '指着', '苗夫人', '没良心', '田归农', '提起', '长剑', '正要', '分心', '刺去',
                   '苗夫人', '一声', '掩面', '哭', '大雨', '中', '直奔', '田归农', '顾不得', '杀', '男孩', '提剑', '追出', '一窜',
                   '一跃', '追到', '苗夫人', '身旁', '劝道', '兰妹', '叫化', '胡说八道', '别理', '苗夫人']
        # context = ['张无忌', '没想到','今天','见到']
        generated_text = context.copy()

        # 预测并添加新的词
        for _ in range(10):  # 假设我们想生成5个新词
            # 将上下文中的词转换为向量
            context_words = [word for word in context if word in model.wv]
            context_vectors = [model.wv[word] for word in context_words]

            if context_vectors:
                # 计算上下文向量的平均值
                avg_vector = np.mean(context_vectors, axis=0)

                # 初始化一个列表来存储每个词的相似度
                similarity_list = []

                # 遍历词汇表中的所有词，计算与平均向量的相似度
                for word in model.wv.index_to_key:
                    vector = model.wv[word]
                    similarity = np.dot(avg_vector, vector) / (np.linalg.norm(avg_vector) * np.linalg.norm(vector))
                    similarity_list.append((word, similarity))

                # 过滤掉与上下文语义不相关的词
                similarity_list = filter_unrelated_words(model,similarity_list, context_vectors, threshold=0.7)

                # 使用Top-k和温度采样选择下一个词
                next_word = top_k_sampling(similarity_list, k=10, temperature=0.7)

                # 避免重复n-gram和单词
                while avoid_repetition(generated_text, next_word, n=2):
                    next_word = top_k_sampling(similarity_list, k=10, temperature=0.7)

                generated_text.append(next_word)
                context.append(next_word)
            else:
                break

        # 输出生成的文本
        print(' '.join(generated_text))

if __name__ == '__main__':
    # data_path = './data_up-3.csv'
    # model_path = train_W2V_model(data_path)
    # print(model_path)
    load_model('W2V-up-3.model', 3)  #model_path  #'W2V.model'