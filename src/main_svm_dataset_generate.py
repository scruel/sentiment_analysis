from data_process import *
from gensim import corpora, models
import config
import os
import gc
import multiprocessing
import logging

logging.basicConfig(level=logging.INFO,
                    format='%(asctime)s [%(levelname)s] <%(processName)s> (%(threadName)s) %(message)s')
logger = logging.getLogger(__name__)
columns = config.columns
data_types = config.data_types


# 非法字过滤
def is_stop_char(word):
    b = ord(word)
    if b < 48:
        return True
    if 58 <= b < 127:
        return True
    return False


# 字向量生成
def generate_char_vec(df):
    cmap = dict()
    for row in df.itertuples():
        for c in row.sentence:
            if not is_stop_char(c):
                if c in cmap:
                    cmap[c] += 1
                else:
                    cmap[c] = 0

    word_vec = sorted(cmap.items(), key=operator.itemgetter(1), reverse=True)
    word_vec = word_vec[10:]
    word_vec = word_vec[: int(0.9 * len(word_vec))]
    word_vec = [k for k, v in word_vec]
    return set(word_vec)


def generate_word_vec(df):
    wmap = dict()
    for row in df.itertuples():
        for word in row.words:
            if word in wmap:
                wmap[word] += 1
            else:
                wmap[word] = 0

    word_vec = sorted(wmap.items(), key=operator.itemgetter(1), reverse=True)
    word_vec = word_vec[10:]
    word_vec = word_vec[: int(0.9 * len(word_vec))]
    word_vec = [k for k, v in word_vec]
    return set(word_vec)


def generate_words_list(df, word_vec, useWord, useChar, useSet):
    words_list = []
    for row in df.itertuples():
        # 分词结果字向量统计
        words = []
        words_set = set()
        if useWord:
            # 包含指示词加权
            for word in row.words + row.indicate_words:
                if word in word_vec and word not in words_set:
                    words.append(word)
                    if useSet:
                        words_set.add(word)
        if useChar:
            for c in row.sentence:
                if c in word_vec and c not in words_set:
                    words.append(c)
                    if useSet:
                        words_set.add(c)
            # 包含指示词加权
            for word in row.indicate_words:
                for c in word:
                    if c in word_vec and c not in words_set:
                        words.append(c)
                        if useSet:
                            words_set.add(c)

        words_list.append(words)
    return words_list


def is_right(score):
    if score == -2:
        return 0
    else:
        return 1


def svm_dataset_generate(useTfidf, useChar, useWord, useSet, useMarkScore, mkey='default', saveDict=False):
    if not (useWord | useChar):
        raise Exception("必须 word 和 char 使用其中之一")
    if useTfidf & useSet:
        raise Exception("tfidf 和 set 只可选其一")
    logger.info("Generating svm dataset...")
    for column in columns:
        generate(column, useTfidf, useChar, useWord, useSet, useMarkScore, mkey)
    logger.info("Done with useTfidf:%s, useChar:%s, useWord:%s, useSet:%s, useMarkScore:%s" % (
        useTfidf, useChar, useWord, useSet, useMarkScore))
    logger.info("*Done svm dataset generate.")


# TODO 二进制判定法
def generate(column, useTfidf, useChar, useWord, useSet, useMarkScore, mkey='default', saveDict=False):
    # if __name__ == '__main__':
    # 生成每个子模型的字向量及文本标识
    logger.info("Processing sub-class: %s" % column)
    mdname2df = dict()
    for data_type in data_types:
        tdf = load_data_from_csv(config.new_dataset_path + data_type + '/' + column + '.csv')
        tdf['words'] = tdf['words'].map(eval)
        tdf['mark_words'] = tdf['mark_words'].map(eval)
        tdf['indicate_words'] = tdf['indicate_words'].map(eval)
        mdname2df[data_type] = tdf

    # dict 生成
    all_df = pd.DataFrame(columns=mdname2df['train'].columns.values.tolist())
    for data_type in data_types:
        all_df = all_df.append(mdname2df[data_type], ignore_index=False)
    logger.info("read finish")
    word_vec = set()
    if useWord:
        word_vec = word_vec.union(generate_word_vec(all_df))
    if useChar:
        word_vec = word_vec.union(generate_char_vec(all_df))
    logger.info("generate_word_vec finish")
    words_list = generate_words_list(all_df, word_vec, useWord, useChar, useSet)
    logger.info("generate_words_list finish")
    dictionary = corpora.Dictionary(words_list)
    if saveDict:
        save_path = config.svm_dataset_path + mkey + "/dict/"
        if not os.path.exists(save_path):
            os.makedirs(save_path)
        dict_path = save_path + column + '.dict'
        dictionary.save_as_text(dict_path)
    logger.info("corpora dictionary finish")

    # 语料库生成
    for data_type in data_types:
        logger.info("Processing dataset: %s" % data_type)
        data_df = mdname2df[data_type]
        words_list = generate_words_list(data_df, word_vec, useWord, useChar, useSet)
        logger.info("generate_words_list finish")
        all_corpus = [dictionary.doc2bow(s) for s in words_list]
        dict_len = len(dictionary)
        if useTfidf:
            tfidf = models.TfidfModel(all_corpus)
            data_df['corpus'] = tfidf[all_corpus]
        else:
            data_df['corpus'] = all_corpus
        logger.info("doc2bow finish")
        # df['sentiments'] = [SnowNLP(x).sentiments for x in df['sentence']]
        dataset_01_list = []
        dataset_3_list = []
        dataset_4_list = []
        for row in data_df.itertuples():
            label = row.score if row.score else 0
            corpus = row.corpus[:]
            indicate_words_len = len(row.indicate_words)
            corpus.append((dict_len, indicate_words_len))
            if useMarkScore:
                mark_score = row.mark_score
                corpus.append((dict_len + 1, mark_score == 1))
                corpus.append((dict_len + 2, mark_score == -1))
            feature_str = ' '.join(['%d:%f' % (k + 1, v) for k, v in corpus])
            dataset_01_list.append('%s %s' % (is_right(label), feature_str))
            if label != -2:
                dataset_3_list.append('%s %s' % (label, feature_str))
            dataset_4_list.append('%s %s' % (label, feature_str))
        logger.info("itertuples finish")
        save_path = config.svm_dataset_path + mkey + "/" + config.svm_01_name + "/" + data_type + '/'
        if not os.path.exists(save_path):
            os.makedirs(save_path)
        with open(save_path + column, 'w',
                  encoding='utf-8', newline='\n') as f:
            f.write('\n'.join(dataset_01_list))

        save_path = config.svm_dataset_path + mkey + "/" + config.svm_3_name + "/" + data_type + '/'
        if not os.path.exists(save_path):
            os.makedirs(save_path)
        with open(save_path + column, 'w',
                  encoding='utf-8', newline='\n') as f:
            f.write('\n'.join(dataset_3_list))

        save_path = config.svm_dataset_path + mkey + "/" + config.svm_4_name + "/" + data_type + '/'
        if not os.path.exists(save_path):
            os.makedirs(save_path)
        with open(save_path + '/' + column, 'w',
                  encoding='utf-8', newline='\n') as f:
            f.write('\n'.join(dataset_4_list))
        gc.collect()
        logger.info("Done save")
    del mdname2df
    gc.collect()
    logger.info("Done process sub-class: %s" % column)
