from data_process import *
import config
import os
import logging
import jieba

logging.basicConfig(level=logging.INFO,
                    format='%(asctime)s [%(levelname)s] <%(processName)s> (%(threadName)s) %(message)s')
logger = logging.getLogger(__name__)
data_path = {"train": config.train_data_path,
             "validation": config.validation_data_path,
             "test": config.test_data_path
             }
label_regex = re.compile(r'[\[({<《【（](.+?)[）】》>})\]]')
columns = config.columns
sentiment_keys = config.sentiment_keys
inspiration_label_dict = None
label_word_map = None


def get_label_word(word):
    global label_word_map
    if not label_word_map:
        label_word_data_df = load_data_from_csv(config.label_words_data_path)
        label_word_data_df = label_word_data_df.loc[label_word_data_df['action_cnt'] != 0]
        label_word_map = dict()
        for index, row in label_word_data_df.iterrows():
            label_word_map[row['word']] = row.to_dict()
    if word in label_word_map:
        return label_word_map[word]
    else:
        return None


# return None if not starts with inspiration word.
def get_inspiration_word_label(sentence):
    global inspiration_label_dict
    if not inspiration_label_dict:
        inspiration_label_dict = dict()
        for index, row in load_data_from_csv(config.inspiration_label_words_data_path).iterrows():
            inspiration_label_dict[row['word']] = row.to_dict()
    r = re.match(label_regex, sentence)
    if not r:
        return r
    v = r.group(1).strip()
    if v in inspiration_label_dict:
        return inspiration_label_dict[v]
    else:
        return None


def dataset_generate(data_type='test', save=True):
    logger.info("Generating new dataset: %s." % data_type)
    data_processor = ConstructDataProcessor(stopwords_data_path=config.stopwords_data_path,
                                            stopwords_ngram_data_path=config.stopwords_ngram_data_path,
                                            concept_mapping_words_data_path=config.concept_mapping_words_data_path, )
    data_df = load_data_from_csv(data_path[data_type])
    id2index = {row.id: row.Index for row in data_df.itertuples()}
    data_df['content'] = data_df['content'].map(format_content)
    # 分段
    data_df['paragraphs'] = data_df['content'].map(
        lambda content: [x.strip() for x in content.splitlines() if x.strip()])
    # 将句子汇入新表
    # content_id, paragraph_id, sentence
    sentence_df = get_sentence_df(data_df)
    # 对每个子句子进行分词
    logger.info("Cutting words...")
    jieba.load_userdict(config.user_dict_path)
    sentence_df['words'] = sentence_df['sentence'].map(
        lambda sentence: [word for word in jieba.lcut(sentence) if word.strip()])
    # 处理 ngram, 停用词等
    logger.info("Processing words...")
    sentence_df['words'] = data_processor.get_words_list_without_stopwords_ngram(sentence_df['words'])
    # 概念映射
    sentence_df['words'] = data_processor.get_mapping_concept_list(sentence_df['words'])
    # ngram
    sentence_df['words'] = get_ngram_words_list(sentence_df['words'], ngram_range=(1, 5))
    # 处理一阶词的停用词, 现在包含所有 gram 的大小的词数.
    sentence_df['words'] = data_processor.get_words_list_without_stopwords(sentence_df['words'])
    # 取出已经完成标记的部分词

    # classification_df_map[column] = pd.DataFrame(columns=['content_id', 'paragraph_id', 'sentence'])
    logger.info("Extra mark words...")
    sentence_df["mark_maps"] = sentence_df['words'].map(lambda l: [get_label_word(w) for w in l])
    sentence_df["mark_maps"] = sentence_df['mark_maps'].map(lambda l: [m for m in l if m])
    for row in sentence_df.itertuples():
        if row.inspiration_word_label:
            row.mark_maps.append(row.inspiration_word_label)
    sentence_df["mark_words"] = sentence_df['mark_maps'].map(lambda mark_maps: [m['word'] for m in mark_maps])
    sentence_df.drop(columns=['inspiration_word_label'], inplace=True)

    logger.info("Classifying...")
    classification_df_map = dict()
    for column in columns:
        # classification_df_map[column] = None
        # 过滤出当前列的情感词
        sentence_df['indicate_words_maps'] = sentence_df['mark_maps'].map(lambda mm: [m for m in mm if m[column]])
        sentence_df['indicate_words'] = sentence_df['indicate_words_maps'].map(
            lambda mark_maps: [m['word'] for m in mark_maps])
        cs_df = sentence_df[sentence_df['indicate_words'].map(len) != 0].copy()
        # 指示词情感获取
        cs_df['mark_score'] = cs_df['indicate_words_maps'].map(get_mark_score)
        cs_df['score'] = cs_df['content_id'].map(lambda index: data_df.at[id2index[index], column])
        cs_df.drop(columns=['mark_maps', 'indicate_words_maps'], inplace=True)
        classification_df_map[column] = cs_df

    # 保存
    if save:
        for column in columns:
            logger.info("Saving %s..." % column)
            save_path = config.new_dataset_path + data_type + '/'
            if not os.path.exists(save_path):
                os.makedirs(save_path)
            classification_df_map[column] \
                .to_csv(save_path + column + '.csv', index=False, sep=',',
                        line_terminator='\n',
                        encoding='utf_8_sig')

    if data_type != 'test':
        logger.info("Measuring..")
        print_all_measurements(data_df, classification_df_map)

    logger.info("*Done dataset_generate.\n")


def get_mark_score(maps):
    score_map = {k: 1 - sentiment_keys.index(k) for k in sentiment_keys}
    score_cnt_map = {k: 0 for k in sentiment_keys}
    for m in maps:
        for sentiment in sentiment_keys:
            score_cnt_map[sentiment] += m[sentiment]
    msm = max(score_cnt_map.items(), key=operator.itemgetter(1))[0]
    return score_map[msm] if score_cnt_map[msm] else -2


def get_sentence_df(df):
    sentences = []
    content_ids = []
    pids = []
    inspiration_word_labels = []
    for row in df.itertuples():
        paragraphs = row.paragraphs
        sentence_set = set()
        for pid in range(len(paragraphs)):
            paragraph = paragraphs[pid]
            # 整合判断是否需要分句
            inspiration_word_label = get_inspiration_word_label(paragraph)
            # 分句
            if inspiration_word_label is not None:
                sentence_list = [paragraph]
            else:
                sentence_list = cut_sentence(paragraph)
            # 分句去重后归并结果(去除刷评论字数等情况)
            for sentence in sentence_list:
                sentence = sentence.strip()
                if sentence and sentence not in sentence_set:
                    sentence_set.add(sentence)
                    sentences.append(sentence)
                    content_ids.append(row.id)
                    inspiration_word_labels.append(inspiration_word_label)
                    pids.append(pid)

    sentence_df = pd.DataFrame({'content_id': content_ids,
                                'paragraph_id': pids,
                                'sentence': sentences,
                                'inspiration_word_label': inspiration_word_labels})
    return sentence_df


def print_all_measurements(data_df, classification_df_map):
    id2index = {row.id: row.Index for row in data_df.itertuples()}
    predict_data_df = data_df.copy()
    actual_data_df = data_df.copy()

    # 分类后的结果, 可用于计算查准率等值
    for column in columns:
        predict_data_df[column] = 0
        classification_df = classification_df_map[column]
        for cid in classification_df['content_id']:
            predict_data_df.at[id2index[cid], column] = 1

    # comp_data_df.head()
    for column in columns:
        actual_data_df[column] = actual_data_df[column].apply(lambda x: 0 if x == -2 else 1)

    for column in columns:
        tp, fp, fn, precision, recall, f1_score = get_ml_score_values(actual_data_df[column], predict_data_df[column])
        logger.info(column)
        logger.info('TP: %d' % tp)
        logger.info('FP: %d' % fp)
        logger.info('FN: %d' % fn)
        logger.info('precision: %4f' % precision)
        logger.info('recall: %4f' % recall)
        logger.info('f1_score: %4f' % f1_score)
        logger.info('\n')
