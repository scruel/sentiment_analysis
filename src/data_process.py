import pandas as pd
import codecs
import operator
import json
import re

digit_regex = re.compile("(\d+)(\.)?(\d+)?$")
label_inline_regex = re.compile(r'([\[({<《【（].+?[）】》>})\]])[\r\n]', re.M)
less_inline_regex = re.compile(r'^(.{1,4})[\r\n]', re.M)
separate_symbols = set(' ,，:：;；.\t')
end_symbols = set('。!！?？~～…')
# end_symbols.add('\r\n')
# end_symbols.add('\n')
all_symbols = separate_symbols.union(end_symbols)


# 加载数据
def load_data_from_csv(file_name, header=0, encoding="utf-8"):
    data_df = pd.read_csv(file_name, header=header, encoding=encoding)
    return data_df


def get_ml_score_values(actual_label, predict_label):
    tp = sum(((predict_label == 1) & (actual_label == 1)))
    fn = sum(((predict_label == 0) & (actual_label == 1)))
    fp = sum(((predict_label == 1) & (actual_label == 0)))
    precision = 0 if (tp + fp) == 0 else tp / (tp + fp)
    recall = 0 if (tp + fn) == 0 else tp / (tp + fn)
    beta = 1
    f1_score = 0 if (precision + recall) == 0 else \
        ((1 + (beta ** 2)) * (precision * recall)) / ((beta ** 2) * precision + recall)
    return tp, fp, fn, precision, recall, f1_score


def cut_sentence(sentence):
    # words = (words).decode('utf8')
    min_sentence_len = 5
    max_sentence_len = 18
    max_separate_symbols_len = 4
    current_word_len = 0
    current_separate_symbols_len = 0
    token = False  # 上一个字符是否是符号
    start = 0
    index = 0
    result = []

    for c in sentence:
        if c in all_symbols:
            # 上一个字符是标点，直接加入到结果中，标点开头情况忽略不计
            if token:
                if index == start:
                    index += 1
                    start = index

                if len(result) and len(result[-1]):
                    result[-1] = result[-1] + c
                if index >= len(sentence) - 1:
                    start = index + 1
                continue

            flag = False
            if c in separate_symbols:
                current_separate_symbols_len += 1
                if current_word_len > max_sentence_len \
                        and current_separate_symbols_len > max_separate_symbols_len:
                    flag = True
            elif c in end_symbols:
                if current_word_len >= min_sentence_len:
                    flag = True

            if flag:
                token = True
                sub_sentence = sentence[start:index + 1]
                if current_word_len < min_sentence_len and len(result):
                    result[-1] = result[-1] + sub_sentence
                else:
                    result.append(sub_sentence)
                index += 1
                start = index
                current_word_len = 0
                current_separate_symbols_len = 0
            else:
                index += 1
        else:
            token = False
            current_word_len += 1
            index += 1
    if start < len(sentence):
        sub_sentence = sentence[start:]
        if current_word_len < min_sentence_len and len(result):
            result[-1] = result[-1] + sub_sentence
        else:
            result.append(sub_sentence)
    return result


def get_ngram_words_list(words_list, ngram_range=(1, 1), words_set=None):
    result_list = []
    for words in words_list:
        result_list.append(get_ngram_words(words, ngram_range, words_set))
    return result_list


def get_ngram_words(words, ngram_range=(1, 1), words_set=None):
    gram_list = []
    min_n, max_n = ngram_range
    for ngram in range(min_n, max_n + 1):
        for i in range(len(words) - ngram + 1):
            sub_words = words[i:i + ngram]
            flag = True
            if words_set:
                for word in sub_words:
                    if word not in words_set:
                        flag = False
                        break
            if flag:
                gram_list.append(''.join(sub_words))
    return gram_list


def extra_ngram_words_list(words_list, sorted_words_frequency_list, topk=8000, ngram_range=(1, 1)):
    useful_words_set = set(x for (x, s) in sorted_words_frequency_list[:topk])
    return get_ngram_words_list(words_list, ngram_range, words_set=useful_words_set)


# 词表制作
def get_words_frequency_map(words_list, encoding="utf-8"):
    words = [word for x in words_list for word in x]
    word_map = {}
    for word in words:
        if word in word_map:
            word_map[word] += 1
        else:
            word_map[word] = 1
    return word_map


def format_content(content):
    content = content.strip()
    if content.startswith('"'):
        content = content[1:]
    if content.endswith('"'):
        content = content[:-1]
    # 防止出现句首启发词独占一行的情况
    content = re.sub(label_inline_regex, '\\1', content)
    content = re.sub(less_inline_regex, '\\1 ', content)
    return content.lower()


def save_word_frequency_list(filename, sorted_list):
    with codecs.open(filename, 'wb', 'utf_8_sig') as f:
        for (key, value) in sorted_list:
            f.write('\"%s\",%d\n' % (key, value))


@DeprecationWarning
def save_words_list_result(filename, content_train, words_list):
    with codecs.open(filename, 'wb', 'utf_8_sig') as f:
        for i in range(len(words_list)):
            f.write(
                '%s\n%s\n\n===========Separator==============\n\n' % (content_train[i], '/ '.join(words_list[i])))


# TODO 懒加载
class ConstructDataProcessor:
    def __init__(self, stopwords_data_path, stopwords_ngram_data_path,
                 concept_mapping_words_data_path,
                 encoding='utf-8'):
        self.stop_words_file = stopwords_data_path

        with open(stopwords_data_path, 'r', encoding=encoding) as f:
            self.stopwords = set([line.strip() for line in f.readlines()])
        self.stopwords.add('\n')
        self.stopwords.add('\r\n')

        with open(stopwords_ngram_data_path, 'r', encoding=encoding) as f:
            self.stopwords_ngram = set([line.strip().lower() for line in f.readlines()])
        self.stopwords_ngram.add('\n')
        self.stopwords_ngram.add('\r\n')

        with open(concept_mapping_words_data_path, 'rb') as f:
            word_mapping_concept_dict_pre = json.load(f, encoding=encoding)

        self.mapping_concept_dict = dict()
        for k, v in word_mapping_concept_dict_pre.items():
            for word in v:
                self.mapping_concept_dict[word.lower()] = k

    def get_mapping_concept_list(self, words_list):
        for words in words_list:
            for i in range(len(words)):
                word = words[i]
                # all word filter by dict.
                if word in self.mapping_concept_dict:
                    words[i] = self.mapping_concept_dict[word]
                elif re.match(digit_regex, word):
                    words[i] = '某数字'
        return words_list

    def get_words_list_without_stopwords(self, words_list):
        words_list = [[word for word in words if word not in self.stopwords] for words in words_list]
        # words_list = [x for x in words_list if len(x)]
        return words_list

    def get_words_list_without_stopwords_ngram(self, words_list):
        words_list = [[word for word in words if word not in self.stopwords_ngram] for words in words_list]
        # words_list = [x for x in words_list if len(x)]
        return words_list
