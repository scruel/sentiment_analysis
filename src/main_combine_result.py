from data_process import *
import config
import logging
import os

logging.basicConfig(level=logging.INFO,
                    format='%(asctime)s [%(levelname)s] <%(processName)s> (%(threadName)s) %(message)s')
logger = logging.getLogger(__name__)
columns = config.columns


def combine_result(class_name, mkey='default'):
    test_data_df = load_data_from_csv(config.test_data_path)

    id2index_map = dict()
    for row in test_data_df.itertuples():
        id2index_map[row.id] = row.Index
    for column in columns:
        logger.info(column)
        ylable_df = load_data_from_csv(config.predict_result_path + mkey + "/" + class_name + "/" + column + '.csv')
        sentence_df = load_data_from_csv(config.new_dataset_path + 'test/' + column + '.csv')
        sentence_df['score'] = ylable_df['y']
        sentence_df['indicate_words'] = sentence_df['indicate_words'].map(eval)
        test_data_df[column] = -2
        test_data_df['content'] = None

        # 加权计算
        sentiment_map = {-2: 0, -1: 0, 0: 0, 1: 0}
        max_sent_len = 3
        curr_sent_len = 0
        last_pid = -1
        last_cid = -1
        last_weight = 0
        # rslist = []
        for row in sentence_df.itertuples():
            if last_cid != row.content_id:
                last_cid = row.content_id
                curr_sent_len = 0
                last_weight = 0
                sentiment_map = {-2: 0, -1: 0, 0: 0, 1: 0}
            if last_pid != row.paragraph_id:
                last_pid = row.paragraph_id
                curr_sent_len = 0

            weight = sum([len(x) for x in row.indicate_words]) * (curr_sent_len + 1)
            sentiment_map[row.score] += weight
            index = id2index_map[row.content_id]
            if sentiment_map[row.score] > last_weight:
                last_weight = sentiment_map[row.score]
                test_data_df.at[index, column] = row.score
            # rslist.append(test_data_df.at[index, column])
            curr_sent_len = (curr_sent_len + 1) % max_sent_len

    save_path = config.test_data_predict_out_path
    if not os.path.exists(save_path):
        os.makedirs(save_path)
    test_data_df.to_csv(save_path + mkey + class_name + '_sentiment_analysis_test_predict_out.csv',
                        encoding="utf_8_sig", index=False, line_terminator='\n')
    logger.info("Done combine result.")
