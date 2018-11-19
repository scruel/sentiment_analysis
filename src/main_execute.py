from main_dataset_generate import dataset_generate
from main_svm_dataset_generate import svm_dataset_generate
from main_liblinear import svm_predict_test
from main_combine_result import combine_result
import config

if __name__ == '__main__':
    save = True
    local = False
    label_dir = 'test'
    print(label_dir)
    dataset_generate(data_type='validation', save=save)
    dataset_generate(data_type='test', save=save)
    dataset_generate(data_type='train', save=save)
    svm_dataset_generate(mkey=label_dir, useTfidf=False, useChar=True, useWord=True, useSet=False, useMarkScore=True)
    svm_predict_test(mkey=label_dir, local=local)
    # 生成最后的结果
    combine_result(config.svm_3_name, mkey=label_dir)
    combine_result(config.svm_4_name, mkey=label_dir)
    print("All done!")
