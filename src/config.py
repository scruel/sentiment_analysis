import os

data_path = os.path.abspath('..') + "/data/"
model_save_path = data_path
new_dataset_path = data_path + "new_dataset/"
svm_dataset_path = new_dataset_path + "svm/"

svm_01_name = "svm_01_classification"
svm_3_name = "svm_3_classification"
svm_4_name = "svm_4_classification"

predict_result_path = data_path + "predict_result/"
test_data_predict_out_path = data_path + "predict_out_result/"

train_data_path = data_path + "sentiment_analysis_trainingset.csv"
validation_data_path = data_path + "sentiment_analysis_validationset.csv"
test_data_path = data_path + "sentiment_analysis_testa.csv"

user_dict_path = data_path + "userdict.txt"
stopwords_data_path = data_path + "stopwords.txt"
stopwords_ngram_data_path = data_path + "stopwords_ngram.txt"
label_words_data_path = data_path + "label_words.csv"
concept_mapping_words_data_path = data_path + "mapping_concept_words.json"
inspiration_label_words_data_path = data_path + "inspiration_label_words.csv"

data_types = ['train', 'validation', 'test']
columns = ['location_traffic_convenience', 'location_distance_from_business_district',
           'location_easy_to_find', 'service_wait_time', 'service_waiters_attitude', 'service_parking_convenience',
           'service_serving_speed', 'price_level', 'price_cost_effective', 'price_discount', 'environment_decoration',
           'environment_noise', 'environment_space', 'environment_cleaness', 'dish_portion', 'dish_taste', 'dish_look',
           'dish_recommendation', 'others_overall_experience', 'others_willing_to_consume_again']
sentiment_keys = ['universal_postitive', 'universal_neutral', 'universal_negetive']
