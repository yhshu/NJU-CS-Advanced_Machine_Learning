import os

import numpy as np
import pandas as pd
import torch
from datasets import load_metric
from pandas import DataFrame
from sklearn import preprocessing
from sklearn.metrics import accuracy_score
from torch.utils.data import TensorDataset
from transformers import AutoModelForSequenceClassification, AutoConfig, Trainer, TrainingArguments, AutoTokenizer

from data_utils import ClothingDataset, subset_to_dataset


def dataframe_preprocess(df: pd.DataFrame):
    return df[
        ['age', 'body type', 'bust size', 'category', 'height', 'item_id', 'rating', 'rented for', 'size', 'user_id',
         'weight']].fillna(method='pad')


def flat_accuracy(preds, labels):
    """
    A function for calculating accuracy scores
    :param preds: predictions
    :param labels: golden labels
    :return: accuracy
    """
    pred_flat = np.argmax(preds, axis=1).flatten()
    labels_flat = labels.flatten()
    return accuracy_score(labels_flat, pred_flat)


def get_text_list(dataframe: DataFrame):
    res = []
    for i in range(0, len(dataframe.values)):
        res.append(dataframe.values[i][0] + ' ' + dataframe.values[i][1])
    return res


def get_label_num(label_literal):
    res = []
    for l in label_literal:
        if l == 'small':
            res.append(0)
        elif l == 'fit':
            res.append(1)
        elif l == 'large':
            res.append(2)
    return res


def get_text_features():
    train_dataset_filepath = './train_dataset'  # split from original training dataset
    dev_dataset_filepath = './dev_dataset'  # split from original training dataset
    test_dataset_filepath = './test_dataset'

    model_name = 'albert-base-v2'  # 'albert-bert-v2', 'roberta-base'
    tokenizer = AutoTokenizer.from_pretrained(model_name)

    if os.path.isfile(train_dataset_filepath) and os.path.isfile(dev_dataset_filepath):  # load from file
        train_dataset = torch.load(train_dataset_filepath)
        dev_dataset = torch.load(dev_dataset_filepath)
    else:  # generate dataloader and save it to file
        train_text_input = train_data[['review_summary', 'review_text']].fillna(method='bfill')
        train_text_input = get_text_list(train_text_input)  # list of str

        train_encodings = tokenizer(train_text_input, truncation='only_first', max_length=128, padding='max_length',
                                    return_tensors='pt')
        train_labels = torch.tensor(get_label_num(train_data['fit'].values))  # list of tensor

        train_dataset = ClothingDataset(train_encodings, train_labels)
        train_size = int(len(train_dataset) * 0.9)
        train_dataset, dev_dataset = torch.utils.data.random_split(train_dataset,
                                                                   [int(train_size),
                                                                    int(len(train_dataset)) - train_size],
                                                                   generator=torch.Generator().manual_seed(29))
        train_dataset = subset_to_dataset(train_dataset)
        dev_dataset = subset_to_dataset(dev_dataset)
        torch.save(train_dataset, train_dataset_filepath)
        torch.save(dev_dataset, dev_dataset_filepath)

    if os.path.isfile(test_dataset_filepath):  # load from file
        test_dataset = torch.load(test_dataset_filepath)
    else:  # generate dataloader and save it to file
        test_text_input = test_data[['review_summary', 'review_text']].fillna(method='bfill')
        test_text_input = get_text_list(test_text_input)

        test_encodings = tokenizer(test_text_input, truncation='only_first', max_length=128, padding='max_length',
                                   return_tensors='pt')
        test_dataset = ClothingDataset(test_encodings, [0 for i in range(len(test_encodings.encodings))])
        torch.save(test_dataset, test_dataset_filepath)

    config = AutoConfig.from_pretrained(model_name, num_labels=3, return_dict=False, max_length=128)
    model = AutoModelForSequenceClassification.from_config(config)
    model.cuda()

    training_args = TrainingArguments('sequence_classification',
                                      do_train=True, do_eval=True, do_predict=True,
                                      # evaluation_strategy='steps', eval_steps=100,
                                      evaluation_strategy='epoch',
                                      per_device_train_batch_size=32, per_device_eval_batch_size=8,
                                      learning_rate=5e-5,
                                      no_cuda=False)
    metric = load_metric("accuracy")

    def compute_metrics(eval_pred):
        logits, labels = eval_pred
        predictions = np.argmax(logits, axis=-1)
        return metric.compute(predictions=predictions, references=labels)

    trainer = Trainer(model, args=training_args,
                      train_dataset=train_dataset, eval_dataset=dev_dataset,
                      compute_metrics=compute_metrics)
    trainer.train()


def get_non_text_features():
    # fill N/A values
    train_input = train_data[
        ['age', 'body type', 'bust size', 'category', 'height', 'item_id', 'rating', 'rented for', 'size', 'user_id',
         'weight', 'review_summary', 'review_text']].fillna(method='bfill')
    train_input.info()
    train_output = train_data['fit']
    test_input = test_data[
        ['age', 'body type', 'bust size', 'category', 'height', 'item_id', 'rating', 'rented for', 'size', 'user_id',
         'weight', 'review_summary', 'review_text']].fillna(method='bfill')
    test_input.info()

    # column 'height'
    def get_height_value(col):
        col_split = col.split('\'')
        for i in range(0, len(col_split)):
            col_split[i] = col_split[i].strip('"').strip(' ')
        return 12 * int(col_split[0]) + int(col_split[1])

    train_input['height'] = train_input['height'].apply(get_height_value)
    test_input['height'] = test_input['height'].apply(get_height_value)

    # column 'weight'
    def get_weight_value(col):
        return col.replace('lbs', '').strip(' ')

    train_input['weight'] = train_input['weight'].apply(get_weight_value)
    test_input['weight'] = test_input['weight'].apply(get_weight_value)

    # keywords of column 'review_summary', 'review_text'
    def get_review_keywords(col):
        col = col.lower()
        if 'tight' in col or 'small' in col or 'short' in col or 'snug' in col:
            return 'small'
        if 'big' in col or 'hide' in col or 'large' in col or 'loose' in col:
            return 'large'
        return 'unk'  # unknown

    train_input['review_summary'] = train_input['review_summary'].apply(get_review_keywords)
    test_input['review_summary'] = test_input['review_summary'].apply(get_review_keywords)
    train_input['review_text'] = train_input['review_text'].apply(get_review_keywords)
    test_input['review_text'] = test_input['review_text'].apply(get_review_keywords)

    def get_num_in_str(col):
        res = ''
        for ch in col:
            if ch in '0123456789':
                res += ch
        return int(res)

    def get_first_ch_in_str(col):
        for ch in col:
            if ch.isalpha():
                return ch
        return 'unk'  # unknown

    train_input['bust size 1'] = train_input['bust size'].apply(get_num_in_str)
    train_input['bust size 2'] = train_input['bust size'].apply(get_first_ch_in_str)
    test_input['bust size 1'] = test_input['bust size'].apply(get_num_in_str)
    test_input['bust size 2'] = test_input['bust size'].apply(get_first_ch_in_str)
    train_input.drop('bust size', axis=1, inplace=True)
    test_input.drop('bust size', axis=1, inplace=True)


def one_hop_encoding(train_input, test_input):
    encoder = preprocessing.OneHotEncoder(handle_unknown='ignore')
    encoder.fit(train_input)
    train_input = encoder.transform(train_input.toarray())
    test_input = encoder.transform(test_input.toarray())
    return train_input, test_input


if __name__ == '__main__':
    os.environ['CUDA_VISIBLE_DEVICES'] = '4'
    os.environ['MPLCONFIGDIR'] = '/data/yhshu/matplotlib'  # wayne

    train_file_path = '../product_fit/train.txt'
    test_file_path = '../product_fit/test.txt'
    test_res_file_path = '../product_fit/output_MF20330067.txt'

    train_data = pd.read_csv(train_file_path)
    test_data = pd.read_csv(test_file_path)
    # test_res = pd.read_csv(test_res_file_path)
    print('#sample of training data: ' + str(len(train_data)))
    print('#sample of testing data: ' + str(len(test_data)))

    # get_text_features()
    get_non_text_features()
