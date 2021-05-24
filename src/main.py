# Created by Yiheng Shu, MF20330067
# yhshu@smail.nju.edu.cn


import os

import numpy as np
import pandas as pd
import torch
import xgboost as xgb
from datasets import load_metric
from pandas import DataFrame
from sklearn import preprocessing
from sklearn.metrics import accuracy_score, f1_score
from torch.utils.data import TensorDataset
from transformers import AutoModelForSequenceClassification, AutoConfig, Trainer, TrainingArguments, AutoTokenizer, \
    TextClassificationPipeline, pipeline

from data_utils import ClothingDataset, subset_to_dataset


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


def transformer_train(model_dir_path=None):
    if model_dir_path is not None and os.path.isfile(model_dir_path + '/pytorch_model.bin'):
        print('[INFO] the model has been trained')
        return

    model_name = 'cardiffnlp/twitter-roberta-base-sentiment'
    tokenizer = AutoTokenizer.from_pretrained(model_name)

    train_dataset_filepath = './train_dataset'  # split from original training dataset
    dev_dataset_filepath = './dev_dataset'  # split from original training dataset

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

    config = AutoConfig.from_pretrained(model_name, num_labels=3, return_dict=False, max_length=128)
    model = AutoModelForSequenceClassification.from_pretrained(model_name, config=config)
    model.cuda()

    training_args = TrainingArguments('sequence_classification',
                                      do_train=True, do_eval=True, do_predict=True,
                                      evaluation_strategy='steps', eval_steps=500,
                                      # evaluation_strategy='epoch',
                                      per_device_train_batch_size=16, per_device_eval_batch_size=8,
                                      learning_rate=5e-5,
                                      no_cuda=False, load_best_model_at_end=True)

    # metric = load_metric("accuracy")

    def compute_metrics(eval_pred):
        logits, labels = eval_pred
        predictions = np.argmax(logits, axis=-1)
        # return metric.compute(predictions=predictions, references=labels)
        return {'f1': f1_score(labels, predictions, average='macro')}

    trainer = Trainer(model, args=training_args,
                      train_dataset=train_dataset, eval_dataset=dev_dataset,
                      compute_metrics=compute_metrics)
    trainer.train()
    trainer.save_model(model_dir_path)


def label_encoding(lbl, train_data, columns: list, test_data=None):
    for col in columns:
        train_data[col] = lbl.fit_transform(train_data[col].astype(str))
        if test_data is not None:
            test_data[col] = lbl.fit_transform(test_data[col].astype(str))
    return train_data, test_data


def get_xgb_processed_data():
    # print information before pre-processing
    train_data.info()
    print(train_data.describe())
    test_data.info()
    print(test_data.describe())

    # fill N/A values
    train_input = train_data[
        ['age', 'body type', 'bust size', 'category', 'height', 'rating', 'rented for', 'size',
         'weight', 'review_summary', 'review_text']].fillna(method='bfill')
    train_output = train_data['fit']
    test_input = test_data[
        ['age', 'body type', 'bust size', 'category', 'height', 'rating', 'rented for', 'size',
         'weight', 'review_summary', 'review_text']].fillna(method='bfill')

    # column 'height'
    def get_height_value(col):
        col_split = col.split('\'')
        for i in range(0, len(col_split)):
            col_split[i] = col_split[i].strip('"').strip(' ')
        return 12 * int(col_split[0]) + int(col_split[1])

    train_input['height'] = train_input['height'].apply(get_height_value).astype(float)
    test_input['height'] = test_input['height'].apply(get_height_value).astype(float)

    # column 'weight'
    def get_weight_value(col):
        return col.replace('lbs', '').strip(' ')

    train_input['weight'] = train_input['weight'].apply(get_weight_value).astype(float)
    test_input['weight'] = test_input['weight'].apply(get_weight_value).astype(float)

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

    train_input['bust size 1'] = train_input['bust size'].apply(get_num_in_str).astype(float)
    train_input['bust size 2'] = train_input['bust size'].apply(get_first_ch_in_str)
    test_input['bust size 1'] = test_input['bust size'].apply(get_num_in_str).astype(float)
    test_input['bust size 2'] = test_input['bust size'].apply(get_first_ch_in_str)
    train_input.drop('bust size', axis=1, inplace=True)
    test_input.drop('bust size', axis=1, inplace=True)

    # convert dataframe object into categories number: body type, category, rented for, review_summary, review_text, bust size 2
    lbl = preprocessing.LabelEncoder()
    train_input, test_input = label_encoding(lbl, train_data=train_input, test_data=test_input,
                                             columns=['body type', 'category', 'rented for', 'review_summary',
                                                      'review_text', 'bust size 2'])
    train_output = train_output.map({'small': 0, 'fit': 1, 'large': 2})

    # print information after a part of pre-processing
    train_input.info()
    print(train_input.describe())
    test_input.info()
    print(test_input.describe())

    # split the original train set to train set and dev set
    train_input_new = train_input.sample(frac=0.9, random_state=0)
    train_output_new = train_output.sample(frac=0.9, random_state=0)
    dev_input = train_input[~train_input.index.isin(train_input_new.index)]
    dev_output = train_output[~train_output.index.isin(train_output_new.index)]
    assert len(train_input_new) == len(train_output_new) and len(dev_input) == len(dev_output)

    # pre-process finished, convert pandas dataframe to xgboost dmatrix
    dtrain = xgb.DMatrix(train_input_new, label=train_output_new, nthread=8)
    ddev = xgb.DMatrix(dev_input, label=dev_output, nthread=8)
    dtest = xgb.DMatrix(test_input, nthread=8)

    return dtrain, ddev, dtest


def one_hop_encoding(train_input, test_input):
    encoder = preprocessing.OneHotEncoder(handle_unknown='ignore')
    encoder.fit(train_input)
    train_input = encoder.transform(train_input.toarray())
    test_input = encoder.transform(test_input.toarray())
    return train_input, test_input


def pred_to_output(ypred):
    res = []
    for val in ypred:
        if val == 0:
            res.append('small')
        elif val == 1:
            res.append('fit')
        elif val == 2:
            res.append('large')
    return res


def run_xgboost():
    dtrain, ddev, dtest = get_xgb_processed_data()
    eval_list = [(ddev, 'eval'), (dtrain, 'train')]
    num_round = 100

    model_filename = 'model.bin'
    # if os.path.isfile(model_filename):
    #     bst = xgb.Booster({'nthread': 8})
    #     bst.load_model(model_filename)
    # else:
    params = dict()
    params['gpu_id'] = 0
    params['tree_method'] = 'gpu_hist'
    # params['tree_method'] = 'auto'
    params['objective'] = 'multi:softmax'
    params['num_class'] = 3
    params['max_depth'] = 7
    params['subsample'] = 1
    # params['num_parallel_tree'] = 2
    # params['sketch_eps'] = 0.03
    # params['eval_metric'] = ['auc']
    bst = xgb.train(params, dtrain, num_round, eval_list, early_stopping_rounds=10)
    bst.save_model(model_filename)
    bst.dump_model('dump.raw.txt')

    ytrain = bst.predict(dtrain)
    yeval = bst.predict(ddev)
    print('train macro F1: ' + str(f1_score(dtrain.get_label(), ytrain, average='macro')))
    print('train weighted F1: ' + str(f1_score(dtrain.get_label(), ytrain, average='weighted')))
    print('train F1: ' + str(f1_score(dtrain.get_label(), ytrain, average=None)))
    print('dev macro F1: ' + str(f1_score(ddev.get_label(), yeval, average='macro')))
    print('dev weighted F1: ' + str(f1_score(ddev.get_label(), yeval, average='weighted')))
    print('dev F1: ' + str(f1_score(ddev.get_label(), yeval, average=None)))

    ypred = bst.predict(dtest)
    test_res_file = open(test_res_file_path, 'w')
    ypred = pred_to_output(ypred.tolist())
    for val in ypred:
        print(val, file=test_res_file)
    test_res_file.close()
    xgb.plot_importance(bst)


def transformer_predict(model_dir_path, test_data):
    test_dataset_filepath = './test_dataset'

    model = AutoModelForSequenceClassification.from_pretrained(model_dir_path)
    tokenizer = AutoTokenizer.from_pretrained('cardiffnlp/twitter-roberta-base-sentiment')

    # if os.path.isfile(test_dataset_filepath):  # load from file
    #     test_dataset = torch.load(test_dataset_filepath)
    # else:  # generate dataloader and save it to file
    test_text_input = test_data[['review_summary', 'review_text']].fillna(method='bfill')
    test_text_input = get_text_list(test_text_input)

    # test_encodings = tokenizer(test_text_input, truncation='only_first', max_length=128, padding='max_length',
    # return_tensors = 'pt')
    # test_dataset = ClothingDataset(test_encodings, [0 for i in range(0, len(test_encodings.encodings))])
    # torch.save(test_dataset, test_dataset_filepath)

    nlp = pipeline('text-classification', model=model, tokenizer=tokenizer)

    test_res_file = open(test_res_file_path + '1', 'w')
    for idx in range(0, len(test_text_input)):
        predictions = nlp(test_text_input[idx])
        if predictions[0]['label'] == 'LABEL_0':
            print('small', file=test_res_file)
        elif predictions[0]['label'] == 'LABEL_1':
            print('fit', file=test_res_file)
        elif predictions[0]['label'] == 'LABEL_2':
            print('large', file=test_res_file)
    test_res_file.close()


if __name__ == '__main__':
    os.environ['CUDA_VISIBLE_DEVICES'] = '3, 4'
    # os.environ['MPLCONFIGDIR'] = '/data/yhshu/matplotlib'  # wayne
    os.environ['MPLCONFIGDIR'] = '/home2/yhshu/yhshu/workspace/aml_homework/matplotlib'  # sophia

    train_file_path = '../product_fit/train.txt'
    test_file_path = '../product_fit/test.txt'
    test_res_file_path = '../product_fit/output_MF20330067.txt'
    transformer_model_dir_path = '/home2/yhshu/yhshu/workspace/aml_homework/src/sequence_classification'

    train_data = pd.read_csv(train_file_path)
    test_data = pd.read_csv(test_file_path)
    # test_res = pd.read_csv(test_res_file_path)
    print('#sample of training data: ' + str(len(train_data)))
    print('#sample of testing data: ' + str(len(test_data)))

    transformer_train(transformer_model_dir_path)
    transformer_predict(transformer_model_dir_path, test_data)
    # run_xgboost()
