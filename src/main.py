import numpy as np
import pandas as pd
import torch
from pandas import DataFrame
from sklearn import preprocessing
from sklearn.metrics import accuracy_score
from torch.utils.data import TensorDataset, DataLoader
from transformers import RobertaForSequenceClassification, AdamW, get_linear_schedule_with_warmup

from src.pre_trained import Transformer


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
    # for i in range(0, len(dataframe.values)):
    for i in range(0, 100):
        res.append(dataframe.values[i][0] + ' ' + dataframe.values[i][1])
    return res


def get_text_features():
    X_train_text = train_data[['review_summary', 'review_text']].fillna(method='bfill')
    X_train_text = get_text_list(X_train_text)
    X_test_text = test_data[['review_summary', 'review_text']].fillna(method='bfill')
    X_test_text = get_text_list(X_test_text)

    y_train = train_data['fit']
    y_test = pd.read_csv(test_res_file_path, header=None)[0]

    epochs = 4
    batch_size = 32

    transformer = Transformer()
    train_input_ids = transformer.encode_text_list(X_train_text)
    train_dataset = TensorDataset(train_input_ids, y_train)
    test_input_ids = transformer.encode_text_list(X_test_text)
    test_dataset = TensorDataset(test_input_ids, y_test)
    train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    test_dataloader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

    model = RobertaForSequenceClassification.from_pretrained('roberta-base', num_labels=3)
    model.cuda()

    optimizer = AdamW(model.parameters(), lr=2e-5)
    total_steps = len(train_dataloader) * epochs
    scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=0, num_training_steps=total_steps)

    for epoch in range(epochs):
        model.train()
        total_loss, total_val_loss = 0, 0
        total_eval_accuracy = 0
        for step, batch in enumerate(train_dataloader):
            model.zero_grad()
            loss, logits = model(batch[0].to(device), token_type_ids=None, attention_mask=(batch[0] > 0).to(device),
                                 labels=batch[1].to(device))
            total_loss += loss.item()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            scheduler.step()

        model.eval()
        for i, batch in enumerate(test_dataloader):
            with torch.no_grad():
                loss, logits = model(batch[0].to(device), token_type_ids=None, attention_mask=(batch[0] > 0).to(device),
                                     labels=batch[1].to(device))

                total_val_loss += loss.item()

                logits = logits.detach().cpu().numpy()
                label_ids = batch[1].to('cpu').numpy()
                total_eval_accuracy += flat_accuracy(logits, label_ids)

        avg_train_loss = total_loss / len(train_dataloader)
        avg_val_loss = total_val_loss / len(test_dataloader)
        avg_val_accuracy = total_eval_accuracy / len(test_dataloader)

        print(f'Train loss     : {avg_train_loss}')
        print(f'Validation loss: {avg_val_loss}')
        print(f'Accuracy: {avg_val_accuracy:.2f}')
        print('\n')


def get_non_text_features():
    X_train = train_data[
        ['age', 'body type', 'bust size', 'category', 'height', 'item_id', 'rating', 'rented for', 'size', 'user_id',
         'weight']].fillna(method='bfill')
    X_train.info()
    y_train = train_data['fit']
    X_test = test_data[
        ['age', 'body type', 'bust size', 'category', 'height', 'item_id', 'rating', 'rented for', 'size', 'user_id',
         'weight']].fillna(method='bfill')
    X_test.info()
    y_test = pd.read_csv(test_res_file_path, header=None)[0]

    encoder = preprocessing.OneHotEncoder(handle_unknown='ignore')
    encoder.fit(X_train)
    X_train = encoder.transform(X_train.toarray())
    X_test = encoder.transform(X_test.toarray())


if __name__ == '__main__':
    device = 'cuda:1'
    train_file_path = '../product_fit/train.txt'
    test_file_path = '../product_fit/test.txt'
    test_res_file_path = '../product_fit/output_AB1234567.txt'

    # train_dataloader = Dataloader(train_file_path, train=True)
    # test_dataloader = Dataloader(test_file_path, train=False)
    train_data = pd.read_csv(train_file_path)
    test_data = pd.read_csv(test_file_path)
    # test_res = pd.read_csv(test_res_file_path)
    print('#sample of training data: ' + str(len(train_data)))
    print('#sample of testing data: ' + str(len(test_data)))

    get_text_features()
