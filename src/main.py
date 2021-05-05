import pandas as pd
from deepforest import CascadeForestClassifier
from sklearn import preprocessing
from sklearn.metrics import accuracy_score


def dataframe_preprocess(df: pd.DataFrame):
    return df[
        ['age', 'body type', 'bust size', 'category', 'height', 'item_id', 'rating', 'rented for', 'size', 'user_id',
         'weight']].fillna(method='pad')


if __name__ == '__main__':
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

    model = CascadeForestClassifier(random_state=1)
    model.fit(X_train, y_train.values)
    y_pred = model.predict(X_test)
    acc = accuracy_score(y_test, y_pred) * 100
    print("\nTesting Accuracy: {:.3f} %".format(acc))
