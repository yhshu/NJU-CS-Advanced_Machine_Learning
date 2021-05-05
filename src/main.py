from src.Dataloader import Dataloader
import pandas as pd

if __name__ == '__main__':
    train_file_path = '../product_fit/train.txt'
    test_file_path = '../product_fit/test.txt'

    # train_dataloader = Dataloader(train_file_path, train=True)
    # test_dataloader = Dataloader(test_file_path, train=False)
    train_data = pd.read_csv(train_file_path)
    test_data = pd.read_csv(test_file_path)
    print('#sample of training data: ' + str(len(train_data)))
    print('#sample of testing data: ' + str(len(test_data)))
