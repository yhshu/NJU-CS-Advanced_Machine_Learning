from src.Dataloader import Dataloader

if __name__ == '__main__':
    train_file_path = '../product_fit/train.txt'
    test_file_path = '../product_fit/test.txt'

    train_dataloader = Dataloader(train_file_path, train=True)
    test_dataloader = Dataloader(test_file_path, train=False)
