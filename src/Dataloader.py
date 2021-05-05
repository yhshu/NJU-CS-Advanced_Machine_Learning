from src.sample import Sample


def file_split(file_lines: list):
    res = []
    for i in range(len(file_lines)):
        line = file_lines[i]
        if line is '' or not line[0].isdigit() or ('.0' not in line[0:4]):
            res[-1] += line
        else:
            res.append(line)
    return res


class Dataloader:
    sample_list: list
    len: int
    train: bool  # train or test

    def __init__(self, filepath: str, train: bool):
        self.len = 0
        self.sample_list = []

        file = open(filepath, 'r')
        header = file.readline()
        header_split = header.split(',')
        file_content = file.read()
        file_lines = file_split(file_content.split('\n'))

        for line in file_lines:
            line_split = line.strip('\n').split(',')
            if train is True:
                data = {'age': line_split[0], 'body type': line_split[1], 'bust size': line_split[2],
                        'category': line_split[3],
                        'height': line_split[5], 'item_id': line_split[6], 'rating': line_split[7],
                        'rented for': line_split[8],
                        'review date': line_split[9] + line_split[10],
                        'review summary': line_split[11], 'size': line_split[-3], 'user id': line_split[-2],
                        'weight': line_split[-1]}
                data['review text'] = line_split[12:-3]
                label = line_split[4]

            else:  # train is False
                data = {'age': line_split[0], 'body type': line_split[1], 'bust size': line_split[2],
                        'category': line_split[3],
                        'height': line_split[4], 'item_id': line_split[5], 'rating': line_split[6],
                        'rented for': line_split[7],
                        'review date': line_split[8] + line_split[9],
                        'review summary': line_split[10], 'size': line_split[-3], 'user id': line_split[-2],
                        'weight': line_split[-1]}
                data['review text'] = line_split[11:-3]  # todo merge
                label = None

            # train: age,body type,bust size,category,fit,height,item_id,rating,rented for,review_date,review_summary,review_text,size,user_id,weight
            # test:  age,body type,bust size,category,height,item_id,rating,rented for,review_date,review_summary,review_text,size,user_id,weight

            if len(data) != 14:
                print('[WARN] the length of the dict is incorrent')
            self.len += 1
            sample = Sample(self.len, data, label)
            self.sample_list.append(sample)
        file.close()
        print('[INFO] #sample: ' + str(len(self.sample_list)))

    def get_sample_by_id(self, id):
        return self.sample_list[id]

    def get_sample_label_by_id(self, id):
        return self.sample_list[id].label
