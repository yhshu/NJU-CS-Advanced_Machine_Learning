import torch
from transformers import RobertaTokenizer


class Transformer:
    def __init__(self):
        self.tokenizer = RobertaTokenizer.from_pretrained('roberta-base')

    def encode_text(self, text: str):
        input_ids = self.tokenizer.encode(text, add_special_tokens=True, truncation='only_first', max_length=128,
                                          padding='max_length',
                                          return_tensors='pt')
        return input_ids

    def encode_text_list(self, text_list: list):
        all_input_ids = []
        for i in range(0, len(text_list)):
            input_ids = self.encode_text(text_list[i])
            all_input_ids.append(input_ids)
            if i % 1000 == 0:
                print('[INFO] encoding text: ' + str(i) + ' / ' + str(len(text_list)))
        all_input_ids = torch.cat(all_input_ids, dim=0)
        return all_input_ids
