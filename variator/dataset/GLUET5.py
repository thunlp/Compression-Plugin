import torch
import transformers
from torch.utils.data import Dataset
from datasets import load_from_disk
import json
import random
import os
import bmtrain as bmt

class T5Classification:
    def __init__(self, mode, tokenizer, args):
        self.tokenizer = tokenizer
        self.max_length = args.max_input_length
        self.data_path = args.data_path

        self.dataset = load_from_disk(self.data_path)[mode]
        self.len = len(self.dataset)

    def process(self, item):
        text = self.fill_template(item)
        tokens = self.tokenizer(text, max_length=self.max_length, padding="max_length", truncation=True)

        return {
            "input_ids": torch.LongTensor(tokens["input_ids"]),
            "attention_mask": torch.LongTensor(tokens["attention_mask"]),
            "labels": torch.tensor(item["label"])
        }

    def fill_template(self, item):
        pass

    def __getitem__(self, index):
        return self.process(self.dataset[index])

    def get_verbalizer(self):
        pass

    def __len__(self):
        return self.len

class MNLI(T5Classification):
    def __init__(self, mode, tokenizer, args):
        super().__init__('validation_matched' if mode == 'valid' else mode, tokenizer, args)

    def fill_template(self, item):
        return f"Sentence 1: {item['premise']} Sentence 2: {item['hypothesis']} Does sentence 1 entails sentence 2? <extra_id_0>."

    def get_verbalizer(self):
        return [self.tokenizer.encode("Yes")[0], self.tokenizer.encode("Maybe")[0], self.tokenizer.encode("No")[0]]

class QQP(T5Classification):
    def __init__(self, mode, tokenizer, args):
        super().__init__('validation' if mode == 'valid' else mode, tokenizer, args)
    def fill_template(self, item):
        return f"Question 1: {item['question1']}\nQuestion 2: {item['question2']}\nAre the two questions paraphrase of each other? <extra_id_0>."
    def get_verbalizer(self):
        return [self.tokenizer.encode("No")[0], self.tokenizer.encode("Yes")[0]]

class QNLI(T5Classification):
    def __init__(self, mode, tokenizer, args):
        super().__init__('validation' if mode == 'valid' else mode, tokenizer, args)
    def fill_template(self, item):
        return f"Question: {item['question']}\nSentence: {item['sentence']}\nDoes the sentence contains the answer to the question? <extra_id_0>."
    def get_verbalizer(self):
        return [self.tokenizer.encode("Yes")[0], self.tokenizer.encode("No")[0]]


class SST2(T5Classification):
    def __init__(self, mode, tokenizer, args):
        super().__init__('validation' if mode == 'valid' else mode, tokenizer, args)
    def fill_template(self, item):
        return f"Sentence: {item['sentence']}\nDoes this sentence express positive or negative emotions? <extra_id_0>."
    def get_verbalizer(self):
        return [self.tokenizer.encode("negative")[0], self.tokenizer.encode("positive")[0]]


class RTE(T5Classification):
    def __init__(self, mode, tokenizer, args):
        super().__init__('validation' if mode == 'valid' else mode, tokenizer, args)
    def fill_template(self, item):
        return f"Sentence 1: {item['sentence1']}\nSentence 2: {item['sentence2']}\nDoes sentence 1 entails sentence 2? <extra_id_0>."
    def get_verbalizer(self):
        return [self.tokenizer.encode("Yes")[0], self.tokenizer.encode("No")[0]]

class MRPC(T5Classification):
    def __init__(self, mode, tokenizer, args):
        super().__init__('validation' if mode == 'valid' else mode, tokenizer, args)
    def fill_template(self, item):
        return f"Sentence 1: {item['sentence1']}\nSentence 2: {item['sentence2']}\nAre the two sentences paraphrase of each other? <extra_id_0>."

    def get_verbalizer(self):
        return [self.tokenizer.encode("No")[0], self.tokenizer.encode("Yes")[0]]

class SQuAD(Dataset):
    def __init__(self, mode, tokenizer, args):
        self.mode = mode

        self.tokenizer = tokenizer
        self.max_length = args.max_input_length
        self.target_length = 32
        
        self.data_path = args.data_path

        if mode == "train":
            data = json.load(open(os.path.join(self.data_path, "train-v1.1.json"), "r", encoding="utf8"))
        else:
            data = json.load(open(os.path.join(self.data_path, "dev-v1.1.json"), "r", encoding="utf8"))
        self.qas = []
        self.context = []
        for doc in data["data"]:
            title = doc["title"]
            for para in doc["paragraphs"]:
                context = para["context"]
                self.context.append(context)
                qas = []
                for qa in para["qas"]:
                    qa.update({"context": len(self.context) - 1})
                    qa["title"] = title
                    qas.append(qa)
                self.qas.extend(qas)
        bmt.print_rank(self.mode, "data size:", len(self.qas))

    def process_valid(self, item):
        context = self.context[item["context"]]
        
        inp = f"Question: {item['question'].strip()}\nContext: {context.strip()}\nAnswer: <extra_id_0>"
        return {
            "input": inp,
            'answers': json.dumps([ans["text"] for ans in item["answers"]]),
        }

    def process(self, item):
        context = self.context[item["context"]]

        inp = f"Question: {item['question'].strip()}\nContext: {context.strip()}\nAnswer: <extra_id_0>"
        target = random.choice(item["answers"])["text"]

        input = self.tokenizer(inp, max_length=self.max_length, padding="max_length", truncation=True, return_tensors='pt')
        target_ids = self.tokenizer.encode(target, max_length=self.target_length - 2, add_special_tokens=False)

        return {
            'input_ids' : input['input_ids'][0],
            'attention_mask' : input['attention_mask'][0],
            'decoder_input_ids': torch.LongTensor([0, 32099] + target_ids + [self.tokenizer.pad_token_id] * (self.target_length - len(target_ids) - 2)),
            'decoder_attention_mask': torch.LongTensor([1] * (len(target_ids) + 2) + [0] * (self.target_length - len(target_ids) - 2)),
            'labels' : torch.LongTensor([-100] + target_ids + [32098] + [-100] * (self.target_length - len(target_ids) - 2)),
        }

    def __getitem__(self, index):
        if self.mode == "train":
            return self.process(self.qas[index])
        else:
            return self.process_valid(self.qas[index])

    def __len__(self):
        return len(self.qas)

T5DATASET = {
    "mnli": MNLI,
    "qqp": QQP,
    "qnli": QNLI,
    "sst2": SST2,
    "rte": RTE,
    "mrpc": MRPC,
    "squad": SQuAD,
}
