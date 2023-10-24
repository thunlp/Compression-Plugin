from torch.utils.data import IterableDataset
from transformers import DataCollatorForLanguageModeling

import os
import random
import torch
import bmtrain as bmt
import torch.nn.functional as F
from .KaraDataset import make_kara_dataset

class WikicorpusT5(IterableDataset):

    def __init__(self, mode, tokenizer, args, tar_len=32):
        self.tokenizer = tokenizer

        self.rank = bmt.rank()
        self.num_gpu = bmt.world_size()

        self.max_length = args.max_input_length
        self.data_path = args.data_path

        self.file_list = []
        for filename in os.listdir(self.data_path):
            if mode in filename:
                self.file_list.append(filename)

        self.dec_length = tar_len
        self.mode = mode

        self.gen = None # torch.manual_seed(233)

    def process(self, line):
        text = line.strip()
        input = self.tokenizer(text + "<extra_id_0>", max_length=self.max_length, padding="max_length", truncation=True, return_tensors='pt')

        return {
            f'input_ids': input['input_ids'][0],
            f'attention_mask': input['attention_mask'][0],
            f'labels': torch.tensor(0),
        }

    def process_valid(self, line):
        text = line.strip()
        doc_ids = self.tokenizer.encode(text, max_length=self.max_length - 1, add_special_tokens=False)
        context, target = self.mask_tokens(doc_ids)

        target = target[:self.dec_length]
        # bmt.print_rank(self.tokenizer.decode(context))
        # bmt.print_rank(self.tokenizer.decode(target))
        # bmt.print_rank("==" * 10)
        return {
            'input_ids': torch.LongTensor(context + [self.tokenizer.pad_token_id] * (self.max_length - len(context))),
            'attention_mask': torch.LongTensor([1] * len(context) + [0] * (self.max_length - len(context))),
            'labels': torch.LongTensor(target[1:] + [-100] * (self.dec_length - len(target) + 1)),
            'decoder_input_ids': torch.LongTensor(target + [0] * (self.dec_length - len(target))),
            'decoder_attention_mask': torch.LongTensor([1] * len(target) + [0] * (self.dec_length - len(target))),
        }

    def mask_tokens(self, doc_ids):
        span_start_ends = self.random_spans_noise_mask(len(doc_ids), noisy_density=0.15, mean_noise_span_length=3)
        start = 0
        context = []
        target = []
        for i, span in enumerate(span_start_ends):
            # "+[0]" placeholder for sentinel
            context.extend(doc_ids[start:span[0]] + [self.tokenizer.vocab_size-i-1])
            target.extend([self.tokenizer.vocab_size-i-1] + doc_ids[span[0]:span[1]])
            start = span[1]

        assert start == len(doc_ids)

        target = [0] + target + [self.tokenizer.vocab_size-i-2]
        context += [1]
        return context, target

    def random_spans_noise_mask(self, length, noisy_density=0.15, mean_noise_span_length=10.0):
        num_noise_tokens = round(length * noisy_density)
        num_noise_tokens = min(max(num_noise_tokens, 1), length - 1)
        num_noise_spans = round(num_noise_tokens / mean_noise_span_length)
        num_noise_spans = max(num_noise_spans, 1)
        num_nonnoise_tokens = length - num_noise_tokens

        def random_segment(seq_length, num_segment):
            # if self.gen is None:
            #     self.gen = torch.manual_seed(2333)
            x = (torch.arange(seq_length - 1) < (num_segment - 1)).long()
            a = torch.randperm(seq_length - 1)#, generator=self.gen)
            x = x[a]
            x = F.pad(x, [1, 0])
            segment_id = torch.cumsum(x, dim=0)
            segment_lengths = torch.zeros(num_segment, dtype=torch.long).scatter_add_(0, segment_id, torch.ones(seq_length, dtype=torch.long))

            return segment_lengths

        noise_span_lengths = random_segment(num_noise_tokens, num_noise_spans)
        nonnoise_span_lengths = random_segment(num_nonnoise_tokens, num_noise_spans)
        interleaved_span_lengths = torch.stack([nonnoise_span_lengths, noise_span_lengths], dim=1).view(num_noise_spans * 2)
        span_start_ends = torch.cumsum(interleaved_span_lengths, dim=0).view(-1, 2)
        return span_start_ends.tolist()

    def iterator(self):
        idx = 0
        random.shuffle(self.file_list)
        for filename in self.file_list:
            with open(os.path.join(self.data_path, filename)) as f:
                data = f.readlines()
                random.shuffle(data)
                for line in data:
                    if idx % self.num_gpu == self.rank:
                        # if self.mode == "valid":
                        yield self.process_valid(line)
                        # else:
                        #     yield self.process(line)
                    idx += 1

    def __iter__(self):
        return self.iterator()

    def __len__(self):
        return len(self.file_list) * 65000


class WikicorpusLLaMA(IterableDataset):
    def __init__(self, mode, tokenizer, args):
        self.tokenizer = tokenizer

        self.rank = bmt.rank()
        self.num_gpu = bmt.world_size()

        self.max_length = args.max_input_length
        self.ctx_length = args.ctx_length
        self.data_path = args.data_path        

        self.wiki = "wikicorpus" in self.data_path
        if self.wiki:
            self.file_list = []
            for filename in os.listdir(self.data_path):
                if mode in filename:
                    self.file_list.append(filename)
        else:
            self.dataset = make_kara_dataset(self.data_path)
        
        self.mode = mode
        
        # self.gen = torch.manual_seed(233)

    def process(self, line):
        text = line.strip()
        tokens = self.tokenizer.encode(text)[:self.max_length]

        pad_num = max(self.max_length - len(tokens), 0)
        front_pad_num = random.randint(max(0, self.ctx_length - len(tokens)), min(pad_num, self.ctx_length))

        input_ids = [0] * front_pad_num + tokens + [0] * (pad_num - front_pad_num)
        attention_mask = [0] * front_pad_num + [1] * len(tokens) + [0] * (pad_num - front_pad_num)
        labels = [-100] * (front_pad_num) + tokens[1:] + [self.tokenizer.eos_token_id] + [-100] * (pad_num - front_pad_num)

        for i in range(self.ctx_length - 1):
            labels[i] = -100

        return {
            f'input_ids': torch.LongTensor(input_ids),
            f'attention_mask': torch.LongTensor(attention_mask),
            f'labels' : torch.LongTensor(labels),
        }

    def iterator(self):
        if not self.wiki:
            for d in self.dataset:
                yield self.process(d['text'])
        else:
            idx = 0
            random.shuffle(self.file_list)
            for filename in self.file_list:
                with open(os.path.join(self.data_path, filename)) as f:
                    data = f.readlines()
                    random.shuffle(data)
                    for line in data:
                        if idx % self.num_gpu == self.rank:
                            # if self.mode == "valid":
                            yield self.process(line)
                            # else:
                            #     yield self.process(line)
                        idx += 1

    def __iter__(self):
        return self.iterator()

    def __len__(self):
        if self.wiki:
            return len(self.file_list) * 65000
        else:
            return len(self.dataset)

