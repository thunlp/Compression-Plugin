from model_center.model import T5
from torch import nn
import torch
import bmtrain as bmt
import math
from model_center.layer import Linear
import types
from model_center.tokenizer import T5Tokenizer

import torch.nn as nn
import math
import bmtrain as bmt
import torch
import torch.nn.functional as F
# from model_center.generation.t5 import T5RandomSampling,T5BeamSearch

import time
from .AdapterLayer import Adapter,LowRankLinear

class MuxLayer(nn.Module):
    def __init__(self, dim_model, mux_num):
        super().__init__()
        self.dim_model = dim_model
        self.mux_num = mux_num

        self.mux_layer = Linear(
            dim_in = dim_model * mux_num,
            dim_out = mux_num,
            init_std=math.sqrt(2 / dim_model / mux_num),
            bias=True
        )

        self.demux_adapter = Adapter(
            dim_in=dim_model * 2,
            dim_mid=64,
            dim_out=dim_model
        )

        self.act = torch.nn.Tanh()

    def mux(self, hidden_states):
        bs, n, hs = hidden_states.size()

        mux_score = hidden_states.view(bs, -1, self.mux_num * hs) # bs, -1, mux_num * hs
        mux_score = self.mux_layer(mux_score).unsqueeze(2) # bs, -1, mux_num
        mux_score = torch.softmax(mux_score, dim=-1)

        x = hidden_states.view(bs, -1, self.mux_num, hs) # bs, -1, mux_num, hs
        x = torch.matmul(mux_score, x).squeeze(-2) # bs, -1, hs

        return x

    def demux(self, hidden_states, ori_hidden):
        bs, _, hs = hidden_states.size()
        # hidden_states: bs, seq_len / mux_num, hs
        x = hidden_states.unsqueeze(2) # bs, -1, 1, hs
        x = x.repeat(1, 1, self.mux_num, 1)
        x = x.view(bs, -1, hs) # bs, seq_len, hs

        tmp = torch.cat((x, ori_hidden), dim=-1)
        adp_x = self.demux_adapter(tmp)
        x = x + adp_x

        return x

class MuxT5Base(nn.Module):
    def __init__(self, model_config, mux_num=4, has_teacher=False, delta_tuning=False):
        super().__init__()
        self.t5 = T5.from_pretrained(model_config)
        self.plm_config = self.t5.config

        self.has_teacher = has_teacher
        bmt.print_rank("with teacher:", self.has_teacher)
        if self.has_teacher:
            self.mux_layer = nn.ModuleList([MuxLayer(self.plm_config.dim_model, mux_num)
                                            for i in range(self.plm_config.num_encoder_layers)])
            bmt.init_parameters(self.mux_layer)

        self.lora_r = 16

        self.delta_tuning = delta_tuning
        if delta_tuning:
            self.insert_lora()

        if self.has_teacher or self.delta_tuning:
            for name, module in self.t5.named_parameters():
                module.requires_grad_(False)
        if self.has_teacher and self.delta_tuning:
            self.freeze_lora()

        self.mux_num = mux_num
        self.teacher = False

        if self.has_teacher:
            self.change_ffn_forward()
        
        self.loss_func = bmt.loss.FusedCrossEntropy()

    def insert_lora(self):
        self.project_q_lora = nn.ModuleList([
            LowRankLinear(self.plm_config.dim_model, self.plm_config.num_heads * self.plm_config.dim_head, r=self.lora_r, lora_alpha=self.lora_r * 2, dtype=self.plm_config.dtype)
            for i in range(self.plm_config.num_encoder_layers)
        ])
        self.project_v_lora = nn.ModuleList([
            LowRankLinear(self.plm_config.dim_model, self.plm_config.num_heads * self.plm_config.dim_head, r=self.lora_r, lora_alpha=self.lora_r * 2, dtype=self.plm_config.dtype)
            for i in range(self.plm_config.num_encoder_layers)
        ])
        self.project_q_lora_dec = nn.ModuleList([
            LowRankLinear(self.plm_config.dim_model, self.plm_config.num_heads * self.plm_config.dim_head, r=self.lora_r, lora_alpha=self.lora_r * 2, dtype=self.plm_config.dtype)
            for i in range(self.plm_config.num_encoder_layers)
        ])
        self.project_v_lora_dec = nn.ModuleList([
            LowRankLinear(self.plm_config.dim_model, self.plm_config.num_heads * self.plm_config.dim_head, r=self.lora_r, lora_alpha=self.lora_r * 2, dtype=self.plm_config.dtype)
            for i in range(self.plm_config.num_encoder_layers)
        ])
        bmt.init_parameters(self.project_q_lora)
        bmt.init_parameters(self.project_v_lora)
        bmt.init_parameters(self.project_q_lora_dec)
        bmt.init_parameters(self.project_v_lora_dec)
        
        def q_lora_linear_forward(
            linear_self,
            x: torch.Tensor
        ):
            ret = linear_self.forward_old(x) # batch, seq_len, dim_model
            lora_ret = self.project_q_lora[linear_self.layer_no](x)
            return ret + lora_ret
        
        def v_lora_linear_forward(
            linear_self,
            x: torch.Tensor
        ):
            ret = linear_self.forward_old(x) # batch, seq_len, dim_model
            lora_ret = self.project_v_lora[linear_self.layer_no](x)
            return ret + lora_ret
        
        def q_lora_linear_forward_dec(
            linear_self,
            x: torch.Tensor
        ):
            ret = linear_self.forward_old(x) # batch, seq_len, dim_model
            lora_ret = self.project_q_lora_dec[linear_self.layer_no](x)
            return ret + lora_ret
        
        def v_lora_linear_forward_dec(
            linear_self,
            x: torch.Tensor
        ):
            ret = linear_self.forward_old(x) # batch, seq_len, dim_model
            lora_ret = self.project_v_lora_dec[linear_self.layer_no](x)
            return ret + lora_ret

        for name, module in self.t5.named_modules(): 
            if name.endswith("self_att.self_attention.project_q"):
                bmt.print_rank("add lora to", name)
                module.forward_old = module.forward
                module.layer_no = int(name.split(".")[-4])
                if "encoder" in name:
                    module.forward = types.MethodType(q_lora_linear_forward, module)
                elif "decoder" in name:
                    module.forward = types.MethodType(q_lora_linear_forward_dec, module)
            elif name.endswith("self_att.self_attention.project_v"):
                bmt.print_rank("add lora to", name)
                module.forward_old = module.forward
                module.layer_no = int(name.split(".")[-4])
                if "encoder" in name:
                    module.forward = types.MethodType(v_lora_linear_forward, module)
                elif "decoder" in name:
                    module.forward = types.MethodType(v_lora_linear_forward_dec, module)

    def freeze_lora(self):
        for name, module in self.project_q_lora.named_parameters():
            module.requires_grad_(False)
        for name, module in self.project_v_lora.named_parameters():
            module.requires_grad_(False)
        for name, module in self.project_q_lora_dec.named_parameters():
            module.requires_grad_(False)
        for name, module in self.project_v_lora_dec.named_parameters():
            module.requires_grad_(False)

    def change_ffn_forward(self):
        def ffn_forward(ffn_self, hidden_states: torch.Tensor):
            bs, n, hs = hidden_states.size()

            x = ffn_self.layernorm_before_ffn(hidden_states)
            if ffn_self.post_layer_norm:
                hidden_states = x

            # add mux operation for ffn layers
            if not self.teacher:
                x_before = x
                x = self.mux_layer[ffn_self.layer_no].mux(x)


            x = ffn_self.ffn(x)

            if ffn_self.dropout is not None:
                x = ffn_self.dropout(x)
            
            # add demux operation for ffn layers
            if not self.teacher:
                x = self.mux_layer[ffn_self.layer_no].demux(x, x_before)

            hidden_states = hidden_states + x

            return hidden_states

        for name, module in self.t5.named_modules():
            if name.endswith("ffn") and len(name.split(".")) == 4 and "encoder" in name:
                bmt.print_rank(name)
                module.layer_no = int(name.split(".")[-2])

                module.forward = types.MethodType(ffn_forward, module)

    def model_forward(self, input_ids, attention_mask, decoder_input_ids, decoder_attention_mask=None, decoder_length=None, teacher=False):
        self.teacher = teacher
        output = self.t5(
            input_ids=input_ids,
            attention_mask=attention_mask,
            decoder_input_ids=decoder_input_ids,
            decoder_attention_mask=decoder_attention_mask,
            decoder_length=decoder_length,
            output_logits=True
        )
        encoder_last_hidden = output.encoder_last_hidden_state
        decoder_last_hidden = output.last_hidden_state
        logits = output.logits
        return encoder_last_hidden, decoder_last_hidden, logits


class MuxT5(MuxT5Base):
    def __init__(self, model_config, mux_num=4, has_teacher=False, delta_tuning=False):
        super().__init__(model_config, mux_num, has_teacher, delta_tuning)
        self.verbalizer = []

    def forward(self, data, only_student=False):
        bs = data["input_ids"].size(0)
        decoder_input_ids = torch.LongTensor([[2, 32099]] * bs).cuda()
        decoder_length = torch.LongTensor([2] * bs).cuda()

        if (not only_student) and self.has_teacher:
            with torch.inference_mode():
                teacher_encoder_last_hidden, teacher_decoder_last_hidden, _ = self.model_forward(data["input_ids"], data["attention_mask"], decoder_input_ids, decoder_length=decoder_length, teacher=True)

            teacher_decoder_last_hidden = teacher_decoder_last_hidden.detach().clone()
            teacher_encoder_last_hidden = teacher_encoder_last_hidden.detach().clone()
        else:
            # teacher_hidden = None
            teacher_decoder_last_hidden = None
            teacher_encoder_last_hidden = None

        student_encoder_last_hidden, student_decoder_last_hidden, output_logits = self.model_forward(data["input_ids"], data["attention_mask"], decoder_input_ids, decoder_length=decoder_length, teacher=False)

        if not ((not only_student) and self.has_teacher):
            student_decoder_last_hidden = None
            student_encoder_last_hidden = None

        logits = output_logits[:,-1] # bs, vocab_size
        scores = logits[:,self.verbalizer]

        labels = data["labels"].view(-1)
        loss = self.loss_func(scores, labels)
        acc = (scores.argmax(dim=1) == labels).sum() / len(labels)

        return {
            'loss' : loss,
            'acc' : acc,
            'num_masks' : bs,

            'teacher_dec_last_hidden_states': teacher_decoder_last_hidden,
            'student_dec_last_hidden_states': student_decoder_last_hidden,
            'teacher_enc_last_hidden_states': teacher_encoder_last_hidden,
            'student_enc_last_hidden_states': student_encoder_last_hidden,
            'predict_labels' : scores.argmax(dim=1)
        }


class MuxT5Pretrain(MuxT5Base):
    def __init__(self, model_config, mux_num=4, has_teacher=False):
        super().__init__(model_config, mux_num, has_teacher, delta_tuning=False)

    def forward(self, data, only_student=False):
        bs = data["input_ids"].size(0)

        if (not only_student) and self.has_teacher:
            with torch.inference_mode():
                teacher_encoder_last_hidden, teacher_decoder_last_hidden, _ = self.model_forward(data["input_ids"], data["attention_mask"], data["decoder_input_ids"], decoder_attention_mask=data["decoder_attention_mask"], teacher=True)
            teacher_decoder_last_hidden = teacher_decoder_last_hidden.detach().clone()
            teacher_encoder_last_hidden = teacher_encoder_last_hidden.detach().clone()
        else:
            teacher_decoder_last_hidden = None
            teacher_encoder_last_hidden = None

        student_encoder_last_hidden, student_decoder_last_hidden, output_logits = self.model_forward(data["input_ids"], data["attention_mask"], data["decoder_input_ids"], decoder_attention_mask=data["decoder_attention_mask"], teacher=False)

        if not((not only_student) and self.has_teacher):
            student_decoder_last_hidden = None
            student_encoder_last_hidden = None

        logits = output_logits * (100 * self.plm_config.dim_model ** -0.5)
        vocab_size = logits.size(-1)

        labels = data["labels"].view(-1)
        loss = self.loss_func(logits.view(-1, vocab_size), labels)
        acc = (logits.view(-1, vocab_size).argmax(dim=1) == labels).sum() / len(labels)

        return {
            'loss' : loss,
            'acc' : acc,
            'num_masks' : bs,

            'teacher_dec_last_hidden_states': teacher_decoder_last_hidden,
            'student_dec_last_hidden_states': student_decoder_last_hidden,
            'teacher_enc_last_hidden_states': teacher_encoder_last_hidden,
            'student_enc_last_hidden_states': student_encoder_last_hidden,
            'predict_labels' : logits.view(-1, vocab_size).argmax(dim=1)
        }

class MuxT5Seq2Seq(MuxT5Base):
    def __init__(self, model_config, mux_num=4, has_teacher=False, delta_tuning=False):
        super().__init__(model_config, mux_num, has_teacher, delta_tuning)

        self.generation = T5BeamSearch(self.t5, T5Tokenizer.from_pretrained(model_config))

    def forward_valid(self, data):
        self.teacher = False
        inference_results = self.generation.generate(data["input"], beam_size=1, max_length=32)

        return {
            "loss": 0,
            "acc": -1,
            "num_masks": len(data["answers"]),
            "predict_labels": inference_results,
            "labels": data["answers"]
        }

    def forward(self, data, only_student=False):
        if "input" in data:
            return self.forward_valid(data)
        bs = data["input_ids"].size(0)
        # decoder_input_ids = torch.LongTensor([[2, 32099]]* bs).cuda()
        # decoder_length = torch.LongTensor([2] * bs).cuda()
        
        
        if (not only_student) and self.has_teacher:
            with torch.inference_mode():
                teacher_encoder_last_hidden, teacher_decoder_last_hidden = self.model_forward(data["input_ids"], data["attention_mask"], data["decoder_input_ids"], decoder_attention_mask=data["decoder_attention_mask"], teacher=True)
                
            teacher_decoder_last_hidden = teacher_decoder_last_hidden.detach().clone()
            teacher_encoder_last_hidden = teacher_encoder_last_hidden.detach().clone()
        else:
            teacher_decoder_last_hidden = None
            teacher_encoder_last_hidden = None

        student_encoder_last_hidden, student_decoder_last_hidden, output_logits = self.model_forward(data["input_ids"], data["attention_mask"], data["decoder_input_ids"], decoder_attention_mask=data["decoder_attention_mask"], teacher=False)

        if not((not only_student) and self.has_teacher):
            student_decoder_last_hidden = None
            student_encoder_last_hidden = None

        logits = output_logits
        vocab_size = logits.size(-1)

        labels = data["labels"].view(-1)
        loss = self.loss_func(logits.view(-1, vocab_size), labels)
        acc = (logits.view(-1, vocab_size).argmax(dim=1) == labels).sum() / len(labels)

        return {
            'loss' : loss,
            'acc' : acc,
            'num_masks' : bs,

            'teacher_dec_last_hidden_states': teacher_decoder_last_hidden,
            'student_dec_last_hidden_states': student_decoder_last_hidden,
            'teacher_enc_last_hidden_states': teacher_encoder_last_hidden,
            'student_enc_last_hidden_states': student_encoder_last_hidden,
        }
