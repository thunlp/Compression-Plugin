import bmtrain as bmt
from scipy.stats import stats
from model_center.tokenizer import T5Tokenizer
from torch.utils.data import DataLoader, DistributedSampler
from argument import getargs
import random
import torch
import time
import os
from model.muxT5 import MuxT5,MuxT5Pretrain,MuxT5Seq2Seq

from dataset.GLUET5 import T5DATASET
from dataset.WikicorpusT5 import WikicorpusT5
from bmtrain.store import _save_to_rank0
from bmtrain.store import DistributedStateDictWrapper

from bmtrain.lr_scheduler.warmup import WarmupLRScheduler
from metric import binary_f1_score,squad_metric

class Constant(WarmupLRScheduler):
    r"""
        After a warmup period during which learning rate increases linearly between 0 and the start_lr,
        The decay period performs :math:`\text{lr}=\text{start_lr}`
    """
    def get_lr_warmup(self, num_iter) -> float:
        return self.start_lr

    def get_lr_decay(self, num_iter) -> float:
        return self.start_lr

def load_model(model, state_dict):
    if bmt.rank() == 0:
        state_dict = DistributedStateDictWrapper(state_dict)
    else:
        state_dict = DistributedStateDictWrapper({})

    ret = model.load_state_dict(
        state_dict,
        strict = False
    )
    torch.cuda.synchronize()
    return ret


def training_checkpoint(model, optimizer, lr_scheduler, global_step, args):
    filename = os.path.join(args.output_path, 'training_checkpoint')
    state_dict = {
        'lr_scheduler' : lr_scheduler.state_dict(),
        'global_step' : global_step,
    }
    bmt.save(model, filename+'.pt')
    torch.save(optimizer.state_dict(), filename+f'.rank-{bmt.rank()}')
    if bmt.rank() == 0:
        torch.save(state_dict, filename+'.bin')


def init(args):
    # init model
    bmt.print_rank("init the model", args.model_config)
    if args.dataset == "wiki":
        model = MuxT5Pretrain(args.model_config, args.mux_num, (args.distil != "no")).cuda()
    elif args.dataset == "squad":
        model = MuxT5Seq2Seq(args.model_config, args.mux_num, (args.distil != "no"), delta_tuning=args.delta_tuning).cuda()
    else:
        model = MuxT5(args.model_config, args.mux_num, (args.distil != "no"), delta_tuning=args.delta_tuning).cuda()
    tokenizer = T5Tokenizer.from_pretrained(args.model_config)

    # load model from checkpoint
    if args.checkpoint:
        bmt.print_rank("try load from", args.checkpoint)
        bmt.print_rank(bmt.load(model, args.checkpoint, strict=False))

    # init dataloader
    dataloader = {}
    for mode in ['train', 'valid']:
        bmt.print_rank("loading dataset", mode)
        if args.dataset == "wiki":
            dataset = WikicorpusT5(mode, tokenizer, args)
            sampler = None
        else:
            dataset = T5DATASET[args.dataset](mode, tokenizer, args)
            sampler = DistributedSampler(
                    dataset,
                    num_replicas=bmt.world_size(),
                    rank=bmt.rank(),
                    seed=args.random_seed
                )

            if args.dataset != "squad":
                model.verbalizer = dataset.get_verbalizer()
                bmt.print_rank("verbalizer:", model.verbalizer)
            if args.dataset == "squad" and mode == "valid":
                sampler = None # generation 不支持多卡
        print("sampler", sampler)
        dataloader[mode] = DataLoader(
            dataset=dataset,
            batch_size=args.batch_size if not (args.dataset == "squad" and mode == "valid") else 32,
            num_workers=4 if mode=='train' and args.dataset != "wiki" else 1,
            sampler=sampler
        )

    param_list = [param for param in model.parameters() if param.requires_grad]
    print('size of parameter list:', len(param_list))

    if args.plugin_path and args.plugin_path != "none":
        bmt.print_rank("try load plugin from", args.plugin_path)
        plugins = torch.load(args.plugin_path)
        plugins_parameter = {key: plugins[key] for key in plugins if key.startswith("mux_layer")}
        assert len(param_list) == len(plugins_parameter)
        bmt.print_rank(load_model(model, plugins_parameter))

    # init optimizer and lr_scheduler
    optimizer = bmt.optim.AdamOptimizer(
        param_list,
        lr=args.lr,
        weight_decay=args.weight_decay,
        # scale = 131072,
    )
    if args.dataset == "wiki":
        lr_scheduler = bmt.lr_scheduler.Linear(
            optimizer,
            start_lr=args.lr,
            warmup_iter=args.warmup_step,
            end_iter=args.training_step,
        )
    else:
        lr_scheduler = Constant(
            optimizer,
            start_lr=args.lr,
            warmup_iter=args.warmup_step,
            end_iter=args.training_step,
        )
    global_step = 0

    optim_manager = bmt.optim.OptimManager(loss_scale=2**20)
    optim_manager.add_optimizer(optimizer, lr_scheduler)

    if args.save_step > 0:
        os.makedirs(args.output_path, exist_ok=True)

    return model, dataloader, optim_manager, optimizer, lr_scheduler, global_step

def valid(model, dataloader, start_time, args):
    model.eval()
    step = 0

    with torch.no_grad():
        labels = []
        predict_labels = []

        total_num = torch.tensor(0.).cuda()
        total_loss = {'loss': 0., 'acc': 0.}
        total_time = 0.
        for k in total_loss.keys():
            total_loss[k] = torch.tensor(total_loss[k]).cuda()

        for data in dataloader['valid']:
            for k in data.keys():
                if isinstance(data[k], torch.Tensor):
                    data[k] = data[k].cuda()

            iter_start_time = time.time()
            model_output = model(data, only_student=True)
            iter_end_time = time.time()
            step += 1

            if args.dataset == "squad":
                labels.extend(model_output["labels"])
                predict_labels.extend(model_output["predict_labels"])
            else:
                labels.append(data['labels'].view(-1))
                predict_labels.append(model_output['predict_labels'].view(-1))

            bmt.print_rank(
                "valid | step: {:6d} | loss: {:.4f} | acc: {:2.2f}% | time: {:2.3f}s | total time: {:7.3f}s".format(
                    step,
                    model_output['loss'],
                    model_output['acc'] * 100,
                    iter_end_time - iter_start_time,
                    iter_end_time - start_time
                )
            )
            total_num += model_output['num_masks']
            total_time += iter_end_time - iter_start_time
            for k in total_loss.keys():
                total_loss[k] = total_loss[k] + model_output[k] * model_output['num_masks']

        total_num = bmt.sum_loss(total_num)
        for k in total_loss.keys():
            total_loss[k] = bmt.sum_loss(total_loss[k])

        if args.dataset != "squad":
            y_true = torch.cat(labels) #.cpu()
            y_pred = torch.cat(predict_labels) # .cpu()

        acc = total_loss['acc'] / total_num
        if args.dataset != "squad":
            f1_score = binary_f1_score(y_pred, y_true)
        else:
            em, f1 = squad_metric(predict_labels, labels)
            acc = em
            f1_score = f1


        bmt.print_rank(
            "valid result | num_masks: {:6d} | loss: {:.4f} | acc: {:2.2f}% | f1: {:2.2f}% | time: {:3.2f}".format(
                int(total_num * bmt.world_size()),
                total_loss['loss'] / total_num,
                acc * 100,
                f1_score * 100,
                total_time,
            )
        )

    if args.dataset == 'mrpc':
        ret = f1_score
    else:
        ret = acc

    model.train()
    return ret

def train(model, dataloader, optim_manager, optimizer, lr_scheduler, global_step, args):
    step = 0
    start_time = time.time()
    step_time = 0.
    optimizer.zero_grad()
    step_loss = {'loss': 0., 'acc': 0.}
    best_valid_res = -1000

    if args.distil != 'no':
        mse_loss_func = torch.nn.functional.mse_loss

    for epoch_num in range(args.epochs):
        if global_step >= args.training_step:
            break
        model.train()
        for data in dataloader['train']:
            for k in data.keys():
                data[k] = data[k].cuda()
            step_start_time = time.time()

            model_output = model(data, only_student = args.distil == "no")

            mse_loss = torch.tensor(0.).cuda()
            if args.distil != 'no':
                if "decoder_attention_mask" in data:
                    # for seq2seq task, there are pad tokens in decoder inputs
                    mse_loss += mse_loss_func(
                        data["decoder_attention_mask"].unsqueeze(-1) * model_output['teacher_dec_last_hidden_states'].to(torch.float32),
                        data["decoder_attention_mask"].unsqueeze(-1) * model_output['student_dec_last_hidden_states'].to(torch.float32),
                    )
                else:
                    # for classification task, only <begin>(<pad> in t5), <extra_id_0> in decoder inputs
                    mse_loss += mse_loss_func(
                        model_output['teacher_dec_last_hidden_states'].to(torch.float32),
                        model_output['student_dec_last_hidden_states'].to(torch.float32),
                    )

                mse_loss += mse_loss_func(
                    data["attention_mask"].unsqueeze(-1) * model_output['teacher_enc_last_hidden_states'].to(torch.float32),
                    data["attention_mask"].unsqueeze(-1) * model_output['student_enc_last_hidden_states'].to(torch.float32),
                )

            for k in step_loss.keys():
                step_loss[k] = step_loss[k] + bmt.sum_loss(model_output[k])

            if args.distil == 'no' or args.distil == 'task':
                loss = (model_output['loss']) / args.grad_accumulation
            elif args.distil == 'task-hidden':
                loss = (model_output['loss'] * args.task_ratio + mse_loss) / args.grad_accumulation
                # loss = (model_output['loss']) / args.grad_accumulation
            elif args.distil == 'hidden':
                loss = (mse_loss) / args.grad_accumulation

            with bmt.inspect.inspect_tensor():
                optim_manager.backward(loss)
            # loss = optimizer.loss_scale(loss)
            # loss.backward()
            step_time += time.time() - step_start_time
            step += 1

            mse_loss = bmt.sum_loss(mse_loss)

            if step % args.grad_accumulation == 0:
                for k in step_loss.keys():
                    step_loss[k] = step_loss[k] / args.grad_accumulation
                grad_norm = optim_manager.clip_grad_norm(optimizer.param_groups, args.clip_grad, norm_type=2)
                
                global_step += 1
                
                if args.inspect_step > 0 and (global_step % args.inspect_step == 0 or global_step == 1):
                    insp_res = bmt.inspect.inspect_model(model, '*')
                    # insp_res += bmt.inspect.inspect_model(model, '*mux*')
                    bmt.print_rank(bmt.inspect.format_summary(insp_res))
                
                optim_manager.step()
                optim_manager.zero_grad()

                bmt.print_rank(
                    "train | epoch: {:2d} | step: {:6d}/{:6d} | loss: {:.4f} | acc: {:.1f}% | grad_norm: {:.3f} | lr: {:.3e} | scale: {:6d} | mse_loss: {:.4f} | time: {:1.2f}s | total time: {:5.1f}s".format(
                        epoch_num,
                        global_step,
                        len(dataloader['train']),
                        step_loss['loss'],
                        step_loss['acc'] * 100,
                        grad_norm,
                        lr_scheduler.get_lr(),
                        int(optim_manager.loss_scale),
                        mse_loss,
                        step_time,
                        time.time() - start_time
                    ), end=''
                )
                if 'debug' in step_loss:
                    for x in step_loss['debug'].view(-1):
                        bmt.print_rank(" | {:.4f}".format(x), end='')
                bmt.print_rank('')

                if args.save_step > 0 and global_step % args.save_step == 0:
                    filename = os.path.join(args.output_path, 'checkpoint_step{}.pt'.format(global_step))
                    bmt.save(model, filename)

                if (args.valid_step > 0 and global_step % args.valid_step == 0) or global_step % len(dataloader["train"]) == len(dataloader["train"]) - 1:# and args.dataset != "wiki":
                    valid_res = valid(model, dataloader, start_time, args)
                    if valid_res > best_valid_res and args.mode=='train':
                        best_valid_res = valid_res
                        filename = os.path.join(args.output_path, 'best_model.pt')
                        bmt.save(model, filename)

                if global_step >= args.training_step:
                    break

                step_time = 0
                for k in step_loss.keys():
                    step_loss[k] = 0.

    filename = os.path.join(args.output_path, 'checkpoint_step{}.pt'.format(global_step))
    bmt.save(model, filename)

def main():
    args = getargs()
    random.seed(args.random_seed)
    torch.manual_seed(args.random_seed)
    bmt.init_distributed(seed=args.random_seed)

    model, dataloader, optim_manager, optimizer, lr_scheduler, global_step = init(args)

    bmt.print_rank('=================')
    bmt.print_rank('begin')
    bmt.print_rank('=================')

    if args.mode == 'valid':
        valid(model, dataloader, time.time(), args)
    else:
        train(model, dataloader, optim_manager, optimizer, lr_scheduler, global_step, args)

if __name__ == '__main__':
    main()
