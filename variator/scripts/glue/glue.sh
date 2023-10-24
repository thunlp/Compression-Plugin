#!/bin/sh

MASTER_ADDR=localhost
MASTER_PORT=34894
NNODES=1
NODE_RANK=0
GPUS_PER_NODE=2

DISTRIBUTED_ARGS="--nproc_per_node $GPUS_PER_NODE \
                  --nnodes $NNODES \
                  --node_rank $NODE_RANK \
                  --master_addr $MASTER_ADDR \
                  --master_port $MASTER_PORT"


mode="train"

model_size=$1
dataset=$2

LR=1e-5
BATCHSIZE=32

OUTPATH=checkpoints/${dataset}-T5/
mkdir -p ${OUTPATH}

seed=10018

if [ $dataset = "mnli" ]; then
max_len=256
total_step=26000
valid_step=2000
fi

if [ $dataset = "rte" ]; then
max_len=256
total_step=26000 # larger than 5 epoches, will only run 5 epoches
valid_step=-1
BATCHSIZE=16
fi

if [ $dataset = "sst2" ]; then
max_len=96
total_step=26000
valid_step=1000
fi

if [ $dataset = "qqp" ]; then
max_len=256
total_step=26000
valid_step=1000
fi


if [ $dataset = "mrpc" ]; then
max_len=96
total_step=26000
valid_step=1000
fi

if [ $dataset = "qnli" ]; then
max_len=356
total_step=26000
valid_step=1000
BATCHSIZE=16
fi

data_path=data/${dataset}/data/

if [ $dataset = "squad" ]; then
max_len=512
total_step=26000
valid_step=100
data_path=data/SQuAD
BATCHSIZE=16
fi

batch_size=$[$BATCHSIZE/$GPUS_PER_NODE]


OPT=""
OPT+=" --random-seed ${seed}"
OPT+=" --mode ${mode}"
OPT+=" --dataset ${dataset}"
OPT+=" --model-config /data/xiaochaojun/PLMs/model-center/t5-${model_size}"
OPT+=" --data-path ${data_path}"

OPT+=" --epochs 5"
OPT+=" --lr ${LR}"
OPT+=" --weight-decay 1e-5"
OPT+=" --warmup-step 50"
OPT+=" --distil no"


OPT+=" --max-input-length ${max_len}"
OPT+=" --batch-size ${batch_size}"
OPT+=" --training-step ${total_step}"
OPT+=" --valid-step ${valid_step}"

OPT+=" --output-path ${OUTPATH}"

CMD="torchrun ${DISTRIBUTED_ARGS} train_t5.py ${OPT}"
echo $CMD
if [ $mode = "train" ]; then 
$CMD 2>&1 | tee ${OUTPATH}/log-${seed}.log
else 
$CMD
fi
