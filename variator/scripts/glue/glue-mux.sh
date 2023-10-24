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
mux_num=4
model_size=$1
dataset=$2
# batch_size=$3
seed=10018
LR=1e-5
BATCHSIZE=32


PLUGINPATH=checkpoints/plugins/plugin-t5_${model_size}-mux_4.pkl
DATAPATH=data/${dataset}/

OUTPATH=checkpoints/${dataset}-T5-mux/
mkdir -p ${OUTPATH}

total_step=26000
valid_step=1000
if [ $dataset = "mnli" ]; then
max_len=256
valid_step=2000
fi

if [ $dataset = "rte" ]; then
max_len=256
valid_step=-1
fi

if [ $dataset = "sst2" ]; then
max_len=96
fi

if [ $dataset = "qqp" ]; then
max_len=256
fi

if [ $dataset = "mrpc" ]; then
max_len=96
fi

if [ $dataset = "qnli" ]; then
max_len=356
fi

batch_size=$[$BATCHSIZE/$GPUS_PER_NODE]

OPT=""
OPT+=" --random-seed ${seed}"
OPT+=" --mode ${mode}"
OPT+=" --dataset ${dataset}"
OPT+=" --model-config t5-${model_size}"
OPT+=" --data-path ${DATAPATH}"

OPT+=" --checkpoint checkpoints/${dataset}-T5/best_model.pt"
OPT+=" --plugin-path ${PLUGINPATH}"


OPT+=" --epochs 50"
OPT+=" --lr ${LR}"
OPT+=" --weight-decay 1e-5"
OPT+=" --warmup-step 50"
OPT+=" --mux-num ${mux_num}"
OPT+=" --distil hidden"

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
