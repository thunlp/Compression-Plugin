#!/bin/sh


#CPU_LIST="32-63"

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
batch_size=32
grad_accumulation=1
model_size=$1

lr=1e-3
mux_num=4

OUTPATH=checkpoints/wiki-T5-${model_size}/
mkdir -p ${OUTPATH}

OPT=""
OPT+=" --random-seed 10018"
OPT+=" --mode ${mode}"
OPT+=" --dataset wiki"
OPT+=" --model t5"
OPT+=" --model-config t5-${model_size}"
OPT+=" --data-path data/wikicorpus_128/"


OPT+=" --distil hidden"

# OPT+=" --mlm-prob 0.15"
OPT+=" --max-input-length 224"
OPT+=" --epochs 5"
OPT+=" --batch-size ${batch_size}"
OPT+=" --grad-accumulation ${grad_accumulation}"
OPT+=" --clip-grad 10"
OPT+=" --lr ${lr}"
OPT+=" --weight-decay 1e-5"
OPT+=" --warmup-step 50"
OPT+=" --training-step 60000"
OPT+=" --valid-step 2000"
OPT+=" --mux-num ${mux_num}"

OPT+=" --save-step 2000"
OPT+=" --output-path ${OUTPATH}"


CMD="torchrun ${DISTRIBUTED_ARGS} train_t5.py ${OPT}"
echo $CMD
if [ $mode = "train" ]; then 
$CMD  | tee -a ${OUTPATH}/log.log
else 
$CMD
fi
