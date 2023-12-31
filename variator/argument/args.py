import argparse

def getargs():
    parser = argparse.ArgumentParser()
    parser.add_argument('--local_rank', type=int, help='local rank', default=-1)
    parser.add_argument('--mode', type=str, default='train')
    parser.add_argument('--dataset', type=str, required=True)
    parser.add_argument('--model', type=str, default="t5")
    parser.add_argument('--model-config', type=str, required=True)

    parser.add_argument('--checkpoint', type=str)
    parser.add_argument('--max-input-length', type=int, default=512)
    parser.add_argument('--epochs', type=int, default=5)
    parser.add_argument('--batch-size', type=int, default=16)
    parser.add_argument('--grad-accumulation', type=int, default=1)
    parser.add_argument('--clip-grad', type=float, default=10)
    parser.add_argument('--lr', type=float, default=1e-5)
    parser.add_argument('--weight-decay', type=float, default=1e-5)
    parser.add_argument('--warmup-step', type=int, default=1)
    parser.add_argument('--training-step', type=int, default=10000000)
    parser.add_argument('--inspect-step', type=int, default=1000)
    parser.add_argument('--valid-step', type=int, default=-1)
    parser.add_argument('--save-step', type=int, default=-1)
    parser.add_argument('--output-path', type=str)
    parser.add_argument('--random-seed', type=int, default=1)
    parser.add_argument('--mux-num', type=int, default=4)
    parser.add_argument('--data-path', type=str, required=True)
    parser.add_argument('--plugin-path', type=str)
    parser.add_argument('--task-ratio', type=float, default=1.0)
    parser.add_argument('--delta-tuning', action="store_true")
    parser.add_argument('--student-config', type=str, default=None)
    parser.add_argument('--ctx-length', type=int, default=None)
    parser.add_argument('--target-length', type=int, default=None)

    parser.add_argument('--distil', type=str, default='no')
    return parser.parse_args()
