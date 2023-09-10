from mmseg.datasets import build_dataloader, build_dataset
import mmcv
import argparse
import os
import pdb


def parse_args():
    parser = argparse.ArgumentParser(
        description='mmseg test (and eval) a model')
    parser.add_argument('config', help='test config file path')
    parser.add_argument(
        '--out', help='out dir')
    parser.add_argument(
        '--eval',
        type=str,
        nargs='+',
        help='evaluation metrics, which depends on the dataset, e.g., "mIoU"'
             ' for generic datasets, and "cityscapes" for Cityscapes')
    args = parser.parse_args()

    return args


args = parse_args()
cfg = mmcv.Config.fromfile(args.config)  # args.config
dataset = build_dataset(cfg.data.test)

outputs = []
files = os.listdir(args.out)
for file in files:
    tmp = mmcv.load(os.path.join(args.out, file))
    outputs+=tmp
# pdb.set_trace()
kwargs = {}
dataset.evaluate(outputs, args.eval, **kwargs)
