import os
from datetime import datetime
import argparse
from typing import Optional
import tensorflow as tf
from utilucas.datasource.s3util import S3Util
from utilucas.datasource.s3loader import S3Loader


def label_smoothing(inputs: tf.Tensor, epsilon: float=0.1):
    last_dim = inputs.get_shape().as_list()[-1]
    return ((1 - epsilon) * inputs) + (epsilon / last_dim)


def get_args():
    argparser = argparse.ArgumentParser(description=__doc__)
    argparser.add_argument(
        '-c', '--config',
        metavar='C',
        default='None',
        help='The Configuration file')
    argparser.add_argument(
        '-d', '--date',
        default=datetime.now().strftime("%Y%m%d"),
        help='The version of model'
    )
    args = argparser.parse_args()
    return args


def get_args_inference():
    argparser = argparse.ArgumentParser()
    argparser.add_argument('-c', '--config',
                           default='None',
                           help='The Configuration File')
    argparser.add_argument('-s', '--sentences',
                           nargs='+',
                           help='Source sentences to reply')
    args = argparser.parse_args()
    return args


def get_args_mcts():
    argparser = argparse.ArgumentParser()
    argparser.add_argument('-p', '--policy',
                           required=True,
                           help='The Configuration File of Policy Network')
    argparser.add_argument('-r', '--reward',
                           required=True,
                           help='The Configuration File of Reward Network')
    argparser.add_argument('-s', '--sentences',
                           nargs='+',
                           help='Source sentences to reply')
    args = argparser.parse_args()
    return args


def get_args_alpha():
    argparser = argparse.ArgumentParser()
    argparser.add_argument('-c', '--config',
                           required=True,
                           help='The Configuration File of Alpha Network')
    argparser.add_argument('-i', '--task_index',
                           required=True,
                           type=int,
                           help='task_index for distributed tensorflow')
    argparser.add_argument('-d', '--date',
                           default=datetime.now().strftime("%Y%m%d"),
                           help='The version of model')
    args = argparser.parse_args()
    return args


def get_args_alpha_ps():
    argparser = argparse.ArgumentParser()
    argparser.add_argument('-c', '--config',
                           required=True,
                           help='The Configuration File of Alpha Network Parameter Server')
    argparser.add_argument('-i', '--task_index',
                           required=True,
                           type=int,
                           help='task_index for distributed tensorflow')
    args = argparser.parse_args()
    return args


def get_model_path(path: str) -> Optional[str]:
    '''
    TensorFlow のモデルパス(〜〜model.ckpt)を返します。
    - path が s3 path の場合：s3 から必要なファイルをダウンロードしローカルのパスを返します
    - path がローカルのファイルパスの場合：それをそのまま返します
    - path がローカルのディレクトリパスの場合：チェックポイントディレクトリとみなして、最新のものを返します。

    :param path: s3 パス、もしくはローカルのファイル・ディレクトリパス
    :return: モデルのパス（model.ckpt まで）
    '''
    if S3Util.is_s3_path(path):
        suffix_list = ['.index', '.meta', '.data-00000-of-00001', '.data-00000-of-00002', '.data-00001-of-00002']
        for suffix in suffix_list:
            try:
                s3_path = path + suffix
                S3Loader.clear_cache(s3_path)
                local_path = S3Loader.download(s3_path)
                model_path = local_path[:- len(suffix)]
            except Exception:
                print('{} does not exist.'.format(s3_path))
        return model_path
    if os.path.isdir(path):
        return tf.train.latest_checkpoint(path)  # may be None
    if os.path.isfile(path):
        return path
    raise Exception('Unknown path: {}'.format(path))
