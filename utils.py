import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
from os.path import join, basename
import tensorflow as tf
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from time import sleep
from random import choice
import numpy as np
import pickle
import subprocess
from utils import *
from time import time
from datetime import datetime
from tensorflow.keras.losses import categorical_crossentropy
from functools import reduce
from comet_ml.query import Metric, Metadata, Parameter, Tag, Other
import comet_ml
import sys
from mpi4py import MPI

mpi = MPI.COMM_WORLD
nproc, rank = mpi.Get_size(), mpi.Get_rank()
localrank = int(os.environ.get('OMPI_COMM_WORLD_LOCAL_RANK', rank))
api = comet_ml.api.API()

def read_comet_config():
    with open('.comet.config', 'r') as f:
        lines = f.readlines()
    config = {}
    for line in lines[1:]:
        key, val = line.split('=')
        val = val.replace('\n', '')
        config[key] = val
    return config

cometconfig = read_comet_config()

def count_available_gpus():
    return str(subprocess.check_output(["nvidia-smi", "-L"])).count('UUID')


def _get_basename(name):
    name = '/'.join(name.split('/')[2:])
    return name.split(':')[0]


def _reshape_labels_like_logits(labels, logits, batchsize, nclass=10):
    return tf.reshape(tf.one_hot(labels, nclass), [batchsize, nclass])


def metrics(labels, logits, batchsize, reverse_ce=False):
    with tf.variable_scope('metrics'):
        labels_reshaped = _reshape_labels_like_logits(labels, logits, batchsize)
        if not reverse_ce:
            xent = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits_v2(
                labels=labels_reshaped, logits=logits), name='xent')
        else:
            preds = tf.nn.softmax(logits)
            xent = tf.reduce_mean(categorical_crossentropy(labels_reshaped, 1 - preds))
        equal = tf.equal(labels, tf.cast(tf.argmax(logits, axis=1), dtype=labels.dtype))
        acc = tf.reduce_mean(tf.to_float(equal), name='acc')
    return xent, acc


def carlini(labels, logits, batchsize, clamp=-100):
    with tf.variable_scope('carlini'):
        labels_reshaped = _reshape_labels_like_logits(labels, logits, batchsize)
        labels_reshaped = tf.cast(labels_reshaped, dtype=logits.dtype)
        target_logit = tf.reduce_sum(logits * labels_reshaped, axis=1)
        second_logit = tf.reduce_max(logits - logits * labels_reshaped, axis=1)
        cw_indiv = tf.maximum(second_logit - target_logit, clamp)
        # return tf.maximum(second_logit - target_logit, clamp)  # , target_logit, second_logit, tmp
        return tf.reduce_mean(cw_indiv)  # , target_logit, second_logit, tmp


def count_params_in_scope():
    scope = tf.get_default_graph().get_name_scope()
    nparam = sum([np.prod(w.shape.as_list()) for w in tf.trainable_variables(scope)])
    # print('scope:', scope, '#params', nparam)
    return nparam


def imagesc(img, title=None, experiment=None, step=None, scale='minmax'):
    if scale == 'minmax':
        img = img - img.ravel().min()
        img = img / img.ravel().max()
    elif type(scale) is float or type(scale) is int:  # good for perturbations
        img = img * .5 / scale + .5
    elif type(scale) is list or type(scale) is tuple:  # good for images
        assert len(scale) == 2, 'scale arg must be length 2'
        lo, hi = scale
        img = (img - lo) / (hi - lo)
    plt.clf()
    plt.imshow(img)
    if title:
        plt.title(title)
    if experiment:
        experiment.log_figure(figure_name=title, step=step)


def pgdstep(img, grad, orig, stepsize=.01, epsilon=.08, perturb=False):
    if perturb: img += (np.random.rand(*img.shape) - .5) * 2 * epsilon
    img += stepsize * np.sign(grad)
    img = np.clip(img, orig - epsilon, orig + epsilon)
    img = np.clip(img, 0, 255)
    return img


def l2_weights(weights):
    return tf.add_n([tf.reduce_sum(weight ** 2) for weight in weights.values() if len(weight.shape.as_list()) > 1])


def tf_preprocess(inputs, batchsize):
    print('data augmentation ON')
    # preprocessing data augmentation
    inputs = tf.pad(inputs, [[0, 0], [4, 4], [4, 4], [0, 0]])
    inputs = tf.random_crop(inputs, [batchsize, 32, 32, 3])
    inputs = tf.map_fn(tf.image.random_flip_left_right, inputs)
    return inputs


def avg_n_dicts(dicts, experiment=None, step=None):
    # given a list of dicts with the same exact schema, return a single dict with same schema whose values are the
    # key-wise average over all input dicts
    means = {}
    for dic in dicts:
        for key in dic:
            if key not in means: means[key] = 0
            means[key] += dic[key] / len(dicts)
    if experiment is not None:
        experiment.log_metrics(means, step=step)
    return means

def merge_n_dicts(dicts):
    # given a list of dicts with mutually exclusive schema, return a dict of all key-value pairs merged
    out = {}
    for d in dicts:
        if d is not None:
            out.update(d)
    return out


def plot_dict_series(dict_series, prefix=None, experiment=None, step=None):
    # given a list of dicts with the same schema, make a series plot for each key in the schema
    # if dict_series is a list of list of dicts, then overlap all plots in the second nested list
    serialized = {}
    for i, timestep in enumerate(dict_series):
        if type(timestep) is dict: timestep = [timestep]
        for dic in timestep:
            for key, val in dic.items():
                if key not in serialized: serialized[key] = []
                if len(serialized[key]) <= i: serialized[key].append([])
                serialized[key][-1].append(val)
    for key, series in serialized.items():
        plt.clf()
        plt.plot(np.array(series))
        plt.title('step {}'.format(step))
        plt.ylabel(key)
        if experiment is not None: experiment.log_figure(figure_name='{}_{}'.format(prefix, key), step=step)


def copy_to_args_from_experiment(args, craftexpt, attrs):
    # given a comet experiment and an args namespace, copy the values of all attributes in attrs from experiment to args
    for param in craftexpt.get_parameters_summary():
        str2bool = dict(true=True, false=False)
        # attrs is a list of attributes that you want to copy over
        if param['name'] in attrs:
            if rank == 0: print(f'from craftexpt copying {param["name"]}: {getattr(args, param["name"])} -> {param["valueCurrent"]}')
            if param['valueCurrent'] == 'null': setattr(args, param['name'], None)
            elif type(getattr(args, param['name'])) is bool: setattr(args, param['name'], str2bool[param['valueCurrent']])
            elif type(getattr(args, param['name'])) is int: setattr(args, param['name'], int(float(param['valueCurrent'])))
            elif type(getattr(args, param['name'])) is str: setattr(args, param['name'], str(param['valueCurrent']))
            elif type(getattr(args, param['name'])) is float: setattr(args, param['name'], float(param['valueCurrent']))
            elif type(getattr(args, param['name'])) is list: setattr(args, param['name'], eval(param['valueCurrent']))
            else: raise Exception('there is an arg that is not of the typical types nor None. This could happen if this arg\'s default in parse.py is None. Fix it')
    return args


def comet_pull_weight(epoch, api, args, rank, deterministic=True):
    for attempt in range(4):
        try:
            tic = time()
            projname = f'{cometconfig["workspace"]}/weightset-{args.net}-{args.weightset}'.lower()
            expts = api.get(projname)
            expts = [expt for expt in expts if int(expt.get_others_summary('nepoch')[0]) == epoch]
            expt = expts[0] if deterministic else choice(expts)
            assets = expt.get_asset_list()
            assert len(assets) == 1, f'{len(assets)} assets found at {expt._get_experiment_url()}. There should only be 1 (the weights)'
            asset = assets[0]
            weights0 = pickle.loads(expt.get_asset(asset['assetId']))
            print(f'rank {rank}: {asset["fileName"]} epoch {epoch} pulled from {expt._get_experiment_url()} in {time() - tic} sec')
            return weights0
        except:
            print(f'comet pull weightset failed: attempt {attempt} epoch {epoch} rank {rank} from https://www.comet.ml/{projname}. Error: {sys.exc_info()}')
            sleep(2)
    raise Exception('failed to pull weights')


def comet_pull_weight_by_key(key, projname, epoch, api, rank, deterministic=True):
    for attempt in range(4):
        try:
            tic = time()
            expt = api.get(cometconfig["workspace"], projname, key)
            assets = expt.get_asset_list()
            assets = assets2dict(assets, 'fileName', 'assetId')
            try:
                weights0 = pickle.loads(expt.get_asset(assets[f'weights0-{epoch}']))
            except:
                name, assetid = assets.popitem()
                print(f'warning: weights0-{epoch} not found. instead, will load {name}')
                weights0 = pickle.loads(expt.get_asset(assetid))
            print(f'rank {rank}: weights0-{epoch} epoch {epoch} pulled from {expt._get_experiment_url()} in {time() - tic} sec')
            return weights0
        except:
            print(f'comet pull weightset failed on attempt {attempt} trying to extract epoch {epoch} at rank {rank}')
            sleep(2)
    raise Exception('failed to pull weights')


def comet_log_asset_weights_and_buffers(epoch, expt, meta, sess):
    file = str(time()).replace('.', '')
    with open(file, 'wb') as f:
        pickle.dump(sess.run((meta.weights0, meta.buffers0)), f)
    expt.log_asset(file, file_name=f'weights0-{epoch}', step=epoch)
    os.remove(file)

def comet_log_asset(experiment, name, asset, step=None):
    fname = str(time()).replace('.', '')
    with open(fname, 'wb') as f:
        pickle.dump(asset, f)
    experiment.log_asset(fname, file_name=name, step=step)
    os.remove(fname)


def transpose_list_of_lists(l):
    return list(map(list, zip(*l)))


def set_available_gpus(args):
    if len(args.gpu) > 0: os.environ['CUDA_VISIBLE_DEVICES'] = ','.join(map(str, args.gpu))
    else: args.gpu = list(range(count_available_gpus()))
    return args.gpu


class Dummy:
    def __getattribute__(self, attr):
        return lambda *arg, **kwarg: None

def lr_schedule(lrnrate, epoch, warmupperiod=5, schedule=[100, 150, 200], max_epoch=250):
    if schedule is None:
        schedule = [max_epoch // 2.667, max_epoch // 1.6, max_epoch // 1.142]
    warmupfactor = min(1, (epoch + 1) / (1e-6 + warmupperiod))
    if epoch < schedule[0]:
        return 1e00 * lrnrate * warmupfactor
    elif epoch < schedule[1]:
        return 1e-1 * lrnrate * warmupfactor
    elif epoch < schedule[2]:
        return 1e-2 * lrnrate * warmupfactor
    else:
        return 1e-3 * lrnrate * warmupfactor

def cr_schedule(craftrate, craftstep, warmupperiod=5, schedule=[20, 40]):
    warmupfactor = min(((craftstep + 1) / warmupperiod) ** 2, 1)
    if craftstep < schedule[0]:
        return 1e00 * craftrate * warmupfactor
    elif craftstep < schedule[1]:
        return 1e-1 * craftrate * warmupfactor
    else:
        return 1e-2 * craftrate * warmupfactor

def epochmass(epoch):
    return min(epoch / 5, 1)

def appendfeats(feats, feat, victimfeed, ybase, ytarget, batchsize):
    # feats is a defaultdict of type list which stores a 50000xNdim matrix of features for the entire dataset
    # feat is the minibatch of features to append
    cleaninputs, cleanlabels = [value for key, value in victimfeed.items() if 'adapter-0/cleaninputs' in str(key)][0]
    cleanmask = [value for key, value in victimfeed.items() if 'cleanmask' in str(key)][0]
    poisonmask = [value for key, value in victimfeed.items() if 'poisonmask' in str(key)][0]
    npoison = sum(poisonmask)
    feats['targetfeats'] = feat[batchsize:]
    feats['targetlabels'] = ytarget
    feats['cleanfeats'].extend(feat[npoison:batchsize])
    feats['poisonfeats'].extend(feat[:npoison])
    feats['cleanlabels'].extend(cleanlabels[cleanmask])
    feats['poisonlabels'].extend(ybase[poisonmask])

def get_featdist(feats):
    targetfeats, poisonfeats = feats['targetfeats'], feats['poisonfeats']
    targetfeat = np.array(targetfeats[:1])
    poisonfeats = np.array(poisonfeats)
    featdist = np.mean(np.linalg.norm(poisonfeats - targetfeat, axis=1))
    return featdist

def uid2craftkey(uid, craftproj):
    conditions = [Parameter('uid') == uid,
                  Metadata('duration') > 10,]
    for attempt in range(4):
        expts = api.query(cometconfig["workspace"], craftproj, reduce(lambda x, y: x & y, conditions))
        assert len(expts) > 0, f'{len(expts)} expts found for this uid'
        timestamp = [expt.end_server_timestamp for expt in expts]
        expt = expts[np.argmax(timestamp)]
        if len(expts) > 1: print(f'there were {len(expts)} expts with same uid. taking most recent one at {expt.url} '
                                 f'finishing at {datetime.fromtimestamp(expt.end_server_timestamp / 1e3 - 3600 * 5)}')
        return expt.id
    raise Exception(f'NO EXPTS FOUND WITH uid {uid} in craftproj {craftproj}')

def get_param(expt, param):
    return expt.get_parameters_summary(param)['valueCurrent']

def print_command_and_args(args):
    command = 'python ' + ' '.join(sys.argv)
    print(command)
    if rank == 0:
        print('\n'.join([f'{key} == {val}' for key, val in sorted(vars(args).items())]))
    return command
    
def assets2dict(assets, keystr, valuestr):
    # helps when doing comet api.get_something_summary() and it returns a list of dicts all with the attribute 'name'
    return {asset[keystr]: asset[valuestr] for asset in assets}

def tf_basename(tensor):
    name = tensor.name
    name = basename(name)
    if ':' in name:
        name = name.split(':')[0]
    return name

def trunc_decimal(val):
    if val > 1e10: return 'inf'
    return int(val * 100) / 100


def package_poisoned_dataset(poisoninputs, xtrain, ytrain, xtarget, ytarget, ytargetadv, xvalid, yvalid, args, craftstep):
    start = int(args.poisonclass / (max(ytrain) + 1) * len(xtrain))
    xtrain[start: start + args.npoison] = poisoninputs
    asset = dict(xtrain=xtrain, ytrain=ytrain, xtarget=xtarget, ytarget=ytarget, ytargetadv=ytargetadv, xvalid=xvalid, yvalid=yvalid)
    file = f'{args.poisondatasetfile}-{craftstep}.pkl'
    with open(file, 'wb') as f: pickle.dump(asset, f)
    print(f'argument -savepoisondataset is ON: poison dataset saved for expt {args.craftkey} craftstep {craftstep} at {file}')
    

def comet_log_asset_apiexpt(expt, fname, asset, step=None):
    with open(fname, 'wb') as f:
        pickle.dump(asset, f)
    expt.log_asset(filename=fname, step=step)
    os.remove(fname)



