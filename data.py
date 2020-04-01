import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'  # silence tensorflow
from tensorflow.keras.datasets import cifar10
import tensorflow as tf
from time import time
import numpy as np
from collections import deque
import socket
from random import choice, choices
import os
from mpi4py import MPI
mpi = MPI.COMM_WORLD

mpi = MPI.COMM_WORLD
rank = mpi.Get_rank()
localrank = int(os.environ.get('OMPI_COMM_WORLD_LOCAL_RANK', rank))

# poisonids = targetids = None

def load_and_apportion_data(mpi, args):

    nperclass = int(args.nbatch * args.batchsize // 10)
    nperclassvalid = 1000
    ntrain = nperclass * 10
    nvalid = nperclassvalid * 10

    # prevent race condition by download dataset on local root of each node while others wait
    localrootrank = None
    if localrank == 0:
        # print('Loading (maybe downloading) CIFAR on local_rank {} of {}'.format(localrank, socket.gethostname()))
        (xtrain, ytrain), (xvalid, yvalid) = truncated_cifar10(nperclass, nperclassvalid, args)
        localrootrank = rank
    gatherresult = mpi.gather(localrootrank, root=0)
    mpi.bcast(gatherresult, root=0)
    # if rank == 0: print('Ranks that loaded first because they were the locally rank 0: ', gatherresult)
    if localrank != 0:
        (xtrain, ytrain), (xvalid, yvalid) = truncated_cifar10(nperclass, nperclassvalid, args)

    # get and save poison and target indices to global scope
    assert len(args.targetids) > 0, f'args.targetids is of length 0'
    args.ntarget = len(args.targetids)
    
    global poisonids, targetids
    poisonids = list(range(nperclass * args.poisonclass, nperclass * args.poisonclass + args.npoison))
    alltargetids = np.array(range(nperclassvalid * args.targetclass, nperclassvalid * args.targetclass + nperclassvalid))
    targetids = alltargetids[args.targetids]

    # apportion data to base, target
    xbase = xtrain[poisonids]
    ybase = ytrain[poisonids]
    xtarget = xvalid[targetids]
    ytarget = yvalid[targetids]
    if args.watermark: xbase = 0.7 * xbase + 0.3 * xtarget[0]
    
    # assign the adversarial label
    if args.ytargetadv == -1: ytargetadv = [args.poisonclass] * args.ntarget
    else: ytargetadv = [args.ytargetadv] * args.ntarget # if ytargetadv -1 then default to base class
    ytargetadv = np.array(ytargetadv)
    
    return xtrain, ytrain, xvalid, yvalid, xbase, ybase, xtarget, ytarget, ytargetadv


def truncated_cifar10(nperclass, nperclassvalid, args):

    # load dataset
    (xtrain, ytrain), (xvalid, yvalid) = cifar10.load_data()

    ytrain, yvalid = np.squeeze(ytrain), np.squeeze(yvalid)
    # datamean, datastd = np.array((0.4914, 0.4822, 0.4465)), np.array((0.2023, 0.1994, 0.2010))

    # take first n examples from train
    inputs, labels = [], []
    counts = [0 for _ in range(10)]
    for x, y in zip(xtrain, ytrain):
        if all([count == nperclass for count in counts]):
            break
        if counts[y] < nperclass:
            # x = x / 255
            # x = (x - datamean) / datastd
            inputs.append(x)
            labels.append(y)
            counts[y] += 1
    order = sorted(range(len(labels)), key=lambda i: labels[i])
    xtrain, ytrain = np.array(inputs, dtype='float32'), np.array(labels)

    # sort the training examples in order of their label. baseline for case if same class for all source images is desired (poisonfrog setting)
    inputs = [inputs[o] for o in order]
    labels = [labels[o] for o in order]
    xtrain, ytrain = np.array(inputs, dtype='float32'), np.array(labels)

    # optional: tile the training examples in groups of 0-9 and repeat. useful for multi-class poison attack
    if args.multiclasspoison:
        xtrain = xtrain.reshape(10, nperclass, *xtrain.shape[1:]).transpose(1, 0, 2, 3, 4).reshape(*xtrain.shape)
        ytrain = ytrain.reshape(10, nperclass).T.reshape(ytrain.shape)
        args.poisonclass = 0 # just a dummy number since all class have poisons now

    # take first n examples from validation
    inputs, labels = [], []
    counts = [0 for _ in range(10)]
    for x, y in zip(xvalid, yvalid):
        if all([count == nperclassvalid for count in counts]):
            break
        if counts[y] < nperclassvalid:
            # x = x / 255
            # x = (x - datamean) / datastd
            inputs.append(x)
            labels.append(y)
            counts[y] += 1
    order = sorted(range(len(labels)), key=lambda i: labels[i])
    xvalid, yvalid = np.array(inputs, dtype='float32'), np.array(labels)

    # sort the testing examples in order of their label. baseline for case if same class for all source images is desired (poisonfrog setting)
    inputs = [inputs[o] for o in order]
    labels = [labels[o] for o in order]
    xvalid, yvalid = np.array(inputs, dtype='float32'), np.array(labels)
    return (xtrain, ytrain), (xvalid, yvalid)


def make_mask(corpusids, npoison):
    global poisonids, targetids
    setpoisonids = set(poisonids)
    setcorpusids = set(corpusids)
    cleanmask =  np.array([i not in setpoisonids for i in corpusids])
    poisonmask = np.array([i in setcorpusids for i in poisonids])
    return poisonmask, cleanmask


def batch_generator(x, y, batchsize, npoison=False, drop_last=True):
    maybe_batchsize = batchsize if drop_last else 0
    permuted = np.random.permutation(len(x))
    i = 0
    while i + maybe_batchsize <= len(permuted):
        corpusids = permuted[i:i + batchsize]
        inputs = x[corpusids]
        labels = y[corpusids]
        if npoison is not False:
            poisonmask, cleanmask = make_mask(corpusids, npoison)
            yield (inputs, labels), cleanmask, poisonmask
        else: yield inputs, labels
        i += batchsize


def batch_queuer(generator, queuesize, nbatch):
    initbatches = [generator.__next__() for i in range(queuesize)]
    assert len(initbatches) == queuesize, 'too few data for generator'
    queue = deque(initbatches)
    for i in range(nbatch):
        yield tuple(queue)
        queue.popleft()
        if i < nbatch - queuesize:
            queue.append(generator.__next__())
        elif i >= nbatch - queuesize:
            queue.append(initbatches[i - nbatch + queuesize])


def feeddict_generator(x, y, lrnrate, meta, args, victim=False, valid=False):
    nbatch = len(x) // args.batchsize
    epochmass = min((epoch / 5) ** 2, 1) if args.epochmass else 1
    for queue in batch_queuer(batch_generator(x, y, args.batchsize, args.npoison), args.nadapt + 1, nbatch):
        trains, cleanmasks, poisonmasks = zip(*queue)
        if not victim:
            craftfeed = {meta.trains: trains,
                         # poisons only in first step, no poisons in unroll steps thereafter
                         meta.cleanmask: cleanmasks[0],
                         meta.poisonmask: poisonmasks[0],
                         meta.lrnrate: lrnrate,
                         meta.epochmass: epochmass,
                         }
            trainfeed = {meta.trains[0]: trains[0],  # no unrolling needed and no poisons
                         meta.lrnrate: lrnrate,
                         meta.augment: False if valid else args.augment # force-disable augmentation during validation
                         }
            assert len(meta.trains) == len(trains), 'length of tensor meta.trains isnt same length as the trains data'
            yield craftfeed, trainfeed, any(poisonmasks[0])
        else:
            victimfeed = {meta.trains[0]: trains[0],
                          # poisons in first step, which is the only step, since there's no unrolling in victim
                          meta.cleanmask: cleanmasks[0],
                          meta.poisonmask: poisonmasks[0],
                          meta.lrnrate: lrnrate,
                          }
            yield victimfeed


def tf_standardize(inputs, datamean, datastd):
    with tf.variable_scope('standardize'):
        return ((inputs / 255) - datamean) / datastd


def toy_data(args):

    # make clean data
    X1 = np.random.randn(args.batchsize // 2, 2) / 2 + 2
    X2 = np.random.randn(args.batchsize // 2, 2) / 2 - 2
    X = np.concatenate([X1, X2], axis=0)
    Y = np.concatenate([np.ones((args.batchsize // 2, 1)), np.zeros((args.batchsize // 2, 1))])
    train = (X, Y)

    # make valid data
    X = np.array([[2, -3]])
    Y = np.zeros((1, 1))
    valid = (X, Y)

    return train, valid
