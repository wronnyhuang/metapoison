print('loading modules')
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'  # silence tensorflow
from comet_ml import Experiment, API
import comet_ml
import tensorflow as tf
from parse import get_parser
from meta import Meta
from data import *
from utils import *
import pickle
import json
from time import time, sleep
from mpi4py import MPI
import warnings
import multiprocessing  # Just for threadcounting in rank0
from subprocess import Popen, STDOUT, PIPE
import socket

# initialize mpi
mpi = MPI.COMM_WORLD
nproc, rank = mpi.Get_size(), mpi.Get_rank()
localrank = int(os.environ.get('OMPI_COMM_WORLD_LOCAL_RANK', rank))

parser = get_parser()
args = parser.parse_args()
args.gpu = set_available_gpus(args)
ncpu = multiprocessing.cpu_count()
print('==> Rank {}/{}, localrank {}, host {}, GPU {}/{}, nCPUs {}'.format(rank, nproc, localrank, socket.gethostname(), localrank % len(args.gpu), len(args.gpu), ncpu))
args.nproc = nproc
args.nmeta = args.nproc * args.nreplay
args.maxepoch = args.nmeta * args.stagger

def craft():
    # comet initialization
    weightapi = API()
    experiment = Dummy()
    if rank == 0:
        experiment = Experiment(project_name=args.craftproj, auto_param_logging=False, auto_metric_logging=False, parse_args=False)
        comet_ml.config.experiment = None
        experiment.add_tag(args.tag)
        experiment.log_parameters(vars(args))
        experiment.log_other('command', print_command_and_args(args))
        print(f'Logged uid: {experiment.get_parameter("uid")}, expt key: {experiment.get_key()}')

    def comet_log_poison():
        poisoninputs = sess.run(meta.poisoninputs, {meta.poisonmask: [True] * args.npoison})
        # log poison assets for victim eval
        fname = str(time()).replace('.', '')
        with open(fname, 'wb') as f:
            pickle.dump(poisoninputs, f)
        experiment.log_asset(fname, file_name='poisoninputs-{}'.format(craftstep), step=craftstep)
        os.remove(fname)
        # log poison figures
        npoison_to_display = 10
        for i in np.linspace(0, args.npoison - 1, npoison_to_display, dtype=int):
            imagesc(poisoninputs[i], title='poison-{}'.format(i), experiment=experiment, step=craftstep, scale=[0, 255])
            imagesc(poisoninputs[i] - xbase[i], title='perturb-{}'.format(i), experiment=experiment, step=craftstep, scale=127.5)
        for i in range(len(xtarget)):
            imagesc(xtarget[i], title='target-{}'.format(i), experiment=experiment, step=craftstep, scale=[0, 255])

    def restart_poison():
        perts = np.random.uniform(-args.eps, args.eps, xbase.shape)
        perts = np.clip(xbase + perts, 0, 255) - xbase
        mpi.Bcast(perts, root=0)
        meta.restart_poison(perts, sess)

    def log_epoch_results(resMs, resLs, craftstep):
        resMgather = mpi.gather(resMs, root=0)
        resLgather = mpi.gather(resLs, root=0)
        if rank == 0:
            resMgather = sum(resMgather, []) # flattens the list of lists
            resLgather = sum(resLgather, [])
            resM, resL = avg_n_dicts(resMgather), avg_n_dicts(resLgather)
            experiment.log_metrics(resM, step=craftstep)
            experiment.log_metrics(resL, step=craftstep)
            [experiment.log_metric(f'xent{i}', r['xent'], step=craftstep) for i, r in enumerate(resLgather)]
            # [experiment.log_metric(f'epoch{i}', r['epoch'], step=craftstep) for i, r in enumerate(resLgather)]
            print(' | '.join(['craftstep {}'.format(craftstep)] + ['elapsed {}'.format(round(time() - tic, 3))] +
                             ['{} {}'.format(key, round(val, 2)) for key, val in resM.items()]))

    def dock_weights_and_buffers(epoch, craftstep):
        if epoch == 0: # randomly initialize
            meta.init_weights(sess, pretrain_weights)
        elif craftstep == 0: # train or load from weightset to correct epoch
            if args.weightsettrain or args.weightset == '': train(epoch)
            else: meta.load_weights(sess, comet_pull_weight(epoch, weightapi, args, rank))
        else: # restore weights from previous replay
            meta.cache_weights(sess, cache=f'replay-{replay}', restore=True)

    print('==> begin crafting poisons on rank {}'.format(rank))
    for craftstep in range(args.ncraftstep):
        # auxiliary tasks
        tic = time()
        if not craftstep % args.restartperiod: restart_poison()
        if not craftstep % args.logperiod and rank == 0: comet_log_poison()
        craftrate = cr_schedule(args.craftrate, craftstep, schedule=[i * args.crdropperiod for i in [1, 2]])

        resMs, resLs = [], []
        for replay in range(args.nreplay):
            epoch = ((rank + replay * args.nproc) * args.stagger + craftstep) % (args.nmeta * args.stagger)
            lrnrate = lr_schedule(args.lrnrate, epoch, args.warmupperiod)
            dock_weights_and_buffers(epoch, craftstep)

            # iterate through all batches in epoch
            for craftfeed, trainfeed, hasPoison in feeddict_generator(xtrain, ytrain, lrnrate, meta, args):
                if args.trajectory == 'clean':
                    if hasPoison: _, resM, = sess.run([meta.accumop, meta.resultM, ], craftfeed)
                    _, resL, = sess.run([meta.trainop, meta.resultL, ], trainfeed)
                elif args.trajectory == 'poison':
                    _, _, resM, resL, = sess.run([meta.accumop, meta.trainop, meta.resultM, meta.resultL, ], craftfeed)
            meta.cache_weights(sess, cache=f'replay-{replay}')
            resL.update(dict(epoch=epoch, craftrate=craftrate))
            resMs.append(resM); resLs.append(resL)

        avg_metagrads = []
        for metagrad_accum in sess.run(meta.metagrads_accum):
            avg_metagrad = np.zeros_like(metagrad_accum)
            mpi.Allreduce(metagrad_accum, avg_metagrad, op=MPI.SUM)
            avg_metagrads.append(avg_metagrad / args.nmeta)
        sess.run([meta.craftop,], {meta.avg_metagrads: tuple(avg_metagrads), meta.craftrate: craftrate})
        log_epoch_results(resMs, resLs, craftstep)
    experiment.send_notification(f'{args.tag} finished', 'finished')
    experiment.end()

    if not args.skipvictim:
        print('==> crafting finished. begin victim.')
        meta.init_weights(sess, pretrain_weights)
        from victim import victim
        argsmod = dict(craftsteps=[craftstep], ntrial=1, Xtag=None)
        kwargs = dict(argsmod=argsmod, sess=sess, meta=meta, xtrain=xtrain, ytrain=ytrain, xvalid=xvalid, yvalid=yvalid, xbase=xbase, ybase=ybase, xtarget=xtarget, ytarget=ytarget, ytargetadv=ytargetadv)
        victim(kwargs)


def train(nepoch):
    tic = time()
    expt = Dummy()
    if args.weightset != '':
        expt = Experiment(project_name=f'weightset-{args.net}-{args.weightset}'.lower(), parse_args=False,
                          auto_param_logging=False, auto_metric_logging=False, log_git_patch=False, log_git_metadata=False)
        comet_ml.config.experiment = None
        expt.log_parameters(vars(args))
        expt.log_other('nepoch', nepoch)
        expt.add_tag(args.tag)

    # train and valid
    meta.init_weights(sess, pretrain_weights) # reinitialize weights and buffers
    print(f'==> begin vanilla train on rank {rank} to epoch {nepoch} at expt {expt._get_experiment_url()}')
    for epoch in range(nepoch):
        lrnrate = lr_schedule(args.lrnrate, epoch, args.warmupperiod)
        for _, trainfeed, _ in feeddict_generator(xtrain, ytrain, lrnrate, meta, args):
            _, resL, = sess.run([meta.trainop, meta.resultL, ], trainfeed)
        expt.log_metrics(resL, step=epoch)
    resVs = [] # begin validation
    for _, validfeed, _ in feeddict_generator(xvalid, yvalid, lrnrate, meta, args, valid=True):
        resV, = sess.run([meta.resultV, ], validfeed)
        resVs.append(resV)
    expt.log_metrics(avg_n_dicts(resVs), step=epoch)
    
    # log weights and buffers of final trained model to comet
    comet_log_asset_weights_and_buffers(nepoch, expt, meta, sess)
    print(' | '.join(['trained to {}'.format(nepoch)] +
                     ['total time {}'.format(round(time() - tic, 3))] +
                     ['{} {}'.format(key, int(val * 100) / 100.) for key, val in resL.items() if 'class' not in key]))
    expt.end()


if __name__ == '__main__':
    # load data and build graph
    print('==> loading data on rank {}'.format(rank))
    xtrain, ytrain, xvalid, yvalid, xbase, ybase, xtarget, ytarget, ytargetadv = load_and_apportion_data(mpi, args)
    print('==> building graph on rank {}'.format(rank))
    meta = Meta(args, xbase, ybase, xtarget, ytarget, ytargetadv)

    # start tf session and initialize variables
    print('==> initializing tf session on rank {}'.format(rank))
    gpu_options = tf.GPUOptions(allow_growth=True, visible_device_list=str(localrank % len(args.gpu)))
    sess = tf.Session(config=tf.ConfigProto(allow_soft_placement=True, gpu_options=gpu_options))
    pretrain_weights = meta.global_initialize(args, sess)
    sess.graph.finalize()

    # begin
    if args.justtrain == 0: craft()
    else: train(args.justtrain)
