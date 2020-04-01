print('loading modules victim')
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'  # silence tensorflow
from comet_ml import Experiment, API
import tensorflow as tf
from parse import get_parser
from meta import Meta
from data import *
from utils import *
import pickle
import json
from time import time, sleep
from socket import gethostname
from collections import defaultdict
from mpi4py import MPI

# initialize mpi
mpi = MPI.COMM_WORLD
nproc, rank = mpi.Get_size(), mpi.Get_rank()
localrank = int(os.environ.get('OMPI_COMM_WORLD_LOCAL_RANK', rank))

api = API()
weightapi = API()
cometconfig = read_comet_config()
parser, attrs = get_parser(True)
for exclude in ['gpu']: attrs.remove(exclude)
args = parser.parse_args()
args.craftkey = args.uid if len(args.uid) == 32 else uid2craftkey(args.uid, args.craftproj)
craftexpt = api.get_experiment(cometconfig["workspace"], args.craftproj, args.craftkey)
copy_to_args_from_experiment(args, craftexpt, attrs)
# if args.Xnvictimepoch > 0: args.nvictimepoch = args.Xnvictimepoch
# if args.Xntrial > 0: args.ntrial = args.Xntrial
if args.Xvictimproj is not None: args.victimproj = args.Xvictimproj
if args.Xtag is not None: args.tag = args.Xtag
if args.Xweightdecay: args.weightdecay = True
if args.Xaugment: args.augment = True
if args.Xbatchsize:
    args.batchsize *= 2
    args.nbatch /= 2
if args.Xlrnrate: args.lrnrate *= 2
if args.Xschedule: args.schedule = [200, 250, 300]
if args.Xnpoison is not None: args.npoison = args.Xnpoison
if args.Xnet is not None: args.net = args.Xnet
args.gpu = set_available_gpus(args)
global meta, sess, xtrain, ytrain, xvalid, yvalid, xbase, ybase, xtarget, ytarget, ytargetadv

def victim(kwargs=None):

    def comet_pull_poison(craftstep):
        for attempt in range(5):
            try:
                bytefile = craftexpt.get_asset(assets[craftstep])
                if localrank == 0: print('==> poisoninputs-{} pulled'.format(craftstep))
                poisoninputs = pickle.loads(bytefile)
                return poisoninputs[:args.npoison]
            except:
                print(f'WARNING: comet pull attempt for craftstep {craftstep} failed on attempt {attempt}')
                sleep(5)

    if kwargs is not None:
        for key in kwargs: globals()[key] = kwargs[key]
        for key in argsmod: setattr(args, key, argsmod[key])

    craftexpt = api.get_experiment(cometconfig["workspace"], args.craftproj, args.craftkey)
    assets = {asset['step']: asset['assetId'] for asset in craftexpt.get_asset_list() if 'poisoninputs-' in asset['fileName']}
    print('==> begin victim train')
    trial = 0
    while args.ntrial is None or trial < args.ntrial:
        for craftstep in args.craftsteps:
            experiment = Experiment(project_name=args.victimproj, auto_param_logging=False, auto_metric_logging=False, parse_args=False)
            experiment.log_parameters(vars(args))
            experiment.set_name(f'{args.craftkey[:5]}-{experiment.get_key()[:5]}')
            experiment.add_tag(args.tag)
            # experiment.add_tag(args.Xtag)
            experiment.log_parameters(dict(craftstep=craftstep, trial=trial))
            experiment.log_other('crafturl', craftexpt.url)
            experiment.log_other('command', 'python ' + ' '.join(sys.argv))
            if localrank == 0: print_command_and_args(args); print('crafturl: ' + craftexpt.url)

            if 'victim.py' in sys.argv[0]:
                poisoninputs = comet_pull_poison(craftstep)
                if poisoninputs is None: experiment.end(); print(f'skipping craftstep {craftstep}'); continue
                if args.savepoisondataset: package_poisoned_dataset(poisoninputs, xtrain, ytrain, xtarget, ytarget, ytargetadv, xvalid, yvalid, args, craftstep); experiment.end(); continue
                # meta.init_weights(sess, pretrain_weights) # what we had before
                meta.global_initialize(args, sess)
                meta.poisoninputs.load(poisoninputs, sess)
            trainstep = 0
            for epoch in range(args.nvictimepoch):
                tic = time()
                lrnrate = lr_schedule(args.lrnrate, epoch, args.warmupperiod, args.schedule)

                # log hidden layer features
                if args.logfeat and epoch == args.nvictimepoch - 1:
                    feats = []
                    for victimfeed in feeddict_generator(xtrain, ytrain, lrnrate, meta, args, victim=True):
                        hiddens = sess.run(meta.hiddens, victimfeed)
                        for i, hidden in enumerate(hiddens):
                            if len(feats) <= i: feats.append(defaultdict(list))
                            feat = np.reshape(hidden, [-1, np.prod(hidden.shape[1:])])
                            appendfeats(feats[i], feat, victimfeed, ybase, ytarget, args.batchsize)
                    for i, feats_layer in enumerate(feats): comet_log_asset(experiment, f'feats_layer{i}', feats_layer, step=epoch)

                # log validation acc
                if epoch in np.round((args.nvictimepoch - 1) * np.linspace(0, 1, args.nvalidpoints) ** 2):
                    resVs = []  # validation
                    for _, validfeed, _ in feeddict_generator(xvalid, yvalid, lrnrate, meta, args, valid=True):
                        resV = sess.run(meta.resultV, validfeed)
                        resVs.append(resV)
                    experiment.log_metrics(avg_n_dicts(resVs), step=trainstep)

                # train one epoch
                for victimfeed in feeddict_generator(xtrain, ytrain, lrnrate, meta, args, victim=True):
                    _, resL = sess.run([meta.trainop, meta.resultL,], victimfeed)
                    if not trainstep % 200: experiment.log_metrics(resL, step=trainstep)
                    trainstep += 1
                    
                experiment.log_metric('elapsed', time() - tic, step=trainstep)
                if args.saveweights: comet_log_asset_weights_and_buffers(epoch, experiment, meta, sess)
                if not epoch % 20 and localrank == 0:
                    print(' | '.join([f'{args.craftkey[:5]}-{args.tag} | trial-{trial} | craftstep-{craftstep} | epoch {epoch} | elapsed {round(time() - tic, 2)}'] +
                                     [f'{key} {trunc_decimal(val)}' for key, val in resL.items() if 'class' not in key] +
                                     [f'{key} {trunc_decimal(val)}' for key, val in resV.items() if 'class' not in key]))
            experiment.end()
        trial += 1


def landscape():
    # load trajectory from file
    # weights is a list of state_dicts, representing the weight trajectory, where the last state_dict is the weights at the minimizer
    # buffers is a list of state_dicts, representing the BN batch statistics trajectory, where the last state_dict is the BN batch statistics at the last iteration
    # buffers0 is only used when evaluating the adversarial loss, since the current-batch statistics are used when computing the training loss
    weights0, buffers0 = comet_pull_weight_by_key(args.uid, args.victimproj, 79, weightapi, rank)
    # alter the weights
    delta = 0  # todo for liam: change this value
    weights = {}
    for key, value in weights0.items():
        weights[key] = value + delta
    # insert the weights and buffers at the minimizer into the computation graph
    meta.load_weights(sess, (weights, buffers0))
    xentT0 = sess.run(meta.xentT0)  # adversarial loss
    xents = []
    for _, trainfeed, _ in feeddict_generator(xtrain, ytrain, 0, meta, args):
        xent = sess.run(meta.xents[0], trainfeed)
        xents.append(xent)
    xent = np.mean(xents)  # training loss
    # todo: use these two variables (xentT0 is the adversarial loss and xent is the training loss) to make surface plots
    pass


if __name__ == '__main__':
    ## load data and build graph
    print('==> loading data')
    xtrain, ytrain, xvalid, yvalid, xbase, ybase, xtarget, ytarget, ytargetadv = load_and_apportion_data(mpi, args)
    print('==> building graph')
    meta = Meta(args, xbase, ybase, xtarget, ytarget, ytargetadv, victim=True)

    # start tf session and initialize variables
    gpu_options = tf.GPUOptions(allow_growth=True, visible_device_list=str(localrank % len(args.gpu)))
    sess = tf.Session(config=tf.ConfigProto(allow_soft_placement=True, gpu_options=gpu_options))
    pretrain_weights = meta.global_initialize(args, sess)
    sess.graph.finalize()

    # begin
    victim()
    # landscape()

