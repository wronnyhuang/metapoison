import argparse

def get_parser(victim=False):
    
    parser = argparse.ArgumentParser(description='Craft poisoned data using MetaPoison')
    parser.add_argument('uid', default=None, type=str, help='Required. Unique ID of the poisons. Used both during crafting and victim evaluation to identify the poisons that are being crafted or later retrieved from comet')
    parser.add_argument('-gpu', default=[], type=int, nargs='+', help="List of GPUs IDs to use. By default will use all available") # if [] then use all gpu
    parser.add_argument('-craftproj', default='craft1', type=str, help='Comet project in which to store the poison crafting experiment')
    parser.add_argument('-victimproj', default='debug', type=str, help='Comet project in which to store the victim evaluation experiments')
    parser.add_argument('-tag', default='', type=str, help='Optional tag for the experiment. Will be logged onto comet')
    parser.add_argument('-bit64', action='store_true', help='Deprecated. Use 64-bit floating point for better reproducibility (default is 32-bit)')
    parser.add_argument('-logperiod', default=10, type=int, help='Period, in epochs, with which to save poisons to comet. E.g. if logperiod=10, then poisons will be saved at craftstep 0, 10, 20, ...')
    parser.add_argument('-skipvictim', action='store_true', help='Skip running victim evaluation. Craft poisons only')
    parser.add_argument('-justtrain', default=0, type=int, help='If justtrain != 0, it go into vanilla training mode--provide justtrain with an int equal to the number of epochs you want to vanilla train. Provide -weightset if you want to store the final weights into comet. This is useful when you want to save a lot of weights in different epoch so that you dont need to pretrain networks in the future')
    # threat model
    parser.add_argument('-npoison', default=200, type=int, help='Number of poisons to craft')
    parser.add_argument('-eps', default=8, type=float, help='Perturbation additive bound, in pixel counts out of 255')
    parser.add_argument('-epscolor', default=[.04, .04, .04], type=float, nargs='+', help='Perturbation color bound, in fraction of the total range, for each LUV color channel')
    parser.add_argument('-smoothcoef', default=1., type=float, help='Smoothing coefficient for the color perturbation. See the lambda coefficient in Laidlaw and Feizi (2019)')
    parser.add_argument('-gridshape', default=[10, 10, 10], type=int, nargs='+', help='Shape, or resolution, of the colorspace grid used for the ReColor function')
    parser.add_argument('-watermark', action='store_true', help='Place a 30%% opacity watermark of the target onto each poison. Used when comparing to Shafahi et al. (2018)')
    # image classes and ids
    parser.add_argument('-targetclass', default=2, type=int, help='Target class ID. See CIFAR-10 class IDs for reference')
    parser.add_argument('-poisonclass', default=5, type=int, help='Poison class ID. See CIFAR-10 class IDs for reference')
    parser.add_argument('-ytargetadv', default=-1, type=int, help='Adversarial class ID. If -1 then use poison class. See CIFAR-10 class IDs for reference')
    parser.add_argument('-targetids', default=[0], type=int, nargs='+', help='Target image IDs. E.g. if targetids = [0], then the target will be the first image of the target class taken from the CIFAR-10 test set. If more than one, then adversarial loss will be the average over all the adversarial losses of the individual targets. This is an additional feature that was not discussed in the MetaPoison paper')
    parser.add_argument('-multiclasspoison', action='store_true', help='Craft poisons from multiple classes (uniformly spread across all classes), rather than from the poison class')
    # runtime/memory budget
    parser.add_argument('-ncraftstep', default=61, type=int, help='Number of craft steps (outer optimization steps)')
    parser.add_argument('-nadapt', default=2, type=int, help='Number of SGD unroll steps (K parameter in Algorithm 1)')
    parser.add_argument('-nreplay', default=1, type=int, help='Number of surrogate models to run in series for each craftstep. The total number of surrogate models (M parameter in Algorithm 1) will be the number of MPI processes (-np parameter) times nreplay. If you have few GPUs, then increase nreplay and decrease the number of MPI processes to avoid running out of memory')
    # neural networks
    parser.add_argument('-net', default='ConvNetBN', type=str, help='Surrogate model architecture. Choose from [ConvNetBN, VGG13, and ResNet]')
    parser.add_argument('-pretrain', default='', type=str, help='Used for the case of fine-tuning. String of this format "<comet_project_name>/<comet_experiment_id>", corresponding to the comet project and experiment where the pretrained weights are stored')
    parser.add_argument('-lrnrate', default=.1, type=float, help='Initial learning rate')
    parser.add_argument('-batchsize', default=125, type=int, help='Batch size')
    parser.add_argument('-nbatch', default=40, type=int, help='Number of batches per epoch. This determines the fraction of CIFAR-10 that you will use at the training set. To use the full training set, batchsize * nbatch must equal 50000')
    parser.add_argument('-warmupperiod', default=5, type=int, help='Linear learning rate warmup period, in epochs')
    parser.add_argument('-optimizer', default='sgd', type=str, help='Optimizer (sgd or mom)')
    parser.add_argument('-weightdecay', action='store_true', help='Use weight decay (with coefficient set to default of 2e-4)')
    parser.add_argument('-augment', action='store_true', help='Use standard CIFAR-10 data augmentation')
    parser.add_argument('-weightsettrain', action='store_true', help='Stagger pretrain the surrogate network weights to their respective starting epochs')
    parser.add_argument('-weightset', default='', type=str, help='If string is empty, stagger pretrain surrogate models to their respective starting epochs. If weightsettrain is on, pretrained surrogate model weights will be stored into a comet project with this name. If weightsettrain is off, pretrained weights from this comet project will be loaded into the surrogate models so as to save time (no more need to pretraining them before crafting)')
    parser.add_argument('-stagger', default=1, type=int, help='Number of epochs each successive surrogate model is staggered from the previous one. Equivalent to T/M in Algorithm 1')
    parser.add_argument('-schedule', default=[100, 150, 200], type=int, nargs='+', help='Learning rate schedule. Learning rate is dropped by 10 at each of these epochs')
    # poison optimization
    parser.add_argument('-craftrate', default=200, type=float, help='Crafting rate, or learning rate of the outer optimization process')
    parser.add_argument('-crdropperiod', default=20, type=int, help='Crafting rate schedule. Crafting rate is dropped by 10 at craftsteps 1 * crdropperiod, 2 * crdropperiod, ...')
    parser.add_argument('-restartperiod', default=9999, type=int, help='Period, in craft steps, with which to restart the poison perturbation')
    parser.add_argument('-objective', default='cwT', type=str, help='Outer objective function. Choose from [cwT, xentT, other]')
    parser.add_argument('-trajectory', default='clean', type=str, help='Whether the weight space trajectory taken by the surrogate models should be based on a clean or poisoned dataset. Choose from [clean, poison]')
    parser.add_argument('-epochmass', action='store_true', help='Down-weight the adversarial losses from models in the first 5 epochs')

    if not victim: return parser

    craftargs = set(vars(parser.parse_known_args()[0]))
    
    # victim specific arguments (by default when running victim.py, all the arguments from the crafting experiment are copied over)
    parser.add_argument('-ntrial', default=1, type=int, help='Number of repeated victim evaluation trials to run')
    parser.add_argument('-nvictimepoch', default=200, type=int, help='Number of epochs to train the victim model for')
    parser.add_argument('-craftsteps', default=[60], type=int, nargs='+', help='Grab the saved poisons crafted at these craftsteps from comet')
    parser.add_argument('-saveweights', action='store_true', help='Save the weights from each epoch')
    parser.add_argument('-savepoisondataset', action='store_true', help='Export the poisoned dataset (poisons concatenated with the clean examples) into a pickle file stored in the local directory')
    parser.add_argument('-poisondatasetfile', default='poisondataset', type=str, help='Filename prefix with which to export the poison dataset. Full filename will be <poisondatasetfile>-<craftstep>.pkl')
    parser.add_argument('-nvalidpoints', default=5, type=int, help='Number of times to measure validation accuracy')
    parser.add_argument('-logfeat', action='store_true', help='Save the intralayer features into comet for feature visualization')
    # Overwrite the settings passed down from the crafting experiment with these new settings
    parser.add_argument('-Xvictimproj', default=None, type=str, help='New comet project in which to store the victim runs')
    parser.add_argument('-Xtag', default=None, type=str, help='New comet tag')
    parser.add_argument('-Xweightdecay', action='store_true', help='Use weight decay during victim evaluation')
    parser.add_argument('-Xaugment', action='store_true', help='Use data augmentation during victim evaluation')
    parser.add_argument('-Xbatchsize', action='store_true', help='Batch size to use during victim evaluation')
    parser.add_argument('-Xlrnrate', action='store_true', help='Learning rate to use during victim evaluation')
    parser.add_argument('-Xschedule', action='store_true', help='Learning rate schedule to use during victim evaluation')
    parser.add_argument('-Xnet', default=None, type=str, help='Network architecture to use during victim evaluation')
    parser.add_argument('-Xnpoison', default=None, type=int, help='Number of poisons to use during victim evaluation. Must be less or equal to the number of poisons crafted. If less, then a random subset of the poisons will be taken')

    return parser, craftargs
