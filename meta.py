import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'  # silence tensorflow import tensorflow as tf
from utils import *
import pickle
import numpy as np
import tensorflow as tf
from recolor import *
from comet_ml import API

weightapi = API()

class Meta():

    def __init__(self, args, xbase, ybase, xtarget, ytarget, ytargetadv, victim=False):
        print(f'Message from meta.py\'s constructor: we got bases of shape {xbase.shape}, base class of {ybase[0]}, target classes of {ytarget}, and adv classes of {ytargetadv}')
        self.victim = victim
        self.args = args
        self.xbase, self.ybase = xbase, ybase
        self.xtarget, self.ytarget, self.ytargetadv = xtarget, ytarget, ytargetadv
        # weights to randomly initialize, True means all weights
        self.coldstart_names = True if args.pretrain == '' else [] #['w6', 'b6']
        self.trainable_names = True  # weights unfrozen during training, True means all weights
        self.cached_weights, self.cached_poisons, self.cached_buffers = {}, {}, {}

        # change to 64 bit for more reproducible results
        self.floattype = tf.float64 if self.args.bit64 else tf.float32
        self.inttype = tf.int64 if self.args.bit64 else tf.int32

        # build graph
        self.build_metalearner_graph()
        self.coldstartop = [tf.variables_initializer([self.weights0[name] for name in self.coldstart_names]),
                            tf.variables_initializer([buffer0 for buffer0 in self.buffers0.values()])]

        # variables initializers. variables in adapter-1, adapter-2, etc are created by keras and unwanted
        unwanteda =  set(tf.global_variables('adapter')) - set(tf.global_variables('adapter-0'))
        unwantedt =  set(tf.global_variables('targeter')) - set(tf.global_variables('targeter-0'))
        unwanted = unwanteda.union(unwantedt)
        unwanted = set(u for u in unwanted if 'moving' not in u.name)
        global_variables = set(tf.global_variables()) - unwanted
        self.modified_global_initializer = tf.variables_initializer(list(global_variables))
        # self.allcoldstartop = tf.variables_initializer(list(global_variables - set(tf.global_variables('poisons'))))

    def build_metalearner_graph(self):

        # define feeds
        self.lrnrate = tf.placeholder(name='lrnrate', shape=[], dtype=self.floattype)
        self.craftrate = tf.placeholder(name='craftrate', shape=[], dtype=self.floattype)
        self.augment = tf.constant(self.args.augment, dtype=tf.bool, name='augment')
        self.epochmass = tf.constant(1, dtype=tf.float32, name='epochmass')
        
        ## define poisons and the boolean mask for minibatching
        with tf.variable_scope('poisons'):
            self.poisonmask = tf.constant([False] * self.args.npoison, dtype=tf.bool, name='poisonmask')
            self.poisonlabels = tf.Variable(tf.constant(self.ybase, dtype=self.inttype), name='poisonlabels')
            poisonlabels_ = tf.boolean_mask(self.poisonlabels, self.poisonmask, axis=0, name='poisonlabels_masked')
            if self.victim:
                self.poisoninputs = tf.Variable(tf.constant(self.xbase, dtype=self.floattype), name='poisoninputs')
                poisoninputs_ = tf.boolean_mask(self.poisoninputs, self.poisonmask, axis=0, name='poisoninputs_masked')
            else:
                self.perts = tf.Variable(tf.zeros_like(self.xbase, dtype=self.floattype), name='perts')
                self.colorperts = tf.Variable(tf.zeros([self.args.npoison, *self.args.gridshape, 3], dtype=tf.float32), name='colorperts')  # shape [npoison, ncolorres, ncolorres, ncolorres, 3]
                perts_ = tf.boolean_mask(self.perts, self.poisonmask, axis=0, name='perts_masked')
                colorperts_ = tf.boolean_mask(self.colorperts, self.poisonmask, axis=0, name='colorperts_masked')
                xbase = tf.constant(self.xbase, dtype=self.floattype, name='xbase')
                poisoninputs_ = tf.boolean_mask(xbase, self.poisonmask, axis=0, name='poisoninputs_masked')
                poisoninputs_, eye, xrefmin, xrefmax = recolor(poisoninputs_, colorperts_, name='poisoninputs_masked_recolored')
                poisoninputs_ = tf.add(poisoninputs_, perts_, name='poisoninputs_masked_perturbed')
                self.poisoninputs = poisoninputs_

        # define target
        inputsT, labelsT, labelsC = tf.constant(self.xtarget, dtype=self.floattype), tf.constant(self.ytargetadv, dtype=self.inttype), tf.constant(self.ytarget, dtype=self.inttype)
        self.trains, self.cleanmasks, self.xents, self.accs, self.xentTs, self.cwTs, self.xentCs, self.xentNTs = [], [], [], [], [], [], [], []

        # select network arcthiecture
        from learners import ConvNet, ConvNetBN, ResNet, KerasModel # these learners require tf1.14
        if self.args.net == 'ConvNet':
            net = ConvNet(self.args)
        elif self.args.net == 'ConvNetBN':
            net = ConvNetBN(self.args)
        elif self.args.net in ['ResNet']:
            net = ResNet(self.args, structure=[3, 3, 3], filters=16, block_type='basic')
        elif self.args.net in ['ResNetnoBN']:
            net = ResNet(self.args, structure=[3, 3, 3], filters=16, block_type='basic', batchnorm=False)
        elif self.args.net in ['ResNetI']:
            # this is more of an imagenet resnet
            net = ResNet(self.args, structure=[2, 2, 2, 2], filters=64, block_type='basic')
        elif self.args.net in ['ResNet50', 'ResNet101', 'ResNet152', 'ResNet50V2',
                               'ResNet101V2', 'ResNet152V2', 'ResNeXt50', 'ResNeXt101',
                               'DenseNet121', 'DenseNet169', 'DenseNet201', 'NASNetMobile', 'NASNetLarge',
                               'InceptionResNetV2', 'InceptionV3', 'VGG11', 'VGG13', 'VGG16', 'VGG19', 'Xception',
                               'DenseNet40', 'MobileNetV2', 'MobileNet']:
            net = KerasModel(self.args, architecture=self.args.net , data='CIFAR10')
        else:
            raise ValueError('Unknown network architecture.')

        ## build metagraph
        for i in range(self.args.nadapt + 1):
            with tf.variable_scope('adapter-' + str(i)):

                # inputs for feed_dict
                inputs = tf.placeholder(name='cleaninputs', shape=[self.args.batchsize, 32, 32, 3], dtype=self.floattype)
                labels = tf.placeholder(name='cleanlabels', shape=[self.args.batchsize], dtype=self.inttype)
                self.trains.append((inputs, labels))

                # inject poisons at base iteration
                if i == 0: self.cleanmask = tf.constant([True] * self.args.batchsize, dtype=tf.bool, name='cleanmask')
                cleaninputs_ = tf.boolean_mask(inputs, self.cleanmask, axis=0, name='cleaninputs_masked')
                cleanlabels_ = tf.boolean_mask(labels, self.cleanmask, axis=0, name='cleanlabels_masked')
                inputs = tf.concat([poisoninputs_, cleaninputs_], axis=0, name='concat_inputs')
                labels = tf.concat([poisonlabels_, cleanlabels_], axis=0, name='concat_labels')
                if self.args.augment: # data augmentation
                    inputs = tf.cond(self.augment, lambda: tf_preprocess(inputs, self.args.batchsize), lambda: inputs)
                inputs = tf.concat([inputs, inputsT], axis=0, name='concat_targ_inputs')
                labels = tf.concat([labels, labelsT], axis=0, name='concat_targ_labels')
                if i > 0:
                    inputs = tf.stop_gradient(inputs, name='inputs_stopgrad')
                    labels = tf.stop_gradient(labels, name='labels_stopgrad')

                # forward pass
                if i == 0:  # construct weight variables (future unrolled weights are nonvariable tensors)
                    self.weights0, self.buffers0 = net.construct_weights()
                    self.weights, self.buffers = self.weights0.copy(), self.buffers0.copy()  # copy the list, but does not deeply copy the tensors
                    if self.trainable_names is True: self.trainable_names = list(self.weights.keys())  # True means all weights
                    if self.coldstart_names is True: self.coldstart_names = list(self.weights.keys())
                logits, hiddens, self.buffers = net.forward(inputs, self.weights, self.buffers)
                if i == 0: self.buffrop = [tf.assign(self.buffers0[key], self.buffers[key]) for key in self.buffers] # op to update the base iteration's batchnorm buffer
                
                xent, acc = metrics(labels[:self.args.batchsize], logits[:self.args.batchsize], self.args.batchsize)
                if self.args.weightdecay: xent += 2e-4 * l2_weights(self.weights)
                self.xents.append(xent)
                self.accs.append(acc)
                if i == 0: self.hiddens = hiddens

                # compute fast weights
                grad_list = tf.gradients(xent, [self.weights[key] for key in self.trainable_names])
                gradients = dict(zip(self.trainable_names, grad_list))
                for key in gradients:
                    assert gradients[key] is not None, "Key {} has no gradient signal.".format(key)
                    self.weights[key] = self.weights[key] - self.lrnrate * gradients[key]

            with tf.variable_scope('targeter-' + str(i)):
                # forward pass on target at current stage of unrolling
                xentT, accT = metrics(labels[self.args.batchsize:], logits[self.args.batchsize:], self.args.ntarget)
                cwT = carlini(labels[self.args.batchsize:], logits[self.args.batchsize:], self.args.ntarget)
                # basic metrics
                self.xentTs.append(xentT)
                self.cwTs.append(cwT)
                # special metrics
                softmax = tf.nn.softmax(logits[self.args.batchsize:])
                neglogprob = -tf.log(softmax)
                neglogprobrev = -tf.log(1 - softmax)
                xentC = tf.add_n([neglogprobrev[i, ytarg] for i, ytarg in enumerate(self.ytarget)]) / len(self.ytarget)
                xentNT = tf.reduce_mean(neglogprobrev[:, int(self.ybase[0])])
                self.xentCs.append(xentC)
                self.xentNTs.append(xentNT)
                if i == 0:
                    self.accT0 = accT
                    self.accC0 = tf.reduce_mean(tf.cast(tf.equal(tf.argmax(softmax, axis=1), self.ytarget), dtype=self.floattype))
                    classes = {}
                    for targid in range(self.args.ntarget):
                        for cls in range(neglogprob.shape.as_list()[1]):
                            classes[f'class-{targid}-{cls}'] = neglogprob[targid, cls]

        self.trains = tuple(self.trains)

        with tf.variable_scope('targeter-0'):
            # target loss on current network without unrolling
            self.xentT0, self.cwT0, self.xentC0 = self.xentTs[0], self.cwTs[0], self.xentCs[0]

        with tf.variable_scope('trainop'):
            with tf.control_dependencies([self.xentT0, self.accT0, self.cwT0]):
                var_list = [value for key, value in self.weights0.items() if key in self.trainable_names]
                if self.args.optimizer == 'sgd': optim = tf.train.GradientDescentOptimizer(self.lrnrate)
                elif self.args.optimizer == 'mom': optim = tf.train.MomentumOptimizer(self.lrnrate, momentum=.9)
                else: optim = tf.train.MomentumOptimizer(self.lrnrate, momentum=.9)
                # else: raise ValueError('Invalid optimizer')
                self.trainop = [optim.minimize(self.xents[0], var_list=var_list), self.buffrop]

        with tf.variable_scope('results'):
            # result dictionaries for learner
            self.resultL = dict(xentT0=self.xentT0,
                                accT0=self.accT0,
                                cwT0=self.cwT0,
                                xent=self.xents[0],
                                acc=self.accs[0],
                                # xentC0=self.xentC0,
                                # accC0=self.accC0,
                                # xentNT0 = self.xentNTs[0],
                                **classes,
                                )
            self.resultV = dict(xentV=self.xents[0],
                                accV=self.accs[0],
                                )

        ## build parts of computation graph specifically for crafting phase
        if not self.victim:
            
            # average target loss over all adapt steps
            with tf.variable_scope('metaloss'):
                self.xentT = tf.add_n(self.xentTs[1:], name='xentT') / len(self.xentTs[1:])
                self.cwT = tf.add_n(self.cwTs[1:], name='cwT') / len(self.cwTs[1:])
                self.xentC = tf.reduce_mean(self.xentCs[1:], name='xentC')
                self.xentNT = tf.add_n(self.xentNTs[1:], name='xentNT') / len(self.xentNTs[1:])
                objectives = dict(xentT=self.xentT,
                                  cwT=self.cwT,
                                  dualxent=self.xentT + self.xentC,
                                  xentC=self.xentC,
                                  indis3=self.xentC + self.xentNT,
                                  )
                self.objective = objectives[self.args.objective]
                self.smoothobj = self.args.smoothcoef * smoothloss(self.colorperts, self.args.gridshape)
                self.objective += self.smoothobj
                self.objective = self.epochmass * self.objective

            with tf.variable_scope('metagrad'):
                # compute metagradient
                self.optim = tf.train.AdamOptimizer(self.craftrate)
                metavars = [self.perts, self.colorperts]
                metagradients = self.optim.compute_gradients(self.objective, var_list=metavars)
                self.metagrads_accum, self.accumop, self.avg_metagrads = [], [], []
                for metagrad, metavar in metagradients:
                    metagrad_accum = tf.get_variable(name=f'metagrad_accum_{tf_basename(metavar)}', shape=metavar.shape.as_list(),
                                                     dtype=self.floattype, initializer=tf.zeros_initializer, trainable=False)
                    self.metagrads_accum.append(metagrad_accum)
                    self.accumop.append(tf.assign_add(metagrad_accum, metagrad, name=f'metagrad_accumop_{tf_basename(metavar)}'))
                    self.avg_metagrads.append(tf.placeholder(shape=metavar.shape.as_list(), dtype=self.floattype, name=f'avg_metagrads_{tf_basename(metavar)}'))
                self.avg_metagrads = tuple(self.avg_metagrads)
                
            with tf.variable_scope('craftop'):
                # apply metagradient
                self.avg_metagradients = [(metagrad, metavar) for metagrad, metavar in zip(self.avg_metagrads, metavars)]
                craftop = self.optim.apply_gradients(self.avg_metagradients, name='craftop')
                # pgd clipping
                with tf.control_dependencies([craftop]):
                    # clip additive perturbation
                    clipop = []
                    clipped = tf.clip_by_value(self.perts, -self.args.eps, self.args.eps)
                    # xbase_recolored, _, _, _ = recolor(xbase, self.colorperts, eye=eye)
                    # clipped = tf.clip_by_value(xbase_recolored + clipped, 0, 255) - xbase_recolored
                    clipop.append(tf.assign(self.perts, clipped, name='clipop_perts'))
                    # clip color space pertrubation
                    # clipped = tf.clip_by_value(self.colorperts, -self.args.epscolor, self.args.epscolor)
                    clipped = [tf.clip_by_value(self.colorperts[:, :, :, :, i], -self.args.epscolor[i], self.args.epscolor[i]) for i in range(3)]
                    eye = tf.tile(tf.expand_dims(eye, 0), [self.args.npoison, 1, 1, 1, 1])
                    clipped = [tf.clip_by_value(eye[:, :, :, :, i] + clipped[i], xrefmin[i], xrefmax[i]) - eye[:, :, :, :, i] for i in range(3)]
                    clipped = tf.stack(clipped, axis=-1)
                    clipop.append(tf.assign(self.colorperts, clipped, name='clipop_colorperts'))
                    # reset metagrad accumulator
                    zeroop = [tf.assign(metagrad_accum, tf.zeros_like(metagrad_accum), name=f'zeroop_{tf_basename(metagrad_accum)}') for metagrad_accum in self.metagrads_accum]
                self.craftop = [craftop, clipop, zeroop]

            with tf.variable_scope('results'):
                # result dictionary for metalearner
                self.resultM = dict(xentT=self.xentT,
                                    cwT=self.cwT,
                                    smoothobj = self.smoothobj,
                                    xentC=self.xentC,
                                    xentNT=self.xentNT,
                                    )

    ## other functions
    def load_weights(self, sess, pretrain_weights):
        if type(pretrain_weights) is tuple: # sometimes we saved both weights (idx 0) and batchnorm buffers (idx 1)
            [self.weights0[key].load(pretrain_weights[0][key], sess) for key in pretrain_weights[0]]
            [self.buffers0[key].load(pretrain_weights[1][key], sess) for key in pretrain_weights[1]]
        else: # sometimes there's no bn buffers
            print('In load_weights(), no batchnorm buffers detected')
            [self.weights0[key].load(pretrain_weights[key], sess) for key in pretrain_weights]

    def init_weights(self, sess, pretrain_weights=None):
        if pretrain_weights is not None: self.load_weights(sess, pretrain_weights)
        sess.run(self.coldstartop)

    def cache_weights(self, sess, cache='default', restore=False):
        if not restore:
            self.cached_weights[cache] = sess.run(self.weights0)
            self.cached_buffers[cache] = sess.run(self.buffers0)
        else:
            [val.load(self.cached_weights[cache][key], sess) for key, val in self.weights0.items()]
            [val.load(self.cached_buffers[cache][key], sess) for key, val in self.buffers0.items()]

    def global_initialize(self, args, sess):
        sess.run(self.modified_global_initializer)
        if args.pretrain:
            projname, key = args.pretrain.split('/')
            pretrain_weights = comet_pull_weight_by_key(key, projname, 200, api, 'allranks')
            self.init_weights(sess, pretrain_weights)
            return pretrain_weights

    def restart_poison(self, perts, sess):
        self.perts.load(perts, sess)
