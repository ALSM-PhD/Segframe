#!/usr/bin/env python3
#-*- coding: utf-8

import importlib
import os

#Filter warnings
import warnings
warnings.filterwarnings('ignore')
    
import numpy as np

#Preparing migration to TF 2.0
import tensorflow as tf
if tf.__version__ >= '1.14.0':
    tf = tf.compat.v1
    from tensorflow.python.util import deprecation
    deprecation._PRINT_DEPRECATION_WARNINGS = False
    tf.logging.set_verbosity(tf.logging.ERROR)
    #tf.disable_v2_behavior()

#Network
from keras.models import Sequential,Model
from keras.layers import Input,Average
from keras import backend, optimizers
from keras.utils import multi_gpu_model
from keras import backend as K

#Locals
from Utils import CacheManager
from Models.GenericModel import GenericModel

class Inception(GenericModel):
    """
    Implements abstract methods from GenericModel.
    Model is the same as in: https://github.com/keras-team/keras-applications/blob/master/keras_applications/inception_resnet_v2.py
    Addapted to provide a Bayesian model
    """
    def __init__(self,config,ds,name=None):
        super().__init__(config,ds,name=name)
        if name is None:
            self.name = "Inception"
        self._modelCache = "{0}-model.h5".format(self.name)
        self._weightsCache = "{0}-weights.h5".format(self.name)
        self._mgpu_weightsCache = "{0}-mgpu-weights.h5".format(self.name)
 
        self.cache_m = CacheManager()
        self.cache_m.registerFile(os.path.join(config.model_path,self._modelCache),self._modelCache)
        self.cache_m.registerFile(os.path.join(config.weights_path,self._weightsCache),self._weightsCache)
        self.cache_m.registerFile(os.path.join(config.weights_path,self._mgpu_weightsCache),self._mgpu_weightsCache)

        self.single = None
        self.parallel = None
        
    def get_model_cache(self):
        """
        Returns path to model cache
        """
        return self.cache_m.fileLocation(self._modelCache)
    
    def get_weights_cache(self):
        """
        Returns path to model cache
        """
        return self.cache_m.fileLocation(self._weightsCache)

    def get_mgpu_weights_cache(self):
        """
        Returns path to model cache
        """
        return self.cache_m.fileLocation(self._mgpu_weightsCache)
    
    def get_npweights_cache(self,add_ext=False):
        """
        Returns path to model cache.

        @param add_ext <boolean>: add numpy file extension to file name.
        """
        if add_ext:
            return "{}.npy".format(self.cache_m.fileLocation(self._weightsCache).split('.')[0])
        else:
            return self.cache_m.fileLocation(self._weightsCache).split('.')[0]

    def get_npmgpu_weights_cache(self,add_ext=False):
        """
        Returns path to model cache

        @param add_ext <boolean>: add numpy file extension to file name.
        """
        if add_ext:
            return "{}.npy".format(self.cache_m.fileLocation(self._mgpu_weightsCache).split('.')[0])
        else:
            return self.cache_m.fileLocation(self._mgpu_weightsCache).split('.')[0]
    
    def register_ensemble(self,m):
        self._model_n = m
        self._weightsCache = "{0}-EM{1}-weights.h5".format(self.name,m)
        self._mgpu_weightsCache = "{0}-EM{1}-mgpu-weights.h5".format(self.name,m)
        self._modelCache = "{0}-EM{1}-model.h5".format(self.name,m)
        
        self.cache_m.registerFile(os.path.join(self._config.weights_path,self._weightsCache),self._weightsCache)
        self.cache_m.registerFile(os.path.join(self._config.weights_path,self._mgpu_weightsCache),self._mgpu_weightsCache)
        self.cache_m.registerFile(os.path.join(self._config.model_path,self._modelCache),self._modelCache)

    def return_model_n(self):
        if hasattr(self,'_model_n'):
            return self._model_n
        else:
            return -1
        
    def build(self,**kwargs):
        """
        @param pre_trained <boolean>: returned model should be pre-trained or not
        @param data_size <int>: size of the training dataset
        """
        model,parallel_model = self._build(**kwargs)
        
        self.single = model
        self.parallel = parallel_model
        
        return (model,parallel_model)

    def build_extractor(self,**kwargs):
        """
        Builds a feature extractor.
        
        Weights should be loaded by caller!

        Key word arguments:
        preload_w: return model with weights already loaded? True -> Yes
        parallel: return parallel model (overrides gpu_count avaliation)? True -> Yes
        """
        #Weight loading for the feature extraction is done latter by requesting party
        kwargs['preload_w'] = False

        if 'parallel' in kwargs and not kwargs['parallel']:
            s,p = self._build(**kwargs)
            return (s,None)
        else:
            return self._build(**kwargs)

    def build_ensemble(self,**kwargs):
        """
        Builds an ensemble of M Inception models.

        Weights are loaded here because of the way ensembles should be built.

        Default build: avareges the output of the corresponding softmaxes

        @param npfile <boolean>: loads weights from numpy files
        """

        if 'npfile' in kwargs:
            npfile = kwargs['npfile']
        else:
            npfile = False

        s_models = []
        p_models = []
        for m in range(self._config.emodels):
            self.register_ensemble(m)
            single,parallel = self._build(**kwargs)
            
            if not parallel is None:
                if npfile and hasattr(model,'get_npmgpu_weights_cache'):
                    parallel.set_weights(np.load(model.get_npmgpu_weights_cache(),allow_pickle=True))
                    if self._config.info:
                        print("[Inception] loaded ensemble weights: {}".format(model.get_npmgpu_weights_cache()))
                elif os.path.isfile(model.get_mgpu_weights_cache()):                    
                    parallel.load_weights(model.get_mgpu_weights_cache(),by_name=True)
                    if self._config.info:
                        print("[Inception] loaded ensemble weights: {}".format(model.get_mgpu_weights_cache()))
            else:
                parallel = None

            if npfile and hasattr(model,'get_npweights_cache'):
                single.set_weights(np.load(model.get_weights_cache(),allow_pickle=True))
                if self._config.info:
                    print("[Inception] loaded ensemble weights: {}".format(model.get_npweights_cache()))
            elif os.path.isfile(model.get_weights_cache()):
                single.load_weights(model.get_weights_cache(),by_name=True)
            else:
                if self._config.info:
                    print("[Inception] Could not load ensemble weights (model {})".format(m))
                single = None
            s_models.append(single)
            p_models.append(parallel)

        s_inputs = [inp for s in s_models for inp in s.inputs]
        s_outputs = [out for s in s_models for out in s.outputs]
        p_models = list(filter(lambda x: not x is None,p_models))
        if len(p_models) > 0:
            p_inputs = [inp for p in p_models for inp in p.inputs]
            p_outputs = [out for p in p_models for out in p.outputs]
        else:
            p_inputs = None
            p_outputs = None

        #Build the ensemble output from individual models
        s_model,p_model = None,None
        ##Single GPU model
        x = Average()(s_outputs)
        s_model = Model(inputs = s_inputs, outputs=x)

        ##Parallel model
        if not p_inputs is None:
            x = Average()(p_outputs)
            p_model = Model(inputs=p_inputs,outputs=x)

        return s_model,p_model
    
    def _build(self,**kwargs):

        width,height,channels = self._check_input_shape()

        if 'data_size' in kwargs:
            self.data_size = kwargs['data_size']

        if 'training' in kwargs:
            training = kwargs['training']
        else:
            training = True
            
        if 'feature' in kwargs:
            feature = kwargs['feature']
        else:
            feature = False

        if 'preload_w' in kwargs:
            preload = kwargs['preload_w']
        else:
            preload = True

        if 'allocated_gpus' in kwargs and not kwargs['allocated_gpus'] is None:
            allocated_gpus = kwargs['allocated_gpus']
        else:
            allocated_gpus = self._config.gpu_count
            
        if backend.image_data_format() == 'channels_first':
            input_shape = (channels, height, width)
        else:
            input_shape = (height, width, channels)

        self.cache_m = CacheManager()
        
        model = self._build_architecture(input_shape,training,feature,preload)
 
        #Check if previous training and LR is saved, if so, use it
        lr_cache = "{0}_learning_rate.txt".format(self.name)
        self.cache_m.registerFile(os.path.join(self._config.cache,lr_cache),lr_cache)
        l_rate = 0.00005
        if os.path.isfile(self.cache_m.fileLocation(lr_cache)) and not self._config.new_net:
            l_rate = float(self.cache_m.read(lr_cache))
            if self._config.info:
                print("Found previous learning rate: {0}".format(l_rate))
        
        #opt = optimizers.SGD(lr=l_rate, decay=1.5e-4, momentum=0.9, nesterov=True)
        opt = optimizers.Adam(lr = l_rate)
        #opt = optimizers.Adadelta(lr=l_rate)

        #Return parallel model if multiple GPUs are available
        parallel_model = None
       
        if allocated_gpus > 1:
            with tf.device('/cpu:0'):
                model.compile(loss='categorical_crossentropy',
                    optimizer=opt,
                    metrics=['accuracy'])
            parallel_model = multi_gpu_model(model,gpus=allocated_gpus)
            parallel_model.compile(loss='categorical_crossentropy',
                                       optimizer=opt,
                                       metrics=['accuracy'],
                                       #options=p_opt, 
                                       #run_metadata=p_mtd
                                       )
        else:
            model.compile(loss='categorical_crossentropy',
                optimizer=opt,
                metrics=['accuracy'],
                #options=p_opt, 
                #run_metadata=p_mtd
                )

        return (model,parallel_model)

    def _build_architecture(self,input_shape,training=None,feature=False,preload=True):
        from . import inception_resnet_v2

        kwargs = {'training':training,
                    'feature':feature,
                    'custom_top':False,
                    'preload':preload,
                    'batch_n':True if self._config.gpu_count <= 1 else False}
        
        inp = Input(shape=input_shape)
                
        inception_body = inception_resnet_v2.InceptionResNetV2(include_top=False,
                                                                weights='imagenet',
                                                                input_tensor=inp,
                                                                input_shape=input_shape,
                                                                pooling='avg',
                                                                classes=self._ds.nclasses,
                                                                **kwargs)
        

        return inception_body
                                
