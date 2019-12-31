#!/usr/bin/env python3
#-*- coding: utf-8

import os,sys
import numpy as np
import importlib
import random
from multiprocessing import Pool, Queue

#Filter warnings
import warnings
warnings.filterwarnings('ignore')
    
from keras.preprocessing.image import ImageDataGenerator
from keras import backend as K
#Preparing migration to TF 2.0
import tensorflow as tf
if tf.__version__ >= '1.14.0':
    tf = tf.compat.v1
    from tensorflow.python.util import deprecation
    deprecation._PRINT_DEPRECATION_WARNINGS = False
    tf.logging.set_verbosity(tf.logging.ERROR)
    #tf.disable_v2_behavior()
    
#Local
from .ALTrainer import ActiveLearningTrainer
from .Predictions import Predictor

#Module
from Utils import Exitcodes,CacheManager

def run_training(config,locations=None):
    """
    Main training function, to work as a new process
    """
    if config.info:
        print("Starting active learning process....")

    if not locations is None:
        cache_m = CacheManager(locations=locations)
    trainer = EnsembleALTrainer(config)
    trainer.run()

class EnsembleALTrainer(ActiveLearningTrainer):
    """
    Implements the structure of active learning:
    - Uses a selection function to acquire new training points;
    - Manages the training/validation/test sets

    Methods that should be overwriten by specific AL strategies:
    - 
    """

    def __init__(self,config):
        """
        @param config <argparse>: A configuration object
        """
        super().__init__(config)
        self.tower = None

    def _initializer(self,gpus,processes):

        #initialize tensorflow session
        gpu_options = None
        if gpus > 0:
            gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=1.0)
            gpu_options.allow_growth = True
            gpu_options.Experimental.use_unified_memory = False
            gpu_options.visible_device_list = ",".join([str(g) for g in range(gpus)])

        sess = tf.Session(config=tf.ConfigProto(
            device_count={"CPU":processes,"GPU":gpus},
            intra_op_parallelism_threads=3, 
            inter_op_parallelism_threads=3,
            log_device_placement=False,
            gpu_options=gpu_options
            ))
        #sess.config = config
        K.set_session(sess)

    def _print_stats(self,train_data,val_data):
        unique,count = np.unique(train_data[1],return_counts=True)
        l_count = dict(zip(unique,count))
        if len(unique) > 2:
            print("Training items:")
            print("\n".join(["label {0}: {1} items" .format(key,l_count[key]) for key in unique]))
        else:
            print("Train labels: {0} are 0; {1} are 1;\n - {2:.2f} are positives".format(l_count[0],l_count[1],(l_count[1]/(l_count[0]+l_count[1]))))
            
        unique,count = np.unique(val_data[1],return_counts=True)
        l_count = dict(zip(unique,count))
        if len(unique) > 2:
            print("Validation items:")
            print("\n".join(["label {0}: {1} items" .format(key,l_count[key]) for key in unique]))
        else:            
            if not 1 in l_count:
                l_count[1] = 0
            print("Validation labels: {0} are 0; {1} are 1;\n - {2:.2f} are positives".format(l_count[0],l_count[1],(l_count[1]/(l_count[0]+l_count[1]))))
            
        print("Train set: {0} items".format(len(train_data[0])))
        print("Validate set: {0} items".format(len(val_data[0])))
        
    def run(self):
        """
        Coordenates the AL process
        """
        from keras import backend as K
        import time
        from datetime import timedelta
        
        #Loaded CNN model and Datasource
        model = self.load_modules()
        self._rex = self._rex.format(model.name)
        #Define initial sets
        self.configure_sets()
        #AL components
        cache_m = CacheManager()
        predictor = Predictor(self._config,keepImg=True,build_ensemble=True)
        function = None
        
        if not self._config.ac_function is None:
            acq = importlib.import_module('AL','AcquisitionFunctions')
            function = getattr(acq,self._config.ac_function)
        else:
            print("You should specify an acquisition function")
            sys.exit(Exitcodes.RUNTIME_ERROR)

        stime = None
        etime = None
        end_train = False
        self._initializer(self._config.gpu_count,self._config.cpu_count)
        
        for r in range(self._config.acquisition_steps):
            if self._config.info:
                print("[EnsembleTrainer] Starting acquisition step {0}/{1}".format(r+1,self._config.acquisition_steps))
                stime = time.time()

            #Save current dataset and report partial result (requires multi load for reading)
            fid = 'al-metadata-{1}-r{0}.pik'.format(r,model.name)
            cache_m.registerFile(os.path.join(self._config.logdir,fid),fid)
            cache_m.dump(((self.train_x,self.train_y),(self.val_x,self.val_y),(self.test_x,self.test_y)),fid)

            self._print_stats((self.train_x,self.train_y),(self.val_x,self.val_y))
            sw_thread = None
            for m in range(self._config.emodels):
                #Some models may take too long to save weights
                if not sw_thread is None:
                    if self._config.info:
                        print("[EnsembleTrainer] Waiting for model weights.", end='')
                    while True:
                        pst = '.'
                        if sw_thread[-1].is_alive():
                            if self._config.info:
                                pst = "{}{}".format(pst,'.')
                                print(pst,end='')
                            sw_thread[-1].join(60.0)
                        else:
                            print('')
                            break
                    
                if hasattr(model,'register_ensemble'):
                    model.register_ensemble(m)
                else:
                    print("Model not ready for ensembling. Implement register_ensemble method")
                    raise AttributeError

                if self._config.info:
                    print("[EnsembleTrainer] Starting model {} training".format(m))
                    
                st = self.train_model(model,(self.train_x,self.train_y),(self.val_x,self.val_y),
                                                set_session=False,stats=False,summary=False,
                                                clear_sess=True,save_numpy=True)
                if sw_thread is None:
                    sw_thread = [st]
                else:
                    sw_thread.append(st)
            
            if r == (self._config.acquisition_steps - 1) or not self.acquire(function,model,acquisition=r,sw_thread=sw_thread):
                if self._config.info:
                    print("[ALTrainer] No more acquisitions are in order")
                end_train = True
                
            #Set load_full to false so dropout is disabled
            predictor.run(self.test_x,self.test_y,load_full=False)
            
            #Attempt to free GPU memory
            K.clear_session()
            
            if self._config.info:
                etime = time.time()
                td = timedelta(seconds=(etime-stime))
                print("Acquisition step took: {0}".format(td))
                
            if end_train:
                return None

    def acquire(self,function,model,**kwargs):
        """
        Adds items to training and validation sets, according to split ratio defined in configuration. 
        Test set is fixed in the begining.

        Returns True if acquisition was sucessful
        """
        from Trainers import ThreadedGenerator
        #An acquisition function should return a NP array with the indexes of all items from the pool that 
        #should be inserted into training and validation sets
        if self.pool_x.shape[0] < self._config.acquire:
            return False

        if kwargs is None:
            kwargs = {}

        kwargs['config'] = self._config

        #Some acquisition functions may need access to GenericModel
        kwargs['model'] = model
        
        if not self._config.tdim is None:
            fix_dim = self._config.tdim
        else:
            fix_dim = self._ds.get_dataset_dimensions()[0][1:] #Only smallest image dimensions matter here

        #Pools are big, use a data generator
        pool_prep = ImageDataGenerator(
            samplewise_center=self._config.batch_norm,
            samplewise_std_normalization=self._config.batch_norm)

        #Acquisition functions that require a generator to load data
        generator_params = {
            'dps':(self.pool_x,self.pool_y),
            'classes':self._ds.nclasses,
            'dim':fix_dim,
            'batch_size':self._config.gpu_count * self._config.batch_size if self._config.gpu_count > 0 else self._config.batch_size,
            'image_generator':pool_prep,
            'shuffle':False, #DO NOT SET TRUE!
            'verbose':self._config.verbose,
            'input_n':1}

        generator = ThreadedGenerator(**generator_params)

        if self._config.verbose > 0:
            print("\nStarting acquisition...(pool size: {})".format(self.pool_x.shape[0]))

        if self._config.verbose > 1:
            print("Starting acquisition using model: {0}".format(hex(id(pred_model))))
        
        pooled_idx = function(None,generator,self.pool_x.shape[0],**kwargs)
        if pooled_idx is None:
            if self._config.info:
                print("[EnsembleTrainer] No indexes returned. Something is wrong.")
            sys.exit(1)
        self.train_x = np.concatenate((self.train_x,self.pool_x[pooled_idx]),axis=0)
        self.train_y = np.concatenate((self.train_y,self.pool_y[pooled_idx]),axis=0)
        self.pool_x = np.delete(self.pool_x,pooled_idx)
        self.pool_y = np.delete(self.pool_y,pooled_idx)

        return True        
