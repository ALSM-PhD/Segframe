#!/usr/bin/env python3
#-*- coding: utf-8

import os,sys
import numpy as np
import importlib
import random

#Filter warnings
import warnings
warnings.filterwarnings('ignore')
    
from keras.preprocessing.image import ImageDataGenerator

#Local
from .GenericTrainer import Trainer
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
    trainer = ActiveLearningTrainer(config)
    trainer.run()
    
class ActiveLearningTrainer(Trainer):
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

        self.pool_x = None
        self.pool_y = None
        self.train_x = None
        self.train_y = None
        self.val_x = None
        self.val_y = None
        self.test_x = None
        self.test_y = None


    def _balance_classes(self,X,Y):
        """
        Returns a tuple (X,Y) of balanced classes

        X and Y are lists
        """
        #Work with NP arrays
        if isinstance(X,list):
            X = np.asarray(X)
        if isinstance(Y,list):
            Y = np.asarray(Y)
            
        #Count the occurrences of each class
        unique,count = np.unique(Y,return_counts=True)

        #Extracts the positions of each class in the dataset
        class_members = {i:np.where(Y == i)[0] for i in unique}

        #Remount the lists
        nX,nY = ([],[])
        mcount = count.min()
        for c in unique:
            if count[c] > mcount:
                ids = np.random.choice(count[c],mcount,replace=False)
                nX.extend(X[class_members[c][ids]])
                nY.extend(Y[class_members[c][ids]])
            else:
                nX.extend(X[class_members[c]])
                nY.extend(Y[class_members[c]])

        #Reshufle all elements
        combined = list(zip(nX,nY))
        random.shuffle(combined)
        nX[:],nY[:] = zip(*combined)

        return nX,nY
        
    def configure_sets(self):
        """
        Creates the initial sets: training (X,Y); example pool; validation set; test set

        All sets are kept as NP arrays
        """
        X,Y = self._ds.load_metadata()

        #Use a sample of the metadata if so instructed
        if self._config.sample != 1.0:
            X,Y = self._ds.sample_metadata(self._config.sample)
            self._ds.check_paths(X,self._config.predst)

        if self._config.balance:
            X,Y = self._balance_classes(X,Y)
            if self._config.info:
                print("[ALTrainer] Using a balanced initial dataset for AL ({} total elements).".format(len(X)))
        elif self._config.info:
            print("[ALTrainer] Using an UNBALANCED initial dataset for AL ({} total elements).".format(len(X)))
            
        #Test set is extracted from the last items and is not changed for the whole run
        t_idx = int(self._config.split[-1:][0] * len(X))
        self.test_x = X[- t_idx:]
        self.test_y = Y[- t_idx:]

        self.pool_x = X[:-t_idx]
        self.pool_y = Y[:-t_idx]

        #Initial training set will be choosen at random from pool
        cache_m = CacheManager()
        if self._config.load_train and not self._config.balance:
            train_idx = cache_m.load('initial_train.pik')
            if not train_idx is None and self._config.info:
                print("[ALTrainer] Using initial training set from {}. This is DANGEROUS. Use the metadata correspondent to the initial set.".format(initial_train))
            
        else:
            if not self._config.load_train and self._config.balance and self._config.info:
                print("[ALTrainer] Dataset balancing and initial train set loading not possible at the same time.")
                
            train_idx = np.random.choice(len(self.pool_x),self._config.init_train,replace=False)
            cache_m.dump(train_idx,'initial_train.pik')
            
        pool_ar_x = np.asarray(self.pool_x)
        pool_ar_y = np.asarray(self.pool_y)
        self.train_x = pool_ar_x[train_idx]
        self.train_y = pool_ar_y[train_idx]

        #Remove choosen elements from the pool
        self.pool_x = np.delete(pool_ar_x,train_idx)
        self.pool_y = np.delete(pool_ar_y,train_idx)
        del(pool_ar_x)
        del(pool_ar_y)
        
        #Initial validation set - keeps the same split ratio for train/val as defined in the configuration
        val_samples = int((self._config.init_train*self._config.split[1])/self._config.split[0])
        val_samples = max(val_samples,100)
        val_idx = np.random.choice(self.pool_x.shape[0],val_samples,replace=False)
        self.val_x = self.pool_x[val_idx]
        self.val_y = self.pool_y[val_idx]
        self.pool_x = np.delete(self.pool_x,val_idx)
        self.pool_y = np.delete(self.pool_y,val_idx)

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
        predictor = Predictor(self._config,keepImg=True)
        function = None
        
        if not self._config.ac_function is None:
            acq = importlib.import_module('AL','AcquisitionFunctions')
            function = getattr(acq,self._config.ac_function)
        else:
            print("You should specify an acquisition function")
            sys.exit(Exitcodes.RUNTIME_ERROR)

        stime = None
        etime = None
        sw_thread = None
        end_train = False
        for r in range(self._config.acquisition_steps):
            if self._config.info:
                print("[ALTrainer] Starting acquisition step {0}/{1}".format(r+1,self._config.acquisition_steps))
                stime = time.time()

            #Save current dataset and report partial result (requires multi load for reading)
            fid = 'al-metadata-{1}-r{0}.pik'.format(r,model.name)
            cache_m.registerFile(os.path.join(self._config.logdir,fid),fid)
            cache_m.dump(((self.train_x,self.train_y),(self.val_x,self.val_y),(self.test_x,self.test_y)),fid)
                
            sw_thread = self.train_model(model,(self.train_x,self.train_y),(self.val_x,self.val_y))            
            
            if r == (self._config.acquisition_steps - 1) or not self.acquire(function,model,acquisition=r,sw_thread=sw_thread):
                if self._config.info:
                    print("[ALTrainer] No more acquisitions are in order")
                end_train = True
                    
            #Some models may take too long to save weights
            if not sw_thread is None and sw_thread.is_alive():
                if self._config.info:
                    print("[ALTrainer] Waiting for model weights...")
                sw_thread.join()
                    
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
            'verbose':self._config.verbose}

        generator = ThreadedGenerator(**generator_params)

        if self._config.gpu_count > 1:
            pred_model = model.parallel
        else:
            pred_model = model.single

        if self._config.verbose > 0:
            print("\nStarting acquisition...(pool size: {})".format(self.pool_x.shape[0]))

        if self._config.verbose > 1:
            print("Starting acquisition using model: {0}".format(hex(id(pred_model))))
        
        pooled_idx = function(pred_model,generator,self.pool_x.shape[0],**kwargs)
        if pooled_idx is None:
            if self._config.info:
                print("[ALTrainer] No indexes returned. Something is wrong.")
            sys.exit(1)
        self.train_x = np.concatenate((self.train_x,self.pool_x[pooled_idx]),axis=0)
        self.train_y = np.concatenate((self.train_y,self.pool_y[pooled_idx]),axis=0)
        self.pool_x = np.delete(self.pool_x,pooled_idx)
        self.pool_y = np.delete(self.pool_y,pooled_idx)

        return True
