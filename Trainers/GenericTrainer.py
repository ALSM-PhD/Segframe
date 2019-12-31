#!/usr/bin/env python3
#-*- coding: utf-8

import importlib
import os,sys
import re
import numpy as np
import threading

#Filter warnings
import warnings
warnings.filterwarnings('ignore')
    
from Datasources.CellRep import CellRep
from Utils import SaveLRCallback,CalculateF1Score,EnsembleModelCallback
from Utils import Exitcodes,CacheManager

#Keras
from keras import backend as K
from keras.preprocessing.image import ImageDataGenerator
# Training callbacks
from keras.callbacks import ModelCheckpoint, ReduceLROnPlateau,LearningRateScheduler
from keras.utils import to_categorical

#Preparing migration to TF 2.0
import tensorflow as tf
if tf.__version__ >= '1.14.0':
    tf = tf.compat.v1
    from tensorflow.python.util import deprecation
    deprecation._PRINT_DEPRECATION_WARNINGS = False
    tf.logging.set_verbosity(tf.logging.ERROR)
    #tf.disable_v2_behavior()

def run_training(config,locations=None):
    """
    Main training function, to work as a new process
    """
    if config.info:
        print("Starting training process....")

    if not locations is None:
        cache_m = CacheManager(locations=locations)
    trainer = Trainer(config)
    trainer.run()

def _reduce_lr_on_epoch(epoch,lr):
    #First 10 epochs, use a smaller LR, than raise to initially defined value
    #if epoch == 0:
    #    lr /= 10

    #if epoch == 9:
    #    lr *= 10
        
    #Reduces LR by a factor of 10 every 30 epochs
    if epoch > 9 and not (epoch%30):
        lr /= 10
    return lr

class Trainer(object):
    """
    Class that implements the training procedures applicable to all
    CNN models.

    Specialized training my be needed for some models and those should be 
    implemented elsewhere.

    @param config <argparse config>: configurations as specified by user
    @param ds <datasource>: some subclass of a GenericDatasource
    """

    def __init__(self,config):
        """
        @param config <parsed configurations>: configurations
        """
        self._config = config
        self._verbose = config.verbose
        self._ds = None
        self._rex = r'{0}-t(?P<try>[0-9]+)e(?P<epoch>[0-9]+).h5'

    def load_modules(self):
        net_name = self._config.network
        if net_name is None or net_name == '':
            print("A network should be specified")
            return Exitcodes.RUNTIME_ERROR

        if self._config.data:
            dsm = importlib.import_module('Datasources',self._config.data)
            self._ds = getattr(dsm,self._config.data)(self._config.predst,self._config.keepimg,self._config)
        else:
            self._ds = CellRep(self._config.predst,self._config.keepimg,self._config)

        net_module = importlib.import_module('Models',net_name)
        net_model = getattr(net_module,net_name)(self._config,self._ds)

        return net_model
    
    def run(self):
        """
        Checks configurations, loads correct module, loads data
        Trains!

        New networks should be inserted as individual modules. Networks should be imported
        by the Models module.
        """

        net_model = self.load_modules()
        
        self._ds.load_metadata()

        self._rex = self._rex.format(net_model.name)

        sw_thread = self.train_model(net_model)
        return sw_thread.join()

    def _choose_generator(self,train_data,val_data):
        """
        Returns a tuple with two batch generators: (train_generator,val_generator)
        The type of generator depends on the config.delay_load option
        """
        train_generator,val_generator = (None,None)

        if self._config.augment:
            train_prep = ImageDataGenerator(
                samplewise_center=self._config.batch_norm,
                samplewise_std_normalization=self._config.batch_norm,
                rotation_range=180,
                width_shift_range=20,
                height_shift_range=20,
                zoom_range=.2,
                #shear_range=.05,
                horizontal_flip=True,
                vertical_flip=True,
                brightness_range=(-20.0,20.0))

            val_prep = ImageDataGenerator(
                samplewise_center=self._config.batch_norm,
                samplewise_std_normalization=self._config.batch_norm,
                brightness_range=(-20.0,20.0))
        else:
            train_prep = ImageDataGenerator(
                samplewise_center=self._config.batch_norm,
                samplewise_std_normalization=self._config.batch_norm)
            val_prep = ImageDataGenerator(
                samplewise_center=self._config.batch_norm,
                samplewise_std_normalization=self._config.batch_norm)
        
        if not self._config.tdim is None:
            fix_dim = self._config.tdim
        else:
            fix_dim = self._ds.get_dataset_dimensions()[0][1:] #Only smallest image dimensions matter here

        if self._config.delay_load:
            from Trainers import ThreadedGenerator
            
            train_generator = ThreadedGenerator(dps=train_data,
                                                classes=self._ds.nclasses,
                                                dim=fix_dim,
                                                batch_size=self._config.batch_size,
                                                image_generator=train_prep,
                                                extra_aug=self._config.augment,
                                                shuffle=True,
                                                verbose=self._verbose)
            
            val_generator = ThreadedGenerator(dps=val_data,
                                                classes=self._ds.nclasses,
                                                dim=fix_dim,
                                                batch_size=self._config.batch_size,
                                                image_generator=val_prep,
                                                extra_aug=self._config.augment,
                                                shuffle=True,
                                                verbose=self._verbose)
        else:
            #Loads training images and validation images
            x_train,y_train = self._ds.load_data(split=None,keepImg=self._config.keepimg,data=train_data)
            
            x_val,y_val = self._ds.load_data(split=None,keepImg=self._config.keepimg,data=val_data)

            #Labels should be converted to categorical representation
            y_train = to_categorical(y_train,self._ds.nclasses)
            y_val = to_categorical(y_val,self._ds.nclasses)
            train_generator = train_prep.flow(x_train,y_train,batch_size=self._config.batch_size,shuffle=True)
            val_generator = val_prep.flow(x_val,y_val,batch_size=1)

        return (train_generator,val_generator)
    
    def train_model(self,model,train_data=None,val_data=None,**kwargs):
        """
        Generic trainer. Receives a GenericModel and trains it
        @param model <GenericModel>
        @param train_data <list>: Should be a collection of image metadata
        @param val_data <list>: Should be a collection of image metadata

        Optional keyword arguments:
        @param set_session <boolean>: configure session here
        @param verbose <int>: set verbosity level for training process. If not specified, use default
        @param summary <boolean>: print model summary
        @param clear_sess <boolean>: clears session and frees GPU memory
        @param allocated_gpus <int>: currently not used
        @param save_numpy <boolean>: save weights in numpy format instead of HDF5
        """
        rcomp = re.compile(self._rex)

        if 'set_session' in kwargs:
            set_session = kwargs['set_session']
        else:
            set_session = True

        if 'stats' in kwargs:
            stats = kwargs['stats']
        else:
            stats = None

        if 'summary' in kwargs:
            summary = kwargs['summary']
        else:
            summary = True

        if 'clear_sess' in kwargs:
            clear_sess = kwargs['clear_sess']
        else:
            clear_sess = False

        if 'allocated_gpus' in kwargs:
            allocated_gpus = kwargs['allocated_gpus']
        else:
            allocated_gpus = None
            
        if 'save_numpy' in kwargs:
            save_numpy = kwargs['save_numpy']
        else:
            save_numpy = False
            
        # session setup
        if set_session:
            session = K.get_session()
            ses_config = tf.ConfigProto(
                device_count={"CPU":self._config.cpu_count,"GPU":self._config.gpu_count},
                intra_op_parallelism_threads=self._config.cpu_count if self._config.gpu_count == 0 else self._config.gpu_count, 
                inter_op_parallelism_threads=self._config.cpu_count if self._config.gpu_count == 0 else self._config.gpu_count,
                log_device_placement=True if self._verbose > 1 else False
                )
            session.config = ses_config
            K.set_session(session)
        
        #Setup of generators, augmentation, preprocessing
        if train_data is None or val_data is None:
            if self._config.sample < 1.0:
                data_sample = self._ds.sample_metadata(self._config.sample)
                train_data,val_data,_ = self._ds.split_metadata(split=self._config.split,data=data_sample)
            else:
                train_data,val_data,_ = self._ds.split_metadata(self._config.split)
            
        if self._verbose > 0 and (stats is None or stats):
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

        train_generator,val_generator = self._choose_generator(train_data,val_data)
        
        single,parallel = model.build(data_size=len(train_data[0]),allocated_gpus=allocated_gpus)
        if not parallel is None:
            training_model = parallel
        else:
            training_model = single
            
        # try to resume the training
        weights = list(filter(lambda f: f.endswith(".h5") and f.startswith(model.name),os.listdir(self._config.weights_path)))
        weights.sort()
        old_e_offset = 0
        if len(weights) > 0 and not self._config.new_net:
            # get last file (which is the furthest on the training) if exists
            ep_weights_file = weights[len(weights)-2]
            match = rcomp.fullmatch(ep_weights_file)
            if match:
                old_e_offset = int(match.group('epoch'))
            else:
                old_e_offset = 0
            # load weights
            try:
                single.load_weights(os.path.join(self._config.weights_path,
                    ep_weights_file))
                if self._verbose > 0:
                    print("Sucessfully loaded previous weights: {0}".format(ep_weights_file))
            except ValueError:
                single.load_weights(os.path.join(self._config.weights_path,"{0}-weights.h5".format(model.name)))
                if self._verbose > 0:
                    print("Sucessfully loaded previous weights from consolidated file.")
            except (ValueError,OSError) as e:
                print("[ALERT] Could not load previous weights, training from scratch")
                if self._verbose > 1:
                    print(e)
                
        wf_header = "{0}-t{1}".format(model.name,old_e_offset+1)

        ### Define special behaviour CALLBACKS
        callbacks = []
        ## ModelCheckpoint
        if self._config.save_w:
            callbacks.append(ModelCheckpoint(os.path.join(
                self._config.weights_path, wf_header + "e{epoch:02d}.h5"), 
                save_weights_only=True, period=5,save_best_only=True,monitor='val_acc'))
        ## ReduceLROnPlateau
        callbacks.append(ReduceLROnPlateau(monitor='loss',factor=0.7,\
                                           patience=10,verbose=self._verbose,\
                                           mode='auto',min_lr=1e-7))
        callbacks.append(LearningRateScheduler(_reduce_lr_on_epoch,verbose=self._verbose))
        ## CalculateF1Score
        if self._config.f1period > 0:
            callbacks.append(CalculateF1Score(val_generator,self._config.f1period,self._config.batch_size,self._config.info))

        if self._config.info and summary:
            print(single.summary())

        training_model.fit_generator(
            generator = train_generator,
            steps_per_epoch = len(train_generator), #// self._config.batch_size,
            epochs = self._config.epochs,
            validation_data = val_generator,
            validation_steps = len(val_generator), #//self._config.batch_size,
            verbose = self._verbose,
            use_multiprocessing = False,
            workers=self._config.cpu_count*2,
            max_queue_size=self._config.batch_size*3,
            callbacks=callbacks,
            )

        if self._verbose > 1:
            print("Done training model: {0}".format(hex(id(training_model))))


        sw_thread = threading.Thread(target=self._save_weights,name='save_weights',args=(model,single,parallel,clear_sess,save_numpy))
        sw_thread.start()
        return sw_thread
        
    def _save_weights(self,model,single,parallel,clear_sess,save_numpy):
        #Save weights for single tower model and for multigpu model (if defined)
        cache_m = CacheManager()
        if self._config.info:
            print("Saving weights, this could take a while...")
        if save_numpy and hasattr(model,'get_npweights_cache'):
            np.save(model.get_npweights_cache(),single.get_weights())
        else:
            single.save_weights(model.get_weights_cache())
            single.save(model.get_model_cache())
            
        if not parallel is None and not model.get_mgpu_weights_cache() is None:
            if save_numpy and hasattr(model,'get_npmgpu_weights_cache'):
                np.save(model.get_npmgpu_weights_cache(),parallel.get_weights())
            else:
                parallel.save_weights(model.get_mgpu_weights_cache())
        cache_m.dump(tuple(self._config.split),'split_ratio.pik')

        if clear_sess:
            K.clear_session()
            
        return Exitcodes.ALL_GOOD
        
