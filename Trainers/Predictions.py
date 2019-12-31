#!/usr/bin/env python3
#-*- coding: utf-8

import importlib
import os,sys
from tqdm import tqdm
import numpy as np

from Datasources.CellRep import CellRep
from .BatchGenerator import SingleGenerator
from Utils import SaveLRCallback
from Utils import Exitcodes,CacheManager,PrintConfusionMatrix

#Keras
from keras import backend as K
from keras.preprocessing.image import ImageDataGenerator
# Training callbacks
from keras.callbacks import ModelCheckpoint, ReduceLROnPlateau
from keras.utils import to_categorical
from keras.models import load_model

#Tensorflow
import tensorflow as tf

#Scikit learn
from sklearn import metrics

def run_prediction(config,locations=None):
    """
    Main training function, to work as a new process
    """
    if config.info:
        print("Starting prediction process....")

    if not locations is None:
        cache_m = CacheManager(locations=locations)
    if config.print_pred:
        print_prediction(config)
    else:
        predictor = Predictor(config)
        predictor.run()

def print_prediction(config):
    cache_m = CacheManager()

    if not os.path.isfile(cache_m.fileLocation('test_pred.pik')):
        return None
    
    #Load predictions
    (expected,Y_pred,nclasses) = cache_m.load('test_pred.pik')
    y_pred = np.argmax(Y_pred, axis=1)
    
    #Output metrics
    if nclasses > 2:
        f1 = metrics.f1_score(expected,y_pred,average='weighted')
    else:
        f1 = metrics.f1_score(expected,y_pred,pos_label=1)
    print("F1 score: {0:.2f}".format(f1))

    m_conf = PrintConfusionMatrix(y_pred,expected,nclasses,config,"TILs")

    #ROC AUC
    #Get positive scores (binary only)
    if nclasses == 2:
        scores = Y_pred.transpose()[1]
        fpr,tpr,thresholds = metrics.roc_curve(expected,scores,pos_label=1)
        print("AUC: {0:f}".format(metrics.roc_auc_score(expected,scores)))

    print("Accuracy: {0:.3f}".format(m_conf[nclasses+2][nclasses]))

    if config.verbose > 1:
        print("False positive rates: {0}".format(fpr))
        print("True positive rates: {0}".format(tpr))
        print("Thresholds: {0}".format(thresholds))
        
class Predictor(object):
    """
    Class responsible for running the predictions and outputing results
    """

    def __init__(self,config,keepImg=False,**kwargs):
        """
        @param config <parsed configurations>: configurations

        Optional keyword arguments:
        @param build_ensemble <boolean>: ask for an ensemble model
        """
        self._config = config
        self._verbose = config.verbose
        self._ds = None
        self._keep = keepImg

        if 'build_ensemble' in kwargs:
            self._ensemble = kwargs['build_ensemble']
        else:
            self._ensemble = False

    def run(self,x_test=None,y_test=None,load_full=True):
        """
        Checks configurations, loads correct module, loads data
        Trains!

        New networks should be inserted as individual modules. Networks should be imported
        by the Models module.

        If provided x_test and y_test data, runs prediction with them.
        """
        net_name = self._config.network
        if net_name is None or net_name == '':
            print("A network should be specified")
            return Exitcodes.RUNTIME_ERROR
                
        if self._config.data:
            dsm = importlib.import_module('Datasources',self._config.data)
            if self._config.testdir:
                self._ds = getattr(dsm,self._config.data)(self._config.testdir,self._config.keepimg,self._config)
            else:
                self._ds = getattr(dsm,self._config.data)(self._config.predst,self._config.keepimg,self._config)
        else:
            self._ds = CellRep(self._config.predst,self._config.keepimg,self._config)

        net_module = importlib.import_module('Models',net_name)
        net_model = getattr(net_module,net_name)(self._config,self._ds)

        if self._config.testdir is None and (x_test is None or y_test is None):
            self._ds.load_metadata()

        self.run_test(net_model,x_test,y_test,load_full)
        
    def run_test(self,model,x_test=None,y_test=None,load_full=True):
        """
        This should be executed after a model has been trained
        """

        cache_m = CacheManager()
        split = None
        if os.path.isfile(cache_m.fileLocation('split_ratio.pik')):
            split = cache_m.load('split_ratio.pik')
        else:
            print("[Predictor] A previously trained model and dataset should exist. No previously defined spliting found.")
            return Exitcodes.RUNTIME_ERROR

        #Priority is for given data as parameters. If None is given, try to load metadata as configured
        if x_test is None or y_test is None:
            if self._config.testdir is None:
                #Load sampled data if required by command line
                if self._config.sample < 1.0:
                    _,_,(x_test,y_test) = self._ds.split_metadata(split=split,data=self._ds.sample_metadata(self._config.sample))
                else:
                    _,_,(x_test,y_test) = self._ds.split_metadata(split)
            else:
                x_test,y_test = self._ds._run_dir(self._config.testdir)

        if self._config.verbose > 0:
            unique,count = np.unique(y_test,return_counts=True)
            l_count = dict(zip(unique,count))
            if len(unique) > 2:
                print("Test items:")
                print("\n".join(["label {0}: {1} items" .format(key,l_count[key]) for key in unique]))
            else:
                if not 1 in l_count:
                    l_count[1] = 0
                print("Test labels: {0} are 0; {1} are 1;\n - {2:.2f} are positives".format(l_count[0],l_count[1],(l_count[1]/(l_count[0]+l_count[1]))))
            print("Test set: {} items".format(len(y_test)))
            
        X,Y = self._ds.load_data(data=(x_test,y_test),keepImg=self._keep)
        if self._config.verbose > 1:
            print("Y original ({1}):\n{0}".format(Y,Y.shape))        
        Y = to_categorical(Y,self._ds.nclasses)

        # session setup
        sess = K.get_session()
        ses_config = tf.ConfigProto(
            device_count={"CPU":self._config.cpu_count,"GPU":self._config.gpu_count},
            intra_op_parallelism_threads=self._config.cpu_count if self._config.gpu_count == 0 else self._config.gpu_count, 
            inter_op_parallelism_threads=self._config.cpu_count if self._config.gpu_count == 0 else self._config.gpu_count,
            log_device_placement=True if self._verbose > 1 else False
            )
        sess.config = ses_config
        K.set_session(sess)
        
        #During test phase multi-gpu mode is not used (maybe done latter)
        if self._ensemble:
            #Weights should be loaded during ensemble build
            if hasattr(model,'build_ensemble'):
                pred_model = model.build_ensemble(training=False,npfile=True)
            else:
                if self._config.info:
                    print('[Predictor] Model not prepared to build ensembles, implement or choose other model')
                return None
        elif load_full and os.path.isfile(model.get_model_cache()):
            try:
                pred_model = load_model(model.get_model_cache())
                if self._config.info:
                    print("Model loaded from: {0}".format(model.get_model_cache()))
            except ValueError:
                pred_model,_ = model.build(training=False,pre_load_w=False)
                pred_model.load_weights(model.get_weights_cache())
                if self._config.info:
                    print("Model weights loaded from: {0}".format(model.get_weights_cache()))
        elif os.path.isfile(model.get_weights_cache()):
                pred_model,_ = model.build(training=False,pre_load_w=False)
                pred_model.load_weights(model.get_weights_cache())
                if self._config.info:
                    print("Model weights loaded from: {0}".format(model.get_weights_cache()))
                    
        else:
            if self._config.info:
                print("No trained model or weights file found")
            return None

        bsize = self._config.batch_size
        stp = round((len(X) / bsize) + 0.5)

        image_generator = ImageDataGenerator(samplewise_center=self._config.batch_norm, 
                                            samplewise_std_normalization=self._config.batch_norm)

        if self._ensemble:
            if not self._config.tdim is None:
                fix_dim = self._config.tdim
            else:
                fix_dim = self._ds.get_dataset_dimensions()[0][1:] #Only smallest image dimensions matter here
            test_generator = SingleGenerator(dps=(X,Y),
                                                classes=self._ds.nclasses,
                                                dim=fix_dim,
                                                batch_size=self._config.batch_size,
                                                image_generator=image_generator,
                                                extra_aug=self._config.augment,
                                                shuffle=False,
                                                verbose=self._verbose,
                                                input_n=self._config.emodels)
        else:
            test_generator = image_generator.flow(x=X,
                                                y=Y,
                                                batch_size=bsize,
                                                shuffle=False)
        
        if self._config.progressbar:
            l = tqdm(desc="Making predictions...",total=stp)

        Y_pred = np.zeros((len(X),self._ds.nclasses),dtype=np.float32)
        for i in range(stp):
            start_idx = i*bsize
            example = test_generator.next()
            Y_pred[start_idx:start_idx+bsize] = pred_model.predict_on_batch(example[0])
            if self._config.progressbar:
                l.update(1)
            elif self._config.info:
                print("Batch prediction ({0}/{1})".format(i,stp))
            if self._config.verbose > 1:
                if not np.array_equal(Y[start_idx:start_idx+bsize],example[1]):
                    print("Datasource label ({0}) and batch label ({1}) differ".format(Y[start_idx:start_idx+bsize],example[1]))

        del(X)
        del(test_generator)
        
        if self._config.progressbar:
            l.close()

        y_pred = np.argmax(Y_pred, axis=1)
        expected = np.argmax(Y, axis=1)

        if self._config.verbose > 0:
            if self._config.verbose > 1:
                np.set_printoptions(threshold=np.inf)
                print("Predicted probs ({1}):\n{0}".format(Y_pred,Y_pred.shape))
            #print("Y ({1}):\n{0}".format(Y,Y.shape))
            print("expected ({1}):\n{0}".format(expected,expected.shape))
            print("Predicted ({1}):\n{0}".format(y_pred,y_pred.shape))
            
        #Save predictions
        cache_m.dump((expected,Y_pred,self._ds.nclasses),'test_pred.pik')

        #Output metrics
        print_prediction(self._config)
