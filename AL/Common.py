#!/usr/bin/env python3
#-*- coding: utf-8

import numpy as np
import os

__doc__ = """
Utility functions for acquisition functions and independent functions
"""
def random_sample(pred_model,generator,data_size,**kwargs):
    """
    Returns a random list of indexes from the given dataset
    """
    if 'config' in kwargs:
        k = kwargs['config'].acquire
    else:
        return None
    
    return np.random.choice(range(data_size),k,replace=False)

def oracle_sample(pred_model,generator,data_size,**kwargs):
    """
    Returns the indexes of images that the current network missclassified with high probability.
    """

    if 'config' in kwargs:
        acquire = kwargs['config'].acquire
        cpu_count = kwargs['config'].cpu_count
        gpu_count = kwargs['config'].gpu_count
    else:
        return None

    if kwargs['config'].info:
        print("Oracle prediction starting...")
        
    #Keep verbosity in 0 to gain speed
    proba = pred_model.predict_generator(generator,
                                             workers=4*cpu_count,
                                             max_queue_size=100*gpu_count,
                                             verbose=0)
            
    pred_classes = proba.argmax(axis=-1)    
    expected = generator.returnLabelsFromIndex()
    miss = np.where(pred_classes != expected)[0]
    miss_prob = np.zeros(shape=expected.shape)
    for k in range(miss.shape[0]):
        miss_prob[miss[k]] = proba[miss[k]][pred_classes[miss[k]]]

    x_pool_idx = np.argsort(miss_prob)[-acquire:]
    
    if kwargs['config'].verbose > 0:
        print('Misses ({}): {}'.format(miss.shape[0]/expected.shape[0],miss))
        print("Probabilities for selected items:\n {}".format(miss_prob[x_pool_idx]))
        print("Selected item's prediction/true label:\n Prediction: {}\n True label: {}".format(pred_classes[x_pool_idx],
                                                                                                   expected[x_pool_idx]))

    return x_pool_idx


def debug_acquisition(s_expected,s_probs,classes,cache_m,config,fidp):
    from Utils import PrintConfusionMatrix
    
    if config.verbose > 0:
        r_class = np.random.randint(0,classes)
        print("Selected item's probabilities for class {2} ({1}): {0}".format(s_probs[r_class],s_probs.shape,r_class))
        prob_mean = np.mean(np.mean(s_probs,axis=-1),axis=-1)
        print("\n".join(["Selected item's mean probabilities for class {}:{}".format(k,prob_mean[k]) for k in range(prob_mean.shape[0])]))
        
    s_pred_all = s_probs[:,:].argmax(axis=0)

    if config.verbose > 0:
        print("Votes: {}".format(s_pred_all))
    #s_pred holds the predictions for each item after a vote
    s_pred = np.asarray([np.bincount(s_pred_all[i]).argmax(axis=0) for i in range(0,s_pred_all.shape[0])])
    if config.verbose > 0:
        print("Classification after vote: {}".format(s_pred))
    PrintConfusionMatrix(s_pred,s_expected,classes,config,"Selected images (AL)")
    if config.save_var:
        cache_m.dump((s_expected,s_probs),fidp)
