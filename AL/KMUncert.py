#!/usr/bin/env python3
#-*- coding: utf-8

import numpy as np
import os
from tqdm import tqdm

from scipy.stats import mode

__doc__ = """
All acquisition functions should receive:
1 - numpy array of items
2 - numpy array of labels
3 - number of items to query
4 - keyword arguments specific for each function (if needed)

Returns: numpy array of element indexes
"""

def km_uncert(bayesian_model,generator,data_size,**kwargs):
    """
    Cluster in K centroids and extract N samples from each cluster, based on maximum bayesian_varratios
    uncertainty.

    Function needs to extract the following configuration parameters:
    model <keras.Model>: model to use for predictions
    generator <keras.Sequence>: data generator for predictions
    data_size <int>: number of data samples
    mc_dp <int>: number of dropout iterations
    cpu_count <int>: number of cpu cores (used to define number of generator workers)
    gpu_count <int>: number of gpus available
    verbose <int>: verbosity level
    pbar <boolean>: user progress bars
    sw_threads <thread Object>: if a thread object is passed, you must wait its conclusion before loading weights
    """
    from sklearn.cluster import KMeans
    from sklearn.decomposition import PCA
    import importlib
    import copy
    import time
    from datetime import timedelta
    from Utils import CacheManager

    cache_m = CacheManager()
    
    if 'config' in kwargs:
        config = kwargs['config']
        gpu_count = config.gpu_count
        cpu_count = config.cpu_count
        verbose = config.verbose
        pbar = config.progressbar
        query = config.acquire
        clusters = config.clusters
    else:
        return None
    
    if 'acquisition' in kwargs:
        acq = kwargs['acquisition']
    else:
        acq = config.acquisition_steps
        
    if 'model' in kwargs:
        model = kwargs['model']
    else:
        print("[km_uncert] GenericModel is needed by km_uncert. Set model kw argument")
        return None

    ## UNCERTAINTY CALCULATION FIRST 
    #Any uncertainty function could be used
    n_config = copy.copy(config)
    n_config.acquire = data_size
    kwargs['config'] = n_config
    un_function = getattr(importlib.import_module('AL'),config.un_function)
    un_indexes = un_function(bayesian_model,generator,data_size,**kwargs)    

    #Models that take to long to save weights might not have finished
    if 'sw_thread' in kwargs:
        if config.ffeat is None and kwargs['sw_thread'].is_alive():
            if config.info:
                print("[km_uncert] Waiting for model weights to become available...")
            kwargs['sw_thread'].join()
    elif config.info:
        print("[km_uncert] Weights thread not available...trying to load weights")

    if not os.path.isfile(model.get_weights_cache()) and not os.path.isfile(model.get_mgpu_weights_cache()):
        if config.info:
            print("[km_uncert] No trained model or weights file found")
        return None

    if config.recluster > 0 and acq > 0 and (acq % config.recluster) != 0:
        km,acquired = cache_m.load('clusters.pik')
        if config.info:
            print("[km_uncert] Loaded clusters from previous acquisition")
            #TODO: REMOVE
            print("Previous cluster size: {};\nAcquired: {}".format(km.labels_.shape,acquired.shape))
        km.labels_ = np.delete(km.labels_,acquired)
    else:
        #Run feature extraction and clustering
        if hasattr(model,'build_extractor'):
            single_m,parallel_m = model.build_extractor(training=False,feature=True,parallel=False)
        else:
            if config.info:
                print("[km_uncert] Model is not prepared to produce features. No feature extractor")
            return None

        #Model can be loaded from previous acquisition train or from a fixed final model
        if gpu_count > 1 and not parallel_m is None:
            pred_model = parallel_m
            if not config.ffeat is None and os.path.isfile(config.ffeat):
                pred_model.load_weights(config.ffeat,by_name=True)
                if config.info:
                    print("Model weights loaded from: {0}".format(config.ffeat))
            else:
                pred_model.load_weights(model.get_mgpu_weights_cache(),by_name=True)
                if config.info:
                    print("Model weights loaded from: {0}".format(model.get_mgpu_weights_cache()))
        else:
            pred_model = single_m
            if not config.ffeat is None and os.path.isfile(config.ffeat):
                pred_model.load_weights(config.ffeat,by_name=True)
                if config.info:
                    print("Model weights loaded from: {0}".format(config.ffeat))
            else:
                pred_model.load_weights(model.get_weights_cache(),by_name=True)
                if config.info:
                    print("Model weights loaded from: {0}".format(model.get_weights_cache()))
                    
        #Extract features for all images in the pool
        if config.info:
            print("Starting feature extraction ({} batches)...".format(len(generator)))        
        features = pred_model.predict_generator(generator,
                                                workers=4*cpu_count,
                                                max_queue_size=100*gpu_count,
                                                verbose=0)
        features = features.reshape(features.shape[0],np.prod(features.shape[1:]))

        if config.pca > 0:
            if config.info:
                print("Starting PCA decomposition...")

            pca = PCA(n_components = config.pca)
            features = pca.fit_transform(features)
            
        stime = None
        etime = None
        if config.verbose > 0:
            print("Done extraction...starting KMeans")
            stime = time.time()
        
        km = KMeans(n_clusters = clusters, init='k-means++',n_jobs=int(cpu_count/2)).fit(features)
        
        if config.verbose > 0:
            etime = time.time()
            td = timedelta(seconds=(etime-stime))
            print("KMeans took {}".format(td))

    un_clusters = {k:[] for k in range(config.clusters)}

    #Distributes items in clusters in descending order of uncertainty
    for iid in un_indexes:
        un_clusters[km.labels_[iid]].append(iid)

    #Save clusters
    if config.save_var:
        fid = 'al-clustermetadata-{1}-r{0}.pik'.format(acq,model.name)
        cache_m.registerFile(os.path.join(config.logdir,fid),fid)
        cache_m.dump((generator.returnDataAsArray(),un_clusters,un_indexes),fid)
        
    #If debug
    if config.debug:
        expected = generator.returnLabelsFromIndex()
        for k in range(len(un_clusters)):
            ind = np.asarray(un_clusters[k])
            print("Cluster {}, # of items: {}".format(k,ind.shape[0]))
            posa = np.ndarray(shape=(1,),dtype=np.int32)
            for ii in range(min(ind.shape[0],30)):
                if ii == 0:
                    posa[0] = np.where(un_indexes == ind[ii])[0]
                else:
                    posa = np.hstack((posa,np.where(un_indexes == ind[ii])[0]))
            print("Cluster {} first items positions in index array (at most 30): {}".format(k,posa))
            #Check % of items of each class in cluster k
            c_labels = expected[ind]
            unique,count = np.unique(c_labels,return_counts=True)
            l_count = dict(zip(unique,count))
            if len(unique) > 2:
                print("Cluster {} items:".format(k))
                print("\n".join(["label {0}: {1} items" .format(key,l_count[key]) for key in unique]))
            else:
                if c_labels.shape[0] == 1:
                    l_count[c_labels[0] ^ 1] = 0
                print("Cluster {3} labels: {0} are 0; {1} are 1;\n - {2:.2f} are positives".format(l_count[0],l_count[1],(l_count[1]/(l_count[0]+l_count[1])),k))            
            
    ac_count = 0
    acquired = []
    j = 0
    while ac_count < query:
        cln = (ac_count+j) % clusters
        q = un_clusters[cln]
        if len(q) > 0:
            acquired.append(q.pop(0))
            ac_count += 1
        else:
            if verbose > 0:
                print("[km_uncert] Cluster {} exausted, will try to acquire image from cluster {}".format(cln,(cln+1)%clusters))
            j += 1
            continue

    acquired = np.asarray(acquired)
    if config.recluster > 0:
        cache_m.dump((km,acquired),'clusters.pik')
    
    return acquired
