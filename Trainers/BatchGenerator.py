#!/usr/bin/env python3
#-*- coding: utf-8
#Author: André L. S. Meirelles (andre.meirelles@aluno.unb.br)

#Filter warnings
import warnings
warnings.filterwarnings('ignore')
    
import keras
from keras.utils.data_utils import Sequence
from keras.preprocessing.image import Iterator,ImageDataGenerator
from keras import backend as K

#System modules
import concurrent.futures
import numpy as np
import imgaug as ia
from imgaug import augmenters as iaa

class GenericIterator(Iterator):
    """
        RHDIterator is actually a generator, yielding the data tuples from a data source as a correlation list.
        
        # Arguments
        image_data_generator: Instance of `ImageDataGenerator`
        data: tuple (X,Y) where X are samples, Y are corresponding labels to use for random transformations and normalization.
        classes: number of classes
        batch_size: Integer, size of a batch.
        image_generator: Keras ImageGenerator for data augmentation
        extra_aug: Do more data augmentation/normalizations here
        shuffle: Boolean, whether to shuffle the data between epochs.
        seed: Random seed for data shuffling.
        data_mean: float, dataset mean for zero centering
        verbose: verbosity level.
        input_n: number of input sources (for multiple submodels in ensemble)
    """

    def __init__(self,
                     data,
                     classes,
                     dim=None,
                     batch_size=8,
                     image_generator=None,
                     extra_aug=False,
                     shuffle=True,
                     seed=173,
                     data_mean=0.0,
                     verbose=0,
                     input_n=1):

        self.data = data
        self.classes = classes
        self.dim = dim
        self.mean = data_mean
        self.image_generator = None
        self.verbose = verbose
        self.extra_aug = extra_aug
        self.input_n = input_n

        #Keep information of example shape as soon as the information is available
        self.shape = None
        
        if not image_generator is None and isinstance(image_generator,ImageDataGenerator):
            self.image_generator = image_generator
        elif not image_generator is None:
            raise TypeError("Image generator should be an " \
            "ImageDataGenerator instance")

        if isinstance(self.data[0],np.ndarray):
            super(GenericIterator, self).__init__(n=self.data[0].shape[0], batch_size=batch_size, shuffle=shuffle, seed=seed)
        else:
            super(GenericIterator, self).__init__(n=len(self.data[0]), batch_size=batch_size, shuffle=shuffle, seed=seed)


    def returnDataSize(self):
        """
        Returns the number of examples
        """
        return self.n
    

    def next(self):
        """
        For python 2.x.
        # Returns
        The next batch.
        """
            
        # Keeps under lock only the mechanism which advances
        # the indexing of each batch.
        with self.lock:
            index_array = next(self.index_generator)
        # The transformation of images is not under thread lock
        # so it can be done in parallel
        return self._get_batches_of_transformed_samples(index_array)
            
    def returnDataInOrder(self,idx):
        """
        Returns a data batch, starting in position idx
        """
        index_array = [i for i in range(idx,idx+self.batch_size)]
        # Check which element(s) to use
        return self._get_batches_of_transformed_samples(index_array)

    def returnLabelsFromIndex(self,idx=None):
        """
        Returns the labels of the data samples refered by idx. Useful for 
        debuging.

        @param idx <int,ndarray>: a single index or an array of indexes
        """
        Y = np.asarray(self.data[1])
        if idx is None:
            return Y
        
        if isinstance(idx,int) or isinstance(idx,np.ndarray):
            return Y[idx]

    def returnDataAsArray(self):
        """
        Return all data as a tuple of ndarrays: (X,Y)
        """

        return (np.asarray(self.data[0]),np.asarray(self.data[1]))

class SingleGenerator(GenericIterator):
    """
    Generates batches of images, applies augmentation, resizing, centering...the whole shebang.
    """
    def __init__(self, 
                     dps,
                     classes,
                     dim=None,
                     batch_size=8,
                     image_generator=None,
                     extra_aug=False,
                     shuffle=True,
                     seed=173,
                     data_mean=0.0,
                     verbose=0,
                     variable_shape=False,
                     input_n=1):
        
        #Set True if examples in the same dataset can have variable shapes
        self.variable_shape = variable_shape
        
        super(SingleGenerator, self).__init__(data=dps,
                                                classes=classes,
                                                dim=dim,
                                                batch_size=batch_size,
                                                image_generator=image_generator,
                                                extra_aug=extra_aug,
                                                shuffle=shuffle,
                                                seed=seed,
                                                data_mean=data_mean,
                                                verbose=verbose,
                                                input_n=input_n)


    def _get_batches_of_transformed_samples(self,index_array):
        """
        Only one argument will be considered. The index array has preference

        #Arguments
           index_array: array of sample indices to include in batch; or
        # Returns 
            a batch of transformed samples
        """
        #For debuging
        if self.verbose > 1:
            print(" index_array: {0}".format(index_array))
            
        # calculate dimensions of each data point
        #Should only create the batches of appropriate size
        if not self.shape is None:
            batch_x = np.zeros(tuple([len(index_array)] + list(self.shape)), dtype=K.floatx())
        else:
            batch_x = None
        y = np.zeros(tuple([len(index_array)]),dtype=int)
                
        # generate a random batch of points
        X = self.data[0]
        Y = self.data[1]
        for i,j in enumerate(index_array):
            t_x = X[j]
            t_y = Y[j]

            #If not an ndarray, readimage
            if not isinstance(t_x,np.ndarray):
                example = t_x.readImage(size=self.dim,verbose=self.verbose)
            else:
                example = t_x
            
            if batch_x is None:
                self.shape = example.shape
                batch_x = np.zeros(tuple([len(index_array)] + list(self.shape)),dtype=K.floatx())
            
            #TEST PURPOSES ONLY - This is slow given the sizes
            #involved
            if not self.image_generator is None:
                example = self.image_generator.random_transform(example,self.seed)
                #example = self.image_generator.standardize(example)

            # add point to x_batch and diagnoses to y
            batch_x[i] = example
            y[i] = t_y
        batch_x = self.image_generator.standardize(batch_x)
        #Center data
        #batch_x -= self.mean
        #Normalize data pixels
        #batch_x /= 255

        if self.variable_shape:
            self.shape = None

        if self.input_n > 1:
            batch_x = [batch_x for _ in range(self.input_n)]
            
        output = (batch_x, keras.utils.to_categorical(y, self.classes))
        return output         

class ThreadedGenerator(GenericIterator):
    """
    Generates batches of images, applies augmentation, resizing, centering...the whole shebang.
    """
    def __init__(self, 
                     dps,
                     classes,
                     dim=None,
                     batch_size=8,
                     image_generator=None,
                     extra_aug=False,
                     shuffle=True,
                     seed=173,
                     data_mean=0.0,
                     verbose=0,
                     variable_shape=False,
                     input_n=1):
        
        #Set True if examples in the same dataset can have variable shapes
        self.variable_shape = variable_shape
        self._executor = None
        self._aug = None
        
        super(ThreadedGenerator, self).__init__(data=dps,
                                                classes=classes,
                                                dim=dim,
                                                batch_size=batch_size,
                                                image_generator=image_generator,
                                                extra_aug=extra_aug,
                                                shuffle=shuffle,
                                                seed=seed,
                                                data_mean=data_mean,
                                                verbose=verbose,
                                                input_n=input_n)


    def _get_batches_of_transformed_samples(self,index_array):
        """
        Only one argument will be considered. The index array has preference

        #Arguments
           index_array: array of sample indices to include in batch; or
        # Returns 
            a batch of transformed samples
        """
        #Start thread pool if not already started
        if self._executor is None:
            workers = round((self.batch_size/3 + (self.batch_size%3>0) +0.5))
            self._executor = concurrent.futures.ThreadPoolExecutor(max_workers=workers)

        #Additional data augmentation
        if self.extra_aug and self._aug is None:
            self._aug = iaa.Sometimes(0.5,
                iaa.ContrastNormalization((0.75,1.5))
                )

        #For debuging
        if self.verbose > 1:
            print(" index_array: {0}".format(index_array))
            
        # calculate dimensions of each data point
        #Should only create the batches of appropriate size
        if not self.shape is None:
            batch_x = np.zeros(tuple([len(index_array)] + list(self.shape)), dtype=K.floatx())
        else:
            batch_x = None
        y = np.zeros(tuple([len(index_array)]),dtype=int)
                
        # generate a random batch of points
        X = self.data[0]
        Y = self.data[1]
        futures = []

        for i,j in enumerate(index_array):
            t_x = X[j]
            t_y = Y[j]
            futures.append(self._executor.submit(self._thread_run_images,t_x,t_y))
            
        for i in range(len(futures)):
            # add point to x_batch and diagnoses to y
            example,t_y = futures[i].result()
            if batch_x is None:
                self.shape = example.shape
                batch_x = np.zeros(tuple([len(index_array)] + list(self.shape)),dtype=K.floatx())            
            batch_x[i] = example
            y[i] = t_y

        #Always normalize
        batch_x = self.image_generator.standardize(batch_x)
        #Apply extra augmentation
        if self.extra_aug:
            batch_x = self._aug(images=batch_x)
        
        del(futures)
        #Center data
        #batch_x -= self.mean
        #Normalize data pixels
        #batch_x /= 255

        if self.variable_shape:
            self.shape = None

        if self.input_n > 1:
            batch_x = [batch_x for _ in range(self.input_n)]
            
        output = (batch_x, keras.utils.to_categorical(y, self.classes))
        return output

    def _thread_run_images(self,t_x,t_y):
        example = t_x.readImage(size=self.dim,verbose=self.verbose)
            
        if not self.image_generator is None:
            example = self.image_generator.random_transform(example,self.seed)
            #example = self.image_generator.standardize(example)

        return (example,t_y)
