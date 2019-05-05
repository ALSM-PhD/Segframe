#!/usr/bin/env python3
#-*- coding: utf-8

import os
import numpy as np
import skimage
from skimage import io

from .SegImage import SegImage

class PImage(SegImage):
    """
    Represents any image handled by OpenCV.
    """
    def __init__(self,path,keepImg=False,origin=None,coord=None,verbose=0):
        """
        @param path <str>: path to image
        @param keepImg <bool>: keep image data in memory
        @param origin <str>: current image is originated from origin
        @param coord <tuple>: coordinates in original image
        """
        super().__init__(path,keepImg,verbose)
        self._coord = coord
        self._origin = origin

    def __str__(self):
        """
        String representation is (coord)-origin if exists, else, file name
        """
        if not (self._coord is None and self._origin is None):
            return "{0}-{1}".format(self._coord,self._origin)
        else:
            return os.path.basename(self._path)

    def __repr__(self):
        return self.__str__()

    def __hash__(self):
        # Hashes current dir and file name
        return hash((self._path.split(os.path.sep)[-2],os.path.basename(self._path)))
    
    def readImage(self,keepImg=None,size=None,verbose=None):
        
        data = None

        if keepImg is None:
            keepImg = self._keep
        elif keepImg:
            #Change seting if we are going to keep the image in memory now
            self.setKeepImg(keepImg)
        if not verbose is None:
            self._verbose = verbose
            
        if self._data is None:
            if self._verbose > 1:
                print("Reading image: {0}".format(self._path))
                
            data = io.imread(self._path);
            if(data.shape[2] > 3): # remove the alpha
                data = data[:,:,0:3];
                
            if not size is None and data.shape != size:
                if self._verbose > 1:
                    print("Resizing image {0} from {1} to {2}".format(os.path.basename(self._path),data.shape,size))
                data = skimage.transform.resize(data,size)
                
            h,w,c = data.shape
            self._dim = (w,h,c)
            
            if self._keep:
                self._data = data
                
        else:
            if self._verbose > 1:
                print("Data already loaded:\n - {0}".format(self._path))
            data = self._data

        return data
    
    def readImageRegion(self,x,y,dx,dy):
        data = None
        
        if self._data is None:
            data = self.readImage()
        else:
            data = self._data
            
        return data[y:(y+dy), x:(x+dx)]

    def setKeepImg(self,keep):
        """
        If image should not be held anymore, delete data
        """
        if keep is None:
            return
        
        if not keep:
            del self._data
            self._data = None

        self._keep = keep
        
    def getImgDim(self):
        """
        Implements abstract method of SegImage
        """
        h,w,c = 0,0,0

        if not self._dim is None:
            return self._dim
        elif not self._data is None:
            h,w,c = self._data.shape
        else:
            data = io.imread(self._path);
            if(data.shape[2] > 3): # remove the alpha
                data = data[:,:,0:3];
            h,w,c = data.shape
            
            if self._keep:
                self._data = data

        self._dim = (w,h,c)
        return self._dim
