#!/usr/bin/env python3
#-*- coding: utf-8

import openslide
from .SegImage import SegImage

class SVSImage( SegImage ):
    """
    SVS images are not handled by OpenCV and need openslide to be handled
    """
    def __init__(self,path,verbose=0):
        """
        @param path <str>: path to image
        """
        super().__init__(path,verbose)

    def readImage(self):
        pass        

    def getImgDim(self):
        pass
