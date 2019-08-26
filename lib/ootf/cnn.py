import numpy as np
import tensorflow as tf
from ootf.nn import NeuralNetwork, default_initializer
 


 
#==================================================================================================  
class ConvolutionalNeuralNetwork(NeuralNetwork):  
    #------------------------------------------------------------------------------------
    def __init__(self, p_nFeatures=[128,256,512,10]):
        #........ |  Instance Attributes | ..............................................
        # // Composite object collections \\
        # ... Weights ...
        self.ConvWeights        = []
        self.ConvBiases         = []
        # ... Layers ...
        self.MaxPoolLayers      = []
        self.AveragePoolLayers  = []
        
        # // Architectural Hyperparameters \\
        self.Features = p_nFeatures
        #................................................................................
        
        # Invoke the inherited logic from ancestor :NeuralNetwork
        super(ConvolutionalNeuralNetwork, self).__init__()
    #------------------------------------------------------------------------------------
    def Convolutional(self, p_tInput, p_nNeuronCount, p_nWindowSize=[3,3], p_nStrides=[1,1], p_bIsPadding=True, p_nPadding=None, p_bHasBias=True, p_tInitializer=None, p_tActivationFunction=None):
      nLayerNum = len(self.ConvWeights) + 1
      
      nInputShape  = p_tInput.get_shape().as_list()
      nInFeatures  = nInputShape[3]          
      nKernelShape = list(p_nWindowSize) + [nInFeatures, p_nNeuronCount]
      nStrides     = [1] + list(p_nStrides) + [1]
      tX = p_tInput
      if p_nPadding is not None:
        sPadding = "VALID"
        tX = self.PadSpatial(p_tInput, p_nWindowSize[0], p_nPadding)
      else:      
        if p_bIsPadding:
          sPadding = "SAME"
        else:
          sPadding = "VALID"
      
      sLayerName = "CONV%d" % nLayerNum
      with tf.variable_scope(sLayerName):
        if p_tInitializer is None:
          p_tInitializer = default_initializer()     
        tW = self.GetParameter(nKernelShape, p_tInitializer) 
        tU = tf.nn.conv2d(tX, tW, strides=nStrides, padding=sPadding) 
        
        if p_bHasBias:
          tB = self.GetParameter([p_nNeuronCount], p_bIsBias=True)
          tU = tU + tB
        
        if p_tActivationFunction is not None:
          # Activation function is a function reference that is passed to the method
          tA = p_tActivationFunction(tU)
        else:
          tA = tU
      
        self.ConvWeights.append(tW)
        if p_bHasBias:
          self.ConvBiases.append(tB)
        
      print("    [%s] Input:%s, Kernel:%s, Output:%s" % (sLayerName, nInputShape, nKernelShape, tA.get_shape().as_list()))
      
      return tA
    #------------------------------------------------------------------------------------
    def MaxPool(self, p_tInput, p_nPoolSize=[2,2], p_nPoolStrides=[2,2], p_bIsPadding=True, p_nPadding=None):
      nLayerNum = len(self.MaxPoolLayers) + 1
      nInputShape  = p_tInput.get_shape().as_list()

      tX = p_tInput
      if p_nPadding is not None:
        sPadding = "VALID"
        tX = self.PadSpatial(p_tInput, p_nPoolSize[0], p_nPadding)
      else:      
        if p_bIsPadding:
          sPadding = "SAME"
        else:
          sPadding = "VALID"
        
      nSize = [1] + list(p_nPoolSize) + [1]
      nStrides = [1] + list(p_nPoolStrides) + [1]
       
      sLayerName = "MAXP%d" % nLayerNum
      with tf.variable_scope(sLayerName):
        tA = tf.nn.max_pool(tX, ksize=nSize, strides=nStrides, padding=sPadding)
        self.MaxPoolLayers.append(tA)
      
      print("    [%s] Input:%s, Pool:%dx%d/%s, Output:%s" % (sLayerName, 
                  nInputShape, p_nPoolSize[0], p_nPoolSize[1], p_nPoolStrides[0], tA.get_shape().as_list()))
      
      return tA 
    # --------------------------------------------------------------------------------------------------------
    def AveragePool(self, p_tInput, p_nPoolSize=(2,2), p_nPoolStrides=(2,2), p_bIsPadding=False):
      nInputShape  = p_tInput.get_shape().as_list()

      if p_bIsPadding:
        sPadding = "SAME"
      else:
        sPadding = "VALID"
        
      nSize = [1] + list(p_nPoolSize) + [1]
      nStrides = [1] + list(p_nPoolStrides) + [1]
      
      sLayerName = "AVGP%d" % (len(self.AveragePoolLayers) + 1)
      with tf.variable_scope(sLayerName):         
        tA = tf.nn.avg_pool(p_tInput, ksize=nSize, strides=nStrides, padding=sPadding)
        self.AveragePoolLayers.append(tA)
        
      print("    [%s] Input:%s, Pool:%dx%d/%s, Output:%s" % (sLayerName, 
                  nInputShape, p_nPoolSize[0], p_nPoolSize[1], p_nPoolStrides[0], tA.get_shape().as_list()))
             
      return tA    
    # --------------------------------------------------------------------------------------------------------
    def Flatten(self, x):
        nFlatConvDims = np.prod(np.asarray(x.get_shape().as_list()[1:]))
        tFlatten = tf.reshape(x, [-1, nFlatConvDims])
        return tFlatten      
    # --------------------------------------------------------------------------------------------------------
    def GlobalAveragePooling(self, p_tInput):
      nInputShape  = p_tInput.get_shape().as_list()
      nPoolSize = [1,nInputShape[1],nInputShape[2],1]
      
      with tf.variable_scope("GAVGPOOL"):
        tA = tf.nn.avg_pool(p_tInput, ksize=nPoolSize, strides=[1,1,1,1], padding="VALID", name="avgpool")
        tA = tf.squeeze(tA, [1,2], name="squeeze")
        
      print("    [GAVGPOOL] Input:%s, Output:%s" % (nInputShape, tA.get_shape().as_list()))
      
      return tA
    # --------------------------------------------------------------------------------------------------------
    def PadSpatial(self, p_tInput, p_nMapDim, p_oPadding=0, p_sName=None):
        tOutput = p_tInput
        
        if (p_oPadding>0):
          nTotalPadding = p_nMapDim - 1
          nPaddingStart  = nTotalPadding // 2
          nPaddingEnd    = nTotalPadding - nPaddingStart
          tOutput = tf.pad(tensor=p_tInput, paddings=[[0, 0], [nPaddingStart, nPaddingEnd], [nPaddingStart, nPaddingEnd], [0, 0]], name=p_sName)
    
        
        return tOutput
    # --------------------------------------------------------------------------------------------------------
    def PadFeatures(self, p_tInput, p_nFeatureDepth, p_sName=None):
        nInputShape = p_tInput.get_shape().as_list()
        
        nPadFeatures = p_nFeatureDepth - nInputShape[3]
        tPad =  tf.pad(p_tInput, [[0, 0], [0, 0], [0, 0], [nPadFeatures // 2, nPadFeatures // 2]], name=p_sName)
        
        return tPad    
    # --------------------------------------------------------------------------------------------------------      
#==================================================================================================