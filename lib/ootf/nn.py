import tensorflow as tf
from tensorflow.contrib.layers import variance_scaling_initializer
from tensorflow.python import control_flow_ops

#------------------------------------------------------------------------------------
def default_initializer():
  return variance_scaling_initializer(factor=1.0, mode='FAN_AVG', uniform=True, dtype=tf.float32) 
#------------------------------------------------------------------------------------



#==================================================================================================
class NeuralNetwork(object):
    BATCH_NORMALIZATION_MOMENTUM = 0.95
  
    #------------------------------------------------------------------------------------
    def __init__(self):
        #........ |  Instance Attributes | ..............................................
        # // Composite object collections \\
        self.FCWeights = []
        self.FCBiases  = []
        self.BatchNormLayers = []
        
        # // Common Tensors \\
        self.IsTraining       = None
        self.GlobalStep       = None
        self.DropOutKeepProb  = None
        self.LearningRate     = None
        self.Momentum         = None
        #................................................................................
        
        # Creates commonly used flags
        self.CreateCommonTensors()
        
        # This is a call to virtual method that descendands override
        self.CreateModel()
    # --------------------------------------------------------------------------------------------------------
    def CreateCommonTensors(self): # virtual
        print(" |_ Creating common neural network tensors")
        with tf.variable_scope("Training"):
            self.IsTraining = tf.placeholder(tf.bool, shape=(), name="IsTraining")
            self.GlobalStep = tf.train.create_global_step()
        
            with tf.variable_scope("HyperParams"):
              self.DropOutKeepProb  = tf.placeholder(tf.float32, shape=(), name="do_keep_prob")
              self.LearningRate     = tf.placeholder(tf.float32, name="lr")
              self.Momentum         = tf.placeholder(tf.float32, name="momentum")
    #------------------------------------------------------------------------------------
    def CreateModel(self): # virtual
        pass
    #------------------------------------------------------------------------------------
    def GetParameter(self, p_tShape, p_tInitializer=tf.initializers.constant(0.0), p_bIsBias=False):
      if p_bIsBias:
        sParamName = "b" 
      else:
        sParamName = "w" 
    
      tParam = tf.get_variable(sParamName, shape=p_tShape, dtype=tf.float32, initializer=p_tInitializer)
      return tParam
    #------------------------------------------------------------------------------------
    def FullyConnected(self, p_tInput, p_nNeuronCount, p_tInitializer=None, p_tActivationFunction=None):
      nInputShape = p_tInput.get_shape().as_list()
      nInputNeurons = nInputShape[-1]
      
      sLayerName = "FC%d" % (len(self.FCWeights) + 1)
      with tf.variable_scope(sLayerName):
        if p_tInitializer is None:
          p_tInitializer = default_initializer()
        tW = self.GetParameter([nInputNeurons, p_nNeuronCount], p_tInitializer)
        tB = self.GetParameter([p_nNeuronCount], p_bIsBias=True ) 
        tU = tf.matmul(p_tInput, tW) + tB
        
        if p_tActivationFunction is not None:
          # Using the function reference that is passed to this method
          tA = p_tActivationFunction(tU)
        else:
          tA = tU
      
        self.FCWeights.append(tW)
        self.FCBiases.append(tB)
      
      print("    [%s] Input:%s, Weights:%s, Output:%s" % (sLayerName, nInputShape, [nInputNeurons, p_nNeuronCount], tA.get_shape().as_list()))
      
      return tA
    # --------------------------------------------------------------------------------------------------------
    def DropOut(self, p_tInput):
      nInputShape = p_tInput.get_shape().as_list()
      tDO = tf.nn.dropout(p_tInput, keep_prob=self.DropOutKeepProb)
      print("    [DO] Input:%s, Output:%s" % (nInputShape, tDO.get_shape().as_list()))
      
      return tDO
    # --------------------------------------------------------------------------------------------------------
    def BatchNormalization(self, p_tInput, p_nBatchNormMomentum=BATCH_NORMALIZATION_MOMENTUM, p_nBatchNormEpsilon=1e-3, p_bIsScalingWithGamma=True):
      """
      Custom implementation of batch normalization layer that will be included in the upcoming machine learning framework by P.I.Kaplanoglou  
      """
      assert self.IsTraining is not None, "Control flags for the neural network are not created"
      
      
      sLayerName = "BN%d" % (len(self.BatchNormLayers) + 1)
      with tf.variable_scope(sLayerName):
        nInputShape = p_tInput.get_shape().as_list()
        nFeatures = nInputShape[-1]
        
        tBeta = tf.get_variable("BN_Beta", shape=[nFeatures], dtype=tf.float32, initializer=tf.initializers.constant(0.0), trainable=True)
        #tBeta = tf.Variable(tf.constant(0.0, shape=[nFeatures], dtype=dtype) , name='BN_beta', trainable=True)
        #self.FCBiases.append(tBeta)
                
        if p_bIsScalingWithGamma:
            tGamma = tf.get_variable("BN_Gamma", shape=[nFeatures], dtype=tf.float32, initializer=tf.initializers.constant(1.0), trainable=True)
            #tGamma = tf.Variable(tf.constant(1.0, shape=[nFeatures], dtype=dtype), name='BN_gamma', trainable=True)
            #self.FCWeights.append(tGamma)
        else:
            tGamma = None
        
        if len(nInputShape) == 4:
            tBatchMean, tBatchVar = tf.nn.moments(p_tInput, [0,1,2], name='BN_moments')
        else:
            tBatchMean, tBatchVar = tf.nn.moments(p_tInput, [0], name='BN_moments')
        tGlobal = tf.train.ExponentialMovingAverage(decay=tf.constant(p_nBatchNormMomentum, dtype=tf.float32), name="BN_global_moments")
        
        def batchMomentsWithUpdate():
            tGlobalMomentsUpdateOp = tGlobal.apply([tBatchMean, tBatchVar])
            with tf.control_dependencies([tGlobalMomentsUpdateOp]):
                return tf.identity(tBatchMean), tf.identity(tBatchVar)
            
        def batchMoments():
            return tf.identity(tBatchMean), tf.identity(tBatchVar)
            
        tGlobalMomentsUpdateOp = tGlobal.apply([tBatchMean, tBatchVar])
        tf.add_to_collection(tf.GraphKeys.UPDATE_OPS, tGlobalMomentsUpdateOp) 
            
        tMean, tVar = control_flow_ops.cond(self.IsTraining, batchMoments,
                                            lambda: (tGlobal.average(tBatchMean), tGlobal.average(tBatchVar)))
        
        tBN = tf.nn.batch_normalization(p_tInput, tMean, tVar, tBeta, tGamma, p_nBatchNormEpsilon)
        
        self.BatchNormLayers.append([tBN, tBeta, tGamma])
        print("    [%s] Input:%s, Output:%s" % (sLayerName, nInputShape, tBN.get_shape().as_list()))
        
        return tBN
    # --------------------------------------------------------------------------------------------------------
            
#==================================================================================================

