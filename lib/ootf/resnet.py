import numpy as np
import tensorflow as tf
from tensorflow.python import control_flow_ops

from ootf.cnn import ConvolutionalNeuralNetwork
from ootf.base import Evaluator





# =======================================================================================================================
class ResidualModule(object):
    # --------------------------------------------------------------------------------------------------------
    def __init__(self, p_oParent, p_nFeatures, p_nStrideOnInput):
        #......................... |  Instance Attributes | .............................
        # // Aggregates \\ 
        self.Parent         = p_oParent
        
        # // Settings \\
        self.Features       = p_nFeatures
        self.StrideOnInput  = p_nStrideOnInput
        
        # // Tensors \\
        self.Input          = None
        self.InputFeatures  = None
        self.Output         = None
        #................................................................................
    # --------------------------------------------------------------------------------------------------------
    def ResidualConnection(self, p_tModuleInput):
        with tf.variable_scope("RESIDUAL"):
          # Spatial downsampling in the first convolutional layer of the module and the skip connection
          if (self.StrideOnInput > 1):
              tSpatialDownsampling = self.Parent.AveragePool(p_tModuleInput, (2,2), (2,2), p_bIsPadding=False)
          else:
              tSpatialDownsampling = p_tModuleInput
                  
          # Pad features if dimensionality of features increases 
          if (self.InputFeatures != self.Features):
              tResidual = self.Parent.PadFeatures(tSpatialDownsampling, self.Features, p_sName="pad_skip")
          else:
              tResidual = tf.identity(tSpatialDownsampling, "skip")
  
        return tResidual
    # --------------------------------------------------------------------------------------------------------
    def Function(self, p_tX):
        self.Input          = p_tX
        self.InputFeatures  = self.Input.get_shape().as_list()[3]

        sModuleName = "RES%d" % (len(self.Parent.Modules) + 1)
        with tf.variable_scope(sModuleName):
            tResidual = self.ResidualConnection(p_tX)
            
            tA = self.Parent.Convolutional(p_tX, self.Features, (3,3), (self.StrideOnInput, self.StrideOnInput), p_nPadding=1, p_bHasBias=False, p_tInitializer=tf.initializers.he_uniform()) #HE INITIALIZATION
            tA = self.Parent.BatchNormalization(tA)
            tA = tf.nn.relu(tA)
            
            tA = self.Parent.Convolutional(tA, self.Features, (3,3), (1,1), p_nPadding=1, p_bHasBias=False, p_tInitializer=tf.initializers.he_uniform())
            tA = self.Parent.BatchNormalization(tA)
            
            tY = tf.nn.relu(tA + tResidual)
                
            self.Output = tY    
        return tY
    # --------------------------------------------------------------------------------------------------------      
# =======================================================================================================================



















   
#==================================================================================================  
class ResNet(ConvolutionalNeuralNetwork):  
    #------------------------------------------------------------------------------------
    def __init__(self, p_nFeatures=[(32,32,3),16,32,64,10], p_nStackSetup=[5,5,5], p_oDataFeed=None):
        #......................... |  Instance Attributes | .............................
        # // Composite object collections \\
        self.Modules = []
        
        # // Aggregates  \\
        self.DataFeed = p_oDataFeed
                
        # // Architectural Hyperparameters \\
        self.StackSetup = p_nStackSetup
        self.DownSampling = [False, True, True, True]

        # // Learning Hyperparameters \\
        #self.Momentum     = 0.9
        self.WeightDecay  = 1e-4
        self.DropOutRate  = 0.5

        # // Tensors \\
        self.Input            = None
        self.Targets          = None
        self.TargetsOneHot    = None
        
        self.Logits           = None
        self.Prediction       = None
        self.PredictedClass   = None
        self.Correct          = None
        self.Accuracy         = None
        
        self.WeightDecayCost  = None
        self.CCECost          = None
        self.CostFunction     = None
        #................................................................................
        
        # Invoke the inherited logic from ancestor :NeuralNetwork
        super(ResNet, self).__init__(p_nFeatures)
    # --------------------------------------------------------------------------------------------------------
    def __getModuleStrides(self, p_nStackIndex):
        nModuleCount = self.StackSetup[p_nStackIndex]
    
        if  self.DownSampling[p_nStackIndex]: 
            nStrides = [2]
        else:
            nStrides = [1]
        nStrides = nStrides + [1]*(nModuleCount -1)
        
        return nStrides        
    # --------------------------------------------------------------------------------------------------------
    def CreateInput(self):
        assert self.DataFeed is not None, "This model works with a TFRecords data feed"
      
        tTrainImageBatch, tTrainLabelBatch = self.DataFeed.TrainingBatches()
        tTestImageBatch, tTestLabelBatch = self.DataFeed.TestingBatches()
        
        tImages, tLabels = control_flow_ops.cond(self.IsTraining,
            lambda: (tTrainImageBatch, tTrainLabelBatch),
            lambda: (tTestImageBatch, tTestLabelBatch))
    
        self.Input = tImages
        self.Targets   = tLabels   
        
        return self.Input, self.Targets 
    #------------------------------------------------------------------------------------
    def CreateModel(self):
      nClassCount = self.Features[-1]
        
      with tf.variable_scope("NeuralNet"):
        #tInput, tTargets = 
        self.CreateInput()
        
        with tf.variable_scope("Targets"):
          self.TargetsOneHot = tf.one_hot(self.Targets, depth=nClassCount, dtype=tf.float32)

        with tf.variable_scope("Stem"):
            tA = self.Convolutional(self.Input, self.Features[1], (3,3), (1,1), p_bIsPadding=True, p_bHasBias=False, p_tInitializer=tf.initializers.he_uniform())
            tA = self.BatchNormalization(tA)
            tA = tf.nn.relu(tA)

        # Creates the stacks of residual modules
        nStackCount = len(self.StackSetup)
        for nStackIndex in range(0, nStackCount):
            nModuleStrides = self.__getModuleStrides(nStackIndex)
            
            sStackName = "Stack%d" % (nStackIndex + 1)
            with tf.variable_scope(sStackName):
                print("  [%s]" % sStackName)              
                for nModuleIndex, nModuleInputStride in enumerate(nModuleStrides):
                    oResBlock = ResidualModule(self, self.Features[nStackIndex + 1], nModuleInputStride)
                    tA = oResBlock.Function(tA)
                    
                    self.Modules.append(oResBlock)
                    print(" |_ Res%d" % (nModuleIndex+1), oResBlock.Input, oResBlock.Output)                                      
                  
        # For each output feature,  averages the values in its spatial activation table. 
        # This results feature activation vector for each image
        tA = self.GlobalAveragePooling(tA)
        
        # Softmax layer (classifier)
        self.Logits     = self.FullyConnected(tA, nClassCount)
        self.Prediction = tf.nn.softmax(self.Logits)       
        
        # Predictions and accuracy
        with tf.variable_scope("Predictions"):
            self.PredictedClass     = tf.argmax(self.Prediction , 1, output_type=tf.int32)
            self.Correct            = tf.equal(self.PredictedClass, tf.cast(self.Targets, tf.int32))
            self.Accuracy           = tf.reduce_mean(tf.cast(self.Correct , tf.float32), name='accuracy')
        
      # Prepare cost function tensors for training
      self.DefineCostFunction()
    # --------------------------------------------------------------------------------------------------------
    def DefineCostFunction(self):
        with tf.variable_scope("Cost"):
            # Multiclass categorical cross entropy (CCE) loss
            tLoss = tf.nn.softmax_cross_entropy_with_logits_v2(logits=self.Logits, labels=self.TargetsOneHot)
            self.CCECost  = tf.reduce_mean(tLoss, name="cce")
            
            # L2 weight decay regularization
            tL2LossesConv = [tf.nn.l2_loss(tKernel) for tKernel in self.ConvWeights]
            tL2LossesFC   = [tf.nn.l2_loss(tWeight) for tWeight in self.FCWeights]
            tL2LossesAll  = tL2LossesConv + tL2LossesFC
            tWeightDecay = tf.constant(self.WeightDecay, tf.float32, name="weight_decay")
            self.WeightDecayCost = tf.multiply(tWeightDecay, tf.add_n(tL2LossesAll), name="l2")

            # Total cost
            self.CostFunction = tf.identity(self.CCECost + self.WeightDecayCost, "total_cost")
    #------------------------------------------------------------------------------------
    def Feed(self, p_nBatchFeatures, p_nBatchTargets=None, p_nLearningRate=None, p_bIsTraining=False):
      oDict = dict()
      
      oDict[self.Input]      = p_nBatchFeatures
      oDict[self.IsTraining] = p_bIsTraining
      
      if p_bIsTraining:
        oDict[self.Targets]      = p_nBatchTargets
        oDict[self.LearningRate] = p_nLearningRate
        oDict[self.DropOutKeepProb] = self.DropOutRate
      else:
        # When the model is trained the neurons are not dropped out
        oDict[self.DropOutKeepProb] = 1.0
        
      return oDict
    # --------------------------------------------------------------------------------------------------------
    def Predict(self, p_oSession, p_oSubSet):   
      nPredictedClasses = np.zeros(p_oSubSet.Labels.shape, np.uint32)
      for nRange, nSamples, nLabels, _ in p_oSubSet:
        nPrediction = p_oSession.run(self.Prediction, feed_dict=self.Feed(nSamples, nLabels))
        nPredictedClasses[nRange] = np.argmax(nPrediction, axis=1).astype(np.uint32)
      
      return nPredictedClasses
    # --------------------------------------------------------------------------------------------------------
    def Evaluate(self, p_oDataSubSet=None, p_nBatchSize=1000, p_oProcess=None, p_oSession=None):
        assert self.DataFeed is not None, "This model works with a TFRecords data feed"
      
        n_val_samples = 10000
        val_batch_size = self.DataFeed.TestBatchSize
        
        n_val_batch = int(  np.ceil(n_val_samples / val_batch_size) )
        
        #val_logits = np.zeros((n_val_samples, 10), dtype=np.float32)
        val_labels = np.zeros((n_val_samples), dtype=np.int64)
        
        pred_labels = np.zeros((n_val_samples), dtype=np.int64)
        val_losses = []
        for i in range(n_val_batch):
            fetches = [self.Prediction, self.Targets, self.CostFunction]
            fetches.append(self.PredictedClass)
            
            if p_oDataSubSet is None:
                oFeedDict = {self.IsTraining: False}
            else:
                oFeedDict = dict()
                oFeedDict[self.Input]       = p_oDataSubSet.Patterns[i * val_batch_size:(i + 1) * val_batch_size,...]
                oFeedDict[self.Targets]     = p_oDataSubSet.Labels[i * val_batch_size:(i + 1) * val_batch_size,...]
                oFeedDict[self.IsTraining]  = False
                
            if p_oSession is not None:
              session_outputs = p_oSession.run(fetches, oFeedDict)
            else:   
              session_outputs = self.Session.run(fetches, oFeedDict)
            
            pred_labels[i * val_batch_size:(i + 1) * val_batch_size] = session_outputs[3]
            val_labels[i * val_batch_size:(i + 1) * val_batch_size] = session_outputs[1]
            val_losses.append(session_outputs[2])
        
        oEval = Evaluator(val_labels, pred_labels)
        # print(nAvgBatchesAccuracy)
        if p_oProcess is None:
            print("Accuracy:", oEval.Accuracy)
        else:
            p_oProcess.Print("Accuracy:", oEval.Accuracy)
        print(oEval.ConfusionMatrix)
                            
        val_loss     = float(np.mean(np.asarray(val_losses)))    
        val_accuracy = oEval.Accuracy
        
        return val_loss, val_accuracy
    # --------------------------------------------------------------------------------------------------------
#==================================================================================================  

  
  
# --------------------------------------------------------------------------------------------------------  
def testReproducibility(p_oModel, p_oSession):
  # The following command has been executed 
  #tf.set_random_seed(2019)

  tW_CONV19 = p_oModel.ConvWeights[-1]
  tW_FC20 = p_oModel.FCWeights[-1]
  assert tW_CONV19.name == "NeuralNet/Stack3/RES9/CONV_2ND/CONV19/w:0", "Architecture is not ResNet20"
  assert tW_FC20.name == "NeuralNet/FC1/w:0", "Architecture is not ResNet20"

  
  nW_CONV19 = tW_CONV19.eval(p_oSession)
  nW_FC20 = tW_FC20.eval(p_oSession)
  
  print("[%s] Initial Weights: Mean=%.6f Std=%.6f" % (tW_CONV19.name, np.round(np.mean(nW_CONV19), 6), np.round(np.std(nW_CONV19), 6)) )
  print("[%s] Initial Weights: Mean=%.6f Std=%.6f" % (tW_FC20.name, np.round(np.mean(nW_FC20), 6), np.round(np.std(nW_FC20), 6)) )
  
  
  assert (np.round(np.mean(nW_CONV19), 6) + 0.00028) <= 1e-5, "CONV19 kernel initial values have different mean than expected"
  assert (np.round(np.std(nW_CONV19), 6) - 0.05897) <= 1e-5, "CONV19 kernel initial values have different std than expecteds"

  assert (np.round(np.mean(nW_FC20), 6) - 0.00532) <= 1e-5, "FC20 kernel initial values have different mean than expected"
  assert (np.round(np.std(nW_FC20), 6) - 0.16359) <= 1e-5, "FC20 kernel initial values have different std than expecteds"
# --------------------------------------------------------------------------------------------------------    
  
  