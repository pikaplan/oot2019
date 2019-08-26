import os
import numpy as np

import tensorflow as tf
from sklearn import metrics

# =======================================================================================================================
class DataSubSet(object):
    # --------------------------------------------------------------------------------------------------------
    def __init__(self):
        self.Samples            = None
        self.Labels             = None
        self.UIDs               = None
        self.ShuffledUIDs       = None
        self.SampleCount        = None
        
        self.__batchSize        = None
        self.BatchCount         = None
        
        self.EndOfData          = True
        self.BatchIndex         = None
        
        self.IsShuffling        = False
    # --------------------------------------------------------------------------------------------------------
    @property
    def BatchSize(self):
      return self.__batchSize
    # --------------------------------------------------------------------------------------------------------
    @BatchSize.setter
    def BatchSize(self, p_nValue):
      self.__batchSize = p_nValue
      self.BatchCount = int(self.SampleCount / self.BatchSize)
      if (self.SampleCount % self.BatchSize) != 0:
        self.BatchCount += 1
    # --------------------------------------------------------------------------------------------------------
    def AppendShard(self, p_nSamples, p_nLabels):
        if self.Samples is None:
            self.Samples = p_nSamples
            self.SampleCount = 0
        else:
            self.Samples = np.concatenate((self.Samples, p_nSamples), axis=0)
        
        if self.Labels is None:
            self.Labels = p_nLabels
        else:
            self.Labels = np.concatenate((self.Labels, p_nLabels), axis=0)
            
        nNextUIDs = np.arange(self.SampleCount, self.SampleCount + p_nSamples.shape[0],  dtype=np.int32)
            
        if self.UIDs is None:
          self.UIDs = nNextUIDs
        else:    
          self.UIDs = np.concatenate((self.UIDs, nNextUIDs), axis=0)
          
        self.SampleCount += p_nSamples.shape[0]
    # --------------------------------------------------------------------------------------------------------
    def GetSamples(self, p_nIndexes):
      
      nSamples = self.Samples[p_nIndexes]
      nLabels  = self.Labels[p_nIndexes]
      nUIDs    = self.UIDs[p_nIndexes]
      
      return nSamples, nLabels, nUIDs
    # --------------------------------------------------------------------------------------------------------
    def __iter__(self):
        assert self.__batchSize is not None, "Batch size not specified"
        
        self.ShuffledUIDs = self.UIDs
        if self.IsShuffling:
            np.random.shuffle(self.ShuffledUIDs)
        
        self.BatchIndex = 0
        self.EndOfData = False      
        return self
    # --------------------------------------------------------------------------------------------------------
    def __next__(self):
        #if self.BatchIndex  < (self.BatchCount - 1):
        if self.BatchIndex  < self.BatchCount:
            nStartRange = self.BatchIndex*self.BatchSize
            nEndRange   = (self.BatchIndex + 1)*self.BatchSize

            if nEndRange >= self.SampleCount:
                nEndRange = self.SampleCount

            if self.IsShuffling:
              nBatchRange = self.ShuffledUIDs[np.arange(nStartRange, nEndRange, dtype=int)]
            else: 
              nBatchRange = np.arange(nStartRange, nEndRange, dtype=int)
            
            nBatchSamples   = self.Samples[nBatchRange,...]
            nBatchLabels    = self.Labels[nBatchRange,...]
            nBatchUIDs      = self.UIDs[nBatchRange]
            
            self.BatchIndex  += 1
            self.EndOfData = (self.BatchIndex == self.BatchCount)
            
            return nBatchRange, nBatchSamples, nBatchLabels, nBatchUIDs
        else:
            raise StopIteration()        
    # --------------------------------------------------------------------------------------------------------
# =======================================================================================================================









# =======================================================================================================================
class ModelState(object):
    #------------------------------------------------------------------------------------
    def __init__(self, p_oSession, p_sFileName):
      self.Session  = p_oSession
      self.FileName = p_sFileName
      oVarList     = tf.global_variables()
      self.Saver   = tf.train.Saver(oVarList, write_version=2)
    #------------------------------------------------------------------------------------            
    def Save(self):
        print(" |__ Saving Weights to " + self.FileName, end="")
        
        self.Saver.save(self.Session, self.FileName)
        print(" -> Saved.")
    #------------------------------------------------------------------------------------
    def Load(self):
      if os.path.exists(self.FileName + ".meta"):
        print(" |__ Restoring Weights from " + self.FileName, end="")
        self.Saver.restore(self.Session, self.FileName)
        print(' -> Restored.')
        bResult = True
      else:
        bResult = False
        
      return bResult
    # --------------------------------------------------------------------------------------------------------            
    
# =======================================================================================================================








#==============================================================================================================================
class Evaluator(object):
    #--------------------------------------------------------------------------------------------------------------
    def __init__(self, p_nActualClasses, p_nPredictedClasses):
        self.ActualClasses      = p_nActualClasses
        self.PredictedClasses   = p_nPredictedClasses
        self.ConfusionMatrix = metrics.confusion_matrix(self.ActualClasses, self.PredictedClasses)        
        self.Accuracy        = metrics.accuracy_score(self.ActualClasses, self.PredictedClasses)
        self.Precision, self.Recall, self.F1Score, self.Support = metrics.precision_recall_fscore_support(self.ActualClasses, self.PredictedClasses, average=None)
        self.AveragePrecision, self.AverageRecall, self.AverageF1Score, self.AverageSupport = metrics.precision_recall_fscore_support(self.ActualClasses, self.PredictedClasses,  average='weighted')
    #--------------------------------------------------------------------------------------------------------------
#==============================================================================================================================

