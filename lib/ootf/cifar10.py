import os
import shutil
import sys
import zipfile
import tarfile
import pickle
from urllib.request import urlretrieve
import numpy as np

# =======================================================================================================================
class DataSetCifar10(object):
    DOWNLOAD_URL = "http://www.cs.toronto.edu/~kriz/cifar-10-python.tar.gz"
    
    # --------------------------------------------------------------------------------------------------------
    def __init__(self, p_cDataSubSet=None, p_sDataFolder="cifar10"):
        self.DataSubSetClass = p_cDataSubSet
        
        self.Training   = None
        self.Validation = None
        self.Testing    = None
        
        self.Samples      = None
        self.Labels       = None
        self.SampleCount  = None
        
        self.TempFolder = "/tmp"
        self.DataFolder = p_sDataFolder
        
        self.BatchesFile                = os.path.join(self.DataFolder, 'batches.meta')
        self.TrainingShardFileTemplate  = os.path.join(self.DataFolder, 'data_batch_%d')
        self.TestFileName               = os.path.join(self.DataFolder, 'test_batch')
        
        self.ClassNames = {  0:"airplane", 1:"automobile", 2:"bird", 3:"cat", 4:"deer" 
                               , 5:"dog", 6: "frog", 7:"horse", 8:"ship", 9:"truck"}
    # --------------------------------------------------------------------------------------------------------            
    def _downloadProgressCallBack(self, count, block_size, total_size):
        pct_complete = float(count * block_size) / total_size
        msg = "\r- Download progress: {0:.1%}".format(pct_complete)
        sys.stdout.write(msg)
        sys.stdout.flush()        
    # --------------------------------------------------------------------------------------------------------
    def __ensureDataSetIsOnDisk(self):
        sSuffix = DataSetCifar10.DOWNLOAD_URL.split('/')[-1]
        sArchiveFileName = os.path.join(self.TempFolder, sSuffix)
        
        if not os.path.isfile(sArchiveFileName):
            sFilePath, _ = urlretrieve(url=DataSetCifar10.DOWNLOAD_URL, filename=sArchiveFileName, reporthook=self._downloadProgressCallBack)
            print()
            print("Download finished. Extracting files.")

            
        if sArchiveFileName.endswith(".zip"):
            zipfile.ZipFile(file=sArchiveFileName, mode="r").extractall(self.TempFolder)
        elif sArchiveFileName.endswith((".tar.gz", ".tgz")):
            tarfile.open(name=sArchiveFileName, mode="r:gz").extractall(self.TempFolder)
        print("Done.")

        shutil.move(os.path.join(self.TempFolder, "./cifar-10-batches-py"), self.DataFolder)

        os.remove(sArchiveFileName)
    # --------------------------------------------------------------------------------------------------------
    def _transposeImageChannels(self, p_nX, p_nShape=(32, 32, 3), p_bIsFlattening=False):
        nResult = np.asarray(p_nX, dtype=np.float32)
        nResult = nResult.reshape([-1, p_nShape[2], p_nShape[0], p_nShape[1]])
        nResult = nResult.transpose([0, 2, 3, 1])
        
        if p_bIsFlattening:
          nResult = nResult.reshape(-1, np.prod(np.asarray(p_nShape)))
        
        return nResult   
    # --------------------------------------------------------------------------------------------------------
    def Load(self):
        if not os.path.exists(self.DataFolder):
            self.__ensureDataSetIsOnDisk()
      
        self.LoadSubset(True)
        self.LoadSubset(False)
    # --------------------------------------------------------------------------------------------------------
    def LoadSubset(self, p_bIsTrainingSubSet=True):
        if p_bIsTrainingSubSet:
            self.Training = self.DataSubSetClass()

            for i in range(5):
                with open(self.TrainingShardFileTemplate % (i+1), 'rb') as oFile:
                    oDict = pickle.load(oFile, encoding='latin1')
                    oFile.close()
                self.Training.AppendShard(self._transposeImageChannels(oDict["data"], (32,32,3)), np.array(oDict['labels'], np.uint8))
        else:
            self.Testing = self.DataSubSetClass() 
            
            with open(self.TestFileName, 'rb') as oFile:
                oDict = pickle.load(oFile, encoding='latin1')
                oFile.close()
            self.Testing.AppendShard(self._transposeImageChannels(oDict["data"], (32,32,3)), np.array(oDict['labels'], np.uint8))
    # --------------------------------------------------------------------------------------------------------
# =======================================================================================================================

