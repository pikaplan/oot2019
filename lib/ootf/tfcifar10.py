import os
import joblib
import pickle
import numpy as np
import tensorflow as tf

tf.logging.set_verbosity(tf.logging.ERROR)

# =======================================================================================================================
class TFDataSetCifar10(object):
    # --------------------------------------------------------------------------------------------------------
    def __init__(self, p_sDataFolder):
        self.DataFolder = p_sDataFolder
                
        # Dataset files
        self.TrainSubSetFileName = os.path.join(self.DataFolder, 'train.tf')
        self.TestSubSetFileName = os.path.join(self.DataFolder, 'test.tf')

        # Mean and Std for Z-Score standardization                
        oPixelMeanStd = joblib.load(os.path.join(self.DataFolder, 'meanstd.pkl'))
        self.PixelMean = oPixelMeanStd['mean']
        self.PixelStd = oPixelMeanStd['std']
        
        # Default values from the paper
        self.TrainBatchSize = 128
        self.TestBatchSize  = 100
        
        print("\n[>] DataSet CIFAR10 prepared")
        print(" |_ Using TFRecords data feed")
    # --------------------------------------------------------------------------------------------------------    
    def __cifar10_input_stream(self, records_path):
        reader = tf.TFRecordReader()
        filename_queue = tf.train.string_input_producer([records_path], None)
        _, record_value = reader.read(filename_queue)
        features = tf.parse_single_example(record_value,
            {
                'image_raw': tf.FixedLenFeature([], tf.string),
                'label': tf.FixedLenFeature([], tf.int64),
            })
        image = tf.decode_raw(features['image_raw'], tf.uint8)
        image = tf.reshape(image, [32,32,3])
        image = tf.cast(image, tf.float32)
        label = tf.cast(features['label'], tf.int64)
        return image, label
    # --------------------------------------------------------------------------------------------------------
    def __normalize_image(self, p_nImage):
        normed_image = (p_nImage - self.PixelMean) / self.PixelStd
        return normed_image 
    # --------------------------------------------------------------------------------------------------------
    def __random_distort_image(self, image):
        # We follow the simple data augmentation in [24] for training: 4 pixels are padded on each side, 
        # and a  32×32  crop is  randomly  sampled  from  the  paddedimage or its horizontal flip.  
      
        #distorted_image = image
        distorted_image = tf.image.pad_to_bounding_box(
            image, 4, 4, 40, 40)    # pad 4 pixels to each side
        distorted_image = tf.random_crop(distorted_image, [32, 32, 3])
        distorted_image = tf.image.random_flip_left_right(distorted_image)
        return distorted_image
    # --------------------------------------------------------------------------------------------------------
    def TrainingBatches(self):
        with tf.variable_scope('train_batch'):
            with tf.device('/cpu:0'):
                train_image, train_label = self.__cifar10_input_stream(self.TrainSubSetFileName)
                train_image = self.__normalize_image(train_image)
                train_image = self.__random_distort_image(train_image)
                train_image_batch, train_label_batch = tf.train.shuffle_batch([train_image, train_label], batch_size=self.TrainBatchSize
                                                                              , num_threads=4, capacity=50000, min_after_dequeue=1000)
        return train_image_batch, train_label_batch
    # --------------------------------------------------------------------------------------------------------
    def TestingBatches(self):
        with tf.variable_scope('evaluate_batch'):
            with tf.device('/cpu:0'):
                test_image, test_label = self.__cifar10_input_stream(self.TestSubSetFileName )
                test_image = self.__normalize_image(test_image)
                test_image_batch, test_label_batch = tf.train.batch([test_image, test_label], batch_size=self.TestBatchSize, 
                                                                    num_threads=1, capacity=10000)
        return test_image_batch, test_label_batch
    # --------------------------------------------------------------------------------------------------------
# =======================================================================================================================




# --------------------------------------------------------------------------------------------------------
def _int64_feature(value):
    return tf.train.Feature(int64_list=tf.train.Int64List(value=[value]))
# --------------------------------------------------------------------------------------------------------
def _bytes_feature(value):
    return tf.train.Feature(bytes_list=tf.train.BytesList(value=[value]))
# --------------------------------------------------------------------------------------------------------
def convert_to_tfdataset(p_sSourceDataFolder, p_sDestDataFolder):
    def save_to_records(save_path, images, labels):
        writer = tf.python_io.TFRecordWriter(save_path)
        for i in range(images.shape[0]):
            image_raw = images[i].tostring()
            example = tf.train.Example(features=tf.train.Features(feature={
                'height'    : _int64_feature(32),
                'width'     : _int64_feature(32),
                'depth'     : _int64_feature(3),
                'label'     : _int64_feature(int(labels[i])),
                'image_raw' : _bytes_feature(image_raw)
                }))
            writer.write(example.SerializeToString())

    # train set
    train_images = np.zeros((50000,3072), dtype=np.uint8)
    trian_labels = np.zeros((50000,), dtype=np.int32)
    for i in range(5):
        sFileName = os.path.join(p_sSourceDataFolder, 'data_batch_%d' % (i+1))
        with open(sFileName, 'rb') as oFile:
                data_batch = pickle.load(oFile, encoding='latin1')
        #data_batch = joblib.load(os.path.join(data_root, 'data_batch_%d' % (i+1)))
        train_images[10000*i:10000*(i+1)] = data_batch['data']
        trian_labels[10000*i:10000*(i+1)] = np.asarray(data_batch['labels'], dtype=np.int32)
    train_images = np.reshape(train_images, [50000,3,32,32])
    train_images = np.transpose(train_images, axes=[0,2,3,1]) # NCHW -> NHWC
    save_to_records(os.path.join(p_sSourceDataFolder, "train.tf"), train_images, trian_labels)
    
    
    # mean and std
    image_mean = np.mean(train_images.astype(np.float32), axis=(0,1,2))
    image_std = np.std(train_images.astype(np.float32), axis=(0,1,2))
    joblib.dump({'mean': image_mean, 'std': image_std}, os.path.join(p_sDestDataFolder, "meanstd.pkl"), compress=5)

    # test set
    sFileName = os.path.join(p_sSourceDataFolder, 'test_batch')
    with open(sFileName, 'rb') as oFile:
        data_batch = pickle.load(oFile, encoding='latin1')
    
    #data_batch = joblib.load(os.path.join(data_root, 'test_batch'))
    test_images = data_batch['data']
    test_images = np.reshape(test_images, [10000,3,32,32])
    test_images = np.transpose(test_images, axes=[0,2,3,1])
    test_labels = np.asarray(data_batch['labels'], dtype=np.int32)
    save_to_records(os.path.join(DEST_DATA_FOLDER, "test.tf"), test_images, test_labels)
# --------------------------------------------------------------------------------------------------------



if __name__ == '__main__':
    SOURCE_DATA_FOLDER = r"/tmp/cifar10"
    DEST_DATA_FOLDER   = r"/tmp/tfcifar10"
  
    convert_to_tfdataset(SOURCE_DATA_FOLDER, DEST_DATA_FOLDER)







