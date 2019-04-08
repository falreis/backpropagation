#### Imports ####
import numpy as np
import tensorflow as tf
from sklearn.datasets import make_blobs

#### SKLearn Blobs ####
class clusterData:
    def __init__(self,
                 n_features = 2, 
                 n_classes = 2, 
                 n_training_samples = 200, 
                 n_testing_samples = 200, 
                 cluster_std = 0.5,
                 center_box = (-2,2),
                 seed = 0):
        #### Params ####
        self.n_features = n_features
        self.n_classes = n_classes
        self.n_training_samples = n_training_samples
        self.n_testing_samples = n_testing_samples
        self.cluster_std = cluster_std
        self.center_box = center_box
        self.seed = seed
        
        #### Training Data ####
        self.x_train, self.y_train_raw = make_blobs(n_samples = self.n_training_samples, 
                                          n_features = self.n_features,
                                          centers = self.n_classes,
                                          center_box = self.center_box,
                                          cluster_std =  self.cluster_std, 
                                          random_state = self.seed)
        # One-hot encode the y values (categorically encoding the data)
        self.y_train = np.zeros((self.y_train_raw.shape[0], self.n_classes))
        self.y_train[np.arange(self.y_train_raw.size), self.y_train_raw] = 1



        #### Testing Data ####
        self.x_test, self.y_test_raw = make_blobs(n_samples = self.n_testing_samples + self.n_training_samples, 
                                    n_features = self.n_features,
                                    centers = self.n_classes,
                                    center_box = self.center_box,
                                    cluster_std =  self.cluster_std, 
                                    random_state = self.seed)
        self.x_test = self.x_test[-self.n_testing_samples:]
        self.y_test_raw = self.y_test_raw[-self.n_testing_samples:]
        # One-hot encode the y values (categorically encoding the data)
        self.y_test = np.zeros((self.y_test_raw.shape[0], self.n_classes))
        self.y_test[np.arange(self.y_test_raw.size), self.y_test_raw] = 1
        
    def trainData(self):
        return self.x_train, self.y_train, self.y_train_raw
    
    def testData(self):
        return self.x_test, self.y_test, self.y_test_raw
    
    
#### MNIST ####
class mnistData:
    def __init__(percent_train = 0.8):
        self.n_features = 784
        self.n_classes = 10
        self.normal_val = 255
        self.percent_train = percent_train

        ##  Loading Data  ##
        mnist = tf.keras.datasets.mnist
        (self.x_train, self.y_train_raw), (self.x_test, self.y_test_raw) = mnist.load_data()

        # Flattening for mlp
        self.x_train = self.x_train.reshape(self.x_train.shape[0], self.n_features)
        self.x_test = self.x_test.reshape(self.x_test.shape[0], self.n_features)

#         print(self.x_train.shape) #--> (60000, 28, 28)
#         print(self.y_train_raw.shape) #--> (60000,)
#         print(self.x_test.shape) #--> (10000, 28, 28)
#         print(self.y_test_raw.shape) #--> (10000,)

        ##  Normalizing Data  ##
        self.x_train = np.divide(self.x_train, self.normal_val)
        self.x_test = np.divide(self.x_test, self.normal_val)

        ##  Splitting Data into Validation and Train Sets  ##
        train_samples = int(self.percent_train * len(self.x_train))
        self.val_x = self.x_train[train_samples:]
        self.x_train = self.x_train[:train_samples]

        target_samples = int(self.percent_train * len(self.y_train_raw))
        self.val_y = self.y_train_raw[target_samples:]
        self.y_train_raw = self.y_train_raw[:target_samples]

        ## One Hot Encoding Y_train for training ##
        self.y_train = np.zeros((self.y_train_raw.shape[0], self.n_classes))
        self.y_train[np.arange(self.y_train_raw.size), self.y_train_raw] = 1

        ## One Hot Encoding Y_test for testing ##
        self.y_test = np.zeros((self.y_test_raw.shape[0], self.n_classes))
        self.y_test[np.arange(self.y_test_raw.size), self.y_test_raw] = 1
        
        def trainData(self):
            return self.x_train, self.y_train, self.y_train_raw
    
        def testData(self):
            return self.x_test, self.y_test, self.y_test_raw