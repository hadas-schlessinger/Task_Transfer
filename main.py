import tensorflow as tf
import scipy.io as sio
import os
import numpy as np
import cv2
import keras
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
from keras.layers import merge, Input
from keras.layers import Dense
from keras.models import Model
from keras.preprocessing import image
from keras.applications.imagenet_utils import preprocess_input
from sklearn.metrics import precision_recall_curve
from sklearn.utils.fixes import signature
from keras.optimizers import SGD
from keras.applications.vgg19 import VGG19
from keras.preprocessing import image
from keras.applications.vgg19 import preprocess_input

data_path = os.path.join(os.getcwd(), 'FlowerData')
test_images_indices = []
#env variables
S = 224
NUMBER_OF_CLASSES = 2
BATCH_SIZE = 16
CNN_BACKEND = 'Tensorflow'
EPOCHS = 5
VERBOSE = 1
NET_NAME = 'ResNet50V2'

# DictinaryLbl = sio.loadmat(matlabfile, mdict=None, appendmat=True)
# lbls = np.transpose(DictinaryLbl['Labels']).tolist()  # list of labels
# helper = keras.utils.to_categorical(lbls, numberofclasses)


def set_and_split_data():
    '''set the images and split them'''
    train = {'data': [],
             'labels': []}
    test = {'data': [],
            'labels': []}
    return train, test


def reconstruct_net(s, num_classes, net_name):
    '''export the net without the last layer'''
    basic_model = ' '
    return basic_model


def train_model(res_net_basic, train, batch_size, epochs, verbose):
    trained_model = ' '
    return trained_model


def test_model(model, test, batch_size, verbose):
    '''test and prints accuracy'''
    pass


def error_type(predictions, test_labels):
    '''returns a vector of all error types'''
    error_type = []
    return error_type


def tuning():
    '''tune hyper parameters to improve the model'''
    pass


def recall_precision_curve():
    '''creates precision curve'''
    pass


def report_results(predictions, error_type_array):
    '''prints the wrong images'''
    pass

################# main ####################
def main():
    error_type_array = []
    # params = get_default_parameters()
    np.random.seed(0)  # seed
    train, test = set_and_split_data()
    # tuning_error_per_set, errors = tuning(train)
    res_net_basic = reconstruct_net(S, NUMBER_OF_CLASSES, NET_NAME)  # preparing the network
    model = train_model(res_net_basic, train, BATCH_SIZE, EPOCHS, VERBOSE) # train and validation stage
    test_model(model, test, BATCH_SIZE, VERBOSE)  # test stage
    predictions = model.predict(test)
    error_type_array = error_type(predictions, test['labels'])  # find the error types
    recall_precision_curve(test['labels'], predictions)  # recall-precision curve
    report_results(predictions, error_type_array)


if __name__ == "__main__":
    main()
