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
from sklearn.model_selection import train_test_split
# from keras.applications.resnet_v2 import resnet_v2
# from keras.applications.resnet_v2 import ResNet50V2
# from sklearn.metrics import precision_recall_curve
# from sklearn.utils.fixes import signature
# from keras.optimizers import SGD
# from keras.applications.vgg19 import VGG19
# from keras.preprocessing import image
# from keras.applications.vgg19 import preprocess_input

test_images_indices = []
# env variables
S = 224
NUMBER_OF_CLASSES = 2
BATCH_SIZE = 16
CNN_BACKEND = 'Tensorflow'
EPOCHS = 5
VERBOSE = 1
NET_NAME = 'ResNet50V2'

data_path = os.path.join(os.getcwd(), 'FlowerData')  # The images path
Mat_file = "C:/Users/noy/PycharmProjects/Task_Transfer/FlowerData/FlowerDataLabels"


def set_and_split_data():
    '''set the images and split them'''
    train = {'data': [],
             'labels': []}
    test = {'data': [],
            'labels': []}

    # dictionary with variable names as keys, and loaded matrices as values.
    dictionary_labels = sio.loadmat(Mat_file, mdict=None, appendmat=True)
    labels = np.transpose(dictionary_labels['Labels']).tolist()  # list of labels
    helper = keras.utils.to_categorical(labels, NUMBER_OF_CLASSES)

    img_pathes_test = []  # 1D array of paths
    images_train = 300  # number of pictures from the data to train

    for i in range(len(helper)):  # for filename in datapath folder which is flowerData
        image_dir_file = data_path + "/" + str(i + 1) + ".jpeg"
        a = image.load_img(image_dir_file, target_size=(224, 224))  # image read
        a = image.img_to_array(a)  # convert the image pixels to a numpy array
        a = np.expand_dims(a, axis=0)
        a = preprocess_input(a)
        if i < images_train:  # divide the 300 first images into train
            train['data'].append(a)  # connect a to the big 4D array of input images
            train['labels'].append(helper[i])  # divide the dictionary into labels vector of train
        else:
            test['data'].append(a)  # accumulate image array
            test['labels'].append(helper[i])  # divide the dictionary into labels vector of test

    # changing the train_images_list into numpy array for fitting VGG16
    train['data'] = np.array(train['data'])
    train['data'] = np.rollaxis(train['data'], 1, 0)
    train['data'] = train['data'][0]

    # changing the test_images_list into numpy array for fitting
    test['data'] = np.array(test['data'])
    test['data'] = np.rollaxis(test['data'], 1, 0)
    test['data'] = test['data'][0]

    # changing the labels into numpy array for fitting
    train['labels'] = np.array(train['labels'])
    test['labels'] = np.array(test['labels'])
    return train, test


def reconstruct_net(s, num_classes, net_name):
    '''export the net without the last layer'''
    basic_model = ' '
    return basic_model


def train_model(res_net_basic, train, batch_size, epochs, verbose):
    train_images, valid_images, train_labels, valid_labels = train_test_split(train['data'], train['labels'], test_size=0.33, shuffle=True)
    trained_model = res_net_basic.fit(train_images, train_labels, batch_size=batch_size, epochs=epochs,
                         verbose=verbose, validation_data=(valid_images, valid_labels), shuffle=True)  # fitting the model
    # hyper parameter  batch_size, epoches
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
    # # tuning_error_per_set, errors = tuning(train)
    # res_net_basic = reconstruct_net(S, NUMBER_OF_CLASSES, NET_NAME)  # preparing the network
    # model = train_model(res_net_basic, train, BATCH_SIZE, EPOCHS, VERBOSE) # train and validation stage
    # test_model(model, test, BATCH_SIZE, VERBOSE)  # test stage
    # predictions = model.predict(test)
    # error_type_array = error_type(predictions, test['labels'])  # find the error types
    # recall_precision_curve(test['labels'], predictions)  # recall-precision curve
    # report_results(predictions, error_type_array)
    # basic_model =keras.applications.resnet_v2.ResNet50V2(include_top=True, weights='imagenet')
    # basic_model =tf.keras.applications.ResNet50V2()


if __name__ == "__main__":
    main()
