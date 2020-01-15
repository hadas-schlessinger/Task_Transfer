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
import sklearn
from sklearn.metrics import precision_recall_curve
from funcsigs import signature
from keras.optimizers import SGD
#from keras.applications.resnet_v2 import ResNet50V2
from keras.preprocessing import image
from sklearn.model_selection import train_test_split

test_images_indices = []
# env variables
S = 224
NUMBER_OF_CLASSES = 2
BATCH_SIZE = 16
EPOCHS = 1
VERBOSE = 1
data_path = os.path.join(os.getcwd(), 'FlowerData')  # The images path
Mat_file = '/Users/hadasch/PycharmProjects/Task_Transfer/FlowerData/FlowerDataLabels.mat'


def set_and_split_data():
    '''set the images and split them'''
    train = {'data': [],
             'labels': []}
    test = {'data': [],
            'labels': []}
    # dictionary with variable names as keys, and loaded matrices as values.
    dictionary_labels = sio.loadmat(Mat_file, mdict=None, appendmat=True)
    helper = keras.utils.to_categorical(np.transpose(dictionary_labels['Labels']).tolist(), NUMBER_OF_CLASSES) # list of labels
    images_train = 300  # number of pictures from the data to train
    for i in range(len(helper)):  # for filename in datapath folder which is flowerData
        image_dir_file = data_path + "/" + str(i + 1) + ".jpeg"
        # image read and converting it image pixels to a numpy array
        a = preprocess_input(np.expand_dims(image.img_to_array(image.load_img(image_dir_file, target_size=(S, S))), axis=0))
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


def reconstruct_net(s, num_classes):
    '''export the net without the last layer'''
    image_input = Input(shape=(s, s, 3))
    basic_model = keras.applications.resnet_v2.ResNet50V2(include_top=True, weights='imagenet', input_tensor=image_input)
    last_layer = basic_model.layers[-2].output
    model_without_last_layer = Model(image_input, Dense(num_classes, activation='sigmoid', name='output')(last_layer))
    model_without_last_layer.summary()
    for layer in model_without_last_layer.layers[:-1]:  # All layers are not trainable besides last one
        layer.trainable = False
    model_without_last_layer.compile(loss='binary_crossentropy', optimizer=SGD(lr=0.01, decay=0.001), metrics=['accuracy'])
    # model_without_last_layer.summary()
    return model_without_last_layer


def train_model(res_net_basic, train, batch_size, epochs, verbose):     # hyper parameter  batch_size, epoches
    train_images, valid_images, train_labels, valid_labels = train_test_split(train['data'], train['labels'], test_size=0.33, shuffle=True)
    return res_net_basic.fit(train_images, train_labels, batch_size=batch_size, epochs=epochs,
                         verbose=verbose, validation_data=(valid_images, valid_labels), shuffle=True)  # fitting the model


def test_model(model, test, batch_size, verbose):
    '''test and prints accuracy'''
    loss, accuracy = model.evaluate(test['data'], test['labels'], batch_size=batch_size, verbose=verbose)
    print(f'The loss is {round(loss,4)}, the accuracy is {round(accuracy*100,4)}% and the error is {round(100 - accuracy*100,4)}%')
    return model.predict(test['data'])  # returns the predictions



def error_type(predictions, test_labels):
    '''returns a vector of all error types'''
    error_type = []
    return error_type


def tuning():
    '''tune hyper parameters to improve the model'''
    pass


def recall_precision_curve(test_labels, pred):
    '''creates precision curve'''
    # generate the recall-precision plot and show it
    precision = dict()
    recall = dict()
    for i in range(2):
        precision[i], recall[i], _ = precision_recall_curve(test_labels[:, i], pred[:, i])
    precision["micro"], recall["micro"], _ = precision_recall_curve(test_labels.ravel(), pred.ravel())
    fig = plt.figure()
    step_kwargs = ({'step': 'post'} if 'step' in signature(plt.fill_between).parameters else {})
    plt.step(recall['micro'], precision['micro'], color='b', alpha=0.2, where='post')
    plt.fill_between(recall["micro"], precision["micro"], alpha=0.2, color='b', **step_kwargs)
    plt.xlabel('Recall'), plt.ylabel('Precision'), plt.title('Precision Recall Curve')
    plt.show()


def report_results(predictions, error_type_array):
    '''prints the wrong images'''
    pass


################# main ####################
def main():
    np.random.seed(0)  # seed
    train, test = set_and_split_data()
    # tuning_error_per_set, errors = tuning(train)
    res_net_new = reconstruct_net(S, NUMBER_OF_CLASSES)  # preparing the network
    train_model(res_net_new, train, BATCH_SIZE, EPOCHS, VERBOSE) # train and validation stage
    predictions = test_model(res_net_new, test, BATCH_SIZE, VERBOSE)  # test stage
    error_type_array = error_type(predictions, test['labels'])  # find the error types
    recall_precision_curve(np.array(test['labels']), np.array(predictions))  # recall-precision curve
    # report_results(predictions, error_type_array)


if __name__ == "__main__":
    main()
