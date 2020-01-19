from __future__ import print_function
import tensorflow as ts
import scipy.io
import pickle
import numpy as np
import keras
from keras.models import Model
from keras.preprocessing import image
#from keras.applications.imagenet_utils import preprocess_input
from keras.optimizers import SGD
import matplotlib.pyplot as plt
import datetime
#from keras.applications.vgg16 import VGG16
from keras.preprocessing import image
#from keras.applications.vgg16 import preprocess_input
#from keras.applications.vgg19 import preprocess_input
from keras.models import Model
import cv2
from PIL import Image
import os
import xlsxwriter
from keras_applications import resnet_v2
from keras_applications import inception_resnet_v2
import datetime
from keras.models import Sequential
from keras.datasets import cifar10
from keras.layers import Dense, Dropout, Flatten, Input, Dense, merge
from keras.layers import Conv2D, MaxPooling2D, BatchNormalization
from keras import optimizers
from keras.optimizers import Adam
import Augmentor
#from keras.applications import VGG16
from keras.preprocessing.image import ImageDataGenerator
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import precision_recall_curve as prc
from numpy import asarray
from keras.applications.resnet_v2 import preprocess_input, decode_predictions
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import precision_recall_curve as prc
import matplotlib.pyplot as plt
import matplotlib.image as mpimg




def createDataAugmentation(AllData, DataParms):
    for ind in DataParms['train_indices']:  # resize all the i,ages in the folder
        imgPath = DataParms['Data_path'] + "/" + str(ind) + ".jpeg"
        imResize = image.load_img(imgPath, target_size=(DataParms['size'], DataParms['size']))  # This loads an image and resizes the image to (224, 224)
        Reverse_image = imResize.transpose(Image.FLIP_LEFT_RIGHT)
        #imgplot = plt.imshow(Reverse_image)
        #plt.show()
        NewImage = image.img_to_array(Reverse_image)  # adds channels: x.shape = (224, 224, 3) for RGB
        NewImage = np.expand_dims(NewImage, axis=0)  # used to add the number of images
        NewImage = keras.applications.resnet_v2.preprocess_input(NewImage)
        AllData['TrainData']['Images'].append(NewImage)
        AllData['TrainData']['Labels'].append(AllData['FullData']['Labels'][ind-1])


def GetData(DataParms):
    # defining the dictionary that will be returned
    AllData = {'TrainData': {}, 'TestData': {}, 'SubTrainData': {}, 'ValidData': {}, 'FullData': {}}  # defining the dictionary that will be returned
    AllData['FullData'] = {'Images': [], 'Labels': []}  # defining a dictionary for TrainData
    AllData['TrainData'] = {'Images': [], 'Labels': []}  # defining a dictionary for TrainData
    AllData['TestData'] = {'Images': [], 'Labels': []}  # defining a dictionary for TestData
    AllData['SubTrainData'] = {'Images': [], 'Labels': []}  # defining a dictionary for SubTrainData
    AllData['ValidData'] = {'Images': [], 'Labels': []}  # defining a dictionary for ValidData

    MatDictionary = scipy.io.loadmat(DataParms['Data_path'] + 'FlowerDataLabels')  # an tenzor Dictionary of all the images Data (image +label)
    AllData['FullData']['Labels'] = MatDictionary['Labels'][0]  # saving All Original labels
    tempImArray = MatDictionary['Data'][0]  # saving All Original labels
    # AllData['FullData']['Labels'] = np.transpose(MatDictionary['Labels'][0]).tolist() # saving All Original labels


    for i in range(len(tempImArray)):  # resize all the i,ages in the folder
        imgPath = DataParms['Data_path'] + "/" + str(i + 1) + ".jpeg"
        imResize = image.load_img(imgPath, target_size=(DataParms['size'], DataParms['size']))  # This loads an image and resizes the image to (224, 224)
        #imgplot = plt.imshow(imResize)
        #plt.show()
        NewImage = image.img_to_array(imResize)  # adds channels: x.shape = (224, 224, 3) for RGB
        NewImage = np.expand_dims(NewImage, axis=0)  # used to add the number of images
        NewImage = keras.applications.resnet_v2.preprocess_input(NewImage)
        AllData['FullData']['Images'].append(NewImage)

    # splitting test train
    for ind in DataParms['test_indices']:
        img = AllData['FullData']['Images'][ind - 1]
        label = AllData['FullData']['Labels'][ind - 1]
        AllData['TestData']['Images'].append(img)
        AllData['TestData']['Labels'].append(label)

    for ind in DataParms['train_indices']:
        img = AllData['FullData']['Images'][ind - 1]
        label = AllData['FullData']['Labels'][ind - 1]
        AllData['TrainData']['Images'].append(img)
        AllData['TrainData']['Labels'].append(label)

    createDataAugmentation(AllData, DataParms)

    return AllData


def SplitTrain(DataParms, AllData):
    subTrainSize = round(len(AllData['TrainData']['Images'])*0.8)
    for counter in range(0,len(AllData['TrainData']['Images'])):
        if counter <= subTrainSize-1:
            AllData['SubTrainData']['Images'].append(AllData['TrainData']['Images'][counter])  # insert image to Images Train dictionary
            AllData['SubTrainData']['Labels'].append(AllData['TrainData']['Labels'][counter])  # insert image label to Labels Train dictionary
        else:
            AllData['ValidData']['Images'].append(AllData['TrainData']['Images'][counter])  # insert image to Images Train dictionary
            AllData['ValidData']['Labels'].append(AllData['TrainData']['Labels'][counter])  # insert image label to Labels Train dictionary

    ImagesArray = np.array(AllData['SubTrainData']['Images'])
    ImagesArray = np.rollaxis(ImagesArray, 1, 0)
    ImagesArray = ImagesArray[0]
    AllData['SubTrainData']['Images'] = ImagesArray

    ImagesArray2 = np.array(AllData['ValidData']['Images'])
    ImagesArray2 = np.rollaxis(ImagesArray2, 1, 0)
    ImagesArray2 = ImagesArray2[0]
    AllData['ValidData']['Images'] = ImagesArray2

    AllData['SubTrainData']['Labels']= np.array(AllData['SubTrainData']['Labels'])
    AllData['ValidData']['Labels'] = np.array(AllData['ValidData']['Labels'])

    ImagesArray = np.array(AllData['TrainData']['Images'])
    ImagesArray = np.rollaxis(ImagesArray, 1, 0)
    ImagesArray = ImagesArray[0]
    AllData['TrainData']['Images'] = ImagesArray

    ImagesArray2 = np.array(AllData['TestData']['Images'])
    ImagesArray2 = np.rollaxis(ImagesArray2, 1, 0)
    ImagesArray2 = ImagesArray2[0]
    AllData['TestData']['Images'] = ImagesArray2

    AllData['TrainData']['Labels']= np.array(AllData['TrainData']['Labels'])
    AllData['TestData']['Labels'] = np.array(AllData['TestData']['Labels'])

    # (hide when project is running)
    with open('AllDataPickle.pickle', 'wb') as pklfile:  # open a new pkl file (with statment sode'nt need to close)  Write to file
        pickle.dump(AllData, pklfile, protocol=pickle.HIGHEST_PROTOCOL)  # insert the data dictionary into the pkl file

    return AllData


def uploadPickle(pickle_name):
    '''
    :param pickle_name: the name of the pickle file
    :return: the pickel file with all data in it
    '''
    with open(pickle_name + '.pickle','rb') as pklfile:  # open a new pkl file (with statment sode'nt need to close)  read from file
        p = pickle.load(pklfile)
    return p


def Create_Net(DataParms, Params):  #there will be ifisssssssssssssssssssssssssssssssss
    # --- declare format of input tensor
    image_input = Input(shape=(DataParms['size'], DataParms['size'], 3))
    # --- creates a full vgg16 model as published
    inital_model = keras.applications.resnet_v2.ResNet50V2(include_top=False, weights='imagenet', input_tensor= image_input, pooling='avg')
    #current_last_layer = inital_model.get_layer('fc2').output
    new_dense_layer = (Dense(1, activation='sigmoid'))(inital_model.output)
    our_model = Model(inital_model.input, new_dense_layer)
    # --- all the layers will "freeze" and will not be trainable except the last layer
    for layer in our_model.layers[:-1]:
        layer.trainable = False
    sgd = SGD(learning_rate=0.01, momentum=0.1)
    #custom_vgg_model.compile(loss='binary_crossentropy', optimizer=opt, metrics=['accuracy'])
    our_model.compile(loss='binary_crossentropy', optimizer=sgd, metrics=['accuracy'])

    with open('ResnetPickle.pickle', 'wb') as pklfile:  # open a new pkl file (with statment sode'nt need to close) Write to file
        pickle.dump(our_model, pklfile, protocol=pickle.HIGHEST_PROTOCOL)  # insert the Hog data dictionary into the pkl file

    return our_model


def Train(model, trainData, testData, train_params):

    # ---- train the net
    # batch_size = number of sampels per gradient update  default 32
    our_model_fit = model.fit(trainData['Images'], trainData['Labels'], epochs=train_params['epochs'], verbose=1,validation_data=(testData['Images'], testData['Labels']), batch_size=train_params['batch_size'])
    # our_model_fit = uploadPickle('ModelsPickle')

    with open('ModelsPickle.pickle', 'wb') as pklfile:  # open a new pkl file (with statment sode'nt need to close) Write to file
        pickle.dump(our_model_fit, pklfile, protocol=pickle.HIGHEST_PROTOCOL)  # insert the Hog data dictionary into the pkl file

    validation_loss_array = our_model_fit.history['val_loss']  # extract all iterations loss
    validation_loss = validation_loss_array[len(validation_loss_array) - 1]  # the loss of the last iteration

    validation_accuracy_array = our_model_fit.history['val_accuracy']  # extract all iterations accuracy
    validation_accuracy = validation_accuracy_array[len(validation_accuracy_array) - 1]  # the accuracy of the last iteration

    train_loss_array = our_model_fit.history['loss']  # extract all iterations loss
    train_loss = train_loss_array[len(train_loss_array) - 1]  # the loss of the last iteration

    train_accuracy_array = our_model_fit.history['accuracy']  # extract all iterations accuracy
    train_accuracy = train_accuracy_array[len(train_accuracy_array) - 1]  # the accuracy of the last iteration

    '''
    print("--the loss of the validation is:  " + str(validation_loss) + " -----------")
    print("--the accuracy of the validation is:  " + str(validation_accuracy) + " -----------")
    print("--the loss of the train is:  " + str(train_loss) + " ----------")
    print("--the accuracy of the train is:  " + str(train_accuracy) + " -----------")
    print("the epoch is:  " + str(epochs))
    '''
    model_results = {'val_loss': validation_loss, 'val_accuracy': validation_accuracy, 'loss': train_loss, 'accuracy': train_accuracy}

    return model, model_results


def Test(TrainModel, testData):

    pred_array = np.array(TrainModel.predict(testData['Images']))
    pred = np.where(pred_array > 0.5, 1, 0)
    score = TrainModel.evaluate(testData['Images'], testData['Labels'])
    print('Test accuracy:', score[1])

    return pred_array, pred, score


def GetDefaultParameters(data_path, train_images_indices, test_images_indices, Imsize, trainsize, testsize, opt, epoch,
                         b_size, layer):
    '''
    :param all parameters: receving all parameters nedded for the algorithem
    :return: parameters dictionary
    '''
    Params = {}
    Params['Data'] = {'Data_path': data_path, 'train_indices': train_images_indices, 'test_indices': test_images_indices, 'size': Imsize, 'trainsize': trainsize, 'testsize': testsize}
    Params['Train'] = {'optimizers': opt, 'epochs': epoch, 'batch_size': b_size, 'layer': layer}  # parameters of training
    return Params


#################### ------------------Main----------------------------#######################################################################

# start with defining decisional parameters
data_path = '/Users/hadasch/PycharmProjects/Task_Transfer/FlowerData/'
train_images_indices = list(range(41, 473))
trainsize = len(train_images_indices)
test_images_indices = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31, 32, 33, 34, 35, 36, 37, 38, 39, 40]
testsize = len(test_images_indices)
Imsize = 224
opt = 'sgd'
epoch = 1
b_size = 10
layer = 1

Params = GetDefaultParameters(data_path, train_images_indices, test_images_indices, Imsize, trainsize, testsize, opt,epoch, b_size, layer)  # get all parameter dictionary
np.random.seed(0)

OurData = GetData(Params['Data'])  # get Images data from the Top folder and organize it into groups
OurData = SplitTrain(Params['Data'], OurData)
# OurData = uploadPickle('AllDataPickle')

ModelPrep = Create_Net(Params['Data'], Params)
# ModelPrep = uploadPickle('ResnetPickle')

TrainedModel, results = Train(ModelPrep, OurData['SubTrainData'], OurData['ValidData'], Params['Train'])

pred = TrainedModel.predict(OurData['TestData']['Images'])
print(pred)
print(OurData['TestData']['Labels'])

Results = Test(TrainedModel, OurData['TestData'])
# Summary = Evaluate(Results, DataRep, OurData)
# ReportResults(Summary, OurData)

