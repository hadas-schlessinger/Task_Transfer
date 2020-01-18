import tensorflow as tf
import scipy.io as sio
import os
import numpy as np
import cv2
import keras
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
from PIL import Image
from keras.layers import merge, Input
from keras.layers import Dense
from keras.models import Model
from keras.preprocessing import image
from keras.applications.imagenet_utils import preprocess_input
import sklearn
from sklearn.metrics import precision_recall_curve
from funcsigs import signature
from keras.optimizers import SGD
from keras.preprocessing import image
from sklearn.model_selection import train_test_split
from sklearn.metrics import plot_precision_recall_curve, average_precision_score


test_images_indices = []
# env variables
S = 224
NUMBER_OF_CLASSES = 1
BATCH_SIZE = 16
EPOCHS = 1
VERBOSE = 1
TRAIN_SIZE = 300  # number of pictures from the data to train
data_path = os.path.join(os.getcwd(), 'FlowerData')  # The images path
Mat_file = '/Users/noy/PycharmProjects/Task_Transfer/FlowerData/FlowerDataLabels.mat'
NEEDS_AUG = True


def set_and_split_data():
    '''set the images and split them'''
    train = {'data': [],
             'labels': []}
    test = {'data': [],
            'labels': []}
    # dictionary with variable names as keys, and loaded matrices as values.
    dictionary_labels = sio.loadmat(Mat_file, mdict=None, appendmat=True)
    labels_from_mat = np.transpose(dictionary_labels['Labels']).tolist()
    labels_with_aug = []

    if NEEDS_AUG:
        for i in range(len(labels_from_mat) + TRAIN_SIZE):
            if i < len(labels_from_mat):
                labels_with_aug.append(labels_from_mat[i])# original images
            else:# augmantation images
                if i < len(labels_from_mat) + TRAIN_SIZE:# augmantation images
                    labels_with_aug.append(labels_from_mat[i - len(labels_from_mat)])
                else:
                    labels_with_aug.append(labels_from_mat[i - len(labels_from_mat) - TRAIN_SIZE])

        helper = keras.utils.to_categorical(labels_with_aug, NUMBER_OF_CLASSES)  # list of labels
    else:
        helper = keras.utils.to_categorical(labels_from_mat, NUMBER_OF_CLASSES)  # list of labels

    for i in range(len(helper)):  # for filename in data path folder which is flowerData
        image_dir_file = data_path + "/" + str(i + 1) + ".jpeg"
        # image read and converting it image pixels to a numpy array
        a = preprocess_input(np.expand_dims(image.img_to_array(image.load_img(image_dir_file, target_size=(S, S))), axis=0))

        # inserts the 300 first and it augmentation images into train
        if i < TRAIN_SIZE or ((i > TRAIN_SIZE) and (i< len(labels_from_mat) + TRAIN_SIZE)):
            train['data'].append(a)  # connect a to the big 4D array of input images
            train['labels'].append(helper[i])  # divide the dictionary into labels vector of train
        else:# inserts the last images into test
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


def reconstruct_net(s, num_classes, activation, optimizer):
    '''export the net without the last layer'''
    image_input = Input(shape=(s, s, 3))
    basic_model = keras.applications.resnet_v2.ResNet50V2(include_top=True, weights='imagenet', input_tensor=image_input)
    last_layer_minus_1 = basic_model.layers[-2].output
    model_without_last_layer = Model(image_input, Dense(num_classes, activation=activation, name='output')(last_layer_minus_1))
    model_without_last_layer.summary()
    for layer in model_without_last_layer.layers[:-1]:  # All layers are not trainable besides last one
        layer.trainable = False
    model_without_last_layer.compile(loss='binary_crossentropy', optimizer=optimizer, metrics=['accuracy'])
    # model_without_last_layer.summary()
    return model_without_last_layer


def train_model(res_net_basic, train, batch_size, epochs, verbose):     # hyper parameter epoches
    train_images, valid_images, train_labels, valid_labels = train_test_split(train['data'], train['labels'], test_size=0.33, shuffle=True)
    return res_net_basic.fit(train_images, train_labels, batch_size=batch_size, epochs=epochs,
                         verbose=verbose, validation_data=(valid_images, valid_labels), shuffle=True)  # fitting the model



def test_model(model, test, batch_size, verbose):
    '''test and prints accuracy'''
    loss, accuracy = model.evaluate(test['data'], test['labels'], batch_size=batch_size, verbose=verbose)
    print(f'The loss is {round(loss,4)}, the accuracy is {round(accuracy*100,4)}% and the error is {round(100 - accuracy*100,4)}%')
    return model.predict(test['data'])  # returns the predictions


def error_type(predictions, test_labels):
    """returns a vector of all error types
    "Type 1: miss-detection: the algorithm thought it is not an flower, but it is
    Type 2: false alarm: the algorithm thought it is an flower, but it is not """
    errors = {'type 1': [], 'type 2': []}
    errors['type 1'] = {'index': [], 'score': []}
    errors['type 2'] = {'index': [], 'score': []}
    score_type_1 = []
    score_type_2 = []
    for i in range(len(predictions)):
        predict_1 = predictions[i][1]
        if predict_1 <= 0.5 and test_labels[i][1] == 1:  # type 1: thought it's not flower(0) but it's (1)
            score_type_1.append((i+301, predict_1))# wich score?
        if predict_1 > 0.5 and test_labels[i][0] == 1:  # type 2: thought it's flower (1) but it's not (0)
            score_type_2.append((i+301, predict_1))
    score_type_1.sort(key = takeSecond)
    score_type_2.sort(key = takeSecond)
    max_score_type_1 = score_type_1[0:min(len(score_type_1),5)]
    max_score_type_2 = score_type_2[0:min(len(score_type_2),5)]
    if len(max_score_type_1) != 0:
        for i in range(5):
            print("Error type 1, ", "Index :" + str(takefirst(max_score_type_1[i])), "Picture number and score: " + str(takeSecond(max_score_type_1[i])))
    else:
        print("There is no type 2 errors")

    if len(max_score_type_2) != 0:
        for i in range(5):
            print("Error type 2, ", "Index :" + str(takefirst(max_score_type_2[i])), "Picture number and score: " + str(takeSecond(max_score_type_2[i])))
    else:
        print("There is no type 2 errors")
    #all errors
    errors['type 1'] = {'index': takefirst(score_type_1), 'distance': takeSecond(score_type_1)}
    errors['type 2'] = {'index': takefirst(score_type_2), 'distance': takeSecond(score_type_2)}
    return errors


def takeSecond(elem):
    return elem[1] #sort by second element function


def takefirst(elem):
    return elem[0] #sort by second element function


def create_plot(x, y, x_name):
    plt.plot(x,y)
    plt.ylabel('Accuracy')
    plt.xlabel(x_name)
    plt.title(f'Accuracy vs. {x_name}')
    plt.ylim(0, 1)
    plt.show()


def tuning(train, batch_size, verbose):
    '''tune hyper parameters to improve the model'''
    # ---- train the net
    acc = []
    for epochs in range(1, 7):
        print(f'epochs = {epochs}')
        model = reconstruct_net(S, NUMBER_OF_CLASSES,'sigmoid',SGD(lr=0.01, decay=0.001))
        hist = train_model(model, train, batch_size, epochs, verbose)
        acc.append(hist.history['val_accuracy'][- 1])
    create_plot(range(1, 7), acc, 'Epochs')
    print(f' tune epochs acc = {acc}')
    chosen_epochs = acc.index(max(acc))+1
    acc.clear()

    activations = ['sigmoid', 'relu', 'softmax', 'elu', 'softsign', 'tanh']
    for activation in activations:
        print(f'activation = {activation}')
        model = reconstruct_net(S, NUMBER_OF_CLASSES, activation, SGD(lr=0.01, decay=0.001))
        hist = train_model(model, train, batch_size, chosen_epochs, verbose)
        #val_acc = hist.history['val_accuracy']
        acc.append(hist.history['val_accuracy'][-1])
    create_plot(activations, acc, 'Activation')
    print(f' tune activation acc = {acc}')
    chosen_activation = activations[acc.index(max(acc))]
    acc.clear()
    learning_rates = [0.01, 0.02, 0.03, 0.04, 0.05]
    decays = [0.001, 0.005, 0.01, 0.1]
    for lr in learning_rates:
        for decay in decays:
            print(f'leraning rate = {lr} and decay = {decay}')
            model = reconstruct_net(S, NUMBER_OF_CLASSES, chosen_activation, SGD(lr=lr, decay=decay))
            hist = train_model(model, train, batch_size, chosen_epochs, verbose)
            # val_acc = hist.history['val_accuracy']
            acc.append(hist.history['val_accuracy'][- 1])
    print(f' tune lr and decay acc = {acc}')
    lr_index = round(acc.index(max(acc)) / 5)
    decey_index = acc.index(max(acc)) / lr_index
    create_plot(range(1,13), acc, 'SGD parameters')

    print('######## Final Tuning Results ############')
    print(f'the chosen epochs is {chosen_epochs}')
    print(f'the chosen activation is {chosen_activation}')
    print(f'max accuracy index is {acc.index(max(acc))}')
    print(f'lr index = {lr_index}, decay index = {decey_index}')
    print(f'lr = {learning_rates[lr_index]}, decay = {decays[decey_index]}')

    # do we need to decide by loss or by accuracy?????

def recall_precision_curve(model,test_samples, test_labels, pred):
    '''creates precision curve'''
    # precision = dict()
    # recall = dict()
    # for i in range(2):
    #     precision[i], recall[i], _ = precision_recall_curve(test_labels[:, i], pred[:, i])
    # precision["micro"], recall["micro"], _ = precision_recall_curve(test_labels.ravel(), pred.ravel())
    # fig = plt.figure()
    # step_kwargs = ({'step': 'post'} if 'step' in signature(plt.fill_between).parameters else {})
    # plt.step(recall['micro'], precision['micro'], color='b', alpha=0.2, where='post')
    # plt.fill_between(recall["micro"], precision["micro"], alpha=0.2, color='b', **step_kwargs)
    # plt.xlabel('Recall'), plt.ylabel('Precision'), plt.title('Precision Recall Curve')
    # plt.show()
    average_precision = average_precision_score(test_labels, pred)
    print('Average precision-recall score: {0:0.2f}'.format(average_precision))
    disp = plot_precision_recall_curve(model, test_samples, test_labels)
    disp.ax_.set_title('2-class Precision-Recall curve: '
                       'AP={0:0.2f}'.format(average_precision))


def create_data_augmentation():
    ''' create images of train tranpose on the y axis'''
    for i in range(1,TRAIN_SIZE+1):
        image_dir_file = data_path + "/" + str(i) + ".jpeg"
        image2 = Image.open(image_dir_file).transpose(Image.FLIP_LEFT_RIGHT)
        image2.save(data_path + '/' + str(i + 472) + ".jpeg")


################# main ####################
def main():
    np.random.seed(0)  # seed
    create_data_augmentation()
    train, test = set_and_split_data()
    tuning(train, BATCH_SIZE, VERBOSE)
    res_net_new = reconstruct_net(S, NUMBER_OF_CLASSES,'sigmoid',SGD(lr=0.01, decay=0.001))  # preparing the network
    train_model(res_net_new, train, BATCH_SIZE, EPOCHS, VERBOSE) # train and validation stage
    predictions = test_model(res_net_new, test, BATCH_SIZE, VERBOSE)  # test stage
    error_type_array = error_type(predictions, test['labels'])  # find the error types
    recall_precision_curve(res_net_new, np.array(test['data']), np.array(test['labels']), np.array(predictions))  # recall-precision curve

if __name__ == "__main__":
    main()
