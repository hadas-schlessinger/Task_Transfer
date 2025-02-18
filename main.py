import tensorflow as tf
import scipy.io as sio
import os
import numpy as np
import cv2
import keras
import matplotlib.pyplot as plt
from PIL import Image
import keras.applications
from keras.layers import Input, Dense
from keras.models import Model
from keras.applications.resnet_v2 import preprocess_input
from sklearn.metrics import precision_recall_curve
from funcsigs import signature
from keras.optimizers import SGD
from keras.preprocessing import image
from sklearn.model_selection import train_test_split


test_images_indices = list(range(301, 473))

# env variables
S = 224
NUMBER_OF_CLASSES = 1
BATCH_SIZE = 16
EPOCHS = 2
VERBOSE = 1
ACTIVATION = 'sigmoid'
data_path = os.path.join(os.getcwd(), 'FlowerData')  # The images path
mat_file = data_path + '/FlowerDataLabels.mat'
NEEDS_AUG = True
LR = 0.03
DECAY = 0.01
LAYERS_TO_TRAIN = -3
THRESHOLD = 0.7

def set_and_split_data():
    '''set the images and split them'''
    train = {'data': [],
             'labels': []}
    test = {'data': [],
            'labels': [],
            }
    # dictionary with variable names as keys, and loaded matrices as values.
    dictionary_labels = sio.loadmat(mat_file, mdict=None, appendmat=True)
    labels_from_mat = np.transpose(dictionary_labels['Labels']).tolist()

    for i in range(len(labels_from_mat)):  # for filename in data path folder which is flowerData
        image_dir_file = data_path + "/" + str(i+1) + ".jpeg"
        # image read and converting it image pixels to a numpy array
        a = preprocess_input(np.expand_dims(image.img_to_array(image.load_img(image_dir_file, target_size=(S, S))), axis=0))
        if i+1 in test_images_indices:  # inserts the test set
            test['data'].append(a)  # accumulate image array
            test['labels'].append(labels_from_mat[i])  # divide the dictionary into labels vector of test
        else:  # inserts the train set and it's augmentation image into train set
            train['data'].append(a)  # connect a to the big 4D array of input images
            train['labels'].append(labels_from_mat[i])  # divide the dictionary into labels vector of train
            aug_dir_file = data_path + "/" + str(i +1000) + ".jpeg"
            a_aug = Image.open(image_dir_file).transpose(Image.FLIP_LEFT_RIGHT) # create image augmentation
            a_aug.save(aug_dir_file)
            a_aug = preprocess_input(np.expand_dims(image.img_to_array(image.load_img(aug_dir_file, target_size=(S, S))), axis=0))
            train['data'].append(a_aug)  # connect a to the big 4D array of input images
            train['labels'].append(labels_from_mat[i])  # divide the dictionary into labels vector of train
    # changing the train anf test into numpy array for fitting
    train['data'] = np.rollaxis(np.array(train['data']), 1, 0)[0]
    test['data'] = np.rollaxis(np.array(test['data']), 1, 0)[0]
    # changing the labels into numpy array for fitting
    train['labels'] = np.array(train['labels'])
    test['labels'] = np.array(test['labels'])
    # split train to train and validation
    train_images, valid_images, train_labels, valid_labels = train_test_split(train['data'], train['labels'],
                                                                              test_size=0.33, shuffle=True)
    train = {
        'data': train_images,
        'labels': train_labels
    }
    validation = {
        'data': valid_images,
        'labels': valid_labels
    }
    return train, test, validation


def reconstruct_net(activation, optimizer, what_to_train):
    '''export the net without the last layer'''
    image_input = Input(shape=(S, S, 3))
    basic_model = keras.applications.resnet_v2.ResNet50V2(include_top=True, weights='imagenet', input_tensor=image_input) # extract the ResNet50V2 net
    last_layer_minus_1 = basic_model.layers[-2].output # takes the last pooling layer
    # compile our model, connected to the original net last pooling layer
    model_without_last_layer = Model(image_input, Dense(NUMBER_OF_CLASSES, activation=activation, name='output')(last_layer_minus_1))
    # model_without_last_layer.summary()
    for layer in model_without_last_layer.layers[:what_to_train]:  # decide what layers to train on
        layer.trainable = False
    model_without_last_layer.compile(loss='binary_crossentropy', optimizer=optimizer, metrics=['accuracy']) # compile our net
    # model_without_last_layer.summary()
    return model_without_last_layer


def train_model(res_net_basic, train_images, train_labels, valid_images, valid_labels, batch_size, epochs, verbose):
    '''train the model'''
    return res_net_basic.fit(train_images, train_labels, batch_size=batch_size, epochs=epochs,
                         verbose=verbose, validation_data=(valid_images, valid_labels), shuffle=True)  # fitting the model


def test_model(model, test, batch_size, verbose):
    '''test and prints accuracy'''
    # print accuracy for threshold 0.5
    loss, accuracy = model.evaluate(test['data'], test['labels'], batch_size=batch_size, verbose=verbose)
    print(f'The loss is {round(loss,4)}, the accuracy is {round(accuracy*100,4)}% '
          f'and the error is {round(100 - accuracy*100,4)}%')
    # print accuracy for chosen threshold
    new_predictions = _fit_threshold(model.predict(test['data']), THRESHOLD)
        #tf.keras.metrics.binary_accuracy(test['labels'], model.predict(test['data']), threshold=0.42)
    new_acc = _calc_accuracy(test['labels'], new_predictions)
    print(f'################################################Calculating new threshold predictions and accuracy')
    print(f'the new prediction labels are {new_predictions}')
    print(f'the new accuracy according to the new threshold is {round(new_acc*100,4)}%')
    return model.predict(test['data'])  # returns the predictions


def _fit_threshold(predictions, threshold):
    new_predictions = []
    [new_predictions.append(0) if predictions[i] <= threshold else new_predictions.append(1) for i in range(len(predictions))]
    return new_predictions


def _calc_accuracy(real_labels, new_labels):
    '''calcs the accuracy for the new threshold'''
    counter = 0
    for i in range(len(new_labels)):
        if real_labels[i] == new_labels[i]:
            counter = counter+1
    return counter/len(new_labels)


def error_type(predictions, test_labels):
    """returns a vector of all error types
    "Type 1: miss-detection: the algorithm thought it is not an flower, but it is
    Type 2: false alarm: the algorithm thought it is an flower, but it is not """
    score_type_1 = []
    score_type_2 = []
    for i in range(len(predictions)):
        pred = predictions[i]
        index_i = test_images_indices[i]
        if pred <= THRESHOLD and test_labels[i] == 1:  # type 1: the net observed a non-flower although it's a flower (1)
            score_type_1.append((index_i, pred))
        if pred > THRESHOLD and test_labels[i] == 0:  # type 2: the net observed a flower although it's not a flower (0)
            score_type_2.append((index_i,pred))
    score_type_1.sort(key=take_second,reverse=False) # sort by min to order by worst error
    score_type_2.sort(key=take_second,reverse=True) # sort by max to order by worst error
    min_score_type_1 = score_type_1[0:min(len(score_type_1),5)]
    max_score_type_2 = score_type_2[0:min(len(score_type_2),5)]

    [print('Error type 1: ', f'Error index: {i+1},', 'Picture index :' + str(take_first(min_score_type_1[i])) +
           ',', 'Score: ' + str(take_second(min_score_type_1[i]))) if len(min_score_type_1) != 0 else print("There is no type 1 errors")
     for i in range(len(min_score_type_1))]
    [print('Error type 2: ', f'Error index: {i+1},', 'Picture index :' + str(take_first(max_score_type_2[i])) + ',',
           'Score: ' + str(take_second(max_score_type_2[i])))  if len(max_score_type_2) != 0 else print("There is no type 1 errors")
     for i in range(len(max_score_type_2))]


def take_second(elem):
    return elem[1] #sort by second element function


def take_first(elem):
    return elem[0] #sort by second element function


def create_plot(x, y, x_name):
    '''create tunning plots'''
    plt.plot(x,y)
    plt.ylabel('Accuracy')
    plt.xlabel(x_name)
    plt.title(f'Accuracy vs. {x_name}')
    plt.ylim(0, 1)
    plt.show()


def _tune_activation(train, validation, batch_size, verbose):
    '''tune the activation function,
    each time we calc the model with different parameter and chose the parameter that gives the best accuracy'''
    acc = []
    activations = ['sigmoid', 'relu', 'softmax', 'elu', 'softsign', 'tanh']
    for activation in activations:
        print(f'activation = {activation}')
        model = reconstruct_net(activation, SGD(lr=0.01, decay=0.001), LAYERS_TO_TRAIN)
        hist = train_model(model, train['data'], train['labels'], validation['data'], validation['labels'], batch_size, EPOCHS, verbose)
        acc.append(hist.history['val_accuracy'][-1]) # take the validation accuracy
    create_plot(activations, acc, 'Activation')
    print(f' tune activation acc = {acc}')
    chosen_activation = activations[acc.index(max(acc))]
    print(f'the chosen activation is {chosen_activation}')
    return chosen_activation


def _tune_layer(train, validation,batch_size, verbose, chosen_activation):
    '''tune the layers to train on,
    each time we calc the model with different parameter and chose the parameter that gives the best accuracy'''
    acc = []
    for layer in [-1, -2, -3, -4, -5]:
        print(f'layer = {layer}')
        model = reconstruct_net(chosen_activation, SGD(lr=0.01, decay=0.001), layer)
        hist = train_model(model,train['data'], train['labels'], validation['data'], validation['labels'], batch_size, EPOCHS, verbose)
        acc.append(hist.history['val_accuracy'][-1]) # take the validation accuracy
    create_plot(range(1,6), acc, 'Number of layers to train')
    print(f' tune layers acc = {acc}')
    chosen_layer = acc.index(max(acc))+1
    print(f'the chosen layer is {chosen_layer}')
    return -1*chosen_layer


def _tune_epochs(train, validation,batch_size, verbose, chosen_activation, chosen_layer):
    '''tune the ephocs,
    each time we calc the model with different parameter and chose the parameter that gives the best accuracy'''
    acc = []
    for epochs in range(1, 8):
        print(f'epochs = {epochs}')
        model = reconstruct_net(chosen_activation, SGD(lr=0.01, decay=0.001), chosen_layer)
        hist = train_model(model, train['data'], train['labels'], validation['data'], validation['labels'], batch_size, epochs, verbose)
        acc.append(hist.history['val_accuracy'][- 1]) # take the validation accuracy
    accuracy_plot(hist)
    print(f' tune epochs acc = {acc}')
    chosen_epochs = acc.index(max(acc)) + 1
    print(f'the chosen epochs is {chosen_epochs}')
    return chosen_epochs


def _tune_gsd(train, validation,batch_size, verbose, chosen_activation, chosen_layer, chosen_epochs):
    '''tune the GSD parameters,
    each time we calc the model with different parameter and chose the parameter that gives the best accuracy'''
    acc = []
    learning_rates = [0.01, 0.03, 0.05]
    decays = [0.001, 0.005, 0.01, 0.1]
    for lr in learning_rates:
        for decay in decays:
            print(f'leraning rate = {lr} and decay = {decay}')
            model = reconstruct_net(chosen_activation, SGD(lr=lr, decay=decay), chosen_layer)
            hist = train_model(model, train['data'], train['labels'], validation['data'], validation['labels'], batch_size, chosen_epochs, verbose)
            acc.append(hist.history['val_accuracy'][- 1]) # take the validation accuracy
    print(f' tune lr and decay acc = {acc}')
    print(f'max accuracy index is {acc.index(max(acc))}')
    # extract the final values
    lr_index = round(acc.index(max(acc)) / 3)
    decey_index = int(acc.index(max(acc)) / lr_index)
    create_plot(range(1, 13), acc, 'SGD parameters')
    print(f'lr index = {lr_index}, decay index = {decey_index}')
    return learning_rates[lr_index], decays[decey_index]


def _tune_threshold(val_labels, val_data, chosen_activation, lr, decay, chosen_layer, train, chosen_epochs):
    '''tune the threshold on the validation only'''
    acc = []
    thresholds = [0.42, 0.45, 0.5, 0.55, 0.6, 0.65, 0.7, 0.75, 0.8]
    # train model
    model = reconstruct_net(chosen_activation, SGD(lr=lr, decay=decay), chosen_layer)
    train_model(model, train['data'], train['labels'], val_data, val_labels, BATCH_SIZE, chosen_epochs, VERBOSE)
    # check different thresholds on the validation set as a test set
    for thrsh in thresholds:
        loss, accuracy = model.evaluate(val_data, val_labels, batch_size=BATCH_SIZE, verbose=VERBOSE)
        print(f'The loss is {round(loss, 4)}, the accuracy is {round(accuracy * 100, 4)}% and the error is {round(100 - accuracy * 100, 4)}%')
        new_predictions = _fit_threshold(model.predict(val_data), thrsh)
            #keras.metrics.binary_accuracy(val_labels, model.predict(val_data), threshold=thrsh)
        new_acc = _calc_accuracy(val_labels, new_predictions)
        print(f'the new accuracy is {new_acc}')
        acc.append(new_acc)
    print(f' tune threshold acc = {acc}')
    chosen_thresh = thresholds[acc.index(max(acc))]
    print(f'threshold is: {chosen_thresh}')
    create_plot(thresholds, acc, 'Threshold')
    return chosen_thresh


def tuning(train, validation, batch_size, verbose):
    '''tune hyper parameters to improve the model'''
    chosen_activation = _tune_activation(train, validation, batch_size, verbose)
    chosen_layer = _tune_layer(train, validation,  batch_size, verbose, chosen_activation)
    chosen_epochs = _tune_epochs(train, validation, batch_size, verbose, chosen_activation, chosen_layer)
    chosen_lr, chosen_decay = _tune_gsd(train, validation,  batch_size, verbose, chosen_activation, chosen_layer, chosen_epochs)
    threshold = _tune_threshold(validation['labels'], validation['data'], chosen_activation, chosen_lr, chosen_decay, chosen_layer, train, chosen_epochs)
    print('######## Final Tuning Results ############')
    print(f'the chosen epochs is {chosen_epochs}')
    print(f'the chosen activation is {chosen_activation}')
    print(f'lr = {chosen_lr}, decay = {chosen_decay}')
    print(f'the chosen threshold is {threshold}')


def recall_precision_curve(test_labels, pred):
    '''creates precision curve'''
    precision = dict()
    recall = dict()
    precision[0], recall[0], _ = precision_recall_curve(test_labels[:, 0], pred[:, 0])
    precision["micro"], recall["micro"], _ = precision_recall_curve(test_labels.ravel(), pred.ravel())
    fig = plt.figure()
    step_kwargs = ({'step': 'post'} if 'step' in signature(plt.fill_between).parameters else {})
    plt.step(recall['micro'], precision['micro'], color='b', alpha=0.2, where='post')
    plt.fill_between(recall["micro"], precision["micro"], alpha=0.2, color='b', **step_kwargs)
    plt.xlabel('Recall'), plt.ylabel('Precision'), plt.title('Precision Recall Curve')
    plt.show()


def show_images(images):
    for i in range(len(images)):
        plt.imshow(images[i])
        plt.show()


def accuracy_plot(history):
    '''summarize history for accuracy of ephocs'''
    plt.plot(history.history['accuracy'])
    plt.plot(history.history['val_accuracy'])
    plt.title('model accuracy')
    plt.ylabel('accuracy')
    plt.xlabel('epoch')
    plt.legend(['train', 'val'], loc='upper left')
    plt.show()


################# main ####################
def main():
    np.random.seed(0)
    print('##################################Step 1: splitting the data into train, test and validation sets ')
    train, test, validation = set_and_split_data()
    # tuning(train,validation, BATCH_SIZE, VERBOSE)
    print('##################################Step 2: re-constructing the net to fit out mission')
    res_net_new = reconstruct_net(ACTIVATION, SGD(lr = LR, decay = DECAY), LAYERS_TO_TRAIN)  # preparing the network
    print('##################################Step 3: training the model')
    train_model(res_net_new, train['data'], train['labels'], validation['data'], validation['labels'], BATCH_SIZE, EPOCHS, VERBOSE) # train stage
    print('##################################Step 4: testing the model')
    predictions = test_model(res_net_new, test, BATCH_SIZE, VERBOSE)  # test stage
    print('##################################Step 5: printing the largest errors')
    error_type(predictions, test['labels'])  # find the error types
    print('##################################Step 6: extracting the precision curve')
    recall_precision_curve(np.array(test['labels']), np.array(predictions))  # recall-precision curve


if __name__ == "__main__":
    main()
