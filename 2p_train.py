import keras.losses
from sklearn.utils import class_weight
from model import *
from data_loading import *
from sklearn.preprocessing import StandardScaler,MaxAbsScaler
from sklearn.metrics import accuracy_score
from keras.optimizers import Adam
from keras.losses import categorical_crossentropy, sparse_categorical_crossentropy
import numpy as np
from keras.utils import to_categorical
from sklearn.utils import compute_class_weight
import scipy.io as sio
import matplotlib.pylab as plt
from sklearn.manifold import TSNE
import keras.backend as K
import tensorflow as tf
from sklearn.metrics import *
from sklearn.metrics import f1_score, accuracy_score
from data_preprocessing import *
import gc
from sklearn.metrics import roc_curve, roc_auc_score, auc
import os
from scipy.io import savemat
from keras.callbacks import ModelCheckpoint, EarlyStopping,ReduceLROnPlateau
import pandas as pd
import sys

from tensorflow.python.client import device_lib
os.environ["CUDA_VISIBLE_DEVICES"] = "0"
print(device_lib.list_local_devices())

def mean_squared_error(y_true, y_pred):
    loss = tf.math.reduce_mean(tf.math.square(y_pred - y_true),axis=-1)
    return loss
def SparseCategoricalCrossentropy(class_weight = None):
    def inner_sparse_categorical_crossentropy(y_true, y_pred):
        scce = tf.keras.losses.SparseCategoricalCrossentropy()
        if class_weight:
            keys_tensor = tf.cast(tf.constant(list(class_weight.keys())), dtype=tf.int32)
            vals_tensor = tf.constant(list(class_weight.values()), tf.float32)
            input_tensor = tf.cast(y_true, dtype=tf.int32)
            init = tf.lookup.KeyValueTensorInitializer(keys_tensor, vals_tensor)
            table = tf.lookup.StaticHashTable(init, default_value=-1)
            sample_weight = table.lookup(input_tensor)
        else:
            sample_weight = None
        return scce(y_true, y_pred, sample_weight)
    return inner_sparse_categorical_crossentropy


F1 = 8
KE = 32
KT = 4
L = 2
FT = 12
pe = 0.2
pt = 0.3
classes = 4
channels = 22
crossValidation = False
batch_size = 64
epochs = 750
lr = 0.0009


def predict(X_train,X_test,y_train_onehot,y_test_onehot,subject,phase,model):
    if phase == 1:
        model.load_weights('weight/out_weights'+str(subject)+'FU(phase1).h5')
    elif phase == 2:
        model.load_weights('weight/out_weights' + str(subject) + 'FU(phase2).h5')
    y_pred, mse = model.predict(X_train)
    y_epred, emse = model.predict(X_test)
    one = np.argmax(y_pred,1)

    y_e = np.argmax(y_epred, 1)
    y_test = np.argmax(y_test_onehot, 1)

    print(confusion_matrix(y_test, y_e), '\n')
    print("subjects:" + str(subject))
    print("accracy:" + str(accuracy_score(y_test, y_e)))
    print("f1-score: " + str(f1_score(y_test, y_e, average='macro')))

    if phase == 1: # target generation
        y = np.argmax(y_train_onehot, 1)
        data1 = mse[np.squeeze(y == 0)]
        data2 = mse[np.squeeze(y == 1)]
        data3 = mse[np.squeeze(y == 2)]
        data4 = mse[np.squeeze(y == 3)]

        data1 = np.mean(data1, axis=0, keepdims=True)
        data2 = np.mean(data2, axis=0, keepdims=True)
        data3 = np.mean(data3, axis=0, keepdims=True)
        data4 = np.mean(data4, axis=0, keepdims=True)

        mat_dic = {"class1": data1, "class2": data2, "class3": data3, "class4": data4}
        savemat("data/target(" + str(subject) + ").mat", mat_dic)

    print(
        "----------------------------------------------------------------------------------------------------------------")
    return max

def train(subject, phase, m):
    X_train, _, y_train, X_test, _, y_test = get_data('data/', subject-1, True, True)

    if phase == 2:
        f = sio.loadmat("data/target(" + str(subject) + ").mat")
        data1 = np.array(f['class1'])
        data2 = np.array(f['class2'])
        data3 = np.array(f['class3'])
        data4 = np.array(f['class4'])

        target = np.ones((4, 1125, 22, 1))
        target[0] = data1
        target[1] = data2
        target[2] = data3
        target[3] = data4


        train_shape = X_train.shape[0]
        test_shape = X_test.shape[0]
        train_label = np.argmax(y_train,-1)

        X_train = X_train.transpose(0,3,2,1)
        X_test = X_test.transpose(0,3,2,1)

        tspot = np.ones((train_shape, 1125, 22, 1))
        for i in range(train_shape):
            if train_label[i] == 0:
                tspot[i] = data1
            elif train_label[i] == 1:
                tspot[i] = data2
            elif train_label[i] == 2:
                tspot[i] = data3
            elif train_label[i] == 3:
                tspot[i] = data4

        test_label = np.argmax(y_test, -1)
        espot = np.ones((test_shape, 1125, 22, 1))
        for i in range(test_shape):
            if test_label[i] == 0:
                espot[i] = data1
            elif test_label[i] == 1:
                espot[i] = data2
            elif test_label[i] == 2:
                espot[i] = data3
            elif test_label[i] == 3:
                espot[i] = data4

    if m == "ATCNet":
        model = ATCNET(
            # Dataset parameters
            n_classes=4,
            in_chans=22,
            in_samples=1125,
            # Sliding window (SW) parameter
            n_windows=5,
            # Attention (AT) block parameter
            attention='mha',  # Options: None, 'mha','mhla', 'cbam', 'se'
            # Convolutional (CV) block parameters
            eegn_F1=16,
            eegn_D=2,
            eegn_kernelSize=64,
            eegn_poolSize=7,
            eegn_dropout=0.3,
            # Temporal convolutional (TC) block parameters
            tcn_depth=2,
            tcn_kernelSize=4,
            tcn_filters=32,
            tcn_dropout=0.3,
            tcn_activation='elu'
        )
    elif m == "TCNet_Fusion":
        model = TCNet_Fusion(
            # Dataset parameters
            n_classes=4,
        )

    loss = [categorical_crossentropy, mean_squared_error]
    opt = Adam(lr=lr)
    if phase == 1:
        callbacks = [
            ModelCheckpoint(monitor='val_loss', filepath='weight/out_weights'+str(subject)+'(phase1).h5',verbose=0, save_best_only=True, save_weight_only=True),
            EarlyStopping(monitor='val_loss', verbose=0, mode='min', patience=300),
        ]
        model.compile(loss=loss, optimizer=opt, metrics=['accuracy'], loss_weights=[1.0, 0.01])
        #model.summary()
        model.fit(x=X_train, y=[y_train, X_train], batch_size=batch_size, epochs=1000, verbose=1, validation_data=(X_test, [y_test, X_test]), callbacks=callbacks)
        predict(X_train, X_test, y_train, y_test, subject, phase, model)

    if phase == 2:
        callbacks = [
            ModelCheckpoint(monitor='val_loss', filepath='weight/out_weights'+str(subject)+'(phase2).h5',verbose=0, save_best_only=True, save_weight_only=True),
            EarlyStopping(monitor='val_loss', verbose=0, mode='min', patience=300),
        ]
        model.compile(loss=loss, optimizer=opt, metrics=['accuracy'], loss_weights=[0.01, 1.0])

        model.fit(x=X_train,y=[y_train,tspot],batch_size=batch_size, epochs=1000, verbose=1, validation_data=(X_test, [y_test,espot]), callbacks=callbacks)
        predict(X_train, X_test, y_train, y_test, subject, phase, model)


for subject in range(9):
    for i in range(0, 10):
        train(subject, phase=1, model="TCNet-Fusion")
        train(subject, phase=2, model="TCNet-Fusion")