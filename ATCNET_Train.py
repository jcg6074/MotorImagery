import keras.losses
from sklearn.utils import class_weight
from ATCNET import *
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
from main import *
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


def predict(X_train,X_test,y_train_onehot,y_test_onehot,subject,max):
    """model = ATCNET(
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
    )"""
    model = TCNet_Fusion(
        # Dataset parameters
        n_classes=4,
    )
    model.load_weights('weight/out_weights'+str(subject)+'FU(phase2).h5')
    y_pred,mse = model.predict(X_train)
    y_epred, emse = model.predict(X_test)
    one = np.argmax(y_pred,1)

    """tsne_np = TSNE(n_components=2, perplexity=5.0, learning_rate=200.0).fit_transform(y_pred)
    class1 = tsne_np[np.squeeze(one == 0)]
    class2 = tsne_np[np.squeeze(one == 1)]
    class3 = tsne_np[np.squeeze(one == 2)]
    class4 = tsne_np[np.squeeze(one == 3)]

    plt.figure(figsize=(15, 15))
    plt.scatter(class1[:, 0], class1[:, 1], label='left hand', c='r')
    plt.scatter(class2[:, 0], class2[:, 1], label='right hand', c='g')
    plt.scatter(class3[:, 0], class3[:, 1], label='feet', c='b')
    plt.scatter(class4[:, 0], class4[:, 1], label='tongue', c='violet')
    plt.xlabel("t[0]")
    plt.ylabel("t[1]")
    plt.legend(loc='lower right')
    plt.savefig("tsne_tri(sub"+str(subject)+"_e).jpg")
    plt.clf()"""

    y_e = np.argmax(y_epred, 1)
    y_test = np.argmax(y_test_onehot, 1)

    print(confusion_matrix(y_test, y_e), '\n')
    print("subjects:" + str(subject))
    print("accracy:" + str(accuracy_score(y_test, y_e)))
    print("f1-score: " + str(f1_score(y_test, y_e, average='macro')))
    """if max < accuracy_score(y_test, y_e):
        max = accuracy_score(y_test, y_e)
        fpr = dict()
        tpr = dict()
        roc_auc = dict()
        for i in range(4):
            fpr[i],tpr[i],_ = roc_curve(y_test_onehot[:,i], y_epred[:,i])
            roc_auc[i] = auc(fpr[i],tpr[i])
        for idx,i in enumerate(range(4)):
            plt.figure(figsize=(15, 15))
            plt.plot(fpr[i], tpr[i], label='ROC curve (area = %0.2f)' % roc_auc[i])
            plt.plot([0, 1], [0, 1], 'k--')
            plt.xlim([0.0, 1.0])
            plt.ylim([0.0, 1.05])
            plt.xlabel('False Positive Rate')
            plt.ylabel('True Positive Rate')
            plt.title('Class %0.0f' % idx)
            plt.legend(loc="lower right")
            plt.savefig("roc_class"+str(i)+"(base).jpg")
            plt.clf()
        print("roc_auc_score: ", roc_auc_score(y_test_onehot, y_epred, multi_class='raise'))"""


    # print(classification_report(y_test_onehot, y_epred))
    # mat_dic = {"EEG": mse, "label": y_pred}
    # savemat("data/mse.mat", mat_dic)
    # sys.stdout.close()
    #if max < accuracy_score(y_test_onehot, y_epred):
     #   max = accuracy_score(y_test_onehot, y_epred)
    """y = np.argmax(y_train_onehot, 1)
    data1 = mse[np.squeeze(y == 0)]
    data2 = mse[np.squeeze(y == 1)]
    data3 = mse[np.squeeze(y == 2)]
    data4 = mse[np.squeeze(y == 3)]

    data1 = np.mean(data1, axis=0, keepdims=True)
    data2 = np.mean(data2, axis=0, keepdims=True)
    data3 = np.mean(data3, axis=0, keepdims=True)
    data4 = np.mean(data4, axis=0, keepdims=True)

    plt.figure(figsize=(15, 15))
    plt.plot(np.reshape(data1,(1125,22)))
    # plt.legend(loc='lower right')
    plt.savefig("class1.jpg")
    plt.clf()
    plt.figure(figsize=(15, 15))
    plt.plot(np.reshape(data2,(1125,22)))
    # plt.legend(loc='lower right')
    plt.savefig("class2.jpg")
    plt.clf()
    plt.figure(figsize=(15, 15))
    plt.plot(np.reshape(data3,(1125,22)))
    # plt.legend(loc='lower right')
    plt.savefig("class3.jpg")
    plt.clf()
    plt.figure(figsize=(15, 15))
    plt.plot(np.reshape(data4,(1125,22)))
    # plt.legend(loc='lower right')
    plt.savefig("class4.jpg")
    plt.clf()

    mat_dic = {"class1": data1, "class2": data2, "class3": data3, "class4": data4}
    savemat("data/middlespotClassFU(" + str(subject) + ")2.mat", mat_dic)"""

    print(
        "----------------------------------------------------------------------------------------------------------------")
    return max

def ttt(subject):
    max=0

    #sys.stdout = open('result.txt', 'a')

    X_train, _, y_train, X_test, _, y_test = get_data('data/', subject-1, True, True)

    dict = {"input": X_train, "label":  y_train}
    np.save("kaggledata/train.npy", dict)
    dict = {"input":  X_test}
    np.save("kaggledata/test.npy", dict)
    y_test = np.argmax(y_test, axis=1)

    df = pd.DataFrame(list(y_test), columns=["true"])
    df.to_csv("kaggledata/solution.csv", index=True)


    f = sio.loadmat("data/middlespotClassFU(" + str(subject) + ")2.mat")
    # f = sio.loadmat("data/middlespot(2).mat")
    data1 = np.array(f['class1'])
    data2 = np.array(f['class2'])
    data3 = np.array(f['class3'])
    data4 = np.array(f['class4'])

    target = np.ones((4, 1125, 22, 1))
    target[0] = data1
    target[1] = data2
    target[2] = data3
    target[3] = data4


    xs = X_train.shape[0]
    xvs = X_test.shape[0]
    yy = np.argmax(y_train,-1)

    X_train = X_train.transpose(0,3,2,1)
    X_test = X_test.transpose(0,3,2,1)

    tspot = np.ones((xs, 1125, 22, 1))
    for i in range(xs):
        if yy[i] == 0:
            tspot[i] = data1
        elif yy[i] == 1:
            tspot[i] = data2
        elif yy[i] == 2:
            tspot[i] = data3
        elif yy[i] == 3:
            tspot[i] = data4
    yy = np.argmax(y_test, -1)
    espot = np.ones((xvs, 1125, 22, 1))
    for i in range(xvs):
        if yy[i] == 0:
            espot[i] = data1
        elif yy[i] == 1:
            espot[i] = data2
        elif yy[i] == 2:
            espot[i] = data3
        elif yy[i] == 3:
            espot[i] = data4

    """model = ATCNET(
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
    )"""
    model = TCNet_Fusion(
        # Dataset parameters
        n_classes=4,
    )

    loss = [categorical_crossentropy, mean_squared_error]
    callbacks = [
        ModelCheckpoint(monitor='val_loss', filepath='weight/out_weights'+str(subject)+'FU(phase2).h5',verbose=0, save_best_only=True, save_weight_only=True),
        EarlyStopping(monitor='val_loss', verbose=0, mode='min', patience=300),
    ]
    opt = Adam(lr=lr)
    model.compile(loss=loss, optimizer=opt, metrics=['accuracy'], loss_weights=[0.01, 1.0])
    model.summary()
    #model.load_weights('weight/out_weights1FU(pro).h5')

    model.fit(x=X_train,y=[y_train,tspot],batch_size=batch_size, epochs=1000, verbose=1, validation_data=(X_test, [y_test,espot]), callbacks=callbacks)
    max = predict(X_train,X_test,y_train,y_test,subject,max)



for subject in range(9):
    for i in range(0, 10):
        ttt(subject)
        gc.collect()