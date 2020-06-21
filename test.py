from google.colab import drive
drive.mount('/content/drive')
from keras.layers import (Conv3D, BatchNormalization, AveragePooling3D, concatenate, Lambda,
                          Activation, Input, GlobalAvgPool3D, Dense)
from keras.regularizers import l2 as l2_penalty
from keras.models import Model

PARAMS = {
    'activation': lambda: Activation('relu'),  # the activation functions
    'bn_scale': True,  # whether to use the scale function in BN
    'weight_decay': 0.0001,  # l2 weight decay
    'kernel_initializer': 'he_uniform',  # initialization
    'first_scale': lambda x: x / 128. - 1.,  # the first pre-processing function
    'dhw': [32, 32, 32],  # the input shape
    'k': 16,  # the `growth rate` in DenseNet
    'bottleneck': 4,  # the `bottleneck` in DenseNet
    'compression': 2,  # the `compression` in DenseNet
    'first_layer': 32,  # the channel of the first layer
    'down_structure': [4, 4, 4],  # the down-sample structure
    'output_size': 2  # the output number of the classification head
}

def _conv_block(x, filters):
    bn_scale = PARAMS['bn_scale']
    activation = PARAMS['activation']
    kernel_initializer = PARAMS['kernel_initializer']
    weight_decay = PARAMS['weight_decay']
    bottleneck = PARAMS['bottleneck']

    x = BatchNormalization(scale=bn_scale, axis=-1)(x)
    x = activation()(x)
    x = Conv3D(filters * bottleneck, kernel_size=(1, 1, 1), padding='same', use_bias=False,
               kernel_initializer=kernel_initializer,
               kernel_regularizer=l2_penalty(weight_decay))(x)
    x = BatchNormalization(scale=bn_scale, axis=-1)(x)
    x = activation()(x)
    x = Conv3D(filters, kernel_size=(3, 3, 3), padding='same', use_bias=True,
               kernel_initializer=kernel_initializer,
               kernel_regularizer=l2_penalty(weight_decay))(x)
    return x

def _dense_block(x, n):
    k = PARAMS['k']

    for _ in range(n):
        conv = _conv_block(x, k)
        x = concatenate([conv, x], axis=-1)
    return x

def _transmit_block(x, is_last):
    bn_scale = PARAMS['bn_scale']
    activation = PARAMS['activation']
    kernel_initializer = PARAMS['kernel_initializer']
    weight_decay = PARAMS['weight_decay']
    compression = PARAMS['compression']

    x = BatchNormalization(scale=bn_scale, axis=-1)(x)
    x = activation()(x)
    if is_last:
        x = GlobalAvgPool3D()(x)
    else:
        *_, f = x.get_shape().as_list()
        x = Conv3D(f // compression, kernel_size=(1, 1, 1), padding='same', use_bias=True,
                   kernel_initializer=kernel_initializer,
                   kernel_regularizer=l2_penalty(weight_decay))(x)
        x = AveragePooling3D((2, 2, 2), padding='valid')(x)
    return x

def get_model(weights=None, **kwargs):
    for k, v in kwargs.items():
        assert k in PARAMS
        PARAMS[k] = v
    print("Model hyper-parameters:", PARAMS)

    dhw = PARAMS['dhw']
    first_scale = PARAMS['first_scale']
    first_layer = PARAMS['first_layer']
    kernel_initializer = PARAMS['kernel_initializer']
    weight_decay = PARAMS['weight_decay']
    down_structure = PARAMS['down_structure']
    output_size = PARAMS['output_size']

    shape = dhw + [1]

    inputs = Input(shape=shape)

    if first_scale is not None:
        scaled = Lambda(first_scale)(inputs)
    else:
        scaled = inputs
    conv = Conv3D(first_layer, kernel_size=(3, 3, 3), padding='same', use_bias=True,
                  kernel_initializer=kernel_initializer,
                  kernel_regularizer=l2_penalty(weight_decay))(scaled)

    downsample_times = len(down_structure)
    for l, n in enumerate(down_structure):
        db = _dense_block(conv, n)
        conv = _transmit_block(db, l == downsample_times - 1)

    if output_size == 1:
        last_activation = 'sigmoid'
    else:
        last_activation = 'softmax'

    outputs = Dense(output_size, activation=last_activation,
                    kernel_regularizer=l2_penalty(weight_decay),
                    kernel_initializer=kernel_initializer)(conv)

    model = Model(inputs, outputs)
    model.summary()

    if weights is not None:
        model.load_weights(weights, by_name=True)
    return model

def get_compiled(loss='categorical_crossentropy', optimizer='adam',
                 metrics=["categorical_accuracy"],
                 weights=None, **kwargs):
    model = get_model(weights=weights, **kwargs)
    model.compile(loss=loss, optimizer=optimizer,
                  metrics=[loss] + metrics)
    return model

# used for finding the optimal model after training\n",
import os
import numpy as np
import tensorflow as tf
import gc

config = tf.compat.v1.ConfigProto()
config.gpu_options.allow_growth = True
sess = tf.compat.v1.Session(config=config)
z = np.ones((117,2))
batch_size = 32
pprcv = 32
pprcvdb2 = int(pprcv/2)
x_test = np.ones((117, pprcv, pprcv, pprcv))

i = 0
for j in range(0,583):
    try:
        tmp = np.load('/content/drive/My Drive/datas/test/test/candidate' + str(j) + '.npz')
        voxel = tmp['voxel']
        seg = tmp['seg']
        x_test[i] = (voxel * seg)[50 - pprcvdb2:50 + pprcvdb2, 50 - pprcvdb2:50 + pprcvdb2, 50 - pprcvdb2:50 + pprcvdb2]
        i = i + 1
    except:
        continue
gc.collect()

x_test = x_test.reshape(x_test.shape[0], pprcv, pprcv, pprcv, 1)
model = get_compiled()
model.load_weights('/content/drive/My Drive/weights/weights.50.h5')
pre_result_1 = model.predict(x_test, batch_size, verbose=1)
model.load_weights('/content/drive/My Drive/weights/weights.30.h5')
pre_result_2 = model.predict(x_test, batch_size, verbose=1)
pre_result = (pre_result_1+pre_result_2)/2
print(pre_result)

my_goal = np.loadtxt("/content/drive/My Drive/datas/sampleSubmission.csv", str, delimiter=",", skiprows=1, usecols=0)
np.savetxt('/content/drive/My Drive/datas/Submission.csv',np.column_stack((my_goal,pre_result[:,0])), delimiter=',', fmt='%s', header='name,predicted', comments='')
