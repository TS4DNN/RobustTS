from tensorflow.keras.optimizers import SGD, Adam
from tensorflow.keras import Model, Input
from tensorflow.keras.layers import Conv2D, MaxPooling2D, AveragePooling2D
from tensorflow.keras.layers import Dense, Dropout, Activation, Flatten,Input
from tensorflow.keras.models import Model, load_model
from svhn_utils import *
from tensorflow.keras.callbacks import ModelCheckpoint
from ResNet import *


def svhn_lenet_5_model():
    nb_classes = 10
    input_tensor = Input((32, 32, 3))
    # 28*28
    temp = Conv2D(filters=6, kernel_size=(5, 5), padding='valid', use_bias=False)(input_tensor)
    temp = Activation('relu')(temp)
    # 24*24
    temp = MaxPooling2D(pool_size=(2, 2))(temp)
    # 12*12
    temp = Conv2D(filters=16, kernel_size=(5, 5), padding='valid', use_bias=False)(temp)
    temp = Activation('relu')(temp)
    # 8*8
    temp = MaxPooling2D(pool_size=(2, 2))(temp)
    # 4*4
    # 1*1
    temp = Flatten()(temp)
    temp = Dropout(0.5)(temp, training=True)
    temp = Dense(120, activation='relu')(temp)
    temp = Dense(84, activation='relu')(temp)
    temp = Dense(nb_classes)(temp)
    output = Activation('softmax', name='predictions')(temp)
    model = Model(input_tensor, output)
    return model


def svhn_lenet5_drop():
    nb_classes = 10
    input_tensor = Input((32, 32, 3))
    # 28*28
    temp = Conv2D(filters=6, kernel_size=(5, 5), padding='valid', use_bias=False)(input_tensor)
    temp = Activation('relu')(temp)
    # 24*24
    temp = MaxPooling2D(pool_size=(2, 2))(temp)
    # 12*12
    temp = Conv2D(filters=16, kernel_size=(5, 5), padding='valid', use_bias=False)(temp)
    temp = Activation('relu')(temp)
    # 8*8
    temp = MaxPooling2D(pool_size=(2, 2))(temp)
    # 4*4
    # 1*1
    temp = Flatten()(temp)
    temp = Dropout(0.5)(temp, training=True)
    temp = Dense(120, activation='relu')(temp)
    temp = Dense(84, activation='relu')(temp)
    temp = Dropout(0.2)(temp, training=True)
    temp = Dense(nb_classes)(temp)
    output = Activation('softmax', name='predictions')(temp)
    model = Model(input_tensor, output)
    return model


def svhn_lenet_1_model():
    nb_classes = 10
    input_tensor = Input((32, 32, 3))
    # 28*28
    temp = Conv2D(filters=6, kernel_size=(5, 5), padding='valid', use_bias=False)(input_tensor)
    temp = Activation('relu')(temp)
    # 24*24
    temp = MaxPooling2D(pool_size=(2, 2))(temp)
    # 12*12
    temp = Conv2D(filters=16, kernel_size=(5, 5), padding='valid', use_bias=False)(temp)
    temp = Activation('relu')(temp)
    # 8*8
    temp = MaxPooling2D(pool_size=(2, 2))(temp)
    # 4*4
    # 1*1
    temp = Flatten()(temp)
    temp = Dropout(0.5)(temp, training=True)
    temp = Dense(nb_classes)(temp)
    output = Activation('softmax', name='predictions')(temp)
    model = Model(input_tensor, output)
    return model


def model_svhn(model_type, save_path):
    (x_train, y_train), (x_test, y_test) = load_data()  # 32*32
    x_train = x_train[:-10000]
    y_train = y_train[:-10000]
    x_test = x_test[:10000]
    y_test = y_test[:10000]
    print('Train:{},Test:{}'.format(len(x_train), len(x_test)))
    print('data success')
    if model_type == 'lenet5':
        model = svhn_lenet_5_model()
        sgd = SGD(lr=0.01, decay=1e-6, momentum=0.9, nesterov=True)
        model.compile(loss='categorical_crossentropy', optimizer=sgd, metrics=['accuracy'])
    if model_type == 'lenet1':
        model = svhn_lenet_1_model()
        sgd = SGD(lr=0.01, decay=1e-6, momentum=0.9, nesterov=True)
        model.compile(loss='categorical_crossentropy', optimizer=sgd, metrics=['accuracy'])
    else:
        input_shape = (32, 32, 3)
        model = resnet20(input_shape, 10)
        model.compile(loss='categorical_crossentropy',
                      optimizer=Adam(lr=1e-3),
                      metrics=['accuracy'])
    model.summary()
    checkpoint = ModelCheckpoint(filepath=save_path, monitor='val_accuracy', mode='auto', save_best_only='True')
    model.fit(x_train, y_train, batch_size=64, epochs=50, validation_data=(x_test, y_test),callbacks=[checkpoint])
    model = load_model(save_path)
    score = model.evaluate(x_test, y_test, verbose=0)
    print(score)


if __name__ == "__main__":
    model_svhn("lenet1", "../models/svhn/lenet1.h5")


