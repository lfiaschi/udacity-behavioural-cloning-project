import keras
from keras.models import Sequential
from keras.layers import Dense, Dropout, Flatten
from keras.layers import Conv2D, MaxPooling2D, Lambda
from keras.layers import Cropping2D


def Simple(input_shape):
    model = Sequential()

    #normalization
    model.add(Lambda(lambda x: x / 127.5 - 1., input_shape=input_shape))
    model.add(Cropping2D(cropping=((70,25), (0,0))))
    model.add(Flatten())
    model.add(Dense(1))

    return model


def LeNetKerasMSE(input_shape, dropout=.5 ):

    """
    Implement classic lenet architecture in keras for regression
    """

    model = Sequential()

    # normalization
    model.add(Lambda(lambda x: x / 127.5 - 1., input_shape=input_shape))
    model.add(Cropping2D(cropping=((70, 25), (0, 0))))

    model.add(Conv2D(6, kernel_size=(5, 5),
                     activation='relu'))

    model.add(MaxPooling2D(pool_size=(2, 2)))

    model.add(Conv2D(16, (5, 5), activation='relu'))

    model.add(MaxPooling2D(pool_size=(2, 2)))

    model.add(Flatten())
    model.add(Dense(128, activation='relu'))
    model.add(Dropout(dropout))
    model.add(Dense(84, activation='relu'))
    model.add(Dropout(dropout))
    model.add(Dense(1, activation='linear'))

    return model