from keras import Input
from keras import models
from keras.layers import Conv2D, MaxPool2D, Dense, Flatten, Dropout


def mnist_conv(input_size):
    X_input = Input(input_size)
    X = Conv2D(32, kernel_size=(3, 3))(X_input)
    X = MaxPool2D(pool_size=(2, 2))(X)
    X = Flatten()(X)
    X = Dense(128, activation='relu')(X)
    X = Dropout(0.2)(X)
    X = Dense(10, activation='softmax')(X)

    model = models.Model(inputs=X_input, outputs=X)

    return model
