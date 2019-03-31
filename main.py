import keras
from net import mnist_conv
from keras.datasets import mnist

# Get the mnist dataset and normalize the data
(x_train, y_train), (x_test, y_test) = mnist.load_data()
x_train = x_train.reshape(x_train.shape[0], 28, 28, 1)
x_test = x_test.reshape(x_test.shape[0], 28, 28, 1)
input_shape = (28, 28, 1)
x_train = x_train.astype('float32')
x_test = x_test.astype('float32')
x_train /= 255
x_test /= 255

# Convert to binary class matrix
y_train = keras.utils.to_categorical(y_train, 10, dtype='float32')
y_test = keras.utils.to_categorical(y_test, 10, dtype='float32')

# Get and compile
model = mnist_conv(input_shape)
model.load_weights('model.h5')
model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
#model.fit(x_train, y_train, epochs=4, batch_size=512)

# Save model
#model.save_weights('model.h5')

test = model.evaluate(x_test, y_test)
print(test)
