from keras.models import Sequential
from keras.layers.convolutional import Conv2D
from keras.layers.convolutional import MaxPooling2D
from keras.layers.core import Activation
from keras.layers.core import Flatten
from keras.layers.core import Dense
from keras import backend as K

class LeNet:
    @staticmethod
    def build(width, height, depth, classes):
        # initialize the model
        model = Sequential()
        inputShape = (height, width, depth)

        # if we are using "channels first", update the input shape
        if K.image_data_format() == "channels_first":
            inputShape = (depth, height, width)
        
        # first set of CONV => RELU => POOL layers
            # input image: 28 x 28 x 1
            # CONV 28x28x20 | 5x5;K = 20
            # ACT  28x28x20
            # POOL 14x14x20 | 2x2
        model.add(Conv2D(20, (5, 5), padding="same", 
            input_shape=inputShape))
        model.add(Activation("relu"))
        model.add(MaxPooling2D(pool_size=(2, 2), strides=(2, 2)))

        # second set of CONV => RELU => POOL layers
            # CONV 14x14x50 | 5x5;K = 20
            # ACT  14x14x50
            # POOL 7x7x50   | 2x2
        model.add(Conv2D(50, (5, 5), padding="same", 
            input_shape=inputShape))
        model.add(Activation("relu"))
        model.add(MaxPooling2D(pool_size=(2, 2), strides=(2, 2)))

        # first (and only) set of FC => RELU layers
            # FC  500
            # ACT 500
        model.add(Flatten())
        model.add(Dense(500))
        model.add(Activation("relu"))

        # softmax classifier
            # FC 10
            # softmax 10
        model.add(Dense(classes))
        model.add(Activation("softmax"))

        # return the constructed network architecture
        return model