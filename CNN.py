from keras.layers import Conv2D
from keras.layers import MaxPooling2D
from keras.layers import Dense
from keras.layers import Flatten
from keras.models import Sequential
from keras.optimizers import SGD
import numpy as np
import idx2numpy

def main():
    #load data
    print("loading the data")
    f4 = './train-images.idx3-ubyte'
    f5 = './train-labels.idx1-ubyte'
    f6 = './t10k-images.idx3-ubyte'
    f7 = './t10k-labels.idx1-ubyte'
    images= idx2numpy.convert_from_file(f4)
    labels = idx2numpy.convert_from_file(f5)
    test_train = idx2numpy.convert_from_file(f6)
    test_label = idx2numpy.convert_from_file(f7)
    #make data of shape 28*28*1 per image
    test_train = test_train.reshape(test_train.shape[0], 28, 28, 1)
    images = images.reshape(images.shape[0], 28, 28, 1)
    images = (images/255).astype('float32')
    #make labels in the form of eg: [0,0,0,0,1,0,0,0,0,0]
    temp = np.zeros(shape= (60000, 10))
    for l in range(60000):
        temp[l, labels[l]] = 1
    labels = temp
    temp1 = np.zeros(shape= (10000, 10))
    for l in range(10000):
        temp1[l, test_label[l]] = 1
    test_label = temp1
    #train the minist data
    accuracy = training_model(images, labels, test_train, test_label)

#creating the cnn model provided in the QUESTION 2b:
def create_model():
    print("Creating the model")
    model = Sequential()
    model.add(Conv2D(32, (3, 3), activation='relu', kernel_initializer='he_uniform', input_shape=(28, 28, 1)))
    model.add(Conv2D(64, (3, 3), activation="relu", kernel_initializer="he_uniform", input_shape=(3, 3, 32)))
    model.add(Conv2D(64, (3, 3), activation='relu', kernel_initializer='he_uniform', input_shape=(3, 3, 64)))
    model.add(MaxPooling2D((2, 2)))
    model.add(Flatten())
    model.add(Dense(7744, activation='relu', kernel_initializer='he_uniform'))
    model.add(Dense(128, activation='relu', kernel_initializer='he_uniform'))
    model.add(Dense(10, activation='softmax'))
    opt = SGD(lr=0.01, momentum=0.9)
    model.compile(optimizer=opt, loss='categorical_crossentropy', metrics=['accuracy'])
    return model

#train the data of minist
def training_model(dataX, dataY, testx, testy):
    # define model according to network provided in question 2b
    model = create_model()
    print("Training the training data...")
    training = model.fit(dataX, dataY, epochs=4, validation_data=(testx, testy))

    # test and calculate the percentage of correct predictions
    predictions = []
    print("Testing the testing data...")
    predections = model.predict_on_batch(testx)
    print("Comparing the output and the real test labels...")
    for x, y in zip(predections, testy):
        predictions.append(x == y)
    print ("the percentage of correct predictions on the test MNIST Data: ", (np.mean(predictions) * 100), "%")


if __name__ == "__main__":
    # Run the module with command-line parameters.
    main()
