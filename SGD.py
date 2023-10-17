import numpy as np
import idx2numpy


#running the main
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
    #make data of shape 784 per image
    test_train = test_train.reshape(test_train.shape[0], 28*28)
    images = images.reshape(images.shape[0], 28*28)
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
    #set number of layer and number of nodes in each layers
    layers = [2, 2, 3, 3, 4]
    NperL = [[784, 600, 10], [784, 420,  10], [784, 128, 64, 10], [784, 500, 250, 10], [784, 500, 250, 100, 10]]
    count = 0
    for l in layers:
        print("This trial is with", layers[count], "layers and ", NperL[count], "number of nodes per layer")
        network = network_initial(layers[count], NperL[count])
    #train
        weight = optimization(images, labels, NperL[count], layers[count], network)

        predictions = []
    #testing and calculating the the percentage of true classifications
        for x, y in zip(test_train, test_label):
            outputarr, z = forwardPass(x, weight, layers[count])
            predected_value = outputarr[len(outputarr)-1]
            predected_value  = np.argmax(predected_value )
            y = np.argmax(y)
            predictions.append(predected_value  == y)

        print("The accuracy of the predictions is:", np.mean(predictions)*100, "%")
        count = count + 1
        print("-------------------------------------------")
        print("-------------------------------------------")

#randomely initialize the weight of the network
def network_initial(layers, NperL):
    network = []
    count = 0
    for l in range(layers):
        network.append(np.random.randn(NperL[count+1],NperL[count]))
        count = count + 1
    return network

#forward pass following the forward pass algorithm
def forwardPass(x_train, weight, layers):
    #set matrices z_i and a_i
    Z = []
    A = []
    count = 0
    A.append(x_train)
    #for the hidden layers with activation relu
    for l in range(layers - 1):
        Z.append(np.dot(weight[count], A[count]))
        A.append(np.maximum(Z[count],0))
        count = count + 1
    #for the output layer with activation softmax
    Z.append(np.dot(weight[count], A[count]))
    A.append(softmax(Z[count]))
    return A,Z

#backward pass using the backward pass algorithm
def backwardPass(y_train, output, A, Z, weight, layers):
    derivativeW = []
    #create [y_i,y_2,....y_r-1,...y_n]
    output[np.argmax(y_train)] = output[np.argmax(y_train)] - 1
    # set error e(l)
    error = output
    derivativeW.append(np.outer(error, A[len(A)-2]))
    count = layers -1
    count1 =  len(Z)- 1 -1
    count3 = len(A)-3
    #set error and weight of hidden layers
    for l in range(layers -1):
        error = np.dot(weight[count].T, error).T * np.where(Z[count1] > 0, 1, 0)
        derivativeW.append(np.outer(error, A[(count3)]))
        count = count - 1
        count1 = count1 - 1
        count3 = count3-1
    #reverse the derivativeW array so that it is W1, w2, w3, since we created it as w3, w2, w1
    derivativeW = derivativeW[::-1]
    return derivativeW


#training using SGD
def optimization(images, labels, NperL, layers, weight, learning=0.001, conversion=10):
    minibatch_size = 100
    #weight = np.array(weight)
    t=0.25
    for iter in range(1, conversion):
        # shuffle
        images, labels = shuffle(images, labels)
        for i in range(0, images.shape[0], minibatch_size):
        # minibatch/chunk
            X_mini = images[i:i + minibatch_size]
            y_mini = labels[i:i + minibatch_size]
            #loop over the mini batch
            for x,y in zip(X_mini, y_mini):
                #forward pass
                A, Z = forwardPass(x, weight, layers)
                #backward pass
                wValues = backwardPass(y, A[len(A) - 1], A, Z, weight, layers)
                wValues= np.array(wValues, dtype=list)
                #update weight
                weight = weight - (learning * wValues)
                #break
        #update learning rate
        learning = learning /pow(iter, t)
        break
    return weight

#softmax
def softmax(x):
    ex = np.exp(x - x.max())
    return ex / np.sum(ex, axis=0)

#shuffling for SGD
def shuffle(a, b):
    shuffled_a = np.empty(a.shape, dtype=a.dtype)
    shuffled_b = np.empty(b.shape, dtype=b.dtype)
    permutation = np.random.permutation(len(a))
    for old_index, new_index in enumerate(permutation):
        shuffled_a[new_index] = a[old_index]
        shuffled_b[new_index] = b[old_index]
    return shuffled_a, shuffled_b

if __name__ == "__main__":
    # Run the module with command-line parameters.
    main()
