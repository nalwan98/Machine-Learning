import numpy as np
import idx2numpy
import pandas as pd
import cv2
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split


#Linear SVM algorothm
def LINEARSVM(data80,data20,target80,target20,converge, plot):
    learning=0.001 #sent n0
    t=0.25 # aconstant used to update learning
    w=np.random.randn(data80.shape[1],1) #randomly choose x0 (which is w0 in this setup)
    ones = np.ones((data80.shape[1],1)) #create the 1 vector
    for iter in range(1,converge): #iterating until it converges
        currentError=0
        for j in range(data80[30,:].shape[0]): #sampling the data80 since it is too large
            tar=target80[j].reshape(1,1)
            x=data80[j].reshape(1,data80[j].shape[0])
            #compute the gradient with linear kernel
            grad = tar*(x.dot(x.T)).dot(w.T) - ones.T
            grad = grad.T
            #project the gradient to the hyperplane yw = 0:
            grad = grad - ((tar*(grad))/np.linalg.norm(tar)).dot(tar)
            # projected gradient descent:
            w = w - learning * grad
            #limit a^(n+1) to [0,C]
            count = 0
            for i in w:
                if i < 0:
                    w[count] = 0
                elif i > 5:
                    w[count] = 5
            count = count + 1

        learning = learning / pow(iter, t)
    return w

def main():
    #load data
    f4 = './train-images.idx3-ubyte'
    f5 = './train-labels.idx1-ubyte'
    f6 = './t10k-images.idx3-ubyte'
    f7 = './t10k-labels.idx1-ubyte'
    images= idx2numpy.convert_from_file(f4)
    labels = idx2numpy.convert_from_file(f5)
    test_train = idx2numpy.convert_from_file(f6)
    test_label = idx2numpy.convert_from_file(f7)
    test_train = test_train.reshape(test_train.shape[0], 28*28)
    images= images[np.where(np.logical_or(labels == 5 , labels ==8))]
    images = images.reshape(images.shape[0], 28*28)
    labels = labels[np.where(np.logical_or(labels == 5 , labels ==8))]
    data80,  data20, target80, target20 = train_test_split(images, labels, test_size = 0.2, random_state=5)
    #print (data20.shape)
    #print (target20.shape)
    print("running svm using the linear kernel")
    LINEARSVM(data80,data20,target80,target20,100, True)
    #RBFSVM(data80,data20,target80,target20,100, True)


if __name__ == "__main__":
    # Run the module with command-line parameters.
    main()
