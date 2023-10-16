import pandas as pd
import numpy as np
import idx2numpy

# calculating the multivariant pdf
def multivariate_normal(x, size, mean, covariance):
    x_m = x - mean
    return (1. / (np.sqrt(abs((2 * np.pi)**size * np.linalg.det(covariance)))) *
            np.exp(-(np.linalg.solve(covariance, x_m).T.dot(x_m)) / 2))

#trainig the data
def train_classifier(train_data, train_labels):
    print("Training the training data")
    Amue = np.mean(train_data[train_labels == 'A'], axis=0)
    Bmue = np.mean(train_data[train_labels == 'B'], axis=0)
    first = train_data[train_labels == 'A']
    second = train_data[train_labels == 'B']
    Acov = 1/first.shape[0]*((first - Amue).T.dot((first - Amue)))
    tempAdiagnal = []
    for l in range(3):
        tempAdiagnal.append(1/first.shape[0]* ((pow((first - Amue),2))).sum(axis=0))
    diagA = np.diag(np.diag(tempAdiagnal))
    print("\n ------------------------------------- \n")
    print("MLE mue vector of class A")
    print(Amue, "\n")
    print("MLE full covariance matrix of class A")
    print(Acov, "\n")
    print("MLE diagnal covariance matrix of class A")
    print(diagA, "\n")
    Bcov = 1/second.shape[0]*((second - Bmue).T.dot((second - Bmue)))
    tempBdiagnal = []
    for l in range(3):
        tempBdiagnal.append(1/second.shape[0]* ((pow((second - Bmue),2))).sum(axis=0))
    diagB = np.diag(np.diag(tempBdiagnal))
    print("\n ------------------------------------- \n")
    print("MLE mue vector of class B")
    print(Bmue, "\n")
    print("MLE full covariance matrix of class B")
    print(Bcov, "\n")
    print("MLE diagnal covariance matrix of class B")
    print(diagB, "\n")
    print("\n ------------------------------------- \n")
    pr_A = train_labels[train_labels=='A'].shape[0]/train_labels.shape[0]
    pr_B = 1.0 - pr_A
    return pr_A, pr_B, Amue, Acov,  Bmue, Bcov, diagA, diagB

#testing the accuracy when the covaraince structure is full
def testing_accuracy_full(test_data, test_labels, pr_A, pr_B, Amue, Acov,  Bmue, Bcov):
    print("Testing the testing data and finding the accuracy when using the full structure \n")
    predictions =[]
    for iter in test_data:
        if (multivariate_normal(iter, iter.shape[0], Amue, Acov)*pr_A) > (multivariate_normal(iter, iter.shape[0], Bmue, Bcov)*pr_B):
            predictions.append('A')
        else:
            predictions.append('B')
    trueFalse = []
    for x, y in zip(predictions, test_labels):
        trueFalse.append(x == y)
    print ("The percentage of correct predictions on the test Data when using the full covariance structure: ", (np.mean(trueFalse) * 100), "%")

#testing the accuracy when the covaraince structure is diagnal
def testing_accuracy_diag(test_data, test_labels, pr_A, pr_B, Amue, Acov,  Bmue, Bcov):
    print("\n ------------------------------------- \n")
    print("Testing the testing data and finding the accuracy when using the diagnal structure \n")
    predictions =[]
    for iter in test_data:
        if (multivariate_normal(iter, iter.shape[0], Amue, Acov)*pr_A) > (multivariate_normal(iter, iter.shape[0], Bmue, Bcov)*pr_B):
            predictions.append('A')
        else:
            predictions.append('B')
    trueFalse = []
    for x, y in zip(predictions, test_labels):
        trueFalse.append(x == y)
    print ("The percentage of correct predictions on the test Data when using the diagnal covariance: ", (np.mean(trueFalse) * 100), "%")

#running the main
def main():
    #loading the data
  train_data = pd.read_csv("train-gaussian.csv")
  test_data = pd.read_csv("test-gaussian.csv")
  train_data = np.array(train_data)
  test_data = np.array(test_data)
   #seperating the labels
  x, y, z, r = train_data.T
  train_labels = x
  m, n, l, o = test_data.T
  test_labels = m
  train_data = np.delete(train_data, 0, axis=1)
  test_data = np.delete(test_data, 0, axis=1)
  train_data = np.array(train_data,dtype='float')
  test_data = np.array(test_data,dtype='float')
  # calling the training function
  pr_A, pr_B, Amue, Acov,  Bmue, Bcov, diagA, diagB = train_classifier(train_data, train_labels)
  #calling the testing function
  testing_accuracy_full(test_data, test_labels, pr_A, pr_B, Amue, Acov,  Bmue, Bcov)
  testing_accuracy_diag(test_data, test_labels, pr_A, pr_B, Amue, diagA,  Bmue, diagB)


if __name__ == "__main__":
    # Run the module with command-line parameters.
    main()
