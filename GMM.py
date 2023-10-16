import pandas as pd
import numpy as np
import idx2numpy
import copy


# calculating the multivariant pdf
def multivariate_normal(x, size, mean, covariance):
    x_m = x - mean
    return  (1. / (np.sqrt(abs((2 * np.pi)**size *np.linalg.det(covariance)))) *
            np.exp(-(x_m @np.linalg.inv(covariance)@x_m.T) / 2))

#testing the accuracy when the covaraince structure is full
def testing_accuracy_full(pr_A, pr_B,test_data, test_labels,Amue, Acov, Aweight, Bmue, Bcov, Bweight):
    print("Testing the testing data and finding the accuracy when using the full structure \n")
    predictions =[]
    sumA = 0
    sumB = 0
    for iter in test_data:
        sumA = 0
        sumB = 0
        for l in range(len(Amue)):
            tempA= Aweight[l] * multivariate_normal(iter, iter.shape[0], Amue[l], Acov[l])
            #print("ttt", Amue[l].shape, Acov[l].shape)
            tempB= Bweight[l] * multivariate_normal(iter, iter.shape[0], Bmue[l], Bcov[l])
        sumA = sumA + tempA
        sumB = sumB + tempB
        #print(sumA, sumB)
        if (sumA * pr_A > (sumB* pr_B)):
            predictions.append('A')
        else:
            predictions.append('B')
    trueFalse = []
    for x, y in zip(predictions, test_labels):
        #print(x,y)
        trueFalse.append(x == y)
    print ("The percentage of correct predictions on the test Data when using the full covariance structure: ", (np.mean(trueFalse) * 100), "%")

#testing the accuracy when the covaraince structure is diagonal
def testing_accuracy_diag(pr_A, pr_B,test_data, test_labels,Amue, Acov, Aweight, Bmue, Bcov, Bweight):
    print("Testing the testing data and finding the accuracy when using the diag structure \n")
    #print(Acov)
    for a  in range(len(Acov)):
        Acov[a] = np.diag(np.diag(Acov[a]))
    for a  in range(len(Bcov)):
        Bcov[a] = np.diag(np.diag(Bcov[a]))
    predictions =[]
    sumA = 0
    sumB = 0
    for iter in test_data:
        sumA = 0
        sumB = 0
        for l in range(len(Amue)):
            tempA= Aweight[l] * multivariate_normal(iter, iter.shape[0], Amue[l], Acov[l])
            #print("ttt", Amue[l].shape, Acov[l].shape)
            tempB= Bweight[l] * multivariate_normal(iter, iter.shape[0], Bmue[l], Bcov[l])
        sumA = sumA + tempA
        sumB = sumB + tempB
        #print(sumA, sumB)
        if (sumA * pr_A > (sumB* pr_B)):
            predictions.append('A')
        else:
            predictions.append('B')
    trueFalse = []
    for x, y in zip(predictions, test_labels):
        #print(x,y)
        trueFalse.append(x == y)
    print ("The percentage of correct predictions on the test Data when using the diag covariance structure: ", (np.mean(trueFalse) * 100), "%")

#top down k means algorithm to use to initialize the GMM
def topDownKmeans(train_data, m):
    k = 1
    cluster= []
    cl1 = []
    cluster.append(cl1)
    centroid = []
    centroid.append(train_data[5])
    count = 0
    while k< m:
        while True:
            old_cluster=copy.deepcopy(cluster)
            cluster=[]
            for l in range(len(old_cluster)):
                cl11 = []
                cluster.append(cl11)
            for xi in train_data:
                temp = []
                temp1 = []
                for cent in centroid:
                    temp.append(np.linalg.norm(xi-cent))
                    temp1.append(cent)
                zipped = zip(temp, temp1)
                zipped = list(zipped)
                sort= sorted(zipped, key = lambda x: x[0])
                #print(np.where(centroid == sort[0][1]))
                num = 1
                for index,value in enumerate(centroid):
                     if (value == sort[0][1]).all():
                         num = index
                         break
                cluster[num].append(xi)
            for i,item in enumerate(centroid):
                centroid[i]= np.mean(cluster[i], axis=0,  dtype=float)
            count= count + 1
            bol= True
            for x,y in zip(old_cluster, cluster):
                if not np.array_equal(x, y):
                    bol = False
                    break
            if (bol):
                break
        ind =max(cluster, key=func)
        #print(ind)
        for index,value in enumerate(cluster):
             if (np.array_equal(value,ind)):
                 num = index
                 break
        cluster.append(cluster[num][len(cluster[num])//2:])
        cluster[num]= (cluster[num][:len(cluster[num])//2])
        centroid.append(np.mean(cluster[len(cluster)-1], axis=0,  dtype=float))
        centroid[num] = np.mean(cluster[num], axis=0,  dtype=float)
        k=k+1
    return centroid, cluster

    #for k in range(m):

#call the kmean to initialize the GMM and use EM algorithm to improve it
def Gmm(train_data, train_labels, test_data, test_labels, m):
    #calling the kmeans algorithm
    print("Initializing the GMMs")
    centroidA, clusterA = topDownKmeans(train_data[train_labels == 'A'], m)
    centroidB, clusterB = topDownKmeans(train_data[train_labels == 'B'], m)
    covarianceA =[]
    covarianceB =[]
    # creating the covariance according to the results of the kmeans
    for l in range(len(clusterA)):
        covarianceA.append(1/len(clusterA[l])*((clusterA[l] - centroidA[l]).T.dot((clusterA[l] - centroidA[l]))))
    for l in range(len(clusterB)):
        covarianceB.append(1/len(clusterB[l])*((clusterB[l] - centroidB[l]).T.dot((clusterB[l] - centroidB[l]))))
    #setting the weights of the gmm randomly, such that they add up to 1
    weightsA = np.random.dirichlet(np.ones(len(clusterA)),size=1)[0]

    weightsB = np.random.dirichlet(np.ones(len(clusterB)),size=1)[0]
    print("training and improving the GMMs")
    #calling the EM algorithm to fix the GMMs
    for l in range(2):
        Q, estep_result = Estep(weightsA, clusterA, covarianceA, centroidA, train_data[train_labels == 'A'])
        weightsA, centroidA, covarianceA, = Mstep(estep_result, weightsA, clusterA, covarianceA, centroidA, train_data[train_labels == 'A'])
    for l in range(2):
        Q, estep_result = Estep(weightsB, clusterB, covarianceB, centroidB, train_data[train_labels == 'B'])
        weightsB, centroidB, covarianceB, = Mstep(estep_result, weightsB, clusterB, covarianceB, centroidB, train_data[train_labels == 'B'])
    #caculate the percentage of A and in B in the data
    pr_A = train_labels[train_labels=='A'].shape[0]/train_labels.shape[0]
    pr_B = 1.0 - pr_A
    #calling the the testing the testing data
    testing_accuracy_full(pr_A, pr_B, test_data, test_labels,centroidA, covarianceA, weightsA, centroidB, covarianceB, weightsB)
    testing_accuracy_diag(pr_A, pr_B, test_data, test_labels,centroidA, covarianceA, weightsA, centroidB, covarianceB, weightsB)

#the Estep that is being called in the def GMM
def Estep(weight, cluster, covariance, mue, data):
    pr = np.zeros((data.shape[0], len(cluster)))
    for c in range(len(cluster)):
        #12.7 in the book
        pr[:,c] = np.sum(weight[c] * multivariate_normal(data, data.shape[1], mue[c], covariance[c]), axis=1)
        temp = np.sum(pr, axis=1)[:,np.newaxis]
        pr = pr/temp
        #12.8 in the book
        summ= np.sum((((data - mue[c]) @ np.linalg.inv(covariance[c])@ (data - mue[c]).T)/2), axis=1)
        summ= summ.reshape(summ.shape[0], 1)
        log= (np.log(np.diagonal(covariance[c]))/2)
        log = log.reshape(log.shape[0], 1).T
        lnw = np.log(weight[c]) - log - summ
        Q = lnw.T @ pr
    return Q, pr

#the Mstep that is being called in the def GMM
def Mstep(estep_result, weight, cluster, covariance, mue, data):
    #w_m_n+1
    new_weight = np.mean(estep_result, axis = 0)
    new_cov = np.zeros((estep_result.shape[1] , data.shape[1] , data.shape[1]))
     #mu_m_n+1
    temp = np.zeros((np.sum(estep_result, axis = 0).shape[0], 1))
    for c in range(np.sum(estep_result, axis = 0).shape[0]):
        temp[c, :] = np.sum(estep_result, axis = 0)[c]

    new_mue = np.dot(estep_result.T, data) / temp
    for l in range(len(cluster)):
        temp_cov = np.matrix(np.diag(estep_result[:,l])) #covaraince
        #sigma_m_n+1
        new_cov[l,:,:]=((data - new_mue[l, :]).T * temp_cov * (data - new_mue[l, :])) / temp[l]
    return new_weight, new_mue, new_cov

#helper function
def func(p):
    return len(p)

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
  # calling the GMM to train and test
  print("Trial: number of gussians component = 2")
  Gmm(train_data, train_labels, test_data, test_labels, 2)
  print("\n ------------------------\n")
  print("Trial: number of gussians component = 4")
  Gmm(train_data, train_labels, test_data, test_labels, 4)
  print("\n ------------------------\n")
  print("Trial: number of gussians component = 8")
  Gmm(train_data, train_labels, test_data, test_labels, 8)
  print("\n ------------------------\n")
  print("Trial: number of gussians component = 16")
  Gmm(train_data, train_labels, test_data, test_labels, 16)

if __name__ == "__main__":
    # Run the module with command-line parameters.
    main()
