import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.decomposition import TruncatedSVD
from sklearn.utils.extmath import randomized_svd
from time import time
from scipy.sparse import csr_matrix
from sklearn.manifold import TSNE
import csv
import matplotlib.pyplot as plt
import math
import re

#cosine distance calculation
def cosine_distance(x, y):
    #using the formula in the book in Q 2.5 and using the definiton of norm
    return np.dot(x, y) / (np.sqrt(np.dot(x, x)) * np.sqrt(np.dot(y, y)))

# pearson correlation coefficient calculation
def pearson_corr(x,y):
    #using the formula in the book in page 79
    return np.clip(np.cov(x,y)/(np.std(x) * (np.std(y))), 0, 1)

# tsne representation using the Tsne library from sklearn
def tsne(model, labels, type, k):
    tsne_lsa_model = TSNE(n_components=2, perplexity=30,learning_rate=100, n_iter=500, verbose=1, random_state=0, angle=0.75)
    tsne_lsa_vectors = tsne_lsa_model.fit_transform(model)
    x = []
    y = []
    for value in tsne_lsa_vectors:
        x.append(value[0])
        y.append(value[1])

    for i in range(len(x)):
        plt.scatter(x[i],y[i])
        plt.annotate(labels[i],xy=(x[i], y[i]),xytext=(5, 2),textcoords='offset points',ha='right',va='bottom')
    #plt.figure(figsize=(20,10))
    plt.savefig('./tsne_' + type +'_'+ str(k) + '.png')
    plt.title("tsne_" + type +'_'+ str(k))
    plt.show()
    plt.clf()
    plt.close('all')

# iterating through non zero item of a csr matrix in scipy (inspired by https://stackoverflow.com/a/37598835)
def nonzerosEntries(matrix, row):
    for i in range(matrix.indptr[row], matrix.indptr[row + 1]):
        yield matrix.indices[i], matrix.data[i]

# Standard SVD which resulted in failure due to consuming too much memory and time
def svd(X):
    #start timer
    t0 = time()
    # run standard SVD
    U, Sigma, VT = np.linalg.svd(X, full_matrices=True, compute_uv=True)
    # calculate time and print it on terminal
    time_used = (time()-t0)/60.
    print("Took {0:.3f} minutes when using the standard SVD for Matrix Factorization.".format(time_used), flush=True)
    return(U,VT)

# Truncated SVD with randomized algorithm
def svd_successful(X,k):
    #start timer
    t0 = time()
    #truncate SVD and fit the matrix
    SVD = TruncatedSVD(n_components=k, algorithm='randomized', n_iter=10, random_state=122)
    SVD.fit(X)
    # calculate time and print it on terminal
    time_used = (time()-t0)/60.
    print("Took {0:.3f} minutes when using the truncated SVD with randomized algorithm for Matrix Factorization.".format(time_used), flush=True)
    return(SVD)

def alternatingAlgorithm(mat, k, lamda, iterations=15):
    #start timer
    t0 = time()
    # randomly initialize U and V
    U = np.random.rand(mat.shape[0], k) * 0.01
    V = np.random.rand(mat.shape[1], k) * 0.01
    # iterate and update U and V according to alternating algorithm unter converged
    for iteration in range(iterations):
        updateuv(mat, U, V, lamda)
        updateuv(mat.T.tocsr(), V, U, lamda)
    # calculate time and print it on terminal
    time_used = (time()-t0)/60.
    print("Took {0:.3f} minutes when using the Alternating Algorithm for Matrix Factorization.".format(time_used), flush=True)
    # return factorization
    return U, V

def updateuv(mat, X, Y, lamda):
    sumVVt = Y.T.dot(Y)
    lamdaI = lamda * np.eye(X.shape[1])
    for u in range(X.shape[0]):
        #temp = sum of vj vjT + lamda I
        temp= sumVVt + lamdaI
        temp1 = np.zeros(X.shape[1])
        for i, data in nonzerosEntries(mat, u):
            #temp1 = sum of (xij vi)
            temp1 += data * Y[i]
        #inverse = (sum of vj vjT + lamda I)^-1
        inverse = np.linalg.inv(temp)
        # X[u] = (sum of vj vjT + lamda I)^-1 (sum of (xij vi))
        X[u] = inverse @ temp1

def pmi(d):
    #start timer
    t0 = time()
    # calculate (sum (sum(fij)) which the denominator of pi and pj
    sum = np.sum(d)
    #initialize Z which is the updated pmi matrix
    Z = np.zeros(d.shape)
    #using np non zero to extract all the non zero elements in the matrix
    m,n = np.nonzero(d)
    print("number of points to update using pmi = ", np.nonzero(d)[0].shape[0])
    #loop through all non zerp elements in the matrix and update them
    for i in range(np.nonzero(d)[0].shape[0]):
        if d[m[i]][n[i]] != 0:
            print(i)
            #calculate pij according to fij/ (sum (sum(fij))
            pij= d[m[i]][n[i]]/sum
            sumi=0
            sumj=0
            sumi = np.sum(d[m[i]])
            #calculate pi according to sum(all j fij)/ (sum (sum(fij))
            pi= sumi/sum
            sumj = np.sum(d[:,n[i]])
            #calculate pj according to sum(all i fij)/ (sum (sum(fij))
            pj= sumj/sum
            #calculate pmi according to log(pij/pi*pj)
            pmi = math.log(pij/(pi*pj),2)
            #assigning values to Z(the new matrix)
            if pmi > 0 :
                Z[m[i]][n[i]]= pmi
    # calculate time and print it on terminal
    time_used = (time()-t0)/60.
    print("Took {0:.3f} minutes to apply pmi to the matrix.".format(time_used), flush=True)
    return Z

#running the main
def main():
    #loading the data
    plt.ioff()
    text_file = open("enwiki8.txt", "r")
    docs = text_file.read().splitlines()
    text_file.close()
    columns = ['list1','list2','list3']
    lines = pd.read_csv("wordsim353_human_scores.csv", names=columns)
    list1 = lines.list1.tolist()
    list2 = lines.list2.tolist()
    list3 = lines.list3.tolist()
    #text_file2 = open("wordsim353_human_scores.txt", "r")
    with open("wordsim353_human_scores.txt")as f:
        wordsim = list(dict.fromkeys((re.findall("[a-zA-Z\-'/]+", f.read()))))
    vec = CountVectorizer(max_features= 10000)
    X = vec.fit_transform(docs[0:5000])
    voc = list(dict.fromkeys(wordsim[30:len(wordsim)-1] + vec.get_feature_names()))
    vec_main = CountVectorizer(vocabulary = voc)
    Y = vec_main.fit_transform(docs)
    df = pd.DataFrame(Y.toarray(), columns=vec_main.get_feature_names())
    print(df)
    data1 = df.to_numpy(dtype=int)

    # Running Standard SVD which resulted in a space Error
    #####print("Standard SVD ")
    #####U,V = svd(data1)
    #--------------------------------------------------------
    # Running SVD with randomized Algorithm with k = 20
    print("\n\nSVD with randomized Algorithm with k = 20")
    Svd20 = svd_successful(data1[:, 0: 1000], 20)

    # Running SVD with randomized Algorithm with k = 50
    print("\n\nSVD with randomized Algorithm with k = 50")
    Svd50 = svd_successful(data1[:, 0: 1000], 50)

    # Running SVD with randomized Algorithm with k = 100
    print("\n\nSVD with randomized Algorithm with k = 100")
    Svd100 = svd_successful(data1[:, 0: 1000], 100)


    data = csr_matrix(data1)
    kv=[20,50,100]
    for k in kv:
        list4 = np.copy(list3)
        print("\n\nmatrix factorization using alternating algorithm for k= ", k)
        U,V =alternatingAlgorithm(data, k, 1.5, iterations=4)
        print("\n\nFinding wordsim pairs and calculating the cosine distances between the pairs")
        t0 = time()
        distances = []
        count = 0
        for i in range(len(list1)-1):
            v1 = -1
            v2 = -1
            for j in range(700):
                if vec_main.get_feature_names()[j] == list1[i]:
                    v1 = j
                if vec_main.get_feature_names()[j] == list2[i]:
                    v2 = j
                if v1 != -1 and v2 != -1:
                    break
            if V[v1].dot(V[v1]) !=0 and V[v2].dot(V[v2]) !=0 :
                distances.append(cosine_distance(V[v1], V[v2]))
            else:
                count = count + 1
                list4 = np.delete(list4, i-count)
        time_used = (time()-t0)/60.
        print("Took {0:.3f} minutes to find the wordsim pairs and calculating the cosine distances between the pairs.".format(time_used), flush=True)
        print("\n\ncalculating Pearson’s correlation coefficient between these cosine distances and human scores")
        print(pearson_corr(distances, list4[0: len(list4)-1]))

        rowsum = np.sum(data1, axis=0)
        word300 = np.argsort(rowsum)[0:300]
        print("\n\ntsne for top 300 words for alternating matrix factorization with k = ", k)
        tsne(V[word300], np.array(vec_main.get_feature_names())[word300], "normal_factorization", k)
    #---------------------------------------------------------

    # pmi running step 2 till 5
    pmi_V = pmi(data1[0:150000,0:3000])

    # Running SVD with randomized Algorithm with k = 20
    print("\n\nSVD with randomized Algorithm with k = 20")
    Svd20 = svd_successful(pmi_V, 20)

    # Running SVD with randomized Algorithm with k = 50
    print("\n\nSVD with randomized Algorithm with k = 50")
    Svd50 = svd_successful(pmi_V, 50)

    # Running SVD with randomized Algorithm with k = 100
    print("\n\nSVD with randomized Algorithm with k = 100")
    Svd100 = svd_successful(pmi_V, 100)

    V = []
    kv=[20, 50, 100]
    for k in kv:
        list4 = np.copy(list3)
        print("\n\nmatrix factorization using alternating algorithm with pmi for k= ", k)
        U,V =alternatingAlgorithm(csr_matrix(pmi_V), k, 1.5, iterations=4)
        print("\n\nFinding wordsim pairs and calculating the cosine distances between the pairs")
        #start timer
        t0 = time()
        distances = []
        count = 0
        for i in range(len(list1)-1):
            v1 = -1
            v2 = -1
            for j in range(700):
                if vec_main.get_feature_names()[j] == list1[i]:
                    v1 = j
                if vec_main.get_feature_names()[j] == list2[i]:
                    v2 = j
                if v1 != -1 and v2 != -1:
                    break
            if V[v1].dot(V[v1]) !=0 and V[v2].dot(V[v2]) !=0 :
                distances.append(cosine_distance(V[v1], V[v2]))
            else:
                count = count + 1
                list4 = np.delete(list4, i-count)
        time_used = (time()-t0)/60.
        print("Took {0:.3f} minutes to find the wordsim pairs and calculating the cosine distances between the pairs.".format(time_used), flush=True)
        print("\n\ncalculating Pearson’s correlation coefficient between these cosine distances and human scores")
        print(pearson_corr(distances, list4[0: len(list4)-1]))
        rowsum = np.sum(pmi_V, axis=0)
        word300 = np.argsort(rowsum)[0:300]
        print("\n\ntsne for top 300 words for alternating pmi matrix factorization with k = ", k)
        tsne(V[word300], np.array(vec_main.get_feature_names())[word300], "pmi_matrix_factorization", k)
    #--------------------------------------------------------


if __name__ == "__main__":
    # Run the module with command-line parameters.
    main()
