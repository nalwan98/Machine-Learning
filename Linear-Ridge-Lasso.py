import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import load_boston
from sklearn.model_selection import train_test_split
import random



#Linear regression algorothm
def LINEAR(data80,data20,target80,target20,converge, lam, plot):
    learning=0.001 #set n0
    t=0.25 # a constant used to update learning
    w=np.random.randn(13,1) #randomly choose x0 (which is w0 in this setup)
    for iter in range(1,converge): #iterating until it converges
        currentError=0

        for i in range(data80.shape[0]):
            k=np.random.randint(1,data80.shape[0]) #choose random k
            #set data according to k
            x=data80[k,:]
            x=x.reshape(1,data80.shape[1])
            #set price according to k
            tar_price=target80[k]
            tar_price=tar_price.reshape(1,1)
            #calculate prediction according to current w (y= w^Tx)
            pred=np.dot(x,w)
            #cacluclate the current error depending on the linear regression error equation
            currentError = currentError + pow((pred - tar_price),2)
            # update w
            w=w-(2/data80.shape[0])*learning*(x.T.dot(pred - tar_price)+ 1/data80.shape[0] *(np.dot(w,lam)))
            #update learning
        learning = learning / pow(iter, t)
        Meanerror = currentError/data80.shape[0]
    #create the predicted values for the test data and plot them with comparision to the real price
    predicted=[]
    for x in data20:
        predicted.append(np.dot(x,w))
    pred=np.array(predicted)

    if plot:
        if lam == 0:
            print("plotting the linear regression algorithm actual prices vs predicted prices")
            plt.scatter(pred,target20)
            plt.xlabel("predicted")
            plt.ylabel("actual")
            plt.savefig('./Linear_actaul_vs_predicted_pricingPlot.jpg')
            plt.grid()
            plt.show()
        else:
            print("plotting the ridge regression algorithm actual prices vs predicted prices")
            plt.scatter(pred,target20)
            plt.xlabel("predicted")
            plt.ylabel("actual")
            plt.savefig('./Ridge_actaul_vs_predicted_pricingPlot.jpg')
            plt.grid()
            plt.show()

    return w, Meanerror


# helper function to find best lambda to use for ridge and LISSO
def FindLamda(data80, data20, target80, target20, converge):

    bestMSE= 100000000
    lamdavalues=[l*0.05 for l in range(1,50)]
    Lambda = 0
    for l in lamdavalues:
        W , Meanerror=LINEAR(data80, data20, target80 ,target20, 100, l, False)
        if Meanerror < bestMSE:
            bestMSE= Meanerror
            Lambda=l
    return(Lambda)

#regression regression algorothm with the help of linear algorithm and FindLamda
def RIGGRESSIONSGD(data80, data20, target80, target20, converge):
    lambd = FindLamda(data80, data20, target80, target20, converge)
    w, Meanerror = LINEAR(data80, data20, target80, target20, converge, lambd, True)
    return w, Meanerror

#helper function for LISSO sign_w
def clip_loops(A):
  for i in range(A.shape[0]):
    for j in range(A.shape[1]):
      if (A[i,j] > 0):
        A[i,j] = 1
      elif (A[i,j] < 0):
        A[i,j] = -1
    #else:
    #    A[i,j] = 0
  return A

#LASSO regression algorothm
def LASSoSGD(data, data80, data20, target80, target20, converge, plot):
    lambd = FindLamda(data80, data20, target80, target20, converge)
    learning=0.001
    t=0.25
    w=np.random.randn(13,1) #randomly choose x0 (which is w0 in this setup)
    for iter in range(1,converge): #iterating until it converges
        N=data80.shape[0]
        currentError=0

        for i in range(data80.shape[0]):
            k=np.random.randint(1,data80.shape[0]) #choose random k
            #set data according to k
            x=data80[k,:]
            x=x.reshape(1,data80.shape[1])
            #set price according to k
            tar_price=target80[k]
            tar_price=tar_price.reshape(1,1)
            #find sign_w for lasso sgn(w)
            sign_w = clip_loops(w)
            #calculate prediction according to current w (y= w^Tx)
            pred=np.dot(x,w)
            currentError = currentError + pow((pred - tar_price),2) + lambd* np.linalg.norm(w, ord=1)
          #update w
            w=w-(learning * (x.dot(x.T).dot(w.T).T - pred.dot(x).T + (lambd* sign_w)))
        #update learning
        learning = learning / pow(iter, t)
        Meanerror = currentError/data80.shape[0]
       #create the predicted values for the test data and plot them with comparision to the real price
    predicted=[]
    for x in data20:
        predicted.append(np.dot(x,w))
    pred=np.array(predicted)

    if plot:
        print("plotting the LISSO regression algorithm actual prices vs predicted prices")
        plt.scatter(pred,target20)
        plt.xlabel("predicted")
        plt.ylabel("actual")
        plt.grid()
        plt.savefig('./LISSO_actaul_vs_predicted_pricingPlot.jpg')
        plt.show()
        plt.clf()
    return w, Meanerror

def main():
    #load data
    print("loading the data")
    data, target = load_boston(return_X_y=True)
    data80,  data20, target80, target20 = train_test_split(data, target, test_size = 0.2, random_state=5)
    # run the algorithms and print the differnt weights we get
    #run Linear on 50 iteration converge
    print("running the linear regression algorithm")
    w = LINEAR(data80,data20,target80,target20,100, 0, True)
    print("printing the weight vector of linear regression")
    print(w)
    #run ridge on 100 iteration converge
    print("running the ridge regression algorithm")
    w = RIGGRESSIONSGD(data80, data20, target80, target20, 100)
    print("printing the weight vector of ridge regression")
    print(w)
    #run LISSO on 100n iteration converge
    print("running the LISSO regression algorithm")
    w = LASSoSGD(data, data80, data20, target, target20, 100, True)
    print("printing the weight vector of LISSO regression")
    print(w)

if __name__ == "__main__":
    # Run the module with command-line parameters.
    main()
