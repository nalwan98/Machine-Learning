import numpy as np
import idx2numpy
import pandas as pd
import cv2
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from scipy.linalg import eigh
from sklearn.manifold import TSNE
import seaborn as sn

# part a of lab project 1
def Pca(images , m, labels, plot, eighV, example, test):
    standardized_data = StandardScaler().fit_transform(images)
    cov = np.cov(standardized_data.T)
    values, vectors = eigh(cov, eigvals=(784 - m,783))
    values = eigh(cov, eigvals_only = True)
    #assigning the projection matrix
    A = vectors.T
    #  A IS THE PROJECTION MATRIX
    #assigning the new coordinates
    new_coordinates = A @ images.T
    # calculate the distortion and return it
    original = A.T @ new_coordinates
    original = original.T
    total_distortion= 0
    for value in range(original.shape[0]):
        total_distortion = total_distortion + np.linalg.norm(original[value] - images[value])

    # develope the first original image back
    if example == True:
        print("generating original pic after PCA, which is saved in folder ")
        original = original.reshape(original.shape[0], 28, 28)
        cv2.imwrite('./original_pic_PCA_' + str(m) +'_.jpg',original[0])

    # plotting if plot is true and m=2
    if plot == True:
        if m != 2:
            print("The plot needs to be in 2 diementions to be plotted")
        else:
            print("plotting 2 diementions PCA")
            new_coordinateso = np.vstack((new_coordinates, labels)).T
            dataframe = pd.DataFrame(data=new_coordinateso, columns=("1st_principal", "2nd_principal", "label"))
            sn.FacetGrid(dataframe, hue="label", height=6).map(plt.scatter, '1st_principal', '2nd_principal').add_legend()
            plt.title("Pca ")
            if test:
                plt.savefig('./2D_PCA_plot_for_test_data_.jpg')
            else:
                plt.savefig('./2D_PCA_plot_for_478_data_.jpg')
            plt.show()
            plt.clf()
            plt.close()
    # plotting the eigne values and variance vs diementions plot if eighV is true
    if eighV == True:
        #creating the plot for the descending eigenvalues of the covariance matrix
        values = sorted(values, reverse = True)
        print("plotting eigenvalues of the covariance matrix descending plot")
        plt.plot(values)
        plt.title("eigenvalues of the covariance matrix descending")
        plt.ylabel('Eigenvalues')
        plt.grid()
        plt.savefig('./PCA_eigenvalues_plot.jpg')
        plt.show()
        plt.clf()
        #creating the plot that shows the variance vs diementions
        percentage = values/ np.sum(values)
        cumulative = np.cumsum(percentage)
        print("plotting Variance vs Dimensions plot")
        plt.plot(cumulative)
        plt.title("Variance vs Dimensions")
        plt.xlabel("number of dimensions")
        plt.ylabel('percentage of variance kept')
        plt.grid()
        plt.savefig('./PCA_variance_vs_diementions_plot.jpg')
        plt.show()
        plt.clf()

    return total_distortion
# part b of lab project 1
def Lda(images, labels, plot):
    #making classes depending on the numbers 4,7,and 8
    arr4 = images[np.where(labels == 4)]
    arr7 = images[np.where(labels == 7)]
    arr8 = images[np.where(labels == 8)]

    #calculating the mean for each class and for all classes together
    mean4 = np.mean(arr4, axis=0 )
    mean4 = mean4.reshape(mean4.shape[0], 1)
    mean7 = np.mean(arr7, axis=0 )
    mean7 = mean7.reshape(mean7.shape[0], 1)
    mean8 = np.mean(arr8, axis=0 )
    mean8 = mean8.reshape(mean8.shape[0], 1)
    meanfull = np.mean(images, axis=0)
    meanfull = meanfull.reshape(meanfull.shape[0], 1)

    #claculating the covariance for each class
    cov4 = np.cov(arr4.reshape(arr4.shape[1], arr4.shape[0]))
    cov7 = np.cov(arr7.reshape(arr7.shape[1], arr7.shape[0]))
    cov8 = np.cov(arr8.reshape(arr8.shape[1], arr8.shape[0]))

    #calculating S_b and S_w
    s_b = (arr4.shape[0]*np.dot((mean4-meanfull),(mean4-meanfull).T))+(arr7.shape[0]*np.dot((mean7-meanfull),(mean7-meanfull).T))+(arr8.shape[0]*np.dot((mean8-meanfull),(mean8-meanfull).T))
    s_w = cov4 + cov7 + cov8

    #calculating s_w^-1 s_b
    matrix = np.linalg.inv(s_w)@s_b
    #calculating the eigenvalues and eigenvectors
    valuesLDA, vectorsLDA = eigh(matrix, eigvals=(782,783))
    #creating the Projection matrix
    Alda = vectorsLDA.T
    #creating the new coordinates
    new_coordinates_LDA1 = Alda @ images.T
    #regenerating the orignal images
    new_coordinates_LDA = Alda.T @ new_coordinates_LDA1
    new_coordinateso = np.vstack((new_coordinates_LDA1, labels)).T
    print("plotting 2 diementions LDA")
    dataframe = pd.DataFrame(data=new_coordinateso, columns=("1st_principal", "2nd_principal", "label"))
    sn.FacetGrid(dataframe, hue="label", height=6).map(plt.scatter, '1st_principal', '2nd_principal').add_legend()
    plt.title("LDA 4,7,8")
    plt.savefig('./2D_LDA_plot_for_478_data_.jpg')
    plt.show()
    plt.clf()
    plt.close()

    #new_coordinates_LDA = new_coordinates_LDA.reshape(new_coordinates_LDA.shape[1], new_coordinates_LDA.shape[0])
    #new_coordinates_LDA = new_coordinates_LDA.reshape(new_coordinates_LDA.shape[0], 28, 28)
    #cv2.imwrite('./original_pic_LDA_.jpg',new_coordinates_LDA[0])

# part c of lab project 1
def Ldac(images, labels, plot):
    #making classes depending on the numbers 4,7,and 8
    arr0 = images[np.where(labels == 0)]
    arr1 = images[np.where(labels == 1)]
    arr2 = images[np.where(labels == 2)]
    arr3 = images[np.where(labels == 3)]
    arr4 = images[np.where(labels == 4)]
    arr5 = images[np.where(labels == 5)]
    arr6 = images[np.where(labels == 6)]
    arr7 = images[np.where(labels == 7)]
    arr8 = images[np.where(labels == 8)]
    arr9 = images[np.where(labels == 9)]

    #calculating the mean for each class and for all classes together
    mean0 = np.mean(arr0, axis=0 )
    mean0 = mean0.reshape(mean0.shape[0], 1)
    mean1 = np.mean(arr1, axis=0 )
    mean1 = mean1.reshape(mean1.shape[0], 1)
    mean2 = np.mean(arr2, axis=0 )
    mean2 = mean2.reshape(mean2.shape[0], 1)
    mean3 = np.mean(arr3, axis=0 )
    mean3 = mean3.reshape(mean3.shape[0], 1)
    mean4 = np.mean(arr4, axis=0 )
    mean4 = mean4.reshape(mean4.shape[0], 1)
    mean5 = np.mean(arr5, axis=0 )
    mean5 = mean5.reshape(mean5.shape[0], 1)
    mean6 = np.mean(arr6, axis=0 )
    mean6 = mean6.reshape(mean6.shape[0], 1)
    mean7 = np.mean(arr7, axis=0 )
    mean7 = mean7.reshape(mean7.shape[0], 1)
    mean8 = np.mean(arr8, axis=0 )
    mean8 = mean8.reshape(mean8.shape[0], 1)
    mean9 = np.mean(arr9, axis=0 )
    mean9 = mean9.reshape(mean9.shape[0], 1)
    meanfull = np.mean(images, axis=0)
    meanfull = meanfull.reshape(meanfull.shape[0], 1)

    #claculating the covariance for each class
    cov0 = np.cov(arr0.reshape(arr0.shape[1], arr0.shape[0]))
    cov1 = np.cov(arr1.reshape(arr1.shape[1], arr1.shape[0]))
    cov2 = np.cov(arr2.reshape(arr2.shape[1], arr2.shape[0]))
    cov3 = np.cov(arr3.reshape(arr3.shape[1], arr3.shape[0]))
    cov4 = np.cov(arr4.reshape(arr4.shape[1], arr4.shape[0]))
    cov5 = np.cov(arr5.reshape(arr5.shape[1], arr5.shape[0]))
    cov6 = np.cov(arr6.reshape(arr6.shape[1], arr6.shape[0]))
    cov7 = np.cov(arr7.reshape(arr7.shape[1], arr7.shape[0]))
    cov8 = np.cov(arr8.reshape(arr8.shape[1], arr8.shape[0]))
    cov9 = np.cov(arr9.reshape(arr9.shape[1], arr9.shape[0]))

    #calculating S_b and S_w
    s_b = (arr0.shape[0]*np.dot((mean0-meanfull),(mean0-meanfull).T))+(arr1.shape[0]*np.dot((mean1-meanfull),(mean1-meanfull).T))+(arr2.shape[0]*np.dot((mean2-meanfull),(mean2-meanfull).T))+(arr3.shape[0]*np.dot((mean3-meanfull),(mean3-meanfull).T))+(arr4.shape[0]*np.dot((mean4-meanfull),(mean4-meanfull).T))+(arr5.shape[0]*np.dot((mean5-meanfull),(mean5-meanfull).T)) +(arr6.shape[0]*np.dot((mean6-meanfull),(mean6-meanfull).T))+(arr7.shape[0]*np.dot((mean7-meanfull),(mean7-meanfull).T))+(arr8.shape[0]*np.dot((mean8-meanfull),(mean8-meanfull).T)) + (arr9.shape[0]*np.dot((mean9-meanfull),(mean9-meanfull).T))
    s_w = cov0+ cov1+ cov2+ cov3+ cov4 + cov5+ cov6+ cov7 + cov8 + cov9

    #calculating s_w^-1 s_b
    matrix = np.linalg.inv(s_w)@s_b
    #calculating the eigenvalues and eigenvectors
    valuesLDA, vectorsLDA = eigh(matrix, eigvals=(782,783))
    #creating the Projection matrix
    Alda = vectorsLDA.T
    #creating the new coordinates
    new_coordinates_LDA1 = Alda @ images.T
    #regenerating the orignal images
    new_coordinates_LDA = Alda.T @ new_coordinates_LDA1
    new_coordinateso = np.vstack((new_coordinates_LDA1, labels)).T
    print("plotting 2 diementions LDA ")
    dataframe = pd.DataFrame(data=new_coordinateso, columns=("1st_principal", "2nd_principal", "label"))
    sn.FacetGrid(dataframe, hue="label", height=6).map(plt.scatter, '1st_principal', '2nd_principal').add_legend()
    plt.title("LDA training test")
    plt.savefig('./2D_LDA_plot_for_test_data_.jpg')
    plt.show()
    plt.clf()

    #new_coordinates_LDA = new_coordinates_LDA.reshape(new_coordinates_LDA.shape[1], new_coordinates_LDA.shape[0])
    #new_coordinates_LDA = new_coordinates_LDA.reshape(new_coordinates_LDA.shape[0], 28, 28)
    #cv2.imwrite('./original_pic_LDA_.jpg',new_coordinates_LDA[0])



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
    test_train = test_train.reshape(test_train.shape[0], 28*28)
    images= images[np.where(np.logical_or(np.logical_or(labels == 4 , labels ==7) , labels == 8))]
    images = images.reshape(images.shape[0], 28*28)
    labels = labels[np.where(np.logical_or(np.logical_or(labels == 4 , labels ==7) , labels == 8))]

############################## QUESTION A ################################################
    #plotting eighenvalues graph
    print("running PCA for part a")
    Pca(images, 2, labels, False, True, False, False)

    #calling Pca for m = 2,10,50,100,200,300 and plotting the distortion
    print("running PCA on different values of m ")
    a= Pca(images, 2, labels, True, False, True, False)
    b= Pca(images, 10, labels, False, False, True, False)
    c= Pca(images, 50, labels, False, False, True, False)
    d= Pca(images, 100, labels, False, False, True, False)
    e= Pca(images, 200, labels, False, False, True, False)
    f= Pca(images, 300, labels, False, False, True, False)

    #plotting the distortion
    #plotting and calculating the total distortion error
    data = np.array([[a, 2], [b, 10], [c, 50], [d, 100], [e, 200], [f, 300]])
    x,y = data.T
    print("plotting the distortion plot")
    plt.scatter(x,y)
    plt.title("total distortion plot")
    plt.xlabel("total distortion made")
    plt.ylabel('differnt values of m')
    plt.grid()
    plt.savefig('./PCA_total_distortion_plot.jpg')
    plt.show()
    plt.clf()
############################## END QUESTION A ################################################

############################## QUESTION B ################################################
    print("Running LDA (part b)")
    LDA = Lda(images, labels, True)
############################## END QUESTION B ################################################

############################## QUESTION C ################################################
    print("starting part c")
    print("running PCA for part c")
    PCA = Pca(test_train, 2, test_label, True, False, False, True)
    print("running LDA for part c")
    LDA = Ldac(test_train, test_label, True)

    print("running Tsne for 4,7,8 data")
    model = TSNE(n_components=2, random_state=0)
    tsne_data = model.fit_transform(images)
    tsne_data = np.vstack((tsne_data.T, labels)).T
    plt.scatter(tsne_data[:,0], tsne_data[:,1], c=tsne_data[:,2])
    plt.savefig('./2D_Tsne_plot_for_4,7,8_data_.jpg')
    plt.show()
    plt.clf()

    print("running Tsne for test data")
    model = TSNE(n_components=2, random_state=0)
    tsne_data = model.fit_transform(test_train)
    tsne_data = np.vstack((tsne_data.T, test_label)).T
    plt.scatter(tsne_data[:,0], tsne_data[:,1], c=tsne_data[:,2])
    plt.savefig('./2D_Tsne_plot_for_test_data_.jpg')
    plt.show()
    plt.clf()
############################## END QUESTION C ################################################

if __name__ == "__main__":
    # Run the module with command-line parameters.
    main()
