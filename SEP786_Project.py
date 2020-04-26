import numpy as np
from sklearn.preprocessing import MinMaxScaler
import matplotlib.pyplot as plt
import sklearn.discriminant_analysis
from sklearn.naive_bayes import GaussianNB
import time

## get the index of the minimum value of ndarray, which is not in minList
def getMinValue(ndarray, minlist):
    minValue = min(set(ndarray)-set(minlist))
    return np.where(ndarray==minValue)[0][0]

## Generate confusion matrix based on prediction and real results
def getConfusionMatrix(prediction, real, positive=1, negative=0):
    tp = tn = fp = fn = 0
    for i in range(real.shape[0]):
        if prediction[i]==real[i]==positive:
            tp+=1
        elif prediction[i]==real[i]==negative:
            tn+=1
        elif prediction[i]==positive and real[i]==negative:
            fp+=1
        elif prediction[i]==negative and real[i]==positive:
            fn+=1
    return [[tp/real.shape[0], fp/real.shape[0]], [fn/real.shape[0], tn/real.shape[0]]]

if __name__ == '__main__':
    # read in data
    data = np.genfromtxt('messidor_features.csv', delimiter=',')

    # normalize the data
    scaler = MinMaxScaler(feature_range=(0, 1))
    scaled = scaler.fit_transform(data.astype('float32'))

    # separate training and dataset
    np.random.shuffle(scaled)
    train = scaled[:850,:]
    test = scaled[850:,:]
    train_x = train[:,:19]
    train_y = train[:,19]
    test_x = test[:,:19]
    test_y = test[:,19]

    minList = []                            #list of the smallest Eigenvalues
    minIndexList = []                       #list of the index of the smallest eigenvalues
    mseList = []                            #list of mean square error
    pcaDim = []                             #list of PCA dimensions
    pcaClassificationError_LDA = []         #Classification Error using LDA based on feature selection using PCA
    backwardClassificationError_LDA = []    #Classification Error using LDA based on feature selection using backward search
    pcaClassificationError_GNB = []         #Classification Error using Gaussian Naive Bayes on feature selection using PCA
    backwardClassificationError_GNB = []    #Classification Error using Gaussian Naive Bayes on feature selection using backward search
    computationTime_LDA_training = []       #Computation Time of training using Linear Discrimination Analysis
    computationTime_GNB_training = []       #Computation Time of training using Gaussian Naive Bayes
    computationTime_LDA_testing = []        #Computation Time of testing using Linear Discrimination Analysis
    computationTime_GNB_testing = []        #Computation Time of testing using Gaussian Naive Bayes
    confusionMatrix_PCA_LDA = []            #confusion matrix for Linear Discrimination Analysis based on feature selection using PCA
    confusionMatrix_PCA_GNB = []            #confusion matrix for Gaussian Naive Bayes based on feature selection using PCA
    confusionMatrix_BWS_LDA = []            #confusion matrix for Linear Discrimination Analysis based on feature selection using Backward Search
    ConfusionMatrix_BWS_GNB = []            #confusion matrix for Gaussian Naive Bayes based on feature selection using Backward Search

    # eigenvalue decomposition
    W, V = np.linalg.eig(np.dot(train_x.T, train_x))

    # for full dimension
    train_xt = np.dot(train_x, V)           #transformed X
    pcaDim.append(19)

    # get Classification Error for PCA and Backward search using LDA

    # Get training time for LDA (magnified for 1000 times)
    time_start = time.clock()
    lda = sklearn.discriminant_analysis.LinearDiscriminantAnalysis()
    for i in range(1000):                                   #train for 1000 times to get a more accurate time
        lda.fit(train_xt, train_y)
    time_elapsed = (time.clock() - time_start)              #count the computation time
    computationTime_LDA_training.append(time_elapsed)       #store the computation time

    # Get testing time for LDA (magnified for 1000 times)
    time_start = time.clock()
    prediction = np.zeros(test_x.shape[0])
    for i in range(1000):                                   #test for 1000 times to get a more accurate time
        prediction = lda.predict(np.dot(test_x, V))
    time_elapsed = (time.clock() - time_start)              #count the computation time
    computationTime_LDA_testing.append(time_elapsed)        #store the computation time

    # store PCA-LDA classification error, calculate PCA-LDA confusion matrix for full dimension
    classificationError = sum(abs(prediction - test_y))/test_y.shape[0]
    pcaClassificationError_LDA.append(classificationError)  #with or without linear transformation, should render the same classification error
    confusionMatrix_PCA_LDA.append(getConfusionMatrix(prediction, test_y))

    # calculate and store BSW-LDA classification error and confusion matrix for full dimension
    lda.fit(train_x, train_y)
    prediction = lda.predict(test_x)
    classificationError = sum(abs(prediction - test_y)) / test_y.shape[0]
    backwardClassificationError_LDA.append(classificationError)
    confusionMatrix_BWS_LDA.append(getConfusionMatrix(prediction, test_y))

    #get Classification Error for PCA and Backward search using Gaussian Naive Bayes

    # Get training time for GNB (magnified for 1000 times)
    time_start = time.clock()
    gnb = GaussianNB()
    for i in range(1000):                                   #train for 1000 times to get a more accurate time
        gnb.fit(train_xt, train_y)
    time_elapsed = (time.clock() - time_start)              #count the computation time
    computationTime_GNB_training.append(time_elapsed)       #store the computation time

    # Get testing time for GNB (magnified for 1000 times)
    time_start = time.clock()
    for i in range(1000):                                   #test for 1000 times to get a more accurate time
        prediction = gnb.predict(np.dot(test_x, V))
    time_elapsed = (time.clock() - time_start)              #count the computation time
    computationTime_GNB_testing.append(time_elapsed)        #store the computation time

    # store PCA-GNB classification error, calculate PCA-GNB confusion matrix for full dimension
    classificationError = sum(abs(prediction - test_y)) / test_y.shape[0]
    pcaClassificationError_GNB.append(classificationError)
    confusionMatrix_PCA_GNB.append(getConfusionMatrix(prediction, test_y))

    # calculate and store BSW-GNB classification error and confusion matrix for full dimension
    gnb.fit(train_x, train_y)
    prediction = gnb.predict(test_x)
    classificationError = sum(abs(prediction - test_y)) / test_y.shape[0]
    backwardClassificationError_GNB.append(classificationError)
    ConfusionMatrix_BWS_GNB.append(getConfusionMatrix(prediction, test_y))

    # remove the eigenvector of smallest eigenvalue one by one, and calculate the corresponding training/testing
    # computation time, PCA_LDA classification error, PCA_GNB classification error, corresponding confusion matrix
    for i in range(15):
        minEigValIndex = getMinValue(W, minList)
        minIndexList.append(minEigValIndex)
        minList.append(W.item(minEigValIndex))

        V_new = np.zeros((19, 18-i))
        index = 0
        for j in range(19):
            if j not in minIndexList:
                V_new[:, index] = V[:, j]
                index += 1

        # calculate the transformed x
        train_xt = np.dot(train_x, V_new)
        pcaDim.append(18-i)

        #calculate training computation time for LDA (magnified for 1000 times)
        time_start = time.clock()
        #lda = sklearn.discriminant_analysis.LinearDiscriminantAnalysis()
        for j in range(1000):                               # train for 1000 times to get a more accurate time
            lda.fit(train_xt, train_y)
        time_elapsed = (time.clock() - time_start)          # count the computation time
        computationTime_LDA_training.append(time_elapsed)   # store the computation time

        # calculate testing computation time for LDA (magnified for 1000 times)
        time_start = time.clock()
        for j in range(1000):                               # test for 1000 times to get a more accurate time
            prediction = lda.predict(np.dot(test_x, V_new))
        time_elapsed = (time.clock() - time_start)          # count the computation time
        computationTime_LDA_testing.append(time_elapsed)    # store the computation time

        # calculate and store PCA-LDA classification error and confusion matrix
        classificationError = sum(abs(prediction - test_y))/test_y.shape[0]
        pcaClassificationError_LDA.append(classificationError)
        confusionMatrix_PCA_LDA.append(getConfusionMatrix(prediction, test_y))

        # calculate training computation time for GNB (magnified for 1000 times)
        time_start = time.clock()
        gnb = GaussianNB()
        for j in range(1000):                               # train for 1000 times to get a more accurate time
            gnb.fit(train_xt, train_y)
        time_elapsed = (time.clock() - time_start)          # count the computation time
        computationTime_GNB_training.append(time_elapsed)   # store the computation time

        # calculate testing computation time for GNB (magnified for 1000 times)
        time_start = time.clock()
        for j in range(1000):                               # test for 1000 times to get a more accurate time
            prediction = gnb.predict(np.dot(test_x, V_new))
        time_elapsed = (time.clock() - time_start)          # count the computation time
        computationTime_GNB_testing.append(time_elapsed)    # store the computation time

        # calculate and store PCA-GNB classification error and confusion matrix
        classificationError = sum(abs(prediction - test_y))/test_y.shape[0]
        pcaClassificationError_GNB.append(classificationError)
        confusionMatrix_PCA_GNB.append(getConfusionMatrix(prediction, test_y))

    # graph plot
    plt.subplot(2, 2, 1)
    plt.plot(pcaDim, computationTime_LDA_training, marker = "+", color='b', label="Linear Discrimination Analysis")
    plt.plot(pcaDim, computationTime_GNB_training, marker = "+", color='g', label="Gaussian Naive Bayes")
    plt.legend(loc='upper left')
    plt.title('COMPUTATION TIME(TRAINING): LDA vs GNB')
    plt.xlabel('PCA DIMENSION')
    plt.ylabel('COMPUTATION TIME')
    plt.xlim(19, 4)
    plt.grid(color='r', which='minor', alpha=0.2)
    plt.grid(color='r', which='major', alpha=0.5)

    plt.subplot(2, 2, 2)
    plt.plot(pcaDim, computationTime_LDA_testing, marker = "+", color='b', label="Linear Discrimination Analysis")
    plt.plot(pcaDim, computationTime_GNB_testing, marker = "+", color='g', label="Gaussian Naive Bayes")
    plt.legend(loc='upper left')
    plt.title('COMPUTATION TIME(TESTING): LDA vs GNB')
    plt.xlabel('PCA DIMENSION')
    plt.ylabel('COMPUTATION TIME')
    plt.xlim(19, 4)
    plt.grid(color='r', which='minor', alpha=0.2)
    plt.grid(color='r', which='major', alpha=0.5)

    plt.subplot(2, 2, 3)
    plt.plot(pcaDim, pcaClassificationError_LDA, marker = "+", color='b', label="Linear Discrimination Analysis")
    plt.plot(pcaDim, pcaClassificationError_GNB, marker = "+", color='g', label="Gaussian Naive Bayes")
    plt.legend(loc='upper left')
    plt.title('PCA CLASS ERR: LDA vs GNB')
    plt.xlabel('PCA DIMENSION')
    plt.ylabel('CLASSIFICATION ERROR')
    plt.xlim(19, 4)
    plt.grid(color='r', which='minor', alpha=0.2)
    plt.grid(color='r', which='major', alpha=0.5)
    plt.axis([19, 4, 0.2, 0.5])

    # backward search based on LDA results
    worstFeature = - np.ones(15)
    for i in range(15):
        confusionMatrix = []
        error = np.ones(19)
        for removed in range(19):                                               #remove columns one by one
            Xselection = np.copy(train_x)
            if removed not in worstFeature:
                Xselection[:, removed] = 0
                for j in range(0, i):
                    Xselection[:, int(worstFeature[j])] = 0                     #remove columns in worstFeature
                lda = sklearn.discriminant_analysis.LinearDiscriminantAnalysis()
                lda.fit(Xselection, train_y)
                prediction = lda.predict(test_x)
                error[removed] = sum(abs(prediction - test_y))/test_y.shape[0]  #store prediction error of each test
                confusionMatrix.append(getConfusionMatrix(prediction, test_y))
            else:
                confusionMatrix.append([0,0,0,0])                               #append [0,0,0,0] to confusion matrix if it is already in worstFeature
        worstFeature[i] = np.argmin(error)
        backwardClassificationError_LDA.append(np.amin(error))
        confusionMatrix_BWS_LDA.append(confusionMatrix[int(worstFeature[i])])

    # backward search based on GNB results
    worstFeature = - np.ones(15)
    for i in range(15):
        confusionMatrix = []
        error = np.ones(19)
        for removed in range(19):                                               #remove columns one by one
            Xselection = np.copy(train_x)
            if removed not in worstFeature:
                Xselection[:, removed] = 0
                for j in range(0, i):
                    Xselection[:, int(worstFeature[j])] = 0                     #remove columns in worstFeature
                gnb = GaussianNB()
                gnb.fit(Xselection, train_y)
                prediction = gnb.predict(test_x)
                error[removed] = sum(abs(prediction - test_y))/test_y.shape[0]  #store prediction error of each test
                confusionMatrix.append(getConfusionMatrix(prediction, test_y))
            else:
                confusionMatrix.append([0,0,0,0])                               #append [0,0,0,0] to confusion matrix if it is already in worstFeature
        worstFeature[i] = np.argmin(error)
        backwardClassificationError_GNB.append(np.amin(error))
        ConfusionMatrix_BWS_GNB.append(confusionMatrix[int(worstFeature[i])])

    plt.subplot(2, 2, 4)
    plt.plot(pcaDim, backwardClassificationError_LDA, marker = "+", color='b', label="Linear Discrimination Analysis")
    plt.plot(pcaDim, backwardClassificationError_GNB, marker = "+", color='g', label="Gaussian Naive Bayes")
    plt.legend(loc='upper left')
    plt.title('FS CLASS ERR: LDA vs GNB')
    plt.xlabel('FEATURE SELECTION DIMENSION')
    plt.ylabel('CLASSIFICATION ERROR')
    plt.axis([19, 4, 0.2, 0.5])
    plt.grid(which='both')
    plt.grid(color='r', which='minor', alpha=0.2)
    plt.grid(color='r', which='major', alpha=0.5)

    plt.subplots_adjust(hspace = 0.5, wspace = 0.5)
    plt.show()

    print("done")