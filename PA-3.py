import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.special import expit
import imageio
import glob
from random import randrange
from sklearn.metrics import roc_curve, auc
from sklearn.metrics import confusion_matrix
            
def stochasticGradientDescent(XTrain, yTrain, nEpoch, alpha, lamb):
    w = np.random.uniform(size=(XTrain.shape[1],))
    previousLoss = +float('inf')
    for i in range(nEpoch):
        index = randrange(0,len(XTrain))
        row = XTrain[index]
        hypothesis = expit(np.dot(row, w))
        error = hypothesis - yTrain[index]
        p1 =  yTrain[index] * np.log(hypothesis + 0.00001)
        p2 =  1 - yTrain[index] * np.log(1-hypothesis + 0.00001)
        loss = p1 - p2
        if previousLoss < loss:
            break
        else:
            previousLoss = loss
        gradient = np.dot(error, row) - (lamb * w)
        if lamb!=0:
            gradient[0]=np.sum(error)
        w = w - alpha * (gradient)
    return w


def batchGradientDescent(XTrain, yTrain, nEpoch, alpha, lamb):
    w = np.random.uniform(size=(XTrain.shape[1],))
    for i in range(nEpoch):
            hypothesis = expit(np.dot(XTrain, w))
            error = hypothesis - yTrain
            gradient = np.dot(error, XTrain)
            w = w - alpha * (gradient- (lamb * w)) 
    return w


def loadImages(train_face_path,train_noface_path,test_face_path,test_noface_path):
  
    totalImages = [[] for i in range(4)]
    
    for train_face in glob.glob(train_face_path +'/*.pgm'):
        
        im1 = imageio.imread(train_face).ravel()
        if im1 is not None:
            totalImages[0].append(im1)
        elif im1 is None:
            print ("Error loading: " + train_face)
        continue
    
    for train_non in glob.glob(train_noface_path +'/*.pgm'):
        
        im2 = imageio.imread(train_non).ravel()
        if im2 is not None:
            totalImages[1].append(im2)
        elif im2 is None:
            print ("Error loading: " + train_non)
        continue
    
    for test_face in glob.glob(test_face_path +'/*.pgm'):
       
        im3 = imageio.imread(test_face).ravel()
        if im3 is not None:
            totalImages[2].append(im3)
        elif im3 is None:
            print ("Error loading: " + test_face)
        continue
    
    for test_non in glob.glob(test_noface_path +'/*.pgm'):
        
        im4 = imageio.imread(test_non).ravel()
        if im4 is not None:
            totalImages[3].append(im4)
        elif im4 is None:
            print ("Error loading: " + test_non)
        continue
    
    return totalImages


def summarizeByClass(dataset):
    seperated = seperateByClasses(dataset)
    summaries = {}
    for classValue, instance in seperated.items():
        summaries[classValue] = summarize(instance)
    return summaries

def seperateByClasses(mydataset):
    seperateImages = {}
    for i in range (len(mydataset)):
        vector = mydataset[i]
        if(vector[-1] not in seperateImages):
            seperateImages[vector[-1]] = []
        seperateImages[vector[-1]].append(vector)
    return seperateImages

#To Calculate Mean
def mean(dataset):
    return np.mean(dataset)

def standardDeviation(dataset):
    return np.std(dataset)

def summarize(dataset):
    summaries = [(mean(attribute), standardDeviation(attribute)) for attribute in zip(*dataset)]
    del summaries[-1]
    return summaries

def calculateProbability(x, mean, standardDeviation):
    exponent = np.exp(-(np.power(x-mean,2)/(2*np.power(standardDeviation,2))))
    return (1/ (np.sqrt((2*np.pi) * np.power(standardDeviation,2)))) * exponent

def calculateClassProbabilities(summaries, inputVector):
    probabilities = {}
    np.finfo(np.float128)
    for classValue, classSummaries in summaries.items():
        probabilities[classValue] = 1
        for i in range(len(classSummaries)):
            mean, standardDeviation = classSummaries[i]
            x = inputVector[i]
            probabilities[classValue] += np.log(calculateProbability(x, mean, standardDeviation))
    return probabilities

def predict(summaries, inputVector):
    probabilities = calculateClassProbabilities(summaries, inputVector)
    bestLabel , bestProb = None, -1
    for classValue, probability in probabilities.items():
        if bestLabel is None or probability > bestProb:
            bestProb = probability
            bestLabel = classValue
    return bestLabel

def getPredictions(summaries, testSet):
    predictedArray = []
    for i in range(len(testSet)):
        result = int(predict(summaries, testSet[i]))
        predictedArray.append(result)
    return predictedArray

def getSpecs_Batch(XTrain, yTrain, XTest, yTest):
    result_problem=pd.DataFrame()
    print("Performance of the model trained on the whole train data")
    print("nEpoch,alpha,lambda,Acc,Prec,Rec, F1")
    nEpoch=200
    alpha=0.5
    lamb=0.5
    w=batchGradientDescent(XTrain, yTrain,nEpoch,alpha,lamb)
    predictedVal=np.dot(w,XTest.T)
    #predictedVal = np.array(predictedVal)
    threshMean = np.mean(predictedVal)
    predictedVal[predictedVal > threshMean] = 1.0
    predictedVal[predictedVal <= threshMean] = 0.0
    a,p,r,f1=getSpecs(yTest, predictedVal)
    
    df=pd.DataFrame(data=[[nEpoch,alpha,lamb,a,p,r,f1]],columns=['nEpoch','alpha','lambda','Acc','Prec','Rec','F1'])
    result_problem=result_problem.append(df)
    
    print(result_problem)
    return predictedVal

def getSpecs_Stochastic(XTrain, yTrain, XTest, yTest):
    result_problem=pd.DataFrame()
    print("Performance of the model trained on the whole train data")
    print("nEpoch,alpha,lambda,Acc,Prec,Rec, F1")
    nEpoch=15
    alpha=0.5
    lamb=0.5
    w=stochasticGradientDescent(XTrain,yTrain,nEpoch,alpha,lamb)
    predictedVal=np.dot(w,XTest.T)
    #predictedVal = np.array(predictedVal)
    threshMean = np.mean(predictedVal)
    predictedVal[predictedVal > threshMean] = 1.0
    predictedVal[predictedVal <= threshMean] = 0.0
    a,p,r,f1=getSpecs(yTest, predictedVal)
    df=pd.DataFrame(data=[[nEpoch,alpha,lamb,a,p,r,f1]],columns=['nEpoch','alpha','lambda','Acc','Prec','Rec','F1'])
    result_problem=result_problem.append(df)

    print(result_problem)
    return predictedVal
    
def getSpecs(testSet, predictions):
    confusion = confusion_matrix(testSet, predictions)
    TP = confusion[0][0]
    FP = confusion[0][1]
    FN = confusion[1][0]
    TN = confusion[1][1]
    
    Accuracy = (TP + TN) / (TP + TN + FN + FP)
    Precision = (TP) / (TP + FP)
    Recall = (TP) / (TP + FN)
    F1 = (2 * (Precision * Recall)) / (Precision + Recall)
    
    return Accuracy, Precision, Recall, F1

def plotROCCurve(fpr,tpr,auc):
    plt.figure()
    plt.plot(fpr, tpr, label='Logistic Regression(area=%0.3f'%auc)
    plt.plot([0, 1], [0, 1],'r--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.0])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Receiver operating characteristic')
    plt.legend(loc="lower right")
    plt.savefig('Log_ROC')
    plt.show()
    
list_of_data = loadImages( r'/Volumes/Shared/MAC/UCDenver_CSE/MachineLearning/Assignment3/MIT-CBCL-Face-dataset/train/face',
                   r'/Volumes/Shared/MAC/UCDenver_CSE/MachineLearning/Assignment3/MIT-CBCL-Face-dataset/train/non-face',
                   r'/Volumes/Shared/MAC/UCDenver_CSE/MachineLearning/Assignment3/MIT-CBCL-Face-dataset/test/face',
                   r'/Volumes/Shared/MAC/UCDenver_CSE/MachineLearning/Assignment3/MIT-CBCL-Face-dataset/test/non-face')
print('Data Loaded!!!')


train_face_path = np.array(list_of_data[0])
train_face_path = np.append(train_face_path, np.ones((len(train_face_path),1)), 1)

train_noface_path = np.array(list_of_data[1])
train_noface_path = np.append(train_noface_path, np.zeros((len(train_noface_path),1)), 1)

test_face_path = np.array(list_of_data[2])
test_face_path = np.append(test_face_path, np.ones((len(test_face_path),1)), 1)

test_noface_path = np.array(list_of_data[3])
test_noface_path = np.append(test_noface_path, np.zeros((len(test_noface_path),1)), 1)

Train_dataset = np.append(train_face_path,train_noface_path, 0)
Test_dataset = np.append(test_face_path, test_noface_path,0)

XTrain = Train_dataset[:,:-1]
XTrain = np.c_[np.ones((XTrain.shape[0])),XTrain]
yTrain = Train_dataset[:, -1]
XTest = Test_dataset[:,:-1]
XTest = np.c_[np.ones((XTest.shape[0])),XTest]
yTest = Test_dataset[:,-1]

#Experiment 1 with Naive Bayes classifier (NB)
print("Naive Bayes classifier (NB):")
summaries = summarizeByClass(Train_dataset)
yPredicted_NB = getPredictions(summaries,Test_dataset)
accuracy_NB = getSpecs(Test_dataset[:,-1], yPredicted_NB)
FPR_Naive, TPR_Naive, thresholds_Naive = roc_curve(yTest,yPredicted_NB)
AUC_Naive = auc(FPR_Naive, TPR_Naive)
plotROCCurve(FPR_Naive,TPR_Naive,AUC_Naive)

#Naive bayes with multiple points
'''
predict_proba=[]
for classValue, classSummaries in summaries.items():
    for i in range(len(classSummaries)):
        mean_s, sd = classSummaries[i]
        predict_proba.append(calculateProbability(XTest[i],mean_s,sd))
false_positive_rate, true_positive_rate, thresholds = roc_curve(yTest[0:10], predict_proba[0:10])

'''

#Experiment 2 with Batch gradient descent based logistic regres- sion (BGD-LR):
print("Batch gradient descent based logistic regression (BGD-LR):")
yPredicted_batch = getSpecs_Batch(XTrain, yTrain,XTest,yTest)
FPR_batch, TPR_batch, thresholds_batch= roc_curve(yTest, yPredicted_batch)
AUC_batch = auc(FPR_batch, TPR_batch)
plotROCCurve(FPR_batch,TPR_batch,AUC_batch)


#Experiment 3 with Stochastic gradient descent based logistic regression (SGD-LR):
print("Stochastic gradient descent based logistic regression (SGD-LR):")
yPredicted_stoc = getSpecs_Stochastic(XTrain, yTrain,XTest,yTest)
FPR_stoc, TPR_stoc, thresholds_stoc = roc_curve(yTest, yPredicted_stoc)
AUC_stoc = auc(FPR_stoc, TPR_stoc)
plotROCCurve(FPR_stoc,TPR_stoc,AUC_stoc)

totalPoints = [[FPR_Naive,TPR_Naive,AUC_Naive], [FPR_batch,TPR_batch,AUC_batch], [FPR_stoc,TPR_stoc,AUC_stoc]]
labels = ['Naive', 'Batch', 'Stochastic']
colors = ['blue','purple','green']
plt.title('Batch Descent, Naive Bayes, Stochastic Descent')
for rocPoints in totalPoints:
    plt.plot(rocPoints[0], rocPoints[1], label='Logistic Regression(area=%0.3f'%rocPoints[2])
plt.plot([0, 1], [0, 1],'r--')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.0])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Receiver operating characteristic')
plt.legend(loc="lower right")
plt.savefig('Log_ROC')
plt.show()