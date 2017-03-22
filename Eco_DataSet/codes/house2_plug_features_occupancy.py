# -*- coding: utf-8 -*-
"""
Created on Thu Mar 09 22:21:56 2017

@author: Yicheng
"""

import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import os.path
from sklearn import preprocessing, cross_validation
from sklearn import neural_network, ensemble, svm, neighbors
from sklearn.metrics import confusion_matrix
import itertools
from sklearn.metrics import f1_score

def plot_confusion_matrix(cm, classes,
                          normalize=False,
                          title='Confusion matrix',
                          cmap=plt.cm.Blues):
    """
    This function prints and plots the confusion matrix.
    Normalization can be applied by setting `normalize=True`.
    """
    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title)
    plt.colorbar()
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=45)
    plt.yticks(tick_marks, classes)

    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        print("Normalized confusion matrix")
    else:
        print('Confusion matrix, without normalization')

    print(cm)

    thresh = cm.max() / 2.
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(j, i, cm[i, j],
                 horizontalalignment="center",
                 color="white" if cm[i, j] > thresh else "black")

    plt.tight_layout()
    plt.ylabel('True label')
    plt.xlabel('Predicted label')

def convertDate(date_string):
    date_list = date_string.split("-")
    day=date_list[0]
    month=date_list[1]
    year=date_list[2]
    date_out=year
    if month=="Jan":
        date_out = date_out+"-01-"
    elif month=="Feb":
        date_out = date_out+"-02-"
    elif month=="Mar":
        date_out = date_out+"-03-"
    elif month=="Apr":
        date_out = date_out+"-04-"
    elif month=="May":
        date_out = date_out+"-05-"
    elif month=="Jun":
        date_out = date_out+"-06-"
    elif month=="Jul":
        date_out = date_out+"-07-"
    elif month=="Aug":
        date_out = date_out+"-08-"
    elif month=="Sep":
        date_out = date_out+"-09-"
    elif month=="Oct":
        date_out = date_out+"-10-"
    elif month=="Nov":
        date_out = date_out+"-11-"
    else:
        date_out = date_out+"-12-"
    date_out = date_out + day
    return date_out
    
occupancy_fname_suffix="../house#02/02_occupancy_csv/"
plug_fname_suffix="../house#02/02_plugs_csv/"

#read occupancy csv
df_occupancy1=pd.read_csv(occupancy_fname_suffix + "02_summer.csv")
df_occupancy2=pd.read_csv(occupancy_fname_suffix + "02_winter.csv")

occupancy_list = list()
house_plug_list = list()
for j in range(0,12):
    house_plug_list.append(list())

#use summer data (skip first 2 data in May)
for i in range(2,len(df_occupancy1)):
    
    #occupancy list
    temp = df_occupancy1.iloc[i,1:].tolist()
    occupancy_list.extend(temp)
   
    #power consumption for 12 appliances
    date_string = df_occupancy1.iloc[i,0]
    date_num = convertDate(date_string)
    print("parsing:" + date_num)
    #list of appliances
    #appliance 10 does not has August data
    for j in range(1,13):
        if j < 10:
            house_plug_fname= plug_fname_suffix + '0' + str(j) +'/' + date_num + ".csv"
        else:
            house_plug_fname= plug_fname_suffix + str(j) +'/' + date_num + ".csv"        
        if os.path.isfile(house_plug_fname)==True:
            df_house_plug = pd.read_csv(house_plug_fname,names=['power_consumption'])
            temp = df_house_plug.iloc[:,0].tolist()
            house_plug_list[j-1].extend(temp)
            #print j
            #print len(house_plug_list[j-1])
        else:
            temp = list()
            for k in range(0, 86400): 
                temp.append(np.nan)
            house_plug_list[j-1].extend(temp)
                
       
#use winter data
for i in range(0,len(df_occupancy2)):
    
    #occupancy list
    temp = df_occupancy2.iloc[i,1:].tolist()
    occupancy_list.extend(temp)
   
    #power consumption for 12 appliances
    date_string = df_occupancy2.iloc[i,0]
    date_num = convertDate(date_string)
    print("parsing:" + date_num)
    #list of appliances
    #appliance 10 does not has August data
    for j in range(1,13):
        if j < 10:
            house_plug_fname= plug_fname_suffix + '0' + str(j) +'/' + date_num + ".csv"
        else:
            house_plug_fname= plug_fname_suffix + str(j) +'/' + date_num + ".csv"        
        if os.path.isfile(house_plug_fname)==True:
            df_house_plug = pd.read_csv(house_plug_fname,names=['power_consumption'])
            temp = df_house_plug.iloc[:,0].tolist()
            house_plug_list[j-1].extend(temp)
            #print j
            #print len(house_plug_list[j-1])
        else:
            temp = list()
            for k in range(0, 86400): 
                temp.append(np.nan)
            house_plug_list[j-1].extend(temp)        


#feature extraction
# 15-min slot

house_features_list = list()
occupancy_slot_list = list()
#df = pd.DataFrame(house_plug_list)

print "processing features ..."

#min,max,mean,std,range
for i in range(0,12):

    #n features
    num_of_features = 7
    for k in range(0,num_of_features):
        house_features_list.append(list())    
    
    #for every slot
    for j in range(0, len(house_plug_list[i])/900):
        print "applicace num: " + str(i) + ", slot: " + str(j)
        minValue = (min(house_plug_list[i][900*j:900*(j+1)-1]))
        maxValue = (max(house_plug_list[i][900*j:900*(j+1)-1]))
        rangeValue = maxValue - minValue
        meanValue = np.mean(house_plug_list[i][900*j:900*(j+1)-1])
        stdValue = np.std(house_plug_list[i][900*j:900*(j+1)-1])
        corValue = np.corrcoef(np.array([house_plug_list[i][900*j:900*(j+1)-2], house_plug_list[0][900*j+1:900*(j+1)-1]])) 
        
        #get the time of the day morning/afternoon/evening
        slot_per_day = 96
        time_of_day = 0
        if j%slot_per_day < 32:
            time_of_day = 1
        elif j%slot_per_day < 64:
            time_of_day = 2
        else:
            time_of_day = 3       
        
        house_features_list[i*num_of_features].append(minValue)
        house_features_list[i*num_of_features+1].append(maxValue)
        house_features_list[i*num_of_features+2].append(rangeValue)
        house_features_list[i*num_of_features+3].append(meanValue)
        house_features_list[i*num_of_features+4].append(stdValue)
        house_features_list[i*num_of_features+5].append(corValue[0][1])
        house_features_list[i*num_of_features+6].append(time_of_day)        

for i in range(0, len(occupancy_list)/900):
    occ = np.mean(occupancy_list[900*i:900*(i+1)-1])
    if occ >= 0.5: occ = 1
    else: occ = 0        
    occupancy_slot_list.append(occ)


name = ["min1","max1","range1","mean1","std1","cor1","time1",
        "min2","max2","range2","mean2","std2","cor2","time2",
        "min3","max3","range3","mean3","std3","cor3","time3",
        "min4","max4","range4","mean4","std4","cor4","time4",
        "min5","max5","range5","mean5","std5","cor5","time5",
        "min6","max6","range6","mean6","std6","cor6","time6",
        "min7","max7","range7","mean7","std7","cor7","time7",
        "min8","max8","range8","mean8","std8","cor8","time8",
        "min9","max9","range9","mean9","std9","cor9","time9",
        "min10","max10","range10","mean10","std10","cor10","time10",
        "min11","max11","range11","mean11","std11","cor11","time11",
        "min12","max12","range12","mean12","std12","cor12","time12"]

#list to dataFrame
df_features = pd.DataFrame()
for i in range(0, len(house_features_list)):
    df_features[name[i]] = pd.Series(house_features_list[i])

df_features.replace(np.nan,-99999,inplace=True)

df_occ_slot = pd.DataFrame()
df_occ_slot["occupancy"] = pd.Series(occupancy_slot_list)
         
#define features(X) and labels(y)
X = np.array(df_features)
y = np.array(df_occ_slot["occupancy"])

X_train_validation = X[0: int (0.8 * len(X)),:]
y_train_validation = y[0: int (0.8 * len(X))]

X_test = X[int (0.8 * len(X)):,:]
y_test = y[int (0.8 * len(X)):]
#create training and testing samples
#X_train, X_test, y_train, y_test = cross_validation.train_test_split(X,y, test_size = 0.2)

#define the classifier:


#define the classifier:multi-layer perceptron algorithm that trains using Backpropagation
clf1 = neural_network.MLPClassifier(solver='sgd', alpha=1e-5,hidden_layer_sizes=(10, 8), random_state=1)
clf1 = clf1.fit(X_train_validation , y_train_validation)
#==============================================================================
# layer_importance_1 = clf1.coefs_
# num_layers = clf1.n_layers_
# norm_layer_importance_1 = [100*float(i)/sum(layer_importance_1) for i in layer_importance_1]
# print "number of layers:"
# print num_layers
# print "norm_layer_importance_1:"
# print norm_layer_importance_1
#==============================================================================

y_predict_1 = clf1.predict(X_test) 
cnf_matrix_1 = confusion_matrix(y_test, y_predict_1)
# plot unnormalized confusion matrix
plt.figure()
plot_confusion_matrix(cnf_matrix_1, classes=["unoccupied", "Occupied"], title='unnormalization confusion matrix of MLP Model')
# Plot normalized confusion matrix
plt.figure()
plot_confusion_matrix(cnf_matrix_1, classes=["unoccupied", "Occupied"], normalize=True, title='Normalized confusion matrix of MLP Model')


#we are using the k nearest neighbors classifier from sklearn
clf2 = neighbors.KNeighborsClassifier()
clf2 = clf2.fit(X_train_validation , y_train_validation)
# no coeff info available
y_predict_2 = clf2.predict(X_test) 
cnf_matrix_2 = confusion_matrix(y_test, y_predict_2)
# plot unnormalized confusion matrix
plt.figure()
plot_confusion_matrix(cnf_matrix_2, classes=["unoccupied", "Occupied"], title='unnormalization confusion matrix of KNN Model')
# Plot normalized confusion matrix
plt.figure()
plot_confusion_matrix(cnf_matrix_2, classes=["unoccupied", "Occupied"], normalize=True, title='Normalized confusion matrix of KNN Model')


#define the classifier: SVM
clf3 = svm.SVC(gamma=2, C=1, kernel='linear')
clf3 = clf3.fit(X_train_validation , y_train_validation)
coef_importance_3 = clf3.coef_
norm_coef_importance_3 = [100*float(i)/sum(coef_importance_3) for i in coef_importance_3]
print "norm_coef_importance_3:"
print norm_coef_importance_3

y_predict_3 = clf3.predict(X_test) 
cnf_matrix_3 = confusion_matrix(y_test, y_predict_3)
# plot unnormalized confusion matrix
plt.figure()
plot_confusion_matrix(cnf_matrix_3, classes=["unoccupied", "Occupied"], title='unnormalization confusion matrix of SVM Model')
# Plot normalized confusion matrix
plt.figure()
plot_confusion_matrix(cnf_matrix_3, classes=["unoccupied", "Occupied"], normalize=True, title='Normalized confusion matrix of SVM Model')


#define the classifier:random forest
clf4 = ensemble.RandomForestClassifier(max_depth=20, n_estimators=10, max_features=84)
clf4 = clf4.fit(X_train_validation , y_train_validation)
coef_importance_4 = clf4.feature_importances_
norm_coef_importance_4 = [100*float(i)/sum(coef_importance_4) for i in coef_importance_4]
print "norm_coef_importance_4:"
print norm_coef_importance_4

y_predict_4 = clf4.predict(X_test) 
cnf_matrix_4 = confusion_matrix(y_test, y_predict_4)
# plot unnormalized confusion matrix
plt.figure()
plot_confusion_matrix(cnf_matrix_4, classes=["unoccupied", "Occupied"], title='unnormalization confusion matrix of RandomForest Model')
# Plot normalized confusion matrix
plt.figure()
plot_confusion_matrix(cnf_matrix_4, classes=["unoccupied", "Occupied"], normalize=True, title='Normalized confusion matrix of RandomForest Model')

#train the classfier
#clf.fit(X_train, y_train)

#test
#accuracy = clf.score(X_test, y_test)
#print("accuracy = " + str(accuracy*100) + '%')

scores1 = cross_validation.cross_val_score(clf1,X_train_validation,y_train_validation,cv=10)
f1_MLP = f1_score(y_test, y_predict_1, average='micro')
print("scores by MLPClassifier: ")
print("score1.mean: "+str(np.mean(scores1)) + "score1.var: " + str(np.var(scores1)) )
print ("f1 score for MLP:")
print f1_MLP
MLPline, = plt.plot(scores1, color='b', marker='+', label='MLP')


scores2 = cross_validation.cross_val_score(clf2,X_train_validation,y_train_validation,cv=10)
f2_MLP = f1_score(y_test, y_predict_2, average='micro')
print("scores by KNeighborsClassifier: ")
print("score2.mean: "+str(np.mean(scores2)) + "score2.var: " + str(np.var(scores2)) )
print ("f1 score for KNN:")
print f2_MLP
KNNline, = plt.plot(scores2, color='r', marker='o', label='KNN')

scores3 = cross_validation.cross_val_score(clf3, X_train_validation, y_train_validation,cv=10)
f3_MLP = f1_score(y_test, y_predict_3, average='micro')
print("Cross validation scores by svm: ")
print("score3.mean: "+str(np.mean(scores3)) + "score3.var: " + str(np.var(scores3)) )
print ("f1 score for SVM:")
print f3_MLP
SVMline, = plt.plot(scores3, color='k', marker=',', label='SVM')

scores4 = cross_validation.cross_val_score(clf4, X_train_validation, y_train_validation,cv=10)
f4_MLP = f1_score(y_test, y_predict_4, average='micro')
print("Cross validation scores by RandomForestClassifier: ")
print("score4.mean: "+str(np.mean(scores4)) + "score4.var: " + str(np.var(scores4)) )
print ("f1 score for RandomForest:")
print f4_MLP
RandomForestline, = plt.plot(scores4, color='g', marker='s', label='Random Forest')


plt.legend(handles=[MLPline, KNNline, SVMline, RandomForestline])
plt.ylabel('Cross Validation Accuracy Score')
plt.xlabel('Tests')
plt.title('House#2 Plugs Features Occupancy Prediction')
