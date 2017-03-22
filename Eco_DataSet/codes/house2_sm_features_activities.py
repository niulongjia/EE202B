# -*- coding: utf-8 -*-
"""
Created on Fri Mar 17 20:35:09 2017

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

yearTable = ["2012","2013"]
monthTable = ["01","02","03","04","05","06","07","08","09","10","11","12"]
dayTable = ["01","02","03","04","05","06","07","08","09","10",
       "11","12","13","14","15","16","17","18","19","20",
       "21","22","23","24","25","26","27","28","29","30","31"]

############################# data parsing ###############################    
plug_fname_suffix="../house#02/02_plugs_csv/"


house_sm_list = list()
house_plug_list = list()
#append 4 lists: powerallphases, phase1, phase2, phase3
for j in range(0,4):
    house_sm_list.append(list())

#append 12 lists for appliance states
for j in range(0,4):
    house_plug_list.append(list())

    
for y in range(0,2):
    for m in range(0,12):
        for d in range(0,31):
            #get the date first
            date_num = yearTable[y] + '-' + monthTable[m] + '-' + dayTable[d]             
            #============== power phases data =============# 
            #open the file
            house_sm_fname="../house#02/02_sm_csv/" + date_num + ".csv"
            if os.path.isfile(house_sm_fname)==True:
                print("parsing:" + date_num)
                df_sm=pd.read_csv(house_sm_fname,
                                  names=['powerallphases',
                                         'powerl1',
                                         'powerl2',
                                         'powerl3',
                                         'currentneutral',
                                         'currentl1',
                                         'currentl2',
                                         'currentl3',
                                         'voltagel1',
                                         'voltagel2',
                                         'voltagel3',
                                         'phaseanglevoltagel2l1',
                                         'phaseanglevoltagel3l1',
                                         'phaseanglecurrentvoltagel1',
                                         'phaseanglecurrentvoltagel2',
                                         'phaseanglecurrentvoltagel3'
                                                        ])
                house_sm_powerallphases = df_sm.iloc[:,0]
                house_sm_powerl1 = df_sm.iloc[:,1]
                house_sm_powerl2 = df_sm.iloc[:,2]
                house_sm_powerl3 = df_sm.iloc[:,3]
                
                house_sm_powerallphases_list = house_sm_powerallphases.tolist()
                house_sm_powerl1_list = house_sm_powerl1.tolist()
                house_sm_powerl2_list = house_sm_powerl2.tolist()
                house_sm_powerl3_list = house_sm_powerl3.tolist()        
                
                house_sm_list[0].extend(house_sm_powerallphases_list)
                house_sm_list[1].extend(house_sm_powerl1_list)
                house_sm_list[2].extend(house_sm_powerl2_list)
                house_sm_list[3].extend(house_sm_powerl3_list)
        
            
                #====================== on/off state of appliances =================#
                
                # This is for Dishwasher/Kettle
                thres_dishwasher_kettle = 100
                house2_dishwasher_fname= plug_fname_suffix + "02/" + date_num + ".csv"
                house2_kettle_fname= plug_fname_suffix + "07/" + date_num + ".csv"   
                
                if os.path.isfile(house2_dishwasher_fname) == True and os.path.isfile(house2_kettle_fname) == True:
                    df_house2_dishwasher_power = pd.read_csv(house2_dishwasher_fname,names=['power_consumption'])
                    df_house2_kettle_power = pd.read_csv(house2_kettle_fname,names=['power_consumption'])
                    
                    house2_dishwasher_power = df_house2_dishwasher_power.iloc[:,0]
                    house2_kettle_power = df_house2_kettle_power.iloc[:,0]
                    
                    for i in range(0, 86400):
                        if house2_dishwasher_power[i] > thres_dishwasher_kettle or house2_kettle_power[i] > thres_dishwasher_kettle:
                            house_plug_list[0].append(1)
                        else:
                            house_plug_list[0].append(0)
                else:
                    for k in range(0, 86400):
                        house_plug_list[0].append(np.nan)
                
            
                # This is for Entertainment
                thres_entertain = 100
                house2_entertain_fname= plug_fname_suffix + "05/" + date_num + ".csv"
                if os.path.isfile(house2_entertain_fname) == True:
                    df_house2_entertain_power = pd.read_csv(house2_entertain_fname,names=['power_consumption'])
                    house2_entertain_power = df_house2_entertain_power.iloc[:,0]
                    
                    for i in range(0,86400):
                        if house2_entertain_power[i] == -1:
                            house_plug_list[1].append(0)
                        elif house2_entertain_power[i] > thres_entertain:
                            house_plug_list[1].append(1)
                        else:
                            house_plug_list[1].append(0)
                else:
                    for k in range(0, 86400):
                        house_plug_list[1].append(np.nan)  
                        
                
                # This is for Lamp 
                thres_lamp = 100 # different states ???
                house2_lamp_fname= plug_fname_suffix + "08/" + date_num + ".csv"
                if os.path.isfile(house2_lamp_fname) == True:
                    df_house2_lamp_power = pd.read_csv(house2_lamp_fname,names=['power_consumption'])
                    house2_lamp_power = df_house2_lamp_power.iloc[:,0]
                    
                    for i in range(0,86400):
                        if house2_lamp_power[i] == -1:
                            house_plug_list[2].append(0)
                        elif house2_lamp_power[i] > thres_lamp:
                            house_plug_list[2].append(1)
                        else:
                            house_plug_list[2].append(0)
                else:
                    for k in range(0, 86400):
                        house_plug_list[2].append(np.nan) 
                    
        
                # This is for laptop 
                thres_laptop = 10 # 
                house2_laptop_fname= plug_fname_suffix + "09/" + date_num + ".csv"
                if os.path.isfile(house2_laptop_fname) == True:
                    df_house2_laptop_power = pd.read_csv(house2_laptop_fname,names=['power_consumption'])
                    house2_laptop_power = df_house2_laptop_power.iloc[:,0]
                    
                    for i in range(0,86400):
                        if house2_laptop_power[i] == -1:
                            house_plug_list[3].append(0)
                        elif house2_laptop_power[i] > thres_laptop:
                            house_plug_list[3].append(1)
                        else:
                            house_plug_list[3].append(0)
                else:
                    for k in range(0, 86400):
                        house_plug_list[3].append(np.nan) 


#==================== human activities ======================#
########### Below are building up prediction states #############
# e.g 0001 means laptop state is on, other states are off #
app_state_list = list()
ThT = 60 # if an appliance is active for more than thres(seconds), then prediction state set to 1

#for each slot
for j in range(0, len(house_plug_list[0])/900):
    
    state = 0
    # for each app
    for i in range(0,4):
        count = 0
        for h in range(900*j, 900*(j+1)-1):
            
            if house_plug_list[i][h] == 1:
                count += 1
                
        if count > ThT:
            state = state*2 + 1
        else:
            state = state*2
    
    app_state_list.append(state)
    

        
######################## feature extraction ###########################

# 15-min slot
house_features_list = list()

print "processing features ..."

#==============features==================#
# min, max, mean, std, range, 
# autocorrelation at lag 1,
# number of detected on/off events

# sum of absolute difference among phase 1, 2 and 3

for i in range(0,4):
    
    #number of features
    num_of_features = 7
    for k in range(0,num_of_features):
        house_features_list.append(list())
    
    #for every slot
    for j in range(0, len(house_sm_list[i])/900):
        print "power phase: " + str(i) + ", slot: " + str(j)
        
        minValue = (min(house_sm_list[i][900*j:900*(j+1)-1]))
        maxValue = (max(house_sm_list[i][900*j:900*(j+1)-1]))
        rangeValue = maxValue - minValue
        meanValue = np.mean(house_sm_list[i][900*j:900*(j+1)-1])
        stdValue = np.std(house_sm_list[i][900*j:900*(j+1)-1]) 
        corValue = np.corrcoef(np.array([house_sm_list[i][900*j:900*(j+1)-2], house_sm_list[0][900*j+1:900*(j+1)-1]]))       
        
        # on/off events: 
        # on/off events occur when an appliance is switched on or off. We
        # detect these events using a simple heuristic: If the difference between
        # a sample and its predecessor is bigger than a threshold ThA
        # and this difference remains higher than ThA for at least ThT seconds,
        # an on/off event is detected. We set ThA = 30W and ThT = 30 s.
        ThA = 30
        ThT = 30
        time_count = 0
        num_state_switch = 0        
        for h in range(900*j+1, 900*(j+1)):
            if abs(house_sm_list[i][h] - house_sm_list[i][h-1]) < ThA:
                # an on/off switch happened before
                if time_count > 0:
                    time_count += 1
                if(time_count  == ThT):
                    time_count = 0
                    num_state_switch += 1                    
            #detect one on/off switch
            else:
                #reset the timer
                time_count += 1
                
            
        
        house_features_list[i*num_of_features].append(minValue)
        house_features_list[i*num_of_features+1].append(maxValue)
        house_features_list[i*num_of_features+2].append(rangeValue)
        house_features_list[i*num_of_features+3].append(meanValue)
        house_features_list[i*num_of_features+4].append(stdValue)        
        house_features_list[i*num_of_features+5].append(corValue[0][1])
        house_features_list[i*num_of_features+6].append(num_state_switch)

#sum of absolute difference
num_of_sad = 3
for k in range(0,num_of_sad):
    house_features_list.append(list())
    
for j in range(0, len(house_sm_list[1])/900):
    sad12 = 0
    sad13 = 0
    sad23 = 0
    for h in range(900*j, 900*(j+1)-1):
        sad12 += abs(house_sm_list[1][h] - house_sm_list[2][h])
        sad13 += abs(house_sm_list[1][h] - float(house_sm_list[3][h]))
        sad23 += abs(house_sm_list[2][h] - float(house_sm_list[3][h]))
        
    sad12 = sad12/900.0
    sad13 = sad13/900.0
    sad23 = sad23/900.0  
    
    house_features_list[28].append(sad12)
    house_features_list[29].append(sad13)
    house_features_list[30].append(sad23)
    




#============== training and classification ==================#
name = ["min0","max0","range0","mean0","std0","cor0","onoff0",
        "min1","max1","range1","mean1","std1","cor1","onoff1",
        "min2","max2","range2","mean2","std2","cor2","onoff2",
        "min3","max3","range3","mean3","std3","cor3","onoff3",
        "sad12", "sad13", "sad23"]

#list to dataFrame
df_features = pd.DataFrame()
for i in range(0, len(house_features_list)):
#for i in range(0,1):
    df_features[name[i]] = pd.Series(house_features_list[i])

#replace the null data with -99999
df_features.replace(np.nan,-99999,inplace=True)

df_occ_slot = pd.DataFrame()
df_occ_slot["activity"] = pd.Series(app_state_list)
         
#define features(X) and labels(y)
X = np.array(df_features)
y = np.array(df_occ_slot["activity"])

X_train_validation = X[0: int (0.8 * len(X)),:]
y_train_validation = y[0: int (0.8 * len(X))]

X_test = X[int (0.8 * len(X)):,:]
y_test = y[int (0.8 * len(X)):]

#create training and testing samples
#X_train, X_test, y_train, y_test = cross_validation.train_test_split(X,y, test_size = 0.2)

#define the classifier:multi-layer perceptron algorithm that trains using Backpropagation
clf1 = neural_network.MLPClassifier(solver='lbfgs', alpha=1e-5,hidden_layer_sizes=(10, 8), random_state=1)
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
plot_confusion_matrix(cnf_matrix_1, 
                      classes=["0000", "0001","0010","0011","0100","0101","0110","0111","1000","1001","1010","1011","1100","1101","1110","1111"], 
                    title='unnormalization confusion matrix of MLP Model')
# Plot normalized confusion matrix
plt.figure()
plot_confusion_matrix(cnf_matrix_1, 
                      classes=["0000", "0001","0010","0011","0100","0101","0110","0111","1000","1001","1010","1011","1100","1101","1110","1111"], 
                    normalize=True, 
                    title='Normalized confusion matrix of MLP Model')



#we are using the k nearest neighbors classifier from sklearn
clf2 = neighbors.KNeighborsClassifier(kernel='linear')
clf2 = clf2.fit(X_train_validation , y_train_validation)
# no coeff info available
y_predict_2 = clf2.predict(X_test) 
cnf_matrix_2 = confusion_matrix(y_test, y_predict_2)
# plot unnormalized confusion matrix
plt.figure()
plot_confusion_matrix(cnf_matrix_2, 
                      classes=["0000", "0001","0010","0011","0100","0101","0110","0111","1000","1001","1010","1011","1100","1101","1110","1111"], 
                    title='unnormalization confusion matrix of KNN Model')
# Plot normalized confusion matrix
plt.figure()
plot_confusion_matrix(cnf_matrix_2, 
                      classes=["0000", "0001","0010","0011","0100","0101","0110","0111","1000","1001","1010","1011","1100","1101","1110","1111"], 
                    normalize=True, 
                    title='Normalized confusion matrix of KNN Model')


#define the classifier: SVM
clf3 = svm.LinearSVC()
clf3 = clf3.fit(X_train_validation , y_train_validation)
coef_importance_3 = clf3.coef_
norm_coef_importance_3 = [100*float(i)/sum(coef_importance_3) for i in coef_importance_3]
print "norm_coef_importance_3:"
print norm_coef_importance_3

y_predict_3 = clf3.predict(X_test) 
cnf_matrix_3 = confusion_matrix(y_test, y_predict_3)
# plot unnormalized confusion matrix
plt.figure()
plot_confusion_matrix(cnf_matrix_3, 
                      classes=["0000", "0001","0010","0011","0100","0101","0110","0111","1000","1001","1010","1011","1100","1101","1110","1111"], 
                    title='unnormalization confusion matrix of SVM Model')
# Plot normalized confusion matrix
plt.figure()
plot_confusion_matrix(cnf_matrix_3, 
                      classes=["0000", "0001","0010","0011","0100","0101","0110","0111","1000","1001","1010","1011","1100","1101","1110","1111"], 
                    normalize=True, 
                    title='Normalized confusion matrix of SVM Model')


#define the classifier:random forest
clf4 = ensemble.RandomForestClassifier(max_depth=20, n_estimators=10, max_features=31)
clf4 = clf4.fit(X_train_validation , y_train_validation)
coef_importance_4 = clf4.feature_importances_
norm_coef_importance_4 = [100*float(i)/sum(coef_importance_4) for i in coef_importance_4]
print "norm_coef_importance_4:"
print norm_coef_importance_4

y_predict_4 = clf4.predict(X_test) 
cnf_matrix_4 = confusion_matrix(y_test, y_predict_4)
# plot unnormalized confusion matrix
plt.figure()
plot_confusion_matrix(cnf_matrix_4, 
                      classes=["0000", "0001","0010","0011","0100","0101","0110","0111","1000","1001","1010","1011","1100","1101","1110","1111"], 
                      title='unnormalization confusion matrix of RandomForest Model')
# Plot normalized confusion matrix
plt.figure()
plot_confusion_matrix(cnf_matrix_4, 
                      classes=["0000", "0001","0010","0011","0100","0101","0110","0111","1000","1001","1010","1011","1100","1101","1110","1111"], 
                    normalize=True, 
                    title='Normalized confusion matrix of RandomForest Model')


#train the classfier
#print ("classifier training ...")
#clf.fit(X_train, y_train)

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

scores3 = cross_validation.cross_val_score(clf3,X_train_validation,y_train_validation,cv=10)
f3_MLP = f1_score(y_test, y_predict_3, average='micro')
print("scores by svm: ")
print("score3.mean: "+str(np.mean(scores3)) + "score3.var: " + str(np.var(scores3)) )
print ("f1 score for SVM:")
print f3_MLP
SVMline, = plt.plot(scores3, color='k', marker=',', label='SVM')

scores4 = cross_validation.cross_val_score(clf4,X_train_validation,y_train_validation,cv=10)
f4_MLP = f1_score(y_test, y_predict_4, average='micro')
print("scores by RandomForestClassifier: ")
print("score4.mean: "+str(np.mean(scores4)) + "score4.var: " + str(np.var(scores4)) )
print ("f1 score for RandomForest:")
print f4_MLP
RandomForestline, = plt.plot(scores4, color='g', marker='s', label='Random Forest')


plt.legend(handles=[MLPline, KNNline, SVMline, RandomForestline])
plt.ylabel('Cross Validation Accuracy Score')
plt.xlabel('Tests')
plt.title('House#2 Smart Meter Features Activities Prediction')
