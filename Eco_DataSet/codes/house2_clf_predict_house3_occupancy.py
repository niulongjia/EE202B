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

############################# data parsing ###############################    
occupancy_fname_suffix="../house#02/02_occupancy_csv/"
#read occupancy csv
df_occupancy1=pd.read_csv(occupancy_fname_suffix + "02_summer.csv")
df_occupancy2=pd.read_csv(occupancy_fname_suffix + "02_winter.csv")

occupancy_list = list()
house_sm_list = list()
#append 4 lists: powerallphases, phase1, phase2, phase3
for j in range(0,4):
    house_sm_list.append(list())

#use summer data
for i in range(2,len(df_occupancy1)):
    
    #occupancy list
    temp = df_occupancy1.iloc[i,1:].tolist()
    occupancy_list.extend(temp)
   
    #power phases data
    #get the date first
    date_string = df_occupancy1.iloc[i,0]
    date_num = convertDate(date_string)
    print("parsing:" + date_num)
    #open the file
    house_sm_fname="../house#02/02_sm_csv/" + date_num + ".csv"
    if os.path.isfile(house_sm_fname)==True:
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
    else:
        temp = list()
        for k in range(0, 86400): 
            temp.append(np.nan)
        house_sm_list[0].extend(temp)    
        house_sm_list[1].extend(temp) 
        house_sm_list[2].extend(temp)    
        house_sm_list[3].extend(temp)

#use winter data
for i in range(0,len(df_occupancy2)):
    
    #occupancy list
    temp = df_occupancy2.iloc[i,1:].tolist()
    occupancy_list.extend(temp)
   
    #power phases data
    #get the date first
    date_string = df_occupancy2.iloc[i,0]
    date_num = convertDate(date_string)
    print("parsing:" + date_num)
    #open the file
    house_sm_fname="../house#02/02_sm_csv/" + date_num + ".csv"
    if os.path.isfile(house_sm_fname)==True:
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
    else:
        temp = list()
        for k in range(0, 86400): 
            temp.append(np.nan)
        house_sm_list[0].extend(temp)    
        house_sm_list[1].extend(temp) 
        house_sm_list[2].extend(temp)    
        house_sm_list[3].extend(temp)

######################## feature extraction ###########################

# 15-min slot
house_features_list = list()
occupancy_slot_list = list()

print "processing features ..."

#==============features==================#
# min, max, mean, std, range, 
# autocorrelation at lag 1,
# number of detected on/off events
# time of the day: morning afternoon evening

# sum of absolute difference among phase 1, 2 and 3


for i in range(0,4):
    
    #number of features
    num_of_features = 8
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
                if time_count  == ThT:
                    time_count = 0
                    num_state_switch += 1                    
            #detect one on/off switch
            else:
                #reset the timer
                time_count += 1
        
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
        house_features_list[i*num_of_features+6].append(num_state_switch)
        house_features_list[i*num_of_features+7].append(time_of_day)

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
    

#============== occupancy =================#        
for i in range(0, len(occupancy_list)/900):
    occ = np.mean(occupancy_list[900*i:900*(i+1)-1])
    if occ >= 0.5: occ = 1.0
    else: occ = 0.0        
    occupancy_slot_list.append(occ)

#============== training and classification ==================#
name = ["min0","max0","range0","mean0","std0","cor0","onoff0","time0",
        "min1","max1","range1","mean1","std1","cor1","onoff1","time1",
        "min2","max2","range2","mean2","std2","cor2","onoff2","time2",
        "min3","max3","range3","mean3","std3","cor3","onoff3","time3",
        "sad12", "sad13", "sad23"]

#list to dataFrame
df_features = pd.DataFrame()
for i in range(0, len(house_features_list)):
    df_features[name[i]] = pd.Series(house_features_list[i])

#replace the null data with -99999
df_features.replace(np.nan,-99999,inplace=True)

df_occ_slot = pd.DataFrame()
df_occ_slot["occupancy"] = pd.Series(occupancy_slot_list)
         
#define features(X) and labels(y)
X = np.array(df_features)
Y = np.array(df_occ_slot["occupancy"])

#define the classifier:random forest
clf = ensemble.RandomForestClassifier(max_depth=20, n_estimators=10, max_features=35)
clf = clf.fit(X , Y)

#===============================================================#
#===================== house 1 =================================#
#===============================================================#

############################# data parsing ###############################    
occupancy_fname_suffix="../house#03/03_occupancy_csv/"
#read occupancy csv
df_occupancy1=pd.read_csv(occupancy_fname_suffix + "03_summer.csv")
df_occupancy2=pd.read_csv(occupancy_fname_suffix + "03_winter.csv")

occupancy_list = list()
house_sm_list = list()
#append 4 lists: powerallphases, phase1, phase2, phase3
for j in range(0,4):
    house_sm_list.append(list())

#use summer data
for i in range(2,len(df_occupancy1)):
    
    #occupancy list
    temp = df_occupancy1.iloc[i,1:].tolist()
    occupancy_list.extend(temp)
   
    #power phases data
    #get the date first
    date_string = df_occupancy1.iloc[i,0]
    date_num = convertDate(date_string)
    print("parsing:" + date_num)
    #open the file
    house_sm_fname="../house#03/03_sm_csv/" + date_num + ".csv"
    if os.path.isfile(house_sm_fname)==True:
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
    else:
        temp = list()
        for k in range(0, 86400): 
            temp.append(np.nan)
        house_sm_list[0].extend(temp)    
        house_sm_list[1].extend(temp) 
        house_sm_list[2].extend(temp)    
        house_sm_list[3].extend(temp)

#use winter data
for i in range(0,len(df_occupancy2)):
    
    #occupancy list
    temp = df_occupancy2.iloc[i,1:].tolist()
    occupancy_list.extend(temp)
   
    #power phases data
    #get the date first
    date_string = df_occupancy2.iloc[i,0]
    date_num = convertDate(date_string)
    print("parsing:" + date_num)
    #open the file
    house_sm_fname="../house#03/03_sm_csv/" + date_num + ".csv"
    if os.path.isfile(house_sm_fname)==True:
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
    else:
        temp = list()
        for k in range(0, 86400): 
            temp.append(np.nan)
        house_sm_list[0].extend(temp)    
        house_sm_list[1].extend(temp) 
        house_sm_list[2].extend(temp)    
        house_sm_list[3].extend(temp)
        
######################## feature extraction ###########################

# 15-min slot
house_features_list = list()
occupancy_slot_list = list()

print "processing features ..."

#==============features==================#
# min, max, mean, std, range, 
# autocorrelation at lag 1,
# number of detected on/off events
# time of the day: morning afternoon evening

# sum of absolute difference among phase 1, 2 and 3


for i in range(0,4):
    
    #number of features
    num_of_features = 8
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
                if time_count  == ThT:
                    time_count = 0
                    num_state_switch += 1                    
            #detect one on/off switch
            else:
                #reset the timer
                time_count += 1
        
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
        house_features_list[i*num_of_features+6].append(num_state_switch)
        house_features_list[i*num_of_features+7].append(time_of_day)

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
    

#============== occupancy =================#        
for i in range(0, len(occupancy_list)/900):
    occ = np.mean(occupancy_list[900*i:900*(i+1)-1])
    if occ >= 0.5: occ = 1.0
    else: occ = 0.0        
    occupancy_slot_list.append(occ)

#============== training and classification ==================#
name = ["min0","max0","range0","mean0","std0","cor0","onoff0","time0",
        "min1","max1","range1","mean1","std1","cor1","onoff1","time1",
        "min2","max2","range2","mean2","std2","cor2","onoff2","time2",
        "min3","max3","range3","mean3","std3","cor3","onoff3","time3",
        "sad12", "sad13", "sad23"]

#list to dataFrame
df_features = pd.DataFrame()
for i in range(0, len(house_features_list)):
    df_features[name[i]] = pd.Series(house_features_list[i])

#replace the null data with -99999
df_features.replace(np.nan,-99999,inplace=True)

df_occ_slot = pd.DataFrame()
df_occ_slot["occupancy"] = pd.Series(occupancy_slot_list)
         
#define features(X) and labels(y)
X = np.array(df_features)
Y = np.array(df_occ_slot["occupancy"])

coef_importance = clf.feature_importances_
norm_coef_importance = [100*float(i)/sum(coef_importance) for i in coef_importance]
print "norm_coef_importance:"
print norm_coef_importance

y = clf.predict(X) 
cnf_matrix = confusion_matrix(Y, y)
# plot unnormalized confusion matrix
plt.figure()
plot_confusion_matrix(cnf_matrix, classes=["unoccupied", "Occupied"], title='unnormalization confusion matrix (Use house#02 Random Forest Model to predict house#03)')
# Plot normalized confusion matrix
plt.figure()
plot_confusion_matrix(cnf_matrix, classes=["unoccupied", "Occupied"], normalize=True, title='Normalized confusion matrix (Use house#02 Random Forest Model to predict house#03)')

scores = cross_validation.cross_val_score(clf,X,Y,cv=10)
f4_MLP = f1_score(Y, y, average='micro')
print("scores by RandomForestClassifier: ")
print("score4.mean: "+str(np.mean(scores)) + "score4.var: " + str(np.var(scores)) )
print ("f1 score for RandomForest:")
print f4_MLP