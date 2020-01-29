import sys
import random
import numpy as np
from sklearn import svm
from sklearn import model_selection
'''Set the initial value'''






'''Get feature data from file as a matrix with a row per data instance'''
def getFeatureData(featureFile,bias=0):
    x=[]
    dFile = open(featureFile, 'r')
    i=0
    for line in dFile:
        row = line.split()
        rVec = [float(item) for item in row]
        if bias > 0:
            rVec.insert(0,bias)
        #print('row {} : {}'.format(i,rVec))
        x.append(rVec)
        i += 1
    dFile.close()
    return x

'''Get label data from file as a dictionary with key as data instance index
and value as the class index
'''
def getLabelData(labelFile,hyperPlaneClass=False):
    lFile = open(labelFile, 'r')
    lDict = {}
    for line in lFile:
        row = line.split()
        #print('label : {}'.format(lDict))
        if hyperPlaneClass and int(row[0]) < 0:
            lDict[int(row[1])] = -1
        else:
            lDict[int(row[1])] = int(row[0])
    lFile.close()
    return lDict

'''Get dot product for 2 vector'''
def dotProduct(u,v):
    if len(u) !=len(v):
        print("Error, u and v different length")
    else:
        sum = 0
        for i in range(len(u)):
            sum += u[i] * v[i]
        return sum

'''Get the norm of the vector function'''
def normVector(u):
    sum=0
    for i in range(len(u)):
        sum += u[i]**2
    return sum**(1/2)

''' get the vector of j column'''
def getVector(u,j):
    vectorj=[]
    for i in range(len(u)):
        vectorj.append(u[i][j])
    return vectorj

'''vector minus'''
def vectorMinus(u,v):
    if len(u) !=len(v):
        print("Error, u and v different length")
    else:
        M=[]
        for i in range(len(u)):
            M.append(u[i]-v[i])
        return M
'''unique function'''
def unique(x):
    output=[]
    for i in x:
        if i not in output:
            output.append(i)
    return(output)


'''data mean function'''
def col_mean(x,colsize):
    data_size=len(x)
    if len(x) == 0:
        output=[]
        for j in range(colsize):
            output.append(0)
        return(output)    
    elif type(x[0]) != list:
        return(x)
    else:
        col_size=len(x[0])
        output=[]
        for j in range(col_size):
            col_sum=0
            for i in range(data_size):
                col_sum +=x[i][j]/data_size
            output.append(col_sum)
        return(output)
        
'''sign function'''
def sign(x):
    if x <=0:
        return(-1)
    elif x>0:
        return(1)
    else:
        return(0)

'''Test for local data'''
#f_Feature=getFeatureData(r"C:\Users\user\Desktop\2018- NJIT Life\2019 fall\CS675 Mechine Learning\assignment\data\ionosphere\ionosphere.data")
#f_Label=getLabelData(r"C:\Users\user\Desktop\2018- NJIT Life\2019 fall\CS675 Mechine Learning\assignment\data\ionosphere\ionosphere.trainlabels.0")
#f_Feature=getFeatureData(r"C:\Users\user\Desktop\2018- NJIT Life\2019 fall\CS675 Mechine Learning\assignment\data\breast_cancer\breast_cancer.data")
#f_Label=getLabelData(r"C:\Users\user\Desktop\2018- NJIT Life\2019 fall\CS675 Mechine Learning\assignment\data\breast_cancer\breast_cancer.trainlabels.0")
#f_Feature=getFeatureData(r"C:\Users\user\Desktop\2018- NJIT Life\2019 fall\CS675 Mechine Learning\assignment\data\qsar_biodeg\qsar.data")
#f_Label=getLabelData(r"C:\Users\user\Desktop\2018- NJIT Life\2019 fall\CS675 Mechine Learning\assignment\data\qsar_biodeg\qsar.trainlabels.0")
#f_Feature=getFeatureData(r"C:\Users\user\Desktop\2018- NJIT Life\2019 fall\CS675 Mechine Learning\assignment\data\climate_simulation\climate.data")
#f_Label=getLabelData(r"C:\Users\user\Desktop\2018- NJIT Life\2019 fall\CS675 Mechine Learning\assignment\data\climate_simulation\climate.trainlabels.0")
#f_Feature=getFeatureData(r"C:\Users\user\Desktop\2018- NJIT Life\2019 fall\CS675 Mechine Learning\assignment\data\micromass\micromass.data")
#f_Label=getLabelData(r"C:\Users\user\Desktop\2018- NJIT Life\2019 fall\CS675 Mechine Learning\assignment\data\micromass\micromass.trainlabels.0")
#f_Feature=getFeatureData(r"C:\Users\user\Desktop\2018- NJIT Life\2019 fall\CS675 Mechine Learning\assignment\data\hill_valley\hill_valley.data")
#f_Label=getLabelData(r"C:\Users\user\Desktop\2018- NJIT Life\2019 fall\CS675 Mechine Learning\assignment\data\hill_valley\hill_valley.trainlabels.0")

#k_input=100


'''data for assignment input'''
dfilename=sys.argv[1]
dlabername=sys.argv[2]
k_input=int(sys.argv[3])
f_Feature=getFeatureData(dfilename)
f_Label=getLabelData(dlabername)

'''define col number, data size and training size'''
col_size=len(f_Feature[0])
data_size=len(f_Feature)


'''compute training_x, training_r and missing label  
   and split train data to train group and test group'''
missing_label=list(range(data_size))
x=[]
r=[]
for i in range(data_size):
    if f_Label.get(i)==1 or f_Label.get(i)==0:
        x.append(f_Feature[i])
        r.append(f_Label[i])
        missing_label.remove(i)

train_x,test_x,train_r,test_r = model_selection.train_test_split(x,r,test_size=0.2)
train_size=len(train_r)
test_size=len(test_r)

'''initially Z and Z1 be empty vector'''
Z=np.zeros((train_size,k_input))
Z1=np.zeros((test_size,k_input))
Ztest=np.zeros((len(missing_label),k_input))

for k in range(k_input):
    '''create random vector w each wj is between -1 and 1'''
    w=[]
    for i in range(col_size):
        w.append(random.random()*2-1)

    '''coculate wTx determine max and min'''
    wTx=[]
    for i in range(train_size):
        wTx.append(dotProduct(w,train_x[i]))
        
    w0=random.uniform(min(wTx),max(wTx))

    for i in range(train_size):
        Z[i][k]=(1+sign(wTx[i]+w0))/2

    for i in range(test_size):
        Z1[i][k]=(1+sign(dotProduct(w,test_x[i])+w0))/2
    '''create test z for future prediction''' 
    for i in range(len(missing_label)):
        Ztest[i][k]=(1+sign(dotProduct(w,f_Feature[missing_label[i]])+w0))/2





clf=svm.LinearSVC(max_iter=10000)
C=np.logspace(-3, 3, 7)

originbestscore=0
originbestC=0
newbestscore=0
newbestC=0
for c in C:
    clf.C=c
    if np.mean(model_selection.cross_val_score(clf, train_x, np.array(train_r), cv=10))>originbestscore:
        originbestscore=np.mean(model_selection.cross_val_score(clf, train_x, np.array(train_r), cv=10))
        originbestC=c
    if np.mean(model_selection.cross_val_score(clf, Z, np.array(train_r), cv=10))> newbestscore:
        newbestscore=np.mean(model_selection.cross_val_score(clf, Z, np.array(train_r), cv=10))
        newbestC=c

originbesterror=1-originbestscore
newbesterror=1-newbestscore



clf.C=originbestC
origintesterror=sum(clf.fit(train_x,np.array(train_r)).predict(test_x) != np.array(test_r))/len(test_r)
clf.C=newbestC
newtesterror=sum(clf.fit(Z,np.array(train_r)).predict(Z1) !=np.array(test_r))/len(test_r)

print("For",sys.argv[2],":")
print("Original data: LinearSVC best C = ",originbestC, "best CV error = ",round(originbesterror*100,1),"%, test error = ", round(origintesterror*100,1),"%")
print("Random hyperplane data:")
print("For k =",k_input)
print("LinearSVC best C = ", newbestC, "best CV error = ",round(newbesterror*100,1),"%, test error = ",round(newtesterror*100,1),"%")
print()
print("Prediction for the missing label data :")

'''Prediction for missing label data'''
predictions=clf.fit(Z,np.array(train_r)).predict(Ztest)
for i in range(len(missing_label)):
    print(predictions[i],missing_label[i])
