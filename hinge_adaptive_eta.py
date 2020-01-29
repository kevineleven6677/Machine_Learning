import sys
import random

'''Set the initial value'''
#eta=0.001
eta_list = [1, .1, .01, .001, .0001, .00001, .000001, .0000001, .00000001, .000000001, .0000000001, .00000000001 ]
bestobj = 1000000000000 # infinity
Error=10000000
theta=0.001


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
        if hyperPlaneClass and int(row[0]) <= 0:
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



'''Test for local data'''
#f_Feature=getFeatureData(r"C:\Users\user\Desktop\2018- NJIT Life\2019 fall\CS675 Mechine Learning\assignment\assignment3\testSVM.data")
#f_Label=getLabelData(r"C:\Users\user\Desktop\2018- NJIT Life\2019 fall\CS675 Mechine Learning\assignment\assignment3\testSVM.trainlabels")
#f_Feature=getFeatureData(r"C:\Users\user\Desktop\2018- NJIT Life\2019 fall\CS675 Mechine Learning\assignment\data\breast_cancer\breast_cancer.data")
#f_Label=getLabelData(r"C:\Users\user\Desktop\2018- NJIT Life\2019 fall\CS675 Mechine Learning\assignment\data\breast_cancer\breast_cancer.trainlabels.0")
#f_Feature=getFeatureData(r"C:\Users\user\Desktop\2018- NJIT Life\2019 fall\CS675 Mechine Learning\assignment\data\ionosphere\ionosphere.data")
#f_Label=getLabelData(r"C:\Users\user\Desktop\2018- NJIT Life\2019 fall\CS675 Mechine Learning\assignment\data\ionosphere\ionosphere.trainlabels.0")


'''data for assignment input'''
dfilename=sys.argv[1]
dlabername=sys.argv[2]

f_Feature=getFeatureData(dfilename)
f_Label=getLabelData(dlabername)

'''define col number, data size and training size'''
col_size=len(f_Feature[0])
data_size=len(f_Feature)
training_size=len(f_Label)


'''compute training_x, training_r and missing label  '''
missing_label=list(range(data_size))
training_x=[]
training_r=[]
for i in range(data_size):
    if f_Label.get(i)==1 or f_Label.get(i)==0:
        training_x.append(f_Feature[i])
        training_r.append(2*f_Label[i]-1)
        missing_label.remove(i)


'''Initialize w' and  w0'''
w=[]
for j in range(col_size):
    w.append(random.random()*0.002-0.01)
#w=[0.0001]*col_size

w0= random.random()*0.002-0.01
#w0=0.0001
''' SVM '''
for k in range(100000):
    '''compute the Error vector and idendity'''
    d=[]
    idendity=[]
    for i in range(training_size):
        d.append(max(0,1-training_r[i]*(dotProduct(w,training_x[i])+w0)))
        if training_r[i]*(dotProduct(w,training_x[i])+w0)< 1:
            idendity.append(1)
        else:
            idendity.append(0)
    
    '''Break condition'''
    if abs(Error-sum(d)) < theta:
        break
    
    '''choose best eta'''
    bestobj = 1000000000000 # infinity
    for eta in eta_list:
        '''update the w information for temp error'''
        for j in range(col_size):
            for i in range(training_size):
                w[j] += eta*idendity[i]*training_r[i]*training_x[i][j]
        for i in range(training_size):
            w0 += eta*idendity[i]*training_r[i]
        
        error=0
        for i in range(training_size):
            error += max(0,1-training_r[i]*(dotProduct(w,training_x[i])+w0))
        obj=error
        '''compare the obj and best obj than update the information'''
        if obj<bestobj:
            bestobj=obj
            best_eta=eta
        #print(obj)
        '''remove the eta effect'''
        for j in range(col_size):
            for i in range(training_size):
                w[j] -= eta*idendity[i]*training_r[i]*training_x[i][j]
        for i in range(training_size):
            w0 -= eta*idendity[i]*training_r[i]
    '''update w with best_eta'''
    for j in range(col_size):
        for i in range(training_size):
            w[j] += best_eta*idendity[i]*training_r[i]*training_x[i][j]
                
    for i in range(training_size):
        w0 += best_eta*idendity[i]*training_r[i]
    Error = sum(d)
    print(best_eta,Error)


'''update the dictionary'''
for i in missing_label:
    if dotProduct(f_Feature[i],w)+w0<0:
        f_Label[i]=0
    else:
        f_Label[i]=1

'''print the w and distance from the original'''
print("Hyperplane w' is:")
print("w'=",w)
print("")
print("Hyperplane's distance from origin:")
print("abs(w0/||w'||)=",abs(w0/normVector(w)))
print("")
print("w0=",w0)

if missing_label != []:
    print("Prediction and label is :")
    '''print for the prediction label'''
    for i in missing_label:
        print(f_Label[i],i)
