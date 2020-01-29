import sys
import random
import math


'''Set up the eta and the stoping condition'''
eta=0.01
theta=0.0000001

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

'''vector plus'''
def vectorPlus(u,v):
    if len(u) !=len(v):
        print("Error, u and v different length")
    else:
        M=[]
        for i in range(len(u)):
            M.append(u[i]+v[i])
        return M



'''Test for local data'''
#f_Feature=getFeatureData(r"C:\Users\user\Desktop\2018- NJIT Life\2019 fall\CS675 Mechine Learning\assignment\assignment4\test.data")
#f_Label=getLabelData(r"C:\Users\user\Desktop\2018- NJIT Life\2019 fall\CS675 Mechine Learning\assignment\assignment4\test.trainlabels")
f_Feature=getFeatureData(r"C:\Users\user\Desktop\2018- NJIT Life\2019 fall\CS675 Mechine Learning\assignment\assignment4\test2.data")
f_Label=getLabelData(r"C:\Users\user\Desktop\2018- NJIT Life\2019 fall\CS675 Mechine Learning\assignment\assignment4\test2.trainlabels")


#f_Feature=getFeatureData(r"C:\Users\user\Desktop\2018- NJIT Life\2019 fall\CS675 Mechine Learning\assignment\data\climate_simulation\climate.data")
#f_Label=getLabelData(r"C:\Users\user\Desktop\2018- NJIT Life\2019 fall\CS675 Mechine Learning\assignment\data\climate_simulation\climate.trainlabels.0")

#f_Feature=getFeatureData(r"C:\Users\user\Desktop\2018- NJIT Life\2019 fall\CS675 Mechine Learning\assignment\data\breast_cancer\breast_cancer.data")
#f_Label=getLabelData(r"C:\Users\user\Desktop\2018- NJIT Life\2019 fall\CS675 Mechine Learning\assignment\data\breast_cancer\breast_cancer.trainlabels.0")
#f_Feature=getFeatureData(r"C:\Users\user\Desktop\2018- NJIT Life\2019 fall\CS675 Mechine Learning\assignment\data\ionosphere\ionosphere.data")
#f_Label=getLabelData(r"C:\Users\user\Desktop\2018- NJIT Life\2019 fall\CS675 Mechine Learning\assignment\data\ionosphere\ionosphere.trainlabels.0")


'''data for assignment input'''
#dfilename=sys.argv[1]
#dlabername=sys.argv[2]

#f_Feature=getFeatureData(dfilename)
#f_Label=getLabelData(dlabername)

'''define col number, data size and training size'''
col_size=len(f_Feature[0])+1
data_size=len(f_Feature)
training_size=len(f_Label)


'''compute training_x, training_r and missing label  '''
missing_label=list(range(data_size))
training_x=[]
training_r=[]
for i in range(data_size):
    if f_Label.get(i)==1 or f_Label.get(i)==0:
        training_x.append([1]+f_Feature[i])
        training_r.append(f_Label[i])
        missing_label.remove(i)


'''Initialize w' and  w0'''
w=[]
for j in range(col_size):
    w.append(random.random()*0.002-0.01)
#w=[0.001]*col_size


''' logist regression '''
Error=10000000
for k in range(100000):
    '''compute the negative log likelihood vector '''
    d=[]
    '''Since the negative log likelihood use too much dotproduct with same thing'''
    '''create a vector to storage all dotproduct object'''
    dotobj=[]
    for i in range(training_size):
        dotobj.append(dotProduct(w,training_x[i]))
    for i in range(training_size):
        d.append(training_r[i]*math.log(1+math.exp(-dotobj[i]))
                 -(1-training_r[i])*math.log(math.exp(-dotobj[i]))
                 +(1-training_r[i])*math.log(1+math.exp(-dotobj[i])))
#        d.append(math.log(1+math.e**(-training_r[i]*dotobj[i])))

    '''Break condition'''
    if abs(Error-sum(d)) < theta:
        break
    
    '''Create delta delta_w vector'''
    delta_w=[]
    for j in range(col_size):
        aa=0
        for i in range(training_size):
            aa += eta*(training_r[i]-1/(1+math.e**(-dotobj[i])))*training_x[i][j]
        delta_w.append(aa)
        
    
    
    '''update the w information and update the Error'''
    w=vectorPlus(w,delta_w)
    Error = sum(d)
    


'''update the dictionary'''
for i in missing_label:
    if 1/(1+math.e**(-dotProduct([1]+f_Feature[i],w)))<0.5:
        f_Label[i]=0
    else:
        f_Label[i]=1

'''print the w and distance from the original'''
print("Hyperplane w' is:")
print("w'=",w[1:])
print("")
print("||w'||=", normVector(w[1:]))
print("")
print("Hyperplane's distance from origin:")
print("abs(w0/||w'||)=",abs(w[0]/normVector(w[1:])))
print("")
#print("w0=",w[0])

if missing_label != []:
    print("Prediction and label is :")
    '''print for the prediction label'''
    for i in missing_label:
        print(f_Label[i],i)
