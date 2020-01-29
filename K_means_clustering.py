import sys
import random

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

'''cluster choose function'''
def c_cluster(x,c,k):
    output=[]
    for i in range(len(x)):
        if c[i]==k:
            output.append(x[i])
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
        
'''Test for local data'''
#f_Feature=getFeatureData(r"C:\Users\user\Desktop\2018- NJIT Life\2019 fall\CS675 Mechine Learning\assignment\assignment3\testSVM.data")
#f_Label=getLabelData(r"C:\Users\user\Desktop\2018- NJIT Life\2019 fall\CS675 Mechine Learning\assignment\assignment3\testSVM.trainlabels")
#f_Feature=getFeatureData(r"C:\Users\user\Desktop\2018- NJIT Life\2019 fall\CS675 Mechine Learning\assignment\data\breast_cancer\breast_cancer.data")
#f_Label=getLabelData(r"C:\Users\user\Desktop\2018- NJIT Life\2019 fall\CS675 Mechine Learning\assignment\data\breast_cancer\breast_cancer.trainlabels.0")
#f_Feature=getFeatureData(r"C:\Users\user\Desktop\2018- NJIT Life\2019 fall\CS675 Mechine Learning\assignment\data\ionosphere\ionosphere.data")
#f_Label=getLabelData(r"C:\Users\user\Desktop\2018- NJIT Life\2019 fall\CS675 Mechine Learning\assignment\data\ionosphere\ionosphere.trainlabels.0")
#f_Feature=getFeatureData(r"C:\Users\user\Desktop\2018- NJIT Life\2019 fall\CS675 Mechine Learning\assignment\data\climate_simulation\climate.data")
#f_Label=getLabelData(r"C:\Users\user\Desktop\2018- NJIT Life\2019 fall\CS675 Mechine Learning\assignment\data\climate_simulation\climate.trainlabels.1")
#cluster_size=3


'''data for assignment input'''
dfilename=sys.argv[1]
cluster_size=int(sys.argv[2])
f_Feature=getFeatureData(dfilename)


'''define col number, data size and training size'''
col_size=len(f_Feature[0])
data_size=len(f_Feature)
#training_size=len(f_Label)



'''initialize the cluster'''
cluster=[]
for i in range(data_size):
    cluster.append(random.randint(0,cluster_size-1))


'''initialize the mj'''
mj=[]
for k in range(cluster_size):
    mj.append(col_mean(c_cluster(f_Feature,cluster,k),col_size))
'''initialize the objective'''
obj=0
for k in range(cluster_size):
    sub_data=c_cluster(f_Feature,cluster,k)
    sub_size=len(sub_data)
    for i in range(sub_size):
        obj += normVector(vectorMinus(sub_data[i],mj[k]))**2
        
'''Recompute cluster'''
for i in range(data_size):
    recompute=[]
    for k in range(cluster_size):
        recompute.append(normVector(vectorMinus(f_Feature[i],mj[k])))
    cluster[i]=recompute.index(min(recompute))
''' for loop'''
for h in range(10000000):
    mj=[]
    for k in range(cluster_size):
        mj.append(col_mean(c_cluster(f_Feature,cluster,k),col_size))
    newobj=0
    for k in range(cluster_size):
        sub_data=c_cluster(f_Feature,cluster,k)
        sub_size=len(sub_data)
        if sub_size>0:
            for i in range(sub_size):
                newobj += normVector(vectorMinus(sub_data[i],mj[k]))**2
    if abs(newobj-obj) == 0 :
        break
    obj=newobj
    for i in range(data_size):
        recompute=[]
        for k in range(cluster_size):
            recompute.append(normVector(vectorMinus(f_Feature[i],mj[k])))
        cluster[i]=recompute.index(min(recompute))
        
'''sort the cluster let the lager cluster abtain smaller index'''
cluster_list=[]
for k in range(cluster_size):
    cluster_list.append([])
for k in range(cluster_size):
    for i in range(data_size):
        if cluster[i]==k:
            cluster_list[k].append(i)
cluster_list.sort(key=len,reverse=True)

for k in range(cluster_size):
    for i in cluster_list[k]:
        cluster[i]=k




for i in range(data_size):
    print(cluster[i],i)


for k in range(cluster_size):
   print("The cluster",k," have ",cluster.count(k)," points") 