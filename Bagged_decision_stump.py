import sys
import random

'''Set the initial value'''
#eta=0.001
eta_list = [1, .1, .01, .001, .0001, .00001, .000001, .0000001, .00000001, .000000001, .0000000001, .00000000001 ]
bestobj = 1000000000000 # infinity
Error=10000000
theta=0.001
L=100

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

  
'''weight sample function'''
#pj=[1/len(training_x)]*len(training_x)
#def w_sample(x):
#    total_x=0
#    cumu_x=[]
#    for i in x:
#        total_x += i
#        cumu_x.append(total_x)
#    rnd=random.random()*total_x
#    for i in cumu_x:
#        if rnd < i:
#            return cumu_x.index(i)

'''Test for local data'''
#f_Feature=getFeatureData(r"C:\Users\user\Desktop\2018- NJIT Life\2019 fall\CS675 Mechine Learning\assignment\assignment3\testSVM.data")
#f_Label=getLabelData(r"C:\Users\user\Desktop\2018- NJIT Life\2019 fall\CS675 Mechine Learning\assignment\assignment3\testSVM.trainlabels")
#f_Feature=getFeatureData(r"C:\Users\user\Desktop\2018- NJIT Life\2019 fall\CS675 Mechine Learning\assignment\data\breast_cancer\breast_cancer.data")
#f_Label=getLabelData(r"C:\Users\user\Desktop\2018- NJIT Life\2019 fall\CS675 Mechine Learning\assignment\data\breast_cancer\breast_cancer.trainlabels.0")
#f_Feature=getFeatureData(r"C:\Users\user\Desktop\2018- NJIT Life\2019 fall\CS675 Mechine Learning\assignment\data\ionosphere\ionosphere.data")
#f_Label=getLabelData(r"C:\Users\user\Desktop\2018- NJIT Life\2019 fall\CS675 Mechine Learning\assignment\data\ionosphere\ionosphere.trainlabels.0")
#f_Feature=getFeatureData(r"C:\Users\user\Desktop\2018- NJIT Life\2019 fall\CS675 Mechine Learning\assignment\data\climate_simulation\climate.data")
#f_Label=getLabelData(r"C:\Users\user\Desktop\2018- NJIT Life\2019 fall\CS675 Mechine Learning\assignment\data\climate_simulation\climate.trainlabels.1")



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


'''Best split function'''
def best_split(x,r):
    col_size=len(x[0])
    row_size=len(x)
    output={"Best column":-1,"Best split s":10000}
    min_gini=100
    for j in range(col_size):
        gini=[]
        vector=unique(getVector(x,j))
        vector=[min(vector)-0.5]+vector
        vector.sort()
        for k in range(1,len(vector)-1):
            vector[k]=(vector[k]+vector[k+1])/2
            vector[len(vector)-1]=max(vector)+0.5
            '''if x[i][j] is split point'''
        for k in range(len(vector)):
            lsize=0;rsize=0;ln=0;rn=0
            for i in range(row_size):
                if  x[i][j] <= vector[k]:
                    lsize += 1
                    if r[i] == -1:
                        ln += 1
                elif  x[i][j] > vector[k]:
                    rsize +=1
                    if r[i] == -1:
                        rn += 1
            if lsize == 0:
                gini.append(rn/rsize*(1-rn/rsize))
            elif rsize == 0:
                gini.append(ln/lsize*(1-ln/lsize))
            else:
                gini.append(lsize/row_size*ln/lsize*(1-ln/lsize)+
                                rsize/row_size*rn/rsize*(1-rn/rsize))
        if min(gini) < min_gini:
            output["Best column"]=j
            output["Best split s"]=vector[gini.index(min(gini))]
            maj=0
            for i in range(training_size):
                if training_x[i][j]<vector[gini.index(min(gini))]:
                    maj += training_r[i]
            if maj <0:
                output["Mojority"]=-1
            else:
                output["Mojority"]= 1
            min_gini=min(gini)
    return(output)


vote={}
for i in missing_label:
    vote[i]=0
for l in range(L):          
    '''bootstrap'''
    boot_train=[]
    boot_r=[]
    for i in range(training_size):
        boot_index=random.randint(0,training_size-1)
        boot_train.append(training_x[boot_index])
        boot_r.append(training_r[boot_index])
    split=best_split(boot_train,boot_r)
    #print(split)
    col=split["Best column"]
    s=split["Best split s"]
    maj=split["Mojority"]
    for i in missing_label:
        if f_Feature[i][col]<s:
            vote[i] += maj
        else:
            vote[i] -= maj

for i in missing_label:
    if vote[i]<0:
        f_Label[i]=0
    else:
        f_Label[i]=1
for i in missing_label:
    print(i ,f_Label[i])