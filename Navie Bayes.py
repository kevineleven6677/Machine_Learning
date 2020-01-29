import sys


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

''' Get the mean list from input'''

def mean_function(x):
    
    if type(x[0]) != list:
        b=0
        for i in range(len(x)):
            b += x[i]/len(x)
    else:
        b=[]
        for j in range(len(x[0])):
            a=0.
            for i in range(len(x)):
                a += x[i][j]/len(x)
            b.append(a)
    return b
'''Get the std list from input'''
def std_function(x):
    mx=mean_function(x)
    if type(x[0]) !=list:
        b=0
        for i in range(len(x)):
            b += (x[i]-mx)**2/(len(x))
        return b**(1/2)
    else:
        b=[]
        for j in range(len(x[0])):
            a=0.
            for i in range(len(x)):
                a += (x[i][j]-mx[j])**2/(len(x))
            b.append(a**(1/2))
        return b

'''defined a function to get the Naviebayes value under data class is y'''
def NavieBayes(x,y):
    my=mean_function(y)
    stdy=std_function(y)
    b=0
    for j in range(len(x)):
        b+= ((x[j]-my[j])/stdy[j])**2
    return b


'''Test for the local data'''
#f_Feature=getFeatureData(r"C:\Users\user\Desktop\2018- NJIT Life\2019 fall\CS675 Mechine Learning\assignment\data\breast_cancer\breast_cancer.data")
#f_Label=getLabelData(r"C:\Users\user\Desktop\2018- NJIT Life\2019 fall\CS675 Mechine Learning\assignment\data\breast_cancer\breast_cancer.trainlabels.0")

'''data for assignment input'''
dfilename=sys.argv[1]
dlabername=sys.argv[2]

f_Feature=getFeatureData(dfilename)
f_Label=getLabelData(dlabername)



data_size= len(f_Feature)
missing_label=list(range(data_size))

data_0=[];data_1=[]
for i in range(data_size):
    if f_Label.get(i)==0:
        data_0.append(f_Feature[i])
        missing_label.remove(i)
    elif f_Label.get(i)==1:
        data_1.append(f_Feature[i])
        missing_label.remove(i)


for i in missing_label:
    if NavieBayes(f_Feature[i],data_0)< NavieBayes(f_Feature[i],data_1):
        f_Label[i]=0
    else:
        f_Label[i]=1
    
for i in missing_label:
    print(f_Label[i],i)

