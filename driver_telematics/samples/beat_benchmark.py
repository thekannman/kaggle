##################################
# developed by rcarson
##################################
import os

# get the length of the vector
def getmo(v1):
    return (v1[0]**2+v1[1]**2)**0.5

# get the cos of two vectors    
def getangle(v1,v2):
    if getmo(v1)*getmo(v2)==0:
        return 0
    
    return sum(times(v1,v2))/(getmo(v1)*getmo(v2))

# multiply two vectors
def times(a,b):
    c=[]
    for i,j in zip(a,b):
        c.append(i*j)
    return c
# vector Euclidean distance    
def getD(x,m):
   # x=np.array(x)
   # m=np.array(m)
   # return sum((x-m)**2)**0.5
    s=0
    for i,j in zip(x,m):
        s+=(i-j)**2 
    return s**0.5

# get the mean vector of a driver        
def get_mean_vec(A):
   # s=np.zeros((1,len(A[0])))
   # for i in A:
   #     s+=np.array(i)
   # return s*1.0/len(A)
    s=[]
    for i in A[0]:
        s.append(i)
    for i in range(1,len(A)):
        for c,j in enumerate(A[i]):
            s[c]+=j
    return [i/len(A) for i in s]
# vector a-b    
def minus(a,b):
    c=[]
    for i,j in zip(a,b):
        c.append(i-j)
    return c

def nmean(a):
    return sum(a)/len(a)        
def file_to_score(files,dirx):
    feature={}
    for j in files:
        name=j[:-4]
        feature[name]={}
        feature[name]['vec']=[]
        feature[name]['angle']=[]
        vec=[]
        angle=[]
        #s=np.array(pd.read_csv(dirx+j))
        s=[]
        f=open(dirx+j)
        f.readline()
        for line in f:
            tmp=[]
            xx=line.split(',')
            tmp.append(float(xx[0]))
            tmp.append(float(xx[1]))
            s.append(tmp)
        f.close()
        for i in range(len(s)-1):
            vec.append(minus(s[i+1],s[i]))
        for i in range(len(vec)-1):
            angle.append(getangle(vec[i+1],vec[i]))
        for i in vec:
            feature[name]['vec'].append(getmo(i))
        feature[name]['angle']+=angle
    maxl=max([len(feature[i]['vec']) for i in feature])
    meanv=nmean([nmean(feature[i]['vec']) for i in feature])
    meana=nmean([nmean(feature[i]['angle']) for i in feature])
    for i in feature:
        while len(feature[i]['vec'])<maxl:
            feature[i]['vec'].append(meanv)
            feature[i]['angle'].append(meana)
    v=get_mean_vec([feature[i]['vec'] for i in feature])#[0]
    a=get_mean_vec([feature[i]['angle'] for i in feature])#[0]
    fea2={}
    for i in feature:
        fea2[i]={}
        fea2[i]['vec']=getD(feature[i]['vec'],v)
        fea2[i]['angle']=getD(feature[i]['angle'],a)
    score={}
    maxv=max([fea2[i]['vec'] for i in fea2])
    maxa=max([fea2[i]['angle'] for i in fea2])
    for i in fea2:
        score[i]=fea2[i]['vec']/(maxv/maxa*0.9)+fea2[i]['angle']
    maxs=max(score.values())
    for i in fea2:
        score[i]=1-score[i]/maxs
    return score
def score_to_csv(dirx,score,filen):
    f=open(filen,'a')
    if dirx=='1':
        f.write('driver_trip,prob\n')
    name=[int(i) for i in score.keys()]
    for i in sorted(name):
        f.write(dirx+'_'+str(i)+','+str(score[str(i)])+'\n')
    f.close()
dirs = [f for f in os.listdir('../data/drivers/')]
for i in sorted(dirs):
    files=[f for f in os.listdir('../data/drivers/'+str(i))]
    score=file_to_score(files,'../data/drivers/'+str(i)+'/')
    score_to_csv(str(i),score,'beat_benchmark.csv')
    print 'driver',i,'done'
