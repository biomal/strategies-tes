#Imported libraries
import pandas as pd #For datasets
import statistics #For mean and standard deviation
import os #Create dir
from sklearn.neighbors import KNeighborsClassifier #knn
from sklearn import svm #svm
from sklearn.neural_network import MLPClassifier #mlp
from sklearn.naive_bayes import GaussianNB #nb
from sklearn import tree #dtree
from sklearn.ensemble import RandomForestClassifier #rforest

#Lambdas
diff = lambda l1,l2: list([x for x in l1 if x not in l2]) #diff between lists
uni = lambda l1, l2: list(set().union(l1,l2)) #union of lists
intersect = lambda l1, l2: [val for val in l1 if val in l2] #intersection between lists
grep = lambda s,c: [pos for pos, char in enumerate(s) if char == c] #returns the postions of char (c) inside a string (s)
maxRow = lambda dataset,row : [dataset.columns.tolist()[i] for i, j in enumerate(dataset.iloc[row]) if j == max(dataset.iloc[row])] #returns the columns names that contains the maximum value in each line from the dataset

#Verifies the existance of directory (dirA) and, if it doenst exist, creates it
def createsDir(dirA):
    if not os.path.exists(dirA):
        os.makedirs(dirA)

#Function that returns the ancestors of node, given a set of nodes
#Goes though the list of nodes and, for each one, verifies if the nodes substring (size s) is equal to the current node
def ancestor(node,nodes):
    ret = []
    for i in range(0,len(nodes)):
        if (len(nodes[i])<len(node)) & (node[0:(len(nodes[i])+1)]==nodes[i]+"."):
            ret.append(nodes[i])
    return list(set(ret))

#Function that returns the ancestors of node, from all possible nodes
#Verifies all the ancestors by making substring of dots "."
def ancestorTotal(node):
    ret = []
    pos=grep(node,".")
    while len(pos)>0:
        node=node[0:pos[len(pos)-1]]
        ret.append(node)
        pos=grep(node,".")
    return ret

#Function that returns the descedants of node, given a set of nodes
#Similar to the ancestor function
def descendant(node,nodes):
    ret = []
    for i in range(0,len(nodes)):
        if (len(nodes[i])>len(node)) & (nodes[i][0:len(node)+1]==node+"."):
            ret.append(nodes[i])
    return list(set(ret))

#Function that returns the siblings of node, given a set of nodes
def siblings(node,nodes):
    ret=[]
    pos=grep(node,".")
    if(len(pos)==0):
        parent="root"
    else:
        parent=node[0:pos[len(pos)-1]]
    for i in range(0,len(nodes)):
        pos=grep(nodes[i],".")
        if(len(pos)==0):
            secondParent="root"
        else:
            secondParent=nodes[i][0:pos[len(pos)-1]]
        if((parent==secondParent)&(node!=nodes[i])):
            ret.append(nodes[i])
    return list(set(ret))

def anotherSiblings(node,nodes):
    ret=[]
    ancesNode=ancestorTotal(node)
    for i in range(0,len(nodes)):
        ancesOtherNode=ancestorTotal(nodes[i])
        if((set(ancesNode)==set(ancesOtherNode))&(nodes[i]!=node)):
            ret.append(nodes[i])
        elif(set(ancesNode)<set(ancesOtherNode)):
            pos=len(ancesNode)
            if(ancesOtherNode[len(ancesOtherNode)-pos-1]!=node):
                ret.append(ancesOtherNode[len(ancesOtherNode)-pos-1])
    return list(set(ret))

#Function that returns the siblings and descendants of node, given a set of nodes
def siblingsDescendant(node,nodes):
    ret=[]
    sib=anotherSiblings(node,nodes)
    for i in range(0,len(sib)):
        ret.append(sib[i])
        desc=descendant(sib[i],nodes)
        for j in range(0,len(desc)):
            ret.append(desc[j])
    return list(set(ret))

#Function that returns the cousins of node, given a set of nodes
def cousins(node, nodes):
    ret=[]
    for i in range(0,len(nodes)):
        if((len(grep(nodes[i],"."))==len(grep(node,".")))&(nodes[i]!=node)):
            ret.append(nodes[i])
    return list(set(ret))

#Function that returns the cousins and descedants of node, given a set of nodes
def cousinsDescendant(node, nodes):
    ret=[]
    cos=cousins(node,nodes)
    for i in range(0,len(cos)):
        ret.append(cos[i])
        desc=descendant(cos[i],nodes)
        for j in range(0,len(desc)):
            ret.append(desc[j])
    return list(set(ret))

#Correction in case of issues with float/string
def correction(data):
    for i in range(0,data.shape[0]):
        data.loc[i,'classification']=str(data.loc[i,'classification'])
    return data

#Runs the ML algorithms and saves the probabilities in "prob-fold"
def runAlgorithms(train,test,alg,dirOutputClassifier,currentFolder,nbClasses):
    x=train.copy()
    y=train.copy()
    del x['classification'] #x = train dataset without 'classification' column
    y=y['classification'] #y = 'classification' column
    if alg == "knn":
        clf=KNeighborsClassifier(n_neighbors=5)
        clf.fit(x,y) #treina
        result=pd.DataFrame(clf.predict_proba(test)) #probabilities dataframe
        dirOutputClassifier=dirOutputClassifier+"knn/"
        createsDir(dirOutputClassifier)
        result.to_csv(dirOutputClassifier+"prob-fold"+str(currentFolder)+".txt")
    elif alg == "svm":
        clf=svm.SVC(probability=True)
        clf.fit(x,y)
        result=pd.DataFrame(clf.predict_proba(test))
        dirOutputClassifier=dirOutputClassifier+"svm/"
        createsDir(dirOutputClassifier)
        result.to_csv(dirOutputClassifier+"prob-fold"+str(currentFolder)+".txt")
    elif alg == "mlp":
        clf=MLPClassifier()
        clf.fit(x,y)
        result=pd.DataFrame(clf.predict_proba(test))
        dirOutputClassifier=dirOutputClassifier+"mlp/"
        createsDir(dirOutputClassifier)
        result.to_csv(dirOutputClassifier+"prob-fold"+str(currentFolder)+".txt")
    elif alg == "mlp2":
        clf=MLPClassifier(activation = 'logistic',solver='adam', hidden_layer_sizes=(200,))
        clf.fit(x,y)
        result=pd.DataFrame(clf.predict_proba(test))
        dirOutputClassifier=dirOutputClassifier+"mlp2/"
        createsDir(dirOutputClassifier)
        result.to_csv(dirOutputClassifier+"prob-fold"+str(currentFolder)+".txt")
    elif alg == "mlp3":
        clf=MLPClassifier(activation = 'relu',solver='adam',hidden_layer_sizes=(len(x.columns),1),max_iter=600)
        clf.fit(x,y)
        result=pd.DataFrame(clf.predict_proba(test))
        dirOutputClassifier=dirOutputClassifier+"mlp3/"
        createsDir(dirOutputClassifier)
        result.to_csv(dirOutputClassifier+"prob-fold"+str(currentFolder)+".txt")
    elif alg == "mlp4":
        clf=MLPClassifier(activation = 'relu',solver='adam',hidden_layer_sizes=(len(x.columns),2),max_iter=600)
        clf.fit(x,y)
        result=pd.DataFrame(clf.predict_proba(test))
        dirOutputClassifier=dirOutputClassifier+"mlp4/"
        createsDir(dirOutputClassifier)
        result.to_csv(dirOutputClassifier+"prob-fold"+str(currentFolder)+".txt")
    elif alg == "mlp5":
        clf=MLPClassifier(activation = 'relu',solver='adam',hidden_layer_sizes=(nbClasses,1),max_iter=600)
        clf.fit(x,y)
        result=pd.DataFrame(clf.predict_proba(test))
        dirOutputClassifier=dirOutputClassifier+"mlp5/"
        createsDir(dirOutputClassifier)
        result.to_csv(dirOutputClassifier+"prob-fold"+str(currentFolder)+".txt")
    elif alg == "mlp6":
        clf=MLPClassifier(activation = 'relu',solver='adam',hidden_layer_sizes=(nbClasses,2),max_iter=600)
        clf.fit(x,y)
        result=pd.DataFrame(clf.predict_proba(test))
        dirOutputClassifier=dirOutputClassifier+"mlp6/"
        createsDir(dirOutputClassifier)
        result.to_csv(dirOutputClassifier+"prob-fold"+str(currentFolder)+".txt")
	elif alg == "nb":
        clf=GaussianNB()
        clf.fit(x,y)
        result=pd.DataFrame(clf.predict_proba(test))
        dirOutputClassifier=dirOutputClassifier+"nb/"
        createsDir(dirOutputClassifier)
        result.to_csv(dirOutputClassifier+"prob-fold"+str(currentFolder)+".txt")
    elif alg == "dtree":
        clf = tree.DecisionTreeClassifier()
        clf.fit(x,y)
        result=pd.DataFrame(clf.predict_proba(test))
        dirOutputClassifier=dirOutputClassifier+"dtree/"
        createsDir(dirOutputClassifier)
        result.to_csv(dirOutputClassifier+"prob-fold"+str(currentFolder)+".txt")
    elif alg == "rforest":
        clf = RandomForestClassifier()
        clf.fit(x,y)
        result=pd.DataFrame(clf.predict_proba(test))
        dirOutputClassifier=dirOutputClassifier+"rforest/"
        createsDir(dirOutputClassifier)
        result.to_csv(dirOutputClassifier+"prob-fold"+str(currentFolder)+".txt")

#Creates the datasets according to each strategy and runs ML
def executes(mainDir,strategy,nFolds,data,alg,should_correct=False):
    dirDatasets=mainDir+"Datasets/"+data+"/"
    dirOutputClassifier=mainDir+"Results/"+data+"/"
    createsDir(dirOutputClassifier)
    dirOutputClassifier=dirOutputClassifier+strategy
    createsDir(dirOutputClassifier)
    trainSave=pd.read_csv(dirDatasets+"train"+str(1)+".csv")
    if should_correct==True:
        trainSave=correction(trainSave)
    nodes=trainSave['classification'].unique() #contains the nodes
    for currentFolder in range(1,nFolds+1): #fold by fold
        if should_correct==True:
            trainSave=pd.read_csv(dirDatasets+"train"+str(currentFolder)+".csv",low_memory=False) #saves the train dataset     
            testSave=pd.read_csv(dirDatasets+"test"+str(currentFolder)+".csv",low_memory=False) #saves the test dataset
            trainSave=correction(trainSave)
        else:
            trainSave=pd.read_csv(dirDatasets+"train"+str(currentFolder)+".csv") #saves the train dataset     
            testSave=pd.read_csv(dirDatasets+"test"+str(currentFolder)+".csv") #saves the test dataset
        if (data=="Mips") | (data=="Repbase"):
            del trainSave['id']
            del testSave['id']
        del testSave['classification'] #testSave doesnt contain 'classification' column
        for currentNode in range(1,len(nodes)+1): #node by node
            train=trainSave.copy() 
            test=testSave.copy() 
            dirOutputClassifier=dirOutputClassifier+"Class"+str(currentNode)+"/"
            createsDir(dirOutputClassifier)
            if strategy=="Exclusive_Strategy/":
                positives = [nodes[currentNode-1]]
                negatives = diff(nodes,positives)
            elif strategy=="Inclusive_Strategy/":
                positives = uni([nodes[currentNode-1]],descendant(nodes[currentNode-1],nodes))
                negatives = diff(nodes,uni(positives,ancestor(nodes[currentNode-1],nodes)))
            elif strategy=="Less_Exclusive_Strategy/":
                positives = [nodes[currentNode-1]]
                negatives = diff(nodes,uni(positives,descendant(nodes[currentNode-1],nodes)))
			elif strategy=="Less_Inclusive_Strategy/":
                positives = uni([nodes[currentNode-1]],descendant(nodes[currentNode-1],nodes))
                negatives = diff(nodes,positives)
            elif strategy=="Siblings_Strategy/":
                positives = uni([nodes[currentNode-1]],descendant(nodes[currentNode-1],nodes))
                negatives = siblingsDescendant(nodes[currentNode-1],nodes)
            elif strategy=="Exclusive_Siblings_Strategy/":
                positives = uni([nodes[currentNode-1]],descendant(nodes[currentNode-1],nodes))
                negatives = anotherSiblings(nodes[currentNode-1],nodes)
            elif strategy=="Cousins_Strategy/":
                positives = uni([nodes[currentNode-1]],descendant(nodes[currentNode-1],nodes))
                negatives = cousinsDescendant(nodes[currentNode-1],nodes)
            elif strategy=="Exclusive_Cousins_Strategy/":
                positives = uni([nodes[currentNode-1]],descendant(nodes[currentNode-1],nodes))
                negatives = cousins(nodes[currentNode-1],nodes)
            positivesTotal="^"+positives[0]+"$"
            for i in range(1,len(positives)):
                positivesTotal=positivesTotal+"|^"+positives[i]+"$"
            negativesTotal="^"+negatives[0]+"$"
            for i in range(1,len(negatives)):
                negativesTotal=negativesTotal+"|^"+negatives[i]+"$"
            currentTrain=train.copy()  #se nao, vai alterar o train
            currentTrain=currentTrain[train['classification'].str.contains(positivesTotal)]
            currentTrain.loc[:,'classification']='1'
            currentTrain2=train.copy()
            currentTrain2=currentTrain2[train['classification'].str.contains(negativesTotal)]
            currentTrain2.loc[:,'classification']='0'
            train=pd.concat([currentTrain,currentTrain2])
            train['classification']=train['classification'].astype('category')
            del currentTrain
            del currentTrain2
            runAlgorithms(train,test,alg,dirOutputClassifier,currentFolder,len(nodes))
            print("Rodou "+alg+" fold "+str(currentFolder)+" node "+str(currentNode)+" strategy "+strategy)   
            dirOutputClassifier=mainDir+"Results/"+data+"/"+strategy

#Creates the probabilities matrix and calculates the measures per fold
def result(mainDir,strategy,nFolds,data,alg,should_correct=False):
    dirDatasets=mainDir+"Datasets/"+data+"/"
    dirOutputResults=mainDir+"Results/"+data+"/"+strategy+"result/"
    createsDir(dirOutputResults)
    dirOutputResults=dirOutputResults+alg+"/"
    createsDir(dirOutputResults)
    train=pd.read_csv(dirDatasets+"train"+str(1)+".csv")
    if should_correct==True:
        train=correction(train)
    nodes=list(train['classification'].unique())
    nNodes=len(nodes)
    for fold in range(1,nFolds+1):
        i=0
        Class=pd.read_csv(mainDir+"Results/"+data+"/"+strategy+"Class"+str(i+1)+"/"+alg+"/prob-fold"+str(fold)+".txt")
        Results = pd.DataFrame(index=range(0,Class.shape[0]),columns=nodes, dtype='object')
        for i in range(0,nNodes):
            Class=pd.read_csv(mainDir+"Results/"+data+"/"+strategy+"Class"+str(i+1)+"/"+alg+"/prob-fold"+str(fold)+".txt")
            Results.loc[:,Results.columns.tolist()[i]]=Class.loc[:,'1']
        Results.to_csv(dirOutputResults+"matrizprob-fold"+str(fold)+".txt")
        Ret=Results.copy()
        for i in range(0,Ret.shape[1]):
            parents=ancestor(nodes[i],nodes)
            if(len(parents)>0):
                for j in range(0,Results.shape[0]):
                    value=[]
                    value.append(float(Results.iloc[j,[i]]))
                    for k in range(0,len(parents)):
                        value.append(float(Results.loc[j,parents[k]]))
                    Ret.iloc[j,[i]]=sum(value)/len(value)
        #Ret contains the adjusted probabilities                            
        Results=pd.DataFrame(index=range(0,Ret.shape[0]),columns=["1"])
        for i in range(0,Ret.shape[0]):
            t=maxRow(Ret,i)
            value=[]
            for j in range(0,len(t)):
                value.append(train['classification'].value_counts()[t[j]])
            Results.iloc[i,[0]]=t[value.index(max(value))]
        Results.to_csv(dirOutputResults+"clasfResult-fold"+str(fold)+".txt")
        test=pd.read_csv(dirDatasets+"test"+str(fold)+".csv")
        if should_correct==True:
            test=correction(test)
        hPD,hRD,hPN,hRN=[0,0,0,0]
        for i in range(0,Results.shape[0]):
            real=test.loc[i,"classification"]
            predicted=list(Results.iloc[i,[0]])
            predictedA=uni(predicted,ancestorTotal(predicted[0]))
            realA=uni([real],ancestorTotal(real))
            hPD=hPD+len(predictedA)
            hRD=hRD+len(realA)
            value=intersect(predictedA,realA)
            hPN=hPN+len(value)
            hRN=hRN+len(value)
        hP=float(hPN)/float(hPD)
        hR=float(hRN)/float(hRD)
        hF=(2*hP*hR)/(hP+hR)
        matriz=pd.DataFrame(index=range(0,1),columns=["hP","hR","hF"])
        matriz.iloc[0]=[hP,hR,hF]
        matriz.to_csv(dirOutputResults+"measures-fold"+str(fold)+".txt")

#Calculates the final values (combination of all folds)
def finalValues(mainDir,strategy,nFolds,data,alg):
    dirOriginResults=mainDir+"Results/"+data+"/"+strategy+"result/"+alg+"/"
    dirOutputFinalValues=mainDir+"Results/"+data+"/"+strategy+"finalValues/"
    createsDir(dirOutputFinalValues)
    measures=pd.DataFrame(index=range(0,1),columns=["hP","sdhP","hR","sdhR","hF","sdhF"])
    hPtotal=[]
    hRtotal=[]
    hFtotal=[]
    for fold in range(1,nFolds+1):
		tabela=pd.read_csv(dirOriginResults+"measures-fold"+str(fold)+".txt")
        hPtotal.append(float(tabela.loc[0,"hP"]))
        hRtotal.append(float(tabela.loc[0,"hR"]))
        hFtotal.append(float(tabela.loc[0,"hF"]))
    measures.loc[0,"hP"]=statistics.mean(hPtotal)
    measures.loc[0,"sdhP"]=statistics.stdev(hPtotal)
    measures.loc[0,"hR"]=statistics.mean(hRtotal)
    measures.loc[0,"sdhR"]=statistics.stdev(hRtotal)
    measures.loc[0,"hF"]=statistics.mean(hFtotal)
    measures.loc[0,"sdhF"]=statistics.stdev(hFtotal)
    measures.to_csv(dirOutputFinalValues+"values-"+alg+".txt")

#If the execution for some reasons stops even though not finished yet
def executesBroken(mainDir,strategy,nFolds,data,alg,foldBeg,foldEnd,classBeg,should_correct=False):
    dirDatasets=mainDir+"Datasets/"+data+"/"
    dirOutputClassifier=mainDir+"Results/"+data+"/"
    createsDir(dirOutputClassifier)
    dirOutputClassifier=dirOutputClassifier+strategy
    createsDir(dirOutputClassifier)
    trainSave=pd.read_csv(dirDatasets+"train"+str(1)+".csv")
    if should_correct==True:
        trainSave=correction(trainSave)
    nodes=trainSave['classification'].unique()
    for currentFolder in range(foldBeg,foldEnd): 
        if should_correct==True:
            trainSave=pd.read_csv(dirDatasets+"train"+str(currentFolder)+".csv",low_memory=False)    
            testSave=pd.read_csv(dirDatasets+"test"+str(currentFolder)+".csv",low_memory=False) 
            trainSave=correction(trainSave)
        else:
            trainSave=pd.read_csv(dirDatasets+"train"+str(currentFolder)+".csv")     
            testSave=pd.read_csv(dirDatasets+"test"+str(currentFolder)+".csv") 
        if (data=="Mips") | (data=="Repbase"):
            del trainSave['id']
            del testSave['id']
        del testSave['classification'] 
        for currentNode in range(classBeg,len(nodes)+1): 
            train=trainSave.copy() 
			test=testSave.copy() 
            dirOutputClassifier=dirOutputClassifier+"Class"+str(currentNode)+"/"
            createsDir(dirOutputClassifier)
            if strategy=="Exclusive_Strategy/":
                positives = [nodes[currentNode-1]]
                negatives = diff(nodes,positives)
            elif strategy=="Inclusive_Strategy/":
                positives = uni([nodes[currentNode-1]],descendant(nodes[currentNode-1],nodes))
                negatives = diff(nodes,uni(positives,ancestor(nodes[currentNode-1],nodes)))
            elif strategy=="Less_Exclusive_Strategy/":
                positives = [nodes[currentNode-1]]
                negatives = diff(nodes,uni(positives,descendant(nodes[currentNode-1],nodes)))
            elif strategy=="Less_Inclusive_Strategy/":
                positives = uni([nodes[currentNode-1]],descendant(nodes[currentNode-1],nodes))
                negatives = diff(nodes,positives)
            elif strategy=="Siblings_Strategy/":
                positives = uni([nodes[currentNode-1]],descendant(nodes[currentNode-1],nodes))
                negatives = siblingsDescendant(nodes[currentNode-1],nodes)
            elif strategy=="Exclusive_Siblings_Strategy/":
                positives = uni([nodes[currentNode-1]],descendant(nodes[currentNode-1],nodes))
                negatives = anotherSiblings(nodes[currentNode-1],nodes)
            elif strategy=="Cousins_Strategy/":
                positives = uni([nodes[currentNode-1]],descendant(nodes[currentNode-1],nodes))
                negatives = cousinsDescendant(nodes[currentNode-1],nodes)
            elif strategy=="Exclusive_Cousins_Strategy/":
                positives = uni([nodes[currentNode-1]],descendant(nodes[currentNode-1],nodes))
                negatives = cousins(nodes[currentNode-1],nodes)
            positivesTotal="^"+positives[0]+"$"
            for i in range(1,len(positives)):
                positivesTotal=positivesTotal+"|^"+positives[i]+"$"
            negativesTotal="^"+negatives[0]+"$"
            for i in range(1,len(negatives)):
                negativesTotal=negativesTotal+"|^"+negatives[i]+"$"
            currentTrain=train.copy() 
            currentTrain=currentTrain[train['classification'].str.contains(positivesTotal)]
            currentTrain.loc[:,'classification']='1'
            currentTrain2=train.copy()
            currentTrain2=currentTrain2[train['classification'].str.contains(negativesTotal)]
            currentTrain2.loc[:,'classification']='0'
            train=pd.concat([currentTrain,currentTrain2])
            train['classification']=train['classification'].astype('category')
            del currentTrain
            del currentTrain2
            runAlgorithms(train,test,alg,dirOutputClassifier,currentFolder,len(nodes))
            dirOutputClassifier=mainDir+"Results/"+data+"/"+strategy
