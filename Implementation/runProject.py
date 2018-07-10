from mainProject import *

mainDir="/home/2017/data"
strategies=["Exclusive_Strategy/","Inclusive_Strategy/","Less_Exclsuive_Strategy/","Less_Inclusive_Strategy/"]
nFolds=10
data="Seq"
algs=["mlp6","mlp4","mlp5","mlp3"]

for i in range(0,len(strategies)):
        for j in range(0,len(algs)):
                executes(mainDir,strategies[i],nFolds,data,algs[j])
                result(mainDir,strategies[i],nFolds,data,algs[j])
                finalValues(mainDir,strategies[i],nFolds,data,algs[j])
