#Bibliotecas importadas
import pandas as pd #Para os datasets
import statistics #Para media e desvio-padrao
import os #Para criar diretorios
from sklearn.neighbors import KNeighborsClassifier #knn
from sklearn import svm #svm
from sklearn.neural_network import MLPClassifier #mlp
from sklearn.naive_bayes import GaussianNB #nb
from sklearn import tree #dtree
from sklearn.ensemble import RandomForestClassifier #rforest

#Lambdas utilizados
diff = lambda l1,l2: list([x for x in l1 if x not in l2]) #diferenca de listas
uni = lambda l1, l2: list(set().union(l1,l2)) #uniao de listas
intersect = lambda l1, l2: [val for val in l1 if val in l2] #interseccao de listas
grep = lambda s,c: [pos for pos, char in enumerate(s) if char == c] #retorna as posicoes do char (c) numa string (s)
maxRow = lambda dataset,row : [dataset.columns.tolist()[i] for i, j in enumerate(dataset.iloc[row]) if j == max(dataset.iloc[row])] #retorna os nomes das colunas que contem o maximo valor na linha do dataset

#Verifica a existencia do diretorio (dirA) e se nao houver, cria
def criaDir(dirA):
    if not os.path.exists(dirA):
        os.makedirs(dirA)

#Funcao que retorna os ancestrais de node, dentre os nodes possiveis
#Percorre a lista de nodes e a cada um, verifica se o node como substring do tamanho eh igual ao node atual, se sim, eh ancestral
def ancestor(node,nodes):
    resp = []
    for i in range(0,len(nodes)):
        if (len(nodes[i])<len(node)) & (node[0:(len(nodes[i])+1)]==nodes[i]+"."):
            resp.append(nodes[i])
    return list(set(resp))

#Funcao que retorna os ancestrais de node, dentro todos os possiveis
#Verifica todos os ancestrais possiveis de node, fazendo substring dos pontos
def ancestorTotal(node):
    resp = []
    pos=grep(node,".")
    while len(pos)>0:
        node=node[0:pos[len(pos)-1]]
        resp.append(node)
        pos=grep(node,".")
    return resp

#Funcao que retorna os descendentes de node, dentre os nodes possiveis
#Semelhante a ancestor, mas ao contrario
def descendant(node,nodes):
    resp = []
    for i in range(0,len(nodes)):
        if (len(nodes[i])>len(node)) & (nodes[i][0:len(node)+1]==node+"."):
            resp.append(nodes[i])
    return list(set(resp))

#Funcao que retorna os irmaos de node, dentre os nodes possiveis
def siblings(node,nodes):
    resp=[]
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
            resp.append(nodes[i])
    return list(set(resp))

def anotherSiblings(node,nodes):
    resp=[]
    ancesNode=ancestorTotal(node)
    for i in range(0,len(nodes)):
        ancesOtherNode=ancestorTotal(nodes[i])
        if((set(ancesNode)==set(ancesOtherNode))&(nodes[i]!=node)):
            resp.append(nodes[i])
        elif(set(ancesNode)<set(ancesOtherNode)):
            pos=len(ancesNode)
            if(ancesOtherNode[len(ancesOtherNode)-pos-1]!=node):
                resp.append(ancesOtherNode[len(ancesOtherNode)-pos-1])
    return list(set(resp))

#Funcao que retorna os irmaos e descendentes de irmaos de node, dentre os nodes possiveis
def siblingsDescendant(node,nodes):
    resp=[]
    sib=anotherSiblings(node,nodes)
    for i in range(0,len(sib)):
        resp.append(sib[i])
        desc=descendant(sib[i],nodes)
        for j in range(0,len(desc)):
            resp.append(desc[j])
    return list(set(resp))

#Funcao que retorna os primos de cada node
def cousins(node, nodes):
    resp=[]
    for i in range(0,len(nodes)):
        if((len(grep(nodes[i],"."))==len(grep(node,".")))&(nodes[i]!=node)):
            resp.append(nodes[i])
    return list(set(resp))

def cousinsDescendant(node, nodes):
    resp=[]
    cos=cousins(node,nodes)
    for i in range(0,len(cos)):
        resp.append(cos[i])
        desc=descendant(cos[i],nodes)
        for j in range(0,len(desc)):
            resp.append(desc[j])
    return list(set(resp))

#Correcao caso haja problemas com float/string
def correcao(data):
    for i in range(0,data.shape[0]):
        data.loc[i,'classification']=str(data.loc[i,'classification'])
    return data

#Roda os algoritmos de ML e salva as probabilidades em "prob-fold"
def rodaAlgoritmos(treino,teste,alg,dirSaidaClassificador,foldAtual,nbClasses):
    x=treino.copy()
    y=treino.copy()
    del x['classification'] #x eh o treino sem o classification
    y=y['classification'] #y eh o treino com o classification
    if alg == "knn":
        clf=KNeighborsClassifier(n_neighbors=5)
        clf.fit(x,y) #treina
        resultado=pd.DataFrame(clf.predict_proba(teste)) #dataframe das probabilidades
        dirSaidaClassificador=dirSaidaClassificador+"knn/"
        criaDir(dirSaidaClassificador)
        #dirSaidaClassificador = "/home/bruna/2017/PesquisaBruna/ResultadoPython/Mips/Abordagem_Exclusiva/Classe1/knn/"
        resultado.to_csv(dirSaidaClassificador+"prob-fold"+str(foldAtual)+".txt")
        #dirSaidaClassificador = "/home/bruna/2017/PesquisaBruna/ResultadoPython/Mips/Abordagem_Exclusiva/Classe1/knn/prob-fold1.txt"
    elif alg == "svm":
        clf=svm.SVC(probability=True)
        clf.fit(x,y)
        resultado=pd.DataFrame(clf.predict_proba(teste))
        dirSaidaClassificador=dirSaidaClassificador+"svm/"
        criaDir(dirSaidaClassificador)
        #dirSaidaClassificador = "/home/bruna/2017/PesquisaBruna/ResultadoPython/Mips/Abordagem_Exclusiva/Classe1/svm/"
        resultado.to_csv(dirSaidaClassificador+"prob-fold"+str(foldAtual)+".txt")
        #dirSaidaClassificador = "/home/bruna/2017/PesquisaBruna/ResultadoPython/Mips/Abordagem_Exclusiva/Classe1/svm/prob-fold1.txt"
    elif alg == "mlp":
        clf=MLPClassifier()
        clf.fit(x,y)
        resultado=pd.DataFrame(clf.predict_proba(teste))
        dirSaidaClassificador=dirSaidaClassificador+"mlp/"
        criaDir(dirSaidaClassificador)
        #dirSaidaClassificador = "/home/bruna/2017/PesquisaBruna/ResultadoPython/Mips/Abordagem_Exclusiva/Classe1/mlp/"
        resultado.to_csv(dirSaidaClassificador+"prob-fold"+str(foldAtual)+".txt")
        #dirSaidaClassificador = "/home/bruna/2017/PesquisaBruna/ResultadoPython/Mips/Abordagem_Exclusiva/Classe1/mlp/prob-fold1.txt"
    elif alg == "mlp2":
        clf=MLPClassifier(activation = 'logistic',solver='adam', hidden_layer_sizes=(200,))
        clf.fit(x,y)
        resultado=pd.DataFrame(clf.predict_proba(teste))
        dirSaidaClassificador=dirSaidaClassificador+"mlp2/"
        criaDir(dirSaidaClassificador)
        #dirSaidaClassificador = "/home/bruna/2017/PesquisaBruna/ResultadoPython/Mips/Abordagem_Exclusiva/Classe1/mlp2/"
        resultado.to_csv(dirSaidaClassificador+"prob-fold"+str(foldAtual)+".txt")
        #dirSaidaClassificador = "/home/bruna/2017/PesquisaBruna/ResultadoPython/Mips/Abordagem_Exclusiva/Classe1/mlp2/prob-fold1.txt"   
    elif alg == "mlp3":
        clf=MLPClassifier(activation = 'relu',solver='adam',hidden_layer_sizes=(len(x.columns),1),max_iter=600)
        clf.fit(x,y)
        resultado=pd.DataFrame(clf.predict_proba(teste))
        dirSaidaClassificador=dirSaidaClassificador+"mlp3/"
        criaDir(dirSaidaClassificador)
        #dirSaidaClassificador = "/home/bruna/2017/PesquisaBruna/ResultadoPython/Mips/Abordagem_Exclusiva/Classe1/mlp3/"
        resultado.to_csv(dirSaidaClassificador+"prob-fold"+str(foldAtual)+".txt")
        #dirSaidaClassificador = "/home/bruna/2017/PesquisaBruna/ResultadoPython/Mips/Abordagem_Exclusiva/Classe1/mlp3/prob-fold1.txt"
    elif alg == "mlp4":
        clf=MLPClassifier(activation = 'relu',solver='adam',hidden_layer_sizes=(len(x.columns),2),max_iter=600)
        clf.fit(x,y)
        resultado=pd.DataFrame(clf.predict_proba(teste))
        dirSaidaClassificador=dirSaidaClassificador+"mlp4/"
        criaDir(dirSaidaClassificador)
        #dirSaidaClassificador = "/home/bruna/2017/PesquisaBruna/ResultadoPython/Mips/Abordagem_Exclusiva/Classe1/mlp4/"
        resultado.to_csv(dirSaidaClassificador+"prob-fold"+str(foldAtual)+".txt")
        #dirSaidaClassificador = "/home/bruna/2017/PesquisaBruna/ResultadoPython/Mips/Abordagem_Exclusiva/Classe1/mlp4/prob-fold1.txt"  
    elif alg == "mlp5":
        clf=MLPClassifier(activation = 'relu',solver='adam',hidden_layer_sizes=(nbClasses,1),max_iter=600)
        clf.fit(x,y)
        resultado=pd.DataFrame(clf.predict_proba(teste))
        dirSaidaClassificador=dirSaidaClassificador+"mlp5/"
        criaDir(dirSaidaClassificador)
        #dirSaidaClassificador = "/home/bruna/2017/PesquisaBruna/ResultadoPython/Mips/Abordagem_Exclusiva/Classe1/mlp5/"
        resultado.to_csv(dirSaidaClassificador+"prob-fold"+str(foldAtual)+".txt")
        #dirSaidaClassificador = "/home/bruna/2017/PesquisaBruna/ResultadoPython/Mips/Abordagem_Exclusiva/Classe1/mlp5/prob-fold1.txt"   
    elif alg == "mlp6":
        clf=MLPClassifier(activation = 'relu',solver='adam',hidden_layer_sizes=(nbClasses,2),max_iter=600)
        clf.fit(x,y)
        resultado=pd.DataFrame(clf.predict_proba(teste))
        dirSaidaClassificador=dirSaidaClassificador+"mlp6/"
        criaDir(dirSaidaClassificador)
        #dirSaidaClassificador = "/home/bruna/2017/PesquisaBruna/ResultadoPython/Mips/Abordagem_Exclusiva/Classe1/mlp6/"
        resultado.to_csv(dirSaidaClassificador+"prob-fold"+str(foldAtual)+".txt")
        #dirSaidaClassificador = "/home/bruna/2017/PesquisaBruna/ResultadoPython/Mips/Abordagem_Exclusiva/Classe1/mlp6/prob-fold1.txt"      
	elif alg == "nb":
        clf=GaussianNB()
        clf.fit(x,y)
        resultado=pd.DataFrame(clf.predict_proba(teste))
        dirSaidaClassificador=dirSaidaClassificador+"nb/"
        criaDir(dirSaidaClassificador)
        #dirSaidaClassificador = "/home/bruna/2017/PesquisaBruna/ResultadoPython/Mips/Abordagem_Exclusiva/Classe1/nb/"
        resultado.to_csv(dirSaidaClassificador+"prob-fold"+str(foldAtual)+".txt")
        #dirSaidaClassificador = "/home/bruna/2017/PesquisaBruna/ResultadoPython/Mips/Abordagem_Exclusiva/Classe1/nb/prob-fold1.txt"
    elif alg == "dtree":
        clf = tree.DecisionTreeClassifier()
        clf.fit(x,y)
        resultado=pd.DataFrame(clf.predict_proba(teste))
        dirSaidaClassificador=dirSaidaClassificador+"dtree/"
        criaDir(dirSaidaClassificador)
        #dirSaidaClassificador = "/home/bruna/2017/PesquisaBruna/ResultadoPython/Mips/Abordagem_Exclusiva/Classe1/dtree/"
        resultado.to_csv(dirSaidaClassificador+"prob-fold"+str(foldAtual)+".txt")
        #dirSaidaClassificador = "/home/bruna/2017/PesquisaBruna/ResultadoPython/Mips/Abordagem_Exclusiva/Classe1/dtree/prob-fold1.txt"
    elif alg == "rforest":
        clf = RandomForestClassifier()
        clf.fit(x,y)
        resultado=pd.DataFrame(clf.predict_proba(teste))
        dirSaidaClassificador=dirSaidaClassificador+"rforest/"
        criaDir(dirSaidaClassificador)
        #dirSaidaClassificador = "/home/bruna/2017/PesquisaBruna/ResultadoPython/Mips/Abordagem_Exclusiva/Classe1/rforest/"
        resultado.to_csv(dirSaidaClassificador+"prob-fold"+str(foldAtual)+".txt")
        #dirSaidaClassificador = "/home/bruna/2017/PesquisaBruna/ResultadoPython/Mips/Abordagem_Exclusiva/Classe1/rforest/prob-fold1.txt"

#Cria os datasets de acordo com a abordagem e roda ML
def executa(dirPrincipal,abordagem,nFolds,dados,alg,corrige=False):
    dirAssinaturas=dirPrincipal+"Assinaturas/"+dados+"/"
    #dirAssinaturas = "/home/bruna/2017/PesquisaBruna/Assinaturas/Mips/"
    dirSaidaClassificador=dirPrincipal+"ResultadoPython/"+dados+"/"
    criaDir(dirSaidaClassificador)
    dirSaidaClassificador=dirSaidaClassificador+abordagem
    criaDir(dirSaidaClassificador)
    #dirSaidaClassificador = "/home/bruna/2017/PesquisaBruna/ResultadoPython/Mips/Abordagem_Exclusiva/"
    treinoSave=pd.read_csv(dirAssinaturas+"train"+str(1)+".csv")
    if corrige==True:
        treinoSave=correcao(treinoSave)
    nodes=treinoSave['classification'].unique() #contem os nos
    for foldAtual in range(1,nFolds+1): #percorre os folds
        if corrige==True:
            treinoSave=pd.read_csv(dirAssinaturas+"train"+str(foldAtual)+".csv",low_memory=False) #salva o treino     
            testeSave=pd.read_csv(dirAssinaturas+"test"+str(foldAtual)+".csv",low_memory=False) #salva o teste
            treinoSave=correcao(treinoSave)
        else:
            treinoSave=pd.read_csv(dirAssinaturas+"train"+str(foldAtual)+".csv") #salva o treino     
            testeSave=pd.read_csv(dirAssinaturas+"test"+str(foldAtual)+".csv") #salva o teste
        if (dados=="Mips") | (dados=="Repbase"):
            del treinoSave['id']
            del testeSave['id']
        del testeSave['classification'] #o teste salvo nao tem o classification
        for nodeAtual in range(1,len(nodes)+1): #percorre os nodes
            treino=treinoSave.copy() #treino recebe a copia do treinoSave
            teste=testeSave.copy() #teste recebe a copia do testeSave, isso evita ter que ler novamente
            dirSaidaClassificador=dirSaidaClassificador+"Classe"+str(nodeAtual)+"/"
            #dirSaidaClassificador = "/home/bruna/2017/PesquisaBruna/ResultadoPython/Mips/Abordagem_Exclusiva/Classe1/"
            criaDir(dirSaidaClassificador)
            if abordagem=="Abordagem_Exclusiva/":
                positivos = [nodes[nodeAtual-1]]
                negativos = diff(nodes,positivos)
            elif abordagem=="Abordagem_Inclusiva/":
                positivos = uni([nodes[nodeAtual-1]],descendant(nodes[nodeAtual-1],nodes))
                negativos = diff(nodes,uni(positivos,ancestor(nodes[nodeAtual-1],nodes)))
            elif abordagem=="Abordagem_Menos_Exclusiva/":
                positivos = [nodes[nodeAtual-1]]
                negativos = diff(nodes,uni(positivos,descendant(nodes[nodeAtual-1],nodes)))
			elif abordagem=="Abordagem_Menos_Inclusiva/":
                positivos = uni([nodes[nodeAtual-1]],descendant(nodes[nodeAtual-1],nodes))
                negativos = diff(nodes,positivos)
            elif abordagem=="Abordagem_Irmaos/":
                positivos = uni([nodes[nodeAtual-1]],descendant(nodes[nodeAtual-1],nodes))
                negativos = siblingsDescendant(nodes[nodeAtual-1],nodes)
            elif abordagem=="Abordagem_Irmaos_Exclusiva/":
                positivos = uni([nodes[nodeAtual-1]],descendant(nodes[nodeAtual-1],nodes))
                negativos = anotherSiblings(nodes[nodeAtual-1],nodes)
            elif abordagem=="Abordagem_Primos/":
                positivos = uni([nodes[nodeAtual-1]],descendant(nodes[nodeAtual-1],nodes))
                negativos = cousinsDescendant(nodes[nodeAtual-1],nodes)
            elif abordagem=="Abordagem_Primos_Exclusiva/":
                positivos = uni([nodes[nodeAtual-1]],descendant(nodes[nodeAtual-1],nodes))
                negativos = cousins(nodes[nodeAtual-1],nodes)
            elif abordagem=="Abordagem_So_Primos/":
                positivos = uni([nodes[nodeAtual-1]],descendant(nodes[nodeAtual-1],nodes))
                negativos = diff(cousins(nodes[nodeAtual-1],nodes),siblings(nodes[nodeAtual-1],nodes))
                if len(negativos)==0:
                    negativos = cousins(nodes[nodeAtual-1],nodes)
            positivosTotal="^"+positivos[0]+"$"
            for i in range(1,len(positivos)):
                positivosTotal=positivosTotal+"|^"+positivos[i]+"$"
            negativosTotal="^"+negativos[0]+"$"
            for i in range(1,len(negativos)):
                negativosTotal=negativosTotal+"|^"+negativos[i]+"$"
            treinoAtual=treino.copy()  #se nao, vai alterar o treino
            treinoAtual=treinoAtual[treino['classification'].str.contains(positivosTotal)]
            treinoAtual.loc[:,'classification']='1'
            treinoAtual2=treino.copy()
            treinoAtual2=treinoAtual2[treino['classification'].str.contains(negativosTotal)]
            treinoAtual2.loc[:,'classification']='0'
            treino=pd.concat([treinoAtual,treinoAtual2])
            treino['classification']=treino['classification'].astype('category')
            del treinoAtual
            del treinoAtual2
            rodaAlgoritmos(treino,teste,alg,dirSaidaClassificador,foldAtual,len(nodes))
            print("Rodou "+alg+" fold "+str(foldAtual)+" node "+str(nodeAtual)+" abordagem "+abordagem)   
            dirSaidaClassificador=dirPrincipal+"ResultadoPython/"+dados+"/"+abordagem

#Cria matriz de probabilidades, gera resultado em Simple Prune e calcula medidas por fold
def resultado(dirPrincipal,abordagem,nFolds,dados,alg,corrige=False):
    dirAssinaturas=dirPrincipal+"Assinaturas/"+dados+"/"
    #dirAssinaturas = "/home/bruna/2017/PesquisaBruna/Assinaturas/Mips/"
    dirSaidaResultado=dirPrincipal+"ResultadoPython/"+dados+"/"+abordagem+"resultado/"
    #dirSaidaResultado = "/home/bruna/2017/PesquisaBruna/ResultadoPython/Mips/Abordagem_Exclusiva/resultado/"
    criaDir(dirSaidaResultado)
    dirSaidaResultado=dirSaidaResultado+alg+"/"
    #dirSaidaResultado = "/home/bruna/2017/PesquisaBruna/ResultadoPython/Mips/Abordagem_Exclusiva/resultado/knn/"
    criaDir(dirSaidaResultado)
    treino=pd.read_csv(dirAssinaturas+"train"+str(1)+".csv")
    if corrige==True:
        treino=correcao(treino)
    nodes=list(treino['classification'].unique())
    nNodes=len(nodes)
    for fold in range(1,nFolds+1):
        i=0
        Classe=pd.read_csv(dirPrincipal+"ResultadoPython/"+dados+"/"+abordagem+"Classe"+str(i+1)+"/"+alg+"/prob-fold"+str(fold)+".txt")
        Resultado = pd.DataFrame(index=range(0,Classe.shape[0]),columns=nodes, dtype='object')
        for i in range(0,nNodes):
            Classe=pd.read_csv(dirPrincipal+"ResultadoPython/"+dados+"/"+abordagem+"Classe"+str(i+1)+"/"+alg+"/prob-fold"+str(fold)+".txt")
            Resultado.loc[:,Resultado.columns.tolist()[i]]=Classe.loc[:,'1']
        Resultado.to_csv(dirSaidaResultado+"matrizprob-fold"+str(fold)+".txt")
        Saida=Resultado.copy()
        for i in range(0,Saida.shape[1]):
            pais=ancestor(nodes[i],nodes)
            if(len(pais)>0):
                for j in range(0,Resultado.shape[0]):
                    valor=[]
                    valor.append(float(Resultado.iloc[j,[i]]))
                    for k in range(0,len(pais)):
                        valor.append(float(Resultado.loc[j,pais[k]]))
                    Saida.iloc[j,[i]]=sum(valor)/len(valor)
        #Saida contem as probabilidades ajustadas apos o SimplePrune                              
        Resultado=pd.DataFrame(index=range(0,Saida.shape[0]),columns=["1"])
        for i in range(0,Saida.shape[0]):
            t=maxRow(Saida,i)
            valor=[]
            for j in range(0,len(t)):
                valor.append(treino['classification'].value_counts()[t[j]])
            Resultado.iloc[i,[0]]=t[valor.index(max(valor))]
        Resultado.to_csv(dirSaidaResultado+"clasfResult-fold"+str(fold)+".txt")
        teste=pd.read_csv(dirAssinaturas+"test"+str(fold)+".csv")
        if corrige==True:
            teste=correcao(teste)
        hPD,hRD,hPN,hRN=[0,0,0,0]
        for i in range(0,Resultado.shape[0]):
            real=teste.loc[i,"classification"]
            predito=list(Resultado.iloc[i,[0]])
            preditoA=uni(predito,ancestorTotal(predito[0]))
            realA=uni([real],ancestorTotal(real))
            hPD=hPD+len(preditoA)
            hRD=hRD+len(realA)
            valor=intersect(preditoA,realA)
            hPN=hPN+len(valor)
            hRN=hRN+len(valor)
        hP=float(hPN)/float(hPD)
        hR=float(hRN)/float(hRD)
        hF=(2*hP*hR)/(hP+hR)
        matriz=pd.DataFrame(index=range(0,1),columns=["hP","hR","hF"])
        matriz.iloc[0]=[hP,hR,hF]
        matriz.to_csv(dirSaidaResultado+"medidas-fold"+str(fold)+".txt")

#Calcula os valores finais (combinacao de todos os folds)
def valoresFinais(dirPrincipal,abordagem,nFolds,dados,alg):
    dirOrigemResultado=dirPrincipal+"ResultadoPython/"+dados+"/"+abordagem+"resultado/"+alg+"/"
    #dirOrigemResultado = "/home/bruna/2017/PesquisaBruna/ResultadoPython/Mips/Abordagem_Exclusiva/ResultadoPython/knn/"
    dirSaidaValoresFinais=dirPrincipal+"ResultadoPython/"+dados+"/"+abordagem+"valoresFinais/"
    #dirSaidaValoresFinais = "/home/bruna/2017/PesquisaBruna/ResultadoPython/Mips/Abordagem_Exclusiva/valoresFinais/"
    criaDir(dirSaidaValoresFinais)
    medidas=pd.DataFrame(index=range(0,1),columns=["hP","sdhP","hR","sdhR","hF","sdhF"])
    hPtotal=[]
    hRtotal=[]
    hFtotal=[]
    for fold in range(1,nFolds+1):
		tabela=pd.read_csv(dirOrigemResultado+"medidas-fold"+str(fold)+".txt")
        hPtotal.append(float(tabela.loc[0,"hP"]))
        hRtotal.append(float(tabela.loc[0,"hR"]))
        hFtotal.append(float(tabela.loc[0,"hF"]))
    medidas.loc[0,"hP"]=statistics.mean(hPtotal)
    medidas.loc[0,"sdhP"]=statistics.stdev(hPtotal)
    medidas.loc[0,"hR"]=statistics.mean(hRtotal)
    medidas.loc[0,"sdhR"]=statistics.stdev(hRtotal)
    medidas.loc[0,"hF"]=statistics.mean(hFtotal)
    medidas.loc[0,"sdhF"]=statistics.stdev(hFtotal)
    medidas.to_csv(dirSaidaValoresFinais+"valores-"+alg+".txt")

#Cria os datasets de acordo com a abordagem e roda ML
def executaQuebrado(dirPrincipal,abordagem,nFolds,dados,alg,foldInicio,foldFim,classeInicio,corrige=False):
    dirAssinaturas=dirPrincipal+"Assinaturas/"+dados+"/"
    #dirAssinaturas = "/home/bruna/2017/PesquisaBruna/Assinaturas/Mips/"
    dirSaidaClassificador=dirPrincipal+"ResultadoPython/"+dados+"/"
    criaDir(dirSaidaClassificador)
    dirSaidaClassificador=dirSaidaClassificador+abordagem
    criaDir(dirSaidaClassificador)
    #dirSaidaClassificador = "/home/bruna/2017/PesquisaBruna/ResultadoPython/Mips/Abordagem_Exclusiva/"
    treinoSave=pd.read_csv(dirAssinaturas+"train"+str(1)+".csv")
    if corrige==True:
        treinoSave=correcao(treinoSave)
    nodes=treinoSave['classification'].unique() #contem os nos
    for foldAtual in range(foldInicio,foldFim): #percorre os folds
        if corrige==True:
            treinoSave=pd.read_csv(dirAssinaturas+"train"+str(foldAtual)+".csv",low_memory=False) #salva o treino     
            testeSave=pd.read_csv(dirAssinaturas+"test"+str(foldAtual)+".csv",low_memory=False) #salva o teste
            treinoSave=correcao(treinoSave)
        else:
            treinoSave=pd.read_csv(dirAssinaturas+"train"+str(foldAtual)+".csv") #salva o treino     
            testeSave=pd.read_csv(dirAssinaturas+"test"+str(foldAtual)+".csv") #salva o teste
        if (dados=="Mips") | (dados=="Repbase"):
            del treinoSave['id']
            del testeSave['id']
        del testeSave['classification'] #o teste salvo nao tem o classification
        for nodeAtual in range(classeInicio,len(nodes)+1): #percorre os nodes
            treino=treinoSave.copy() #treino recebe a copia do treinoSave
			teste=testeSave.copy() #teste recebe a copia do testeSave, isso evita ter que ler novamente
            dirSaidaClassificador=dirSaidaClassificador+"Classe"+str(nodeAtual)+"/"
            #dirSaidaClassificador = "/home/bruna/2017/PesquisaBruna/ResultadoPython/Mips/Abordagem_Exclusiva/Classe1/"
            criaDir(dirSaidaClassificador)
            if abordagem=="Abordagem_Exclusiva/":
                positivos = [nodes[nodeAtual-1]]
                negativos = diff(nodes,positivos)
            elif abordagem=="Abordagem_Inclusiva/":
                positivos = uni([nodes[nodeAtual-1]],descendant(nodes[nodeAtual-1],nodes))
                negativos = diff(nodes,uni(positivos,ancestor(nodes[nodeAtual-1],nodes)))
            elif abordagem=="Abordagem_Menos_Exclusiva/":
                positivos = [nodes[nodeAtual-1]]
                negativos = diff(nodes,uni(positivos,descendant(nodes[nodeAtual-1],nodes)))
            elif abordagem=="Abordagem_Menos_Inclusiva/":
                positivos = uni([nodes[nodeAtual-1]],descendant(nodes[nodeAtual-1],nodes))
                negativos = diff(nodes,positivos)
            elif abordagem=="Abordagem_Irmaos/":
                positivos = uni([nodes[nodeAtual-1]],descendant(nodes[nodeAtual-1],nodes))
                negativos = siblingsDescendant(nodes[nodeAtual-1],nodes)
            elif abordagem=="Abordagem_Irmaos_Exclusiva/":
                positivos = uni([nodes[nodeAtual-1]],descendant(nodes[nodeAtual-1],nodes))
                negativos = anotherSiblings(nodes[nodeAtual-1],nodes)
            elif abordagem=="Abordagem_Primos/":
                positivos = uni([nodes[nodeAtual-1]],descendant(nodes[nodeAtual-1],nodes))
                negativos = cousinsDescendant(nodes[nodeAtual-1],nodes)
            elif abordagem=="Abordagem_Primos_Exclusiva/":
                positivos = uni([nodes[nodeAtual-1]],descendant(nodes[nodeAtual-1],nodes))
                negativos = cousins(nodes[nodeAtual-1],nodes)
            elif abordagem=="Abordagem_So_Primos/":
                positivos = uni([nodes[nodeAtual-1]],descendant(nodes[nodeAtual-1],nodes))
                negativos = diff(cousins(nodes[nodeAtual-1],nodes),siblings(nodes[nodeAtual-1],nodes))
                if len(negativos)==0:
                    negativos = cousins(nodes[nodeAtual-1],nodes)
            positivosTotal="^"+positivos[0]+"$"
            for i in range(1,len(positivos)):
                positivosTotal=positivosTotal+"|^"+positivos[i]+"$"
            negativosTotal="^"+negativos[0]+"$"
            for i in range(1,len(negativos)):
                negativosTotal=negativosTotal+"|^"+negativos[i]+"$"
            treinoAtual=treino.copy()  #se nao, vai alterar o treino
            treinoAtual=treinoAtual[treino['classification'].str.contains(positivosTotal)]
            treinoAtual.loc[:,'classification']='1'
            treinoAtual2=treino.copy()
            treinoAtual2=treinoAtual2[treino['classification'].str.contains(negativosTotal)]
            treinoAtual2.loc[:,'classification']='0'
            treino=pd.concat([treinoAtual,treinoAtual2])
            treino['classification']=treino['classification'].astype('category')
            del treinoAtual
            del treinoAtual2
            rodaAlgoritmos(treino,teste,alg,dirSaidaClassificador,foldAtual,len(nodes))
            print("Rodou "+alg+" fold "+str(foldAtual)+" node "+str(nodeAtual)+" abordagem "+abordagem)   
            dirSaidaClassificador=dirPrincipal+"ResultadoPython/"+dados+"/"+abordagem
