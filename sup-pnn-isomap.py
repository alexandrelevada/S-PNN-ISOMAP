#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
A Supervised Probabilistic Nearest Neighbors Isometric Feature Mapping for Dimensionality Reduction Based Metric Learning

Python script to reproduce the experiments in the paper

Created on Wed Jul 24 16:59:33 2019

"""

# Imports
import sys
import time
import warnings
import scipy
import umap
import numpy as np
import scipy as sp
import networkx as nx
import matplotlib.pyplot as plt
import sklearn.datasets as skdata
import sklearn.neighbors as sknn
from numpy import log
from numpy import trace
from numpy import dot
from numpy import sqrt
from numpy.linalg import det
from numpy.linalg import inv
from sklearn import preprocessing
from sklearn import metrics
from sklearn.cross_decomposition import PLSRegression
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.model_selection import train_test_split
from sklearn.model_selection import cross_val_score
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.discriminant_analysis import QuadraticDiscriminantAnalysis
from sklearn.neural_network import MLPClassifier
from sklearn.gaussian_process import GaussianProcessClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import balanced_accuracy_score
from networkx.convert_matrix import from_numpy_array
from networkx.algorithms.tree.mst import minimum_spanning_tree

# To avoid unnecessary warning messages
warnings.simplefilter(action='ignore')

# PCA implementation
def myPCA(dados, d):
    # Eigenvalues and eigenvectors of the covariance matrix
    v, w = np.linalg.eig(np.cov(dados.T))
    # Sort the eigenvalues
    ordem = v.argsort()
    # Select the d eigenvectors associated to the d largest eigenvalues
    maiores_autovetores = w[:, ordem[-d:]]
    # Projection matrix
    Wpca = maiores_autovetores
    # Linear projection into the 2D subspace
    novos_dados = np.dot(Wpca.T, dados.T)
    
    return novos_dados

# Supervised PCA implementation (variation from paper Supervised Principal Component Analysis - Pattern Recognition)
def SupervisedPCA(dados, labels, d):
    dados = dados.T
    m = dados.shape[0]      # number of samples
    n = dados.shape[1]      # number of features
    I = np.eye(n)
    U = np.ones((n, n))
    H = I - (1/n)*U
    L = np.zeros((n, n))
    for i in range(n):
        for j in range(n):
            if labels[i] == labels[j]:
                L[i, j] = 1
    Q1 = np.dot(dados, H)
    Q2 = np.dot(H, dados.T)
    Q = np.dot(np.dot(Q1, L), Q2)
    # Eigenvalues and eigenvectors of the covariance matrix
    v, w = np.linalg.eig(Q)
    # Sort the eigenvalues
    ordem = v.argsort()
    # Select the d eigenvectors associated to the d largest eigenvalues
    maiores_autovetores = w[:, ordem[-d:]]
    # Projection matrix
    Wpca = maiores_autovetores
    # Linear projection into the 2D subspace
    novos_dados = np.dot(Wpca.T, dados)
    return novos_dados

# ISOMAP implementation
def myIsomap(dados, k, d):
    # Generate KNN graph
    knnGraph = sknn.kneighbors_graph(dados, n_neighbors=k, mode='distance')
    # Computes geodesic distances
    A = knnGraph.toarray()
    G = nx.from_numpy_array(A)
    D = nx.floyd_warshall_numpy(G)
    n = D.shape[0]
    # Remove infs
    maximo = np.nanmax(D[D != np.inf])
    D[D == np.inf] = maximo
    # Computes centering matrix H
    H = np.eye(n, n) - (1/n)*np.ones((n, n))
    # Computes the inner products matrix B
    B = -0.5*H.dot(D**2).dot(H)
    # Eigeendecomposition
    lambdas, alphas = sp.linalg.eigh(B)
    # Sort eigenvalues and eigenvectors
    indices = lambdas.argsort()[::-1]
    lambdas = lambdas[indices]
    alphas = alphas[:, indices]
    # Select the d largest eigenvectors
    lambdas = lambdas[0:d]
    alphas = alphas[:, 0:d]
    # Computes the intrinsic coordinates
    output = alphas*np.sqrt(lambdas)
    
    return output

# Supervised ISOMAP-PNN
def SUP_ISOMAP_PNN(dados, target, k, d):
    # Number of samples
    n = dados.shape[0]
    # Generate KNN graph
    knnGraph = sknn.kneighbors_graph(dados, n_neighbors=k, mode='distance')
    # Computes geodesic distances
    A = knnGraph.toarray()
    # Estimates the edge weights by PNN
    G = nx.from_numpy_array(A)

    # Check if the graph is connected
    if not nx.is_connected(G):
        # Build a graph
        CompleteGraph = sknn.kneighbors_graph(dados, n_neighbors=n-1, mode='distance')
        # Adjacency matrix
        W_K = CompleteGraph.toarray()
        # NetworkX format
        K_n = nx.from_numpy_array(W_K)
        # MST
        W_mst = nx.minimum_spanning_tree(K_n)
        mst = [(u, v, d) for (u, v, d) in W_mst.edges(data=True)]
        mst_edges = []
        for edge in mst:
            edge_tuple = (edge[0], edge[1], edge[2]['weight'])
            mst_edges.append(edge_tuple)
        # To assure the k-NNG is connected we add te MST edges
        G.add_weighted_edges_from(mst_edges)

    D = nx.floyd_warshall_numpy(G)      
    # Computes the probabilistic nearest neighbors
    W = A.copy()
    P = np.zeros(W.shape)
    for i in range(n):
        distancias = D[i, :]
        order = distancias.argsort()    
        distancias.sort()
        for j in range(n):
            W[i, j] = (distancias[k+1] - distancias[j])/(distancias[k+1] - distancias[1])   # PNN
        P[i, order[1:nn+1]] = W[i, 1:nn+1] #+ W2[i, 1:nn+1]
        for j in range(n):
            if A[i, j] > 0:
                if target[i] != target[j]:
                    P[i, j] = (A[i, j] + P[i, j])
                else:
                    P[i, j] = min(A[i, j], P[i, j])

    # Creates the definitive distance matrix
    G = nx.from_numpy_array(P)
    D = nx.floyd_warshall_numpy(G)
    # Remove infs if any
    maximo = np.nanmax(D[D != np.inf])
    D[D == np.inf] = maximo
    # Computes centering matrix H
    H = np.eye(n, n) - (1/n)*np.ones((n, n))
    # Computes the inner products matrix B
    B = -0.5*H.dot(D**2).dot(H)
    # Eigeendecomposition
    lambdas, alphas = sp.linalg.eigh(B)
    # Sort eigenvalues and eigenvectors
    indices = lambdas.argsort()[::-1]
    lambdas = lambdas[indices]
    alphas = alphas[:, indices]
    # Select the d largest eigenvectors
    lambdas = lambdas[0:d]
    alphas = alphas[:, 0:d]
    # Computes the intrinsic coordinates
    output = alphas*np.sqrt(lambdas)
    
    return output


'''
 Computes the Silhouette coefficient and the supervised classification
 accuracies for several classifiers: KNN, SVM, NB, DT, QDA, MPL, GPC and RFC
 dados: learned representation (output of a dimens. reduction - DR)
 target: ground-truth (data labels)
 method: string to identify the DR method (PCA, NP-PCAKL, KPCA, ISOMAP, LLE, LAP, ...)
'''
def Classification(dados, target, method, mode='holdout'):
    
    print()
    print('Supervised classification for %s features' %(method))
    print()
    
    lista = []

    # 8 different classifiers
    neigh = KNeighborsClassifier(n_neighbors=7)
    svm = SVC(gamma='auto')
    nb = GaussianNB()
    dt = DecisionTreeClassifier(random_state=42)
    qda = QuadraticDiscriminantAnalysis()
    mpl = MLPClassifier(hidden_layer_sizes=(100,), activation='logistic', max_iter=1000)
    gpc = GaussianProcessClassifier()
    rfc = RandomForestClassifier()

    # 50% for training and 40% for testing
    X_train, X_test, y_train, y_test = train_test_split(dados, target, train_size=0.5, random_state=42)

    # KNN
    neigh.fit(X_train, y_train) 
    #acc = neigh.score(X_test, y_test)
    pred = neigh.predict(X_test)
    acc = balanced_accuracy_score(pred, y_test)
    lista.append(acc)
    #print('KNN accuracy: ', acc)

    # SMV
    svm.fit(X_train, y_train) 
    #acc = svm.score(X_test, y_test)
    pred = svm.predict(X_test)
    acc = balanced_accuracy_score(pred, y_test)
    lista.append(acc)
    #print('SVM accuracy: ', acc)

    # Naive Bayes
    nb.fit(X_train, y_train)
    #acc = nb.score(X_test, y_test)
    pred = nb.predict(X_test)
    acc = balanced_accuracy_score(pred, y_test)
    lista.append(acc)
    #print('NB accuracy: ', acc)

    # Decision Tree
    dt.fit(X_train, y_train)
    #acc = dt.score(X_test, y_test)
    pred = dt.predict(X_test)
    acc = balanced_accuracy_score(pred, y_test)
    lista.append(acc)
    #print('DT accuracy: ', acc)

    # Quadratic Discriminant 
    qda.fit(X_train, y_train)
    #acc = qda.score(X_test, y_test)
    pred = qda.predict(X_test)
    acc = balanced_accuracy_score(pred, y_test)
    lista.append(acc)
    #print('QDA accuracy: ', acc)

    # MPL classifier
    mpl.fit(X_train, y_train)
    #acc = mpl.score(X_test, y_test)
    pred = mpl.predict(X_test)
    acc = balanced_accuracy_score(pred, y_test)
    lista.append(acc)
    #print('MPL accuracy: ', acc)

    # Gaussian Process
    gpc.fit(X_train, y_train)
    #acc = gpc.score(X_test, y_test)
    pred = gpc.predict(X_test)
    acc = balanced_accuracy_score(pred, y_test)
    lista.append(acc)
    #print('GPC accuracy: ', acc)

    # Random Forest Classifier
    rfc.fit(X_train, y_train)
    #acc = rfc.score(X_test, y_test)
    pred = rfc.predict(X_test)
    acc = balanced_accuracy_score(pred, y_test)
    lista.append(acc)
    #print('RFC accuracy: ', acc)
    
    # Computes the Silhoutte coefficient
    sc = metrics.silhouette_score(dados.real, target, metric='euclidean')
    
    # Computes the average accuracy
    average = sum(lista)/len(lista)
    maximo = max(lista)
    
    #print('Silhouette coefficient: ', sc)
    print('Average accuracy: ', average)
    print('Maximum accuracy: ', maximo)
    print()

    return [sc, average, maximo]


# Plot the scatterplots dor the 2D output data
def PlotaDados(dados, labels, metodo):
    nclass = len(np.unique(labels))
    if metodo == 'LDA':
        if nclass == 2:
            return -1
    # Encode the labels as integers
    lista = []
    for x in labels:
        if x not in lista:  
            lista.append(x)     
    # Map labels to numbers
    rotulos = []
    for x in labels:  
        for i in range(len(lista)):
            if x == lista[i]:  
                rotulos.append(i)
    rotulos = np.array(rotulos)
    if nclass > 11:
        cores = ['black', 'gray', 'silver', 'whitesmoke', 'rosybrown', 'firebrick', 'red', 'darksalmon', 'sienna', 'sandybrown', 'bisque', 'tan', 'moccasin', 'floralwhite', 'gold', 'darkkhaki', 'lightgoldenrodyellow', 'olivedrab', 'chartreuse', 'palegreen', 'darkgreen', 'seagreen', 'mediumspringgreen', 'lightseagreen', 'paleturquoise', 'darkcyan', 'darkturquoise', 'deepskyblue', 'aliceblue', 'slategray', 'royalblue', 'navy', 'blue', 'mediumpurple', 'darkorchid', 'plum', 'm', 'mediumvioletred', 'palevioletred', 'black', 'gray', 'silver', 'whitesmoke', 'rosybrown', 'firebrick', 'red', 'darksalmon', 'sienna', 'sandybrown', 'bisque', 'tan', 'moccasin', 'floralwhite', 'gold', 'darkkhaki', 'lightgoldenrodyellow', 'olivedrab', 'chartreuse', 'palegreen', 'darkgreen', 'seagreen', 'mediumspringgreen', 'lightseagreen', 'paleturquoise', 'darkcyan', 'darkturquoise', 'deepskyblue', 'aliceblue', 'slategray', 'royalblue', 'navy', 'blue', 'mediumpurple', 'darkorchid', 'plum', 'm', 'mediumvioletred', 'palevioletred', 'black', 'gray', 'silver', 'whitesmoke', 'rosybrown', 'firebrick', 'red', 'darksalmon', 'sienna', 'sandybrown', 'bisque', 'tan', 'moccasin', 'floralwhite', 'gold', 'darkkhaki', 'lightgoldenrodyellow', 'olivedrab', 'chartreuse', 'palegreen', 'darkgreen', 'seagreen', 'mediumspringgreen', 'lightseagreen', 'paleturquoise', 'darkcyan', 'darkturquoise', 'deepskyblue', 'aliceblue', 'slategray', 'royalblue', 'navy', 'blue', 'mediumpurple', 'darkorchid', 'plum', 'm', 'mediumvioletred', 'palevioletred', 'black', 'gray', 'silver', 'whitesmoke', 'rosybrown', 'firebrick', 'red', 'darksalmon', 'sienna', 'sandybrown', 'bisque', 'tan', 'moccasin', 'floralwhite', 'gold', 'darkkhaki', 'lightgoldenrodyellow', 'olivedrab', 'chartreuse', 'palegreen', 'darkgreen', 'seagreen', 'mediumspringgreen', 'lightseagreen', 'paleturquoise', 'darkcyan', 'darkturquoise', 'deepskyblue', 'aliceblue', 'slategray', 'royalblue', 'navy', 'blue', 'mediumpurple', 'darkorchid', 'plum', 'm', 'mediumvioletred', 'palevioletred']
        np.random.shuffle(cores)
    else:
        cores = ['blue', 'red', 'green', 'black', 'cyan', 'magenta', 'orange', 'darkkhaki', 'brown', 'purple', 'salmon']
    plt.figure(1)
    for i in range(nclass):
        indices = np.where(rotulos==i)[0]
        cor = cores[i]
        plt.scatter(dados[indices, 0], dados[indices, 1], c=cor, alpha=0.4, marker='.')
    nome_arquivo = metodo + '.png'
    plt.title(metodo+' clusters')
    plt.savefig(nome_arquivo)
    plt.close()

#%%%%%%%%%%%%%%%%%%%%  Data loading
X = skdata.fetch_openml(name='SVHN_small', version=1)
#X = skdata.fetch_openml(name='CIFAR_10_small', version=1)
#X = skdata.fetch_openml(name='thoracic_surgery', version=1)
#X = skdata.fetch_openml(name='vowel', version=2)
#X = skdata.fetch_openml(name='balance-scale', version=1)           
#X = skdata.fetch_openml(name='collins', version=4)
#X = skdata.fetch_openml(name='car-evaluation', version=1)
#X = skdata.fetch_openml(name='cmc', version=1)
#X = skdata.fetch_openml(name='monks-problems-1', version=1)
#X = skdata.fetch_openml(name='tr45.wc', version=1)
#X = skdata.fetch_openml(name='eating', version=1)
#X = skdata.fetch_openml(name='wall-robot-navigation', version=1)
#X = skdata.fetch_openml(name='letter', version=1)
#X = skdata.fetch_openml(name='sensory', version=2)
#X = skdata.fetch_openml(name='tic-tac-toe', version=1)
#X = skdata.fetch_openml(name='Olivetti_Faces', version=1)
#X = skdata.fetch_openml(name='Kuzushiji-MNIST', version=1)
#X = skdata.fetch_openml(name='gas-drift', version=1)
#X = skdata.fetch_openml(name='thyroid-dis', version=1)
#X = skdata.fetch_openml(name='Fashion-MNIST', version=1)
#X = skdata.fetch_openml(name='ionosphere', version=1)
#X = skdata.fetch_openml(name='prnn_synth', version=1)
#X = skdata.fetch_openml(name='cnae-9', version=1)
#X = skdata.fetch_openml(name='yeast', version=1)
#X = skdata.fetch_openml(name='mfeat-factors', version=1)
#X = skdata.fetch_openml(name='MNIST_784', version=1)
#X = skdata.load_digits()
#X = skdata.fetch_openml(name='eye_movements', version=1)
#X = skdata.fetch_openml(name='pendigits', version=1)
#X = skdata.fetch_openml(name='fri_c0_500_10', version=2)
#X = skdata.fetch_openml(name='nursery', version=1)
#X = skdata.fetch_openml(name='Indian_pines', version=1)
#X = skdata.fetch_openml(name='ilpd', version=1)
#X = skdata.fetch_openml(name='USPS', version=1)
#X = skdata.fetch_openml(name='vehicle', version=1)
#X = skdata.fetch_openml(name='optdigits', version=1)
#X = skdata.fetch_openml(name='diabetes', version=1)
#X = skdata.fetch_openml(name='lung-cancer', version=1)  
#X = skdata.fetch_openml(name='grub-damage', version=2)             
#X = skdata.fetch_openml(name='waveform-5000', version=1) 

dados = X['data']
target = X['target']


if 'details' in X.keys():    
    if X['details']['name'] == 'optdigits':
        dados, lixo, target, garbage = train_test_split(dados, target, train_size=0.25, random_state=42)    
    if X['details']['name'] == 'satimage':
        dados, lixo, target, garbage = train_test_split(dados, target, train_size=0.25, random_state=42)
    if X['details']['name'] == 'JapaneseVowels':
        dados, lixo, target, garbage = train_test_split(dados, target, train_size=0.25, random_state=42)            
    if X['details']['name'] == 'wall-robot-navigation':
        dados, lixo, target, garbage = train_test_split(dados, target, train_size=0.5, random_state=42)            
    if X['details']['name'] == 'Indian_pines':
        dados, lixo, target, garbage = train_test_split(dados, target, train_size=0.25, random_state=42)
    if X['details']['name'] == 'CIFAR_10_small':
        dados, lixo, target, garbage = train_test_split(dados, target, train_size=0.2, random_state=42)
    elif X['details']['name'] == 'pendigits':
        dados, lixo, target, garbage = train_test_split(dados, target, train_size=0.2, random_state=42)
    elif X['details']['name'] == 'artificial-characters':
        dados, lixo, target, garbage = train_test_split(dados, target, train_size=0.25, random_state=42)
    elif X['details']['name'] == 'nursery':
        dados, lixo, target, garbage = train_test_split(dados, target, train_size=0.25, random_state=42)            
    elif X['details']['name'] == 'eye_movements':
        dados, lixo, target, garbage = train_test_split(dados, target, train_size=0.3, random_state=42)
    elif X['details']['name'] == 'mnist_784':
        dados, lixo, target, garbage = train_test_split(dados, target, train_size=0.05, random_state=42)
    elif X['details']['name'] == 'letter':
        dados, lixo, target, garbage = train_test_split(dados, target, train_size=0.1, random_state=42)
    elif X['details']['name'] == 'Kuzushiji-MNIST':
        dados, lixo, target, garbage = train_test_split(dados, target, train_size=0.05, random_state=42)
    elif X['details']['name'] == 'Fashion-MNIST':
        dados, lixo, target, garbage = train_test_split(dados, target, train_size=0.05, random_state=42)
    elif X['details']['name'] == 'SVHN_small':
        dados, lixo, target, garbage = train_test_split(dados, target, train_size=0.25, random_state=42)
    elif X['details']['name'] == 'USPS':
        dados, lixo, target, garbage = train_test_split(dados, target, train_size=0.25, random_state=42)
    elif X['details']['name'] == 'gas-drift':
        dados, lixo, target, garbage = train_test_split(dados, target, train_size=0.2, random_state=42)
    elif X['details']['name'] == 'CreditCardSubset':
        dados, lixo, target, garbage = train_test_split(dados, target, train_size=0.2, random_state=42)


#%%%%%%%%%%%%%%%%%%%% Supervised classification for ISOMAP-KL features

n = dados.shape[0]
m = dados.shape[1]
c = len(np.unique(target))

#nn = round(np.log2(n))
nn = round(sqrt(n))
#nn = 5

print('N = ', n)
print('M = ', m)
print('C = %d' %c)
print('K = ', nn)
#input()

if type(dados) == scipy.sparse._csr.csr_matrix:
    dados = dados.todense()
    dados = np.asarray(dados)
else:
    # Treat categorical features
    if not isinstance(dados, np.ndarray):
        cat_cols = dados.select_dtypes(['category']).columns
        dados[cat_cols] = dados[cat_cols].apply(lambda x: x.cat.codes)
        dados = dados.to_numpy()
le = LabelEncoder()
le.fit(target)
target = le.transform(target)

# Data standardization (to deal with variables having different units/scales)
dados = preprocessing.scale(dados)

#%%%%%%%%%%%%% PLS
model = PLSRegression(n_components=2)
dados_pls = model.fit_transform(dados, y=target)

#%%%%%%%%%%%% UMAP
model = umap.UMAP(n_components=2, random_state=42)
dados_umap = model.fit_transform(dados, y=target)

#%%%%%%%%%%%%% Supervised PCA
dados_suppca = SupervisedPCA(dados, target, 2)
dados_suppca = dados_suppca.T

#%%%%%%%%%%%% LDA
if c > 2:
    model = LinearDiscriminantAnalysis(n_components=2)
else:
    model = LinearDiscriminantAnalysis(n_components=1)
dados_lda = model.fit_transform(dados, target)

#%%%%%%%%%%% Supervised classification
L_suppca = Classification(dados_suppca.real, target, 'SUP PCA')
L_pls = Classification(dados_pls[0], target, 'PLS')
L_lda = Classification(dados_lda, target, 'LDA')
L_umap = Classification(dados_umap, target, 'S-UMAP')

#%%%%%%%%%%%% Plot data
PlotaDados(dados_pls[0], target, 'PLS')
PlotaDados(dados_umap, target, 'S-UMAP')
PlotaDados(dados_suppca, target, 'SUP PCA')
PlotaDados(dados_lda, target, 'LDA')

# Supervised PNN-ISOMAP
dados_isopnn = SUP_ISOMAP_PNN(dados, target, nn, 2)
L_isopnn = Classification(dados_isopnn, target, 'S-PNN-ISO')
PlotaDados(dados_isopnn, target, 'S-PNN-ISO')