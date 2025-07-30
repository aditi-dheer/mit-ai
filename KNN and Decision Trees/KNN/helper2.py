import numpy as np
import matplotlib.pyplot as plt
import matplotlib
import pandas as pd
from scipy.io import loadmat
import os

import sys
import traceback


def visualize_knn(num_neighbors):
    
    global N,k,hp,x,hl,xt,ht
    filename = os.path.join(os.getcwd(), "..", "..", "data", "Iris_Data.csv")
    iris_df = pd.read_csv(filename)
    #iris_df = iris_df.head(50)
    #col_one_arr = iris_df['petal_length'].to_numpy()
    #col_two_arr = iris_df['petal_width'].to_numpy()
    #x = iris_df[['petal_length','petal_width']].to_numpy()
    
    iris_df = iris_df.sample(n=50, replace=False, random_state=1)
    x = iris_df[['petal_length','petal_width']].to_numpy()
    
    fig = plt.figure(frameon=False)
   
    ax = fig.add_subplot(111)
    plt.rcParams['toolbar'] = 'None' 
    ax.set_xlabel('Petal Length')
    ax.set_ylabel('Petal Width')
    ax.yaxis.set_major_locator(matplotlib.ticker.MultipleLocator(0.25))
    ax.xaxis.set_major_locator(matplotlib.ticker.MultipleLocator(0.5))
    plt.xlim(0,7)
    plt.ylim(0,3)
    
    k=num_neighbors;
    print(str(k))
    hp=plt.plot(x[:,0],x[:,1],'bo')
    xt=None
    hl=None
    ht=None

  
    def onclick(event):
        global N,k,hp,x,hl,xt,ht

        if ht is None:
            ht=plt.plot(event.xdata,event.ydata,'ro')
        ht[0].set_data(event.xdata,event.ydata)
        xt=np.array([ht[0].get_data()])
        
        S = np.sum(x**2, axis=1, keepdims=True)
        R = np.sum(xt**2, axis=1, keepdims=True).T
        G = innerproduct(x, xt)
        t = S + R - 2*G
        D = np.sqrt(np.maximum(t, 0))
        indices = np.argsort(D, axis=0)
        dists = np.sort(D, axis=0)
        inds = indices[:k,:]
        dists = dists[:k,:]

        #inds,dists=findknn(x,xt,k); # find k nearest neighbors
        xdata=[]
        ydata=[]

        for i in range(k):
                xdata.append(xt[0,0])
                xdata.append(x[inds[i,0],0])
                xdata.append(None)
                ydata.append(xt[0,1])
                ydata.append(x[inds[i,0],1])
                ydata.append(None)
        if hl is None:
            hl=plt.plot(xdata,ydata,'r-')
            plt.title('%i-Nearest Neighbors' % k)
        else:
            hl[0].set_data(xdata,ydata)


    cid = fig.canvas.mpl_connect('button_press_event', onclick)
    plt.title('Click to add test point')
    

def plotimage(xdim, ydim, M,d=2): # plot an image and draw a red/blue/green box around it (specified by "d")
    Q=np.zeros((xdim+2,ydim+2))
    Q[1:-1,1:-1]=M
    Q=np.repeat(Q[:,:,np.newaxis],3,axis=2)
    Q[[0,-1],:,d]=1
    Q[:,[0,-1],d]=1
    plt.imshow(Q, cmap=plt.cm.binary_r)

def plotfaces(X, xdim=38, ydim=31 ):
    n, d = X.shape
    m=np.ceil(np.sqrt(n))
    for i in range(n):
        plt.subplot(m,m,i+1)
        plt.imshow(X[i, :].reshape(ydim, xdim).T, cmap=plt.cm.binary_r)
        plt.axis('off')

def visualize_knn_boundary(knnclassifier):
    global globalK
    globalK=np.ones(1, dtype=int);
    Xdata=[]
    ldata=[]
    w=[]
    b=[]
    line=None
    stepsize=1;

    fig = plt.figure()
    ax = fig.add_subplot(111)
    plt.xlim(0,1)
    plt.ylim(0,1)

    def visboundary(k, Xdata, ldata, ax, knnclassifier):
        RES=50;
        grid = np.linspace(0,1,RES);
        X, Y = np.meshgrid(grid, grid)
        xTe=np.array([X.flatten(),Y.flatten()])
        Z=knnclassifier(np.array(Xdata),np.array(ldata).T,xTe.T,k[0])
        Z=Z.reshape(X.shape)
        boundary=ax.contourf(X, Y, np.sign(Z),colors=[[0.5, 0.5, 1], [1, 0.5, 0.5]])
        plt.draw()
        sys.__stdout__.write(str(Xdata))
        return(boundary)

    def onclickkdemo(event, Xdata, ldata, ax, knnclassifier):
        global globalK
        if event.key == 'p': # add positive point
            ax.plot(event.xdata,event.ydata,'or')
            label=1
            pos=np.array([event.xdata,event.ydata])
            ldata.append(label);
            Xdata.append(pos)
        if event.key==None: # add negative point
            ax.plot(event.xdata,event.ydata,'ob')
            label=-1
            pos=np.array([event.xdata,event.ydata])
            ldata.append(label);
            Xdata.append(pos)
        if event.key == 'h':
            globalK=globalK % (len(ldata)-1)+1;
        visboundary(globalK, Xdata, ldata, ax, knnclassifier)
        plt.title('k=%i' % globalK)
    cid = fig.canvas.mpl_connect('button_press_event', lambda event: onclickkdemo(event, Xdata, ldata, ax, knnclassifier))


def findknn_grader(xTr,xTe,k):
    D = l2distance(xTr, xTe)
    indices = np.argsort(D, axis=0)
    dists = np.sort(D, axis=0)
    return indices[:k,:], dists[:k,:]
  
def accuracy_grader(truth,preds):
    
    truth = truth.flatten()
    preds = preds.flatten()

    if len(truth) == 0 and len(preds) == 0:
        output = 0
        return output
    return np.mean(truth == preds)

def innerproduct(X,Z=None):
    if Z is None: # case when there is only one input (X)
        return innerproduct(X, X)
    else:  # case when there are two inputs (X,Z)
        return np.dot(X, Z.T)

def l2distance(X,Z=None):
    if Z is None:
        D = l2distance(X, X)
    else:  # case when there are two inputs (X,Z)
        S = np.sum(X**2, axis=1, keepdims=True)
        R = np.sum(Z**2, axis=1, keepdims=True).T
        G = innerproduct(X, Z)
        t = S + R - 2*G
        D = np.sqrt(np.maximum(t, 0))
    return D
