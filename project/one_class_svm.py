# -*- coding: utf-8 -*-
"""
Created on Wed Dec 27 19:39:38 2017

@author: asus
"""
import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import make_blobs
from scipy.optimize import fmin_l_bfgs_b
from sklearn.metrics.pairwise import linear_kernel
#import cvxopt

# generate data
def make_blob_dataset(n_samples, contamination=0.05, random_state=42):
    rng = np.random.RandomState(random_state)
    X_inliers = 0.3 * rng.normal(size=(int(n_samples * (1. - contamination)), 2)) + 2
    X_outliers = rng.uniform(low=-1, high=5, size=(int(n_samples * contamination), 2))
    X = np.concatenate((X_inliers, X_outliers), axis=0)
    rng.shuffle(X)
    return X


# defining qp function, directly copying from the old question
def qp(P, q, A, b, C, verbose=True):
    # Gram matrix
    n = P.shape[0]
    P = cvxopt.matrix(P)
    q = cvxopt.matrix(q)
    A = cvxopt.matrix(A)
    b = cvxopt.matrix(b)
    G = cvxopt.matrix(np.concatenate([np.diag(np.ones(n) * -1),
                                     np.diag(np.ones(n))], axis=0))
    h = cvxopt.matrix(np.concatenate([np.zeros(n), C * np.ones(n)]))

    # Solve QP problem
    cvxopt.solvers.options['show_progress'] = verbose
    solution = cvxopt.solvers.qp(P, q, G, h, A, b, solver='mosec')
 
    return np.ravel(solution['x'])



def compute_rho(K, mu_support, idx_support):
    i = np.argmin(mu_support)
    K_support = K[idx_support[i],idx_support]
    rho = np.dot(mu_support,K_support)
    return rho


def kernel(X1,X2):    
    return linear_kernel(X1,X2)


# Using the qp solver for the problem, 
# Since windows only supports cvxopt 3.4 My pc 3.6 NoN, Testing by yourself

def ocsvm_solver(K,r_c=10):
    n = np.shape(K)[0]
    P = K
    q = -1.0* np.ones(n)
    A = np.zeros(n).reshape(1,-1)
    b = 0.0
    C = r_c/n
    mu = qp(P, q, A, b, C, verbose=False)
    idx_support = np.where(np.abs(mu) > 1e-5)[0]
    mu_support = mu[idx_support]
    return mu_support, idx_support

X = make_blob_dataset(300)
#plt.scatter(X[:, 0], X[:, 1], color='b'); 

K = kernel(X, X)
#mu_support, idx_support = ocsvm_solver(K,c=10.0)
#rho = compute_rho(K,mu_support,idx_support)
#plot_ocsvm(X, mu_support, idx_support, rho)

def prox_non_acc(C,X,gram, lr = 0.001,max_iter = 100):
    
     n = X.shape[0]
     # initialize parameter
     mu = np.random.rand(n)
     threshold = C / n
     for i in range(max_iter):
         grad_g = gram.dot(mu)-np.ones(n)
         #print(grad_g)
         # gradient desecent for function g
         mu_temp = mu - lr * grad_g
         
         # projection step
         idx1 = mu_temp < 0
         idx2 = mu_temp > threshold
         mu_temp[idx1] = 0
         mu_temp[idx2] = threshold
         mu = mu_temp
         print(mu)
     return mu
 
       
#mu = prox_non_acc(C=100,X=X,gram=K)       
        

def coordinate(C,X,gram,max_iter=30):
    # first compute the gradient of f(u) w.r.t u_i
    # The results which I computed are grad(u) = (Ku)_i - 1 = K_i u -1 (K_i ith column in kernel matrix )
    # By the coordinate approach we have
    # K_i([u_i,u_(-i)]) = K_{i,i}u_i + K_{i,-i}u_{i,-i} = 1
    # Rewrite as u_i = (1-K_{i,-i}U_{-i})/(K_{i,i})
    # The most naive approach, the smart update rule... je ne sais pas
    
     n = X.shape[0]
     # initialize parameter
     mu = np.random.rand(n)
     threshold = C / n
     
     for i in range(max_iter):
        # iteration times
        k_sup = gram[i,:]
        print(mu)
        
        for cor in range(n):
            # optimizing each coordiante
            
            # generating K_{i,-i} and u_{i,-i}
            k_cor = k_sup[cor]
            k_noncor = np.delete(k_sup,cor)
            
            mu_cor = mu[cor]
            mu_noncor = np.delete(mu,cor)
            
            update_cor = (1-np.dot(k_noncor,mu_noncor))/(k_cor*mu_cor)
            
            # projection step
            if update_cor <0:
                update_cor = 1e-5
            elif update_cor > threshold:
                update_cor = threshold
            
            # updating the coordiante
            mu[cor] = update_cor
            
     return mu

#mu =  coordinate(C=100,X=X,gram=K) 


class bfgs:
    
     def __init__(self, C=100):
         
         self.C = C
         self.mu = None
    
    
     def func(self, params, *args):
        # compute loss function
        
        # args
        K = args
        
        # parameters
        mu = params
        
        K = np.asarray(K)
        # computing loss function
        loss = np.dot(np.dot(mu.T,K),mu)-np.sum(mu)
        print(loss)
        
        return loss
    
     def func_grad(self,params,*args):
        # compute gradient function
        
        # args
        K = args
        
        
        # parameters
        mu = params
        
        K = np.asarray(K)
        
        N_sample, _ = K.shape
        
        gradient = np.dot(K,mu) - np.ones(N_sample)
        
        
        return gradient
    
     def fit(self,X,K):
        
         # initialize parameters
        N_sample = X.shape[0]
     
        mu = np.random.rand(N_sample)
        
        threshold = self.C / N_sample
        
        # setting the range
        interval = np.zeros([N_sample,2])
        
        interval[:,1]= threshold
        
        para_optimal,_,info = fmin_l_bfgs_b(self.func,x0=mu,fprime=self.func_grad,args=K,
                                            approx_grad=False,bounds=interval,maxiter=30)
        
        
        self.mu = para_optimal
        
        return para_optimal
    
    

clf = bfgs()
mu = clf.fit(X,K)
print(mu[:20])
    
            
            
            
        
    
 

