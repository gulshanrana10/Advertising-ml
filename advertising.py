# -*- coding: utf-8 -*-
"""
Created on Mon Jul 30 19:06:23 2018

@author: Gulshan Rana
"""

#import csv
#import matplotlib.pyplot as plt
import numpy as np
from sklearn.model_selection import train_test_split
#from sklearn import cross_validation
#import scipy as sc
from sympy import Symbol,Derivative
sym=Symbol
der=Derivative
import random
#hdir(C:\Users\Gulshan Rana\AppData\Roaming\Microsoft\Windows\Start Menu\Programs\Anaconda3 (64-bit))

#add= "C:\Users\Gulshan Rana\Desktop\data_2ngenre.csv"
     
dataset= np.genfromtxt("advertising.csv", delimiter=",")

dataset= dataset[1:,]
#print(dataset)

x= dataset[0:,0:4]
y_raw= dataset[0: ,-1]
#print(x.shape, y_raw.shape)
#print(y_raw)
p=np.transpose(x)
p[0]=(p[0]-65)/15.85
p[2]=(p[2]-55000)/65487.5
p[3]=(p[3]-180)/43.9
x=np.transpose(p)
#print(x)
y= np.zeros((1000, 1))    #rreason of double brkt
for i in range(1000):
    y[i][0]= y_raw[i]
 
x_train, x_test, y_train, y_test= train_test_split(x, y, test_size=0.2)
#print(y)
def sigmoid(z):
     return(1/(1+np.exp(-z)))

def hypothesis(x,theta):
    h=np.dot(x,theta)
    return(sigmoid(h))
#print(x_train)
theta=np.random.rand(x_train.shape[1],1)  #28*1
H=hypothesis(x_train, theta)  #H=h
alpha=0.01
#print(x_train)
#print(theta)
#print(H)
m= len(y)
#print(m)
#print((H-y).shape,x.shape)

nditer=3000

for i in range(nditer):
    H=hypothesis(x_train,theta)
    theta=theta-alpha*(1/m)*np.dot(np.transpose(x_train),(H-y_train))
    

op=hypothesis(x_test,theta)
#print(op.shape[0])
#print(op)
#print(np.sum(op)/200)
c=0;
for i in range(op.shape[0]):
    if op[i][0]>0.45:
       op[i][0]=1;
    else:
        op[i][0]=0;
    if op[i]==y_test[i]:
        c=c+1
print(op)
print(c/op.shape[0])
#print(theta)
   
    
    



'''def cost(y,h,x,L):
      #  x is the array, m is the array lenth(in this case 200), y=0 or 1, h is hypothesis,L= regularization const
    
    i=0
    J=np.zeros((),dtype=float)
    y=input("enter the prediction y= 0 or 1: ")
    for i in range(m):
         J = J +(-(1/m)*((y*(np.log(h[i]))+((1-y)*(np.log(1-h[i]))))
    return J 
print(cost(y_train,H,x_train,1000))

for i in range(201):
    H= np.append(sigmoid(h[i]))
        
       
        

X = list(csv.reader(open('data_2ngenre.csv','r')))




def sigma(i,j,f):
    i = input("enter the initial value of sigma(i): ")
    j = input("enter the final value of sigma(j): ")
    
    while i<(j+1):
        
        also to do mean normalisation,X=200*30,theta=1*30



def mingrad(j,A):  j=cost fnc, A= learning rate
    p=0
    r=0
    n=input("enter the no. of itterations: ")
    for r<n:
        while p<30:
    
            temp[p]=theta[p]-A*der(cost,theta[p]).doit()
        
            theta[p]=temp(p)
            p+=1
            return temp(p)
        for i in temp:
            theta=temp
    return theta
'''
     

    
        

