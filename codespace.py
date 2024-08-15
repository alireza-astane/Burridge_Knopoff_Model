import numpy as np
import matplotlib.pyplot as plt
from numba import njit
import scipy 
import datetime
import json
from multiprocessing import Process


L = 50
u = 0.0001
k1 = 1
k2 = 0.04
eps = 0.01  
alpha = 0.1
F0 = 1
m = 1
h = 0.1
eps2 = 0.007     


@njit 
def sum2(a):
    sh = a.shape[0]
    b =  np.zeros(sh)
    b[0] = a[0]
    for i in range(1,sh):
        b[i] = b[i-1] + a[i]

    return b


@njit
def f(x,v,t,l,alpha):
     k2= 1/(l-1)
     
     xIntoLeft = np.roll(x,-1)
     xIntoRight = np.roll(x,1)
     xIntoLeft[-1] = 0
     xIntoRight[0] = 0
     return np.where(v==0,
     np.where(np.abs(-x*(2*k1 + k2) + (xIntoRight + xIntoLeft)*k1 + k2*u*t) < F0, -x*(2*k1 + k2) + (xIntoRight + xIntoLeft)*k1 + k2*u*t ,F0*(1-eps) ),
     F0*(1-eps)/(1+alpha*np.abs(v)))


@njit
def acc(x,v,t,l,alpha):
    k2= 1/(l-1)
    xIntoLeft = np.roll(x,-1)
    xIntoRight = np.roll(x,1)
    xIntoLeft[-1] = 0
    xIntoRight[0] = 0

    return (-x*(2*k1 + k2) + (xIntoRight + xIntoLeft)*k1 + k2*u*t - f(x,v,t,l,alpha) )/m



@njit
def tenstion(x,v,t,l,alpha):
    return m*acc(x,v,t,l,alpha) + f(x,v,t,l,alpha)


@njit
def step(x,v,t,l,alpha):
    k2= 1/(l-1)

    if np.all(v==0):
        t += (F0-np.max(tenstion(x,v,t,l,alpha)))/(k2*u)


        k_1 = v
        l_1 = acc(x,v,t,l,alpha)

        k_2 = v + l_1*h/2
        l_2 = acc(x+k_1*h/2,v + l_1*h/2,t + h/2,l,alpha)

        k_3 = v + l_2*h/2
        l_3 = acc(x+k_2*h/2,v + l_2*h/2,t + h/2,l,alpha)

        k_4 = v + l_3/2
        l_4 = acc(x+k_3*h,v + l_3*h, t + h,l,alpha)

        x = x +  (k_1 + 2*k_2 + 2*k_3 + k_4)*h/6
        v = v +  (l_1 + 2*l_2 + 2*l_3 + l_4)*h/6

        v = np.where(v<0, 0, v)
        

    else: 
        k_1 = v
        l_1 = acc(x,v,t,l,alpha)

        k_2 = v + l_1*h/2
        l_2 = acc(x+k_1*h/2,v + l_1*h/2,t + h/2,l,alpha)

        k_3 = v + l_2*h/2
        l_3 = acc(x+k_2*h/2,v + l_2*h/2,t + h/2,l,alpha)

        k_4 = v + l_3/2
        l_4 = acc(x+k_3*h,v + l_3*h, t + h,l,alpha)

        x = x +  (k_1 + 2*k_2 + 2*k_3 + k_4)*h/6
        v = v +  (l_1 + 2*l_2 + 2*l_3 + l_4)*h/6

        v = np.where(v<0, 0, v)

    return x, v,t+h

@njit(cache=True)
def np_any_axis1(x):
    """Numba compatible version of np.any(x, axis=1)."""
    out = np.zeros(x.shape[0], dtype=np.bool8)
    for i in range(x.shape[1]):
        out = np.logical_or(out, x[:, i])
    return out


@njit
def run(x,v,t,steps,l,alpha):
    xdata = np.zeros((steps,l),dtype=np.float64)
    vdata = np.zeros((steps,l),dtype=np.float64)
    tenstionData = np.zeros((steps,l),dtype=np.float64)
    for i in range(steps):
        x,v,t = step(x,v,t,l,alpha)
        xdata[i] = x
        vdata[i] = v
        tenstionData[i] = tenstion(x,v,t,l,alpha)

    slipping = np_any_axis1(vdata>0)

    slippingR = np.roll(slipping,+1)
    slippingL = np.roll(slipping,-1)

    slippingR[0] = False
    slippingL[-1] = False 

    ends  = np.logical_xor( slipping ,(slipping *slippingL))
    starts  = np.logical_xor( slipping ,(slipping *slippingR))

    totalX = np.sum(xdata,axis = 1)

    eventsSize = totalX[np.argwhere(ends)[:,0]] - totalX[np.argwhere(starts)[:,0]]
    eventsBlocks = np.sum(xdata[ends]!=xdata[starts],axis=1).reshape(-1,1)



    arr1 = vdata>0
    arr2 = arr1 > np.roll(arr1,1)
    arr3 = np.sum(arr2,axis=1)
    arr4 = sum2(arr3)

    arr5 = arr4[np.argwhere(starts)[:,0]]
    arr6 = arr4[np.argwhere(ends)[:,0]]

    eventsBlocks2 = arr6-arr5


    return x,v,t,eventsSize,eventsBlocks,eventsBlocks2,tenstionData,ends


@njit
def getData():

    x = 0.1*np.random.uniform(-1,1,L)
    v = np.zeros(L)
    t = 0


    # TsData 

    TsData= np.empty((0,50))
    eventsSizes = np.empty((0))
    eventsBlocks = np.empty((0,50)).astype(np.int64)

    for i in range(10):
        x,v,t,eventsSize,eventsBlocks,eventsBlocks2,tenstionData,ends = run(x,v,t,100_000,L,alpha)


        Ts = tenstionData[ends][np.argwhere(eventsBlocks2[:] >= 500)[:,0]]
        TsData = np.concatenate((TsData,Ts),axis = 0)


        eventsSizes = np.concatenate((eventsSizes,eventsSize))
        eventsBlocks = np.concatenate((eventsBlocks,eventsBlock))

        print(i)

    return  eventsSizes, eventsBlocks,TsData


def function():
    eventsSizes, eventsBlocks,TsData = getData()
    np.save(f"TsDataAlpha={alpha},L={L}",TsData)
    np.save(f"EventsSizesAlpha={alpha},L={L}",eventsSizes)
    np.save(f"eventsBlocksAlpha={alpha},L={L}",eventsBlocks)
    print("completed")


if __name__ =="__main__":
    t0 = Process(target=function,args=())
    t1 = Process(target=function,args=())
    t2 = Process(target=function,args=())
    t3 = Process(target=function,args=())
    t4 = Process(target=function,args=())
    t5 = Process(target=function,args=())
    t6 = Process(target=function,args=())
    t7 = Process(target=function,args=())
    t8 = Process(target=function,args=())
    t9 = Process(target=function,args=())
    t10 = Process(target=function,args=())
    t11 = Process(target=function,args=())
    
    t0.start()
    t1.start()
    t2.start()
    t3.start()
    t4.start()
    t5.start()
    t6.start()
    t7.start()
    t8.start()
    t9.start()
    t10.start()
    t11.start()

    t0.join()
    t1.join()
    t2.join()
    t3.join()
    t4.join()
    t5.join()
    t6.join()
    t7.join()
    t8.join()
    t9.join()
    t10.join()
    t11.join()

