from turtle import width
from functions import *
from matplotlib import pyplot as plt

"""
t   = np.linspace(0,110,num=110)
Hd  = np.zeros(np.size(t))
THd = 40
Hd[50:85] = (t[50:85]-t[50])/THd
Hd[85:] = Hd[84]-(t[85:]-t[84])/THd

plt.plot(t[84:],Hd[84:],color='#FFB000',lw=2.5)
plt.plot(t[:51],np.zeros(np.size(t[:51])),color='#648FFF',lw=2.5)
plt.plot(t[50:85],Hd[50:85],color='#DC267F',lw=2.5)
#plt.ylim([-0.8,0.8])
"""
timeRise = np.array([80,70,60,50,40,30,20,16,14,13])
yRise    = np.array([0.851,0.829,0.787,0.748,0.689,0.597,0.429,0.316,0.244,0.202])
timeFall = np.array([30.5,32.5,35,40,45,50,60,70,80])
yFall    = np.array([1.48,1.43,1.39,1.33,1.29,1.26,1.22,1.18,1.16])
plt.plot(timeRise,yRise,'-o',color='#DC267F')
plt.plot(timeFall,yFall,'-o',color='#FFB000')

plt.grid()
plt.show()