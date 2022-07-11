from functions import *
import pandas as pd

colNames = ['time','deltaQ']
f318 = pd.read_csv('results/labData/rawData_fittings/rawData/f3-18.csv',names=colNames,header=None)
f321 = pd.read_csv('results/labData/rawData_fittings/rawData/f3-21.csv',names=colNames,header=None)
f325 = pd.read_csv('results/labData/rawData_fittings/rawData/f3-25.csv',names=colNames,header=None)
f329 = pd.read_csv('results/labData/rawData_fittings/rawData/f3-29.csv',names=colNames,header=None)
run_ID = ['f3-18','f3-21','f3-25','f3-29']

for i,df in enumerate([f318,f321,f325,f329]):
    labPar,labFit,labError = interpolate(expLab,df['time'],df['deltaQ'])
    labFit=np.asarray(labFit)
    print(run_ID[i],1/labPar[2])
    plt.subplot(2,2,i+1)
    plt.plot(df['time'],df['deltaQ'],label='lab data')
    plt.plot(df['time'],labFit      ,label='fit'     )
    plt.xlabel(r'$t^*$ [h]')
    plt.ylabel(r'$\Delta Q$ [-]')
    plt.title(run_ID[i])
    plt.grid()
    plt.legend()

plt.tight_layout()
plt.show()