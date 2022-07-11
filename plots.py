import pandas as pd
import matplotlib as mpl
from matplotlib import pyplot as plt
from functions import *

mpl.rcParams['font.family'] = "sans-serif"
plt.rcParams.update({
    "mathtext.fontset": "cm" })
                        
fs=14
lw=2

plotWhat = 'labData'  # deltaQtime,labData,Loops,Tbif_L,Tbif_beta

if plotWhat == 'Tbif2_new':
    df = pd.read_csv('results/Tbif_L/output_Tbif2_new.csv')
    df = df[df.beta0 == 17]
    df.set_index('L',inplace=True)
    df.sort_index(0,inplace=True)
    plt.plot(df.Tbif2_new,lw=lw)

elif plotWhat == 'deltaQtime':
    deltaQtime = np.loadtxt('deltaQtime.txt')    
    tAdim  = deltaQtime[:,0]
    deltaQ = deltaQtime[:,1]
    plt.plot(tAdim,deltaQ,label='computed')

    flexIndex = flexFinder(deltaQ[1:],tAdim[2:-1]-tAdim[1:-2])

    exp1End = int(0.8*flexIndex)
    exp1Par,exp1Fit,exp1Error = interpolate(exponential,tAdim[:exp1End],deltaQ[:exp1End],ic=[deltaQ[0], 0.2])
    print(1/exp1Par[1])
    plt.plot(tAdim[:exp1End],exp1Fit,label='first stage exponential fit')

    exp2End = np.where(deltaQ==np.max(deltaQ))[0][0]
    exp2Par,exp2Fit,exp2Error = interpolate(expLab,deltaQtime[flexIndex:exp2End,0],deltaQtime[flexIndex:exp2End,1],ic=[deltaQ[flexIndex], deltaQ[exp2End], 0.2])
    plt.plot(tAdim[flexIndex:exp2End],exp2Fit,label='second stage exponential fit')
    print(1/exp2Par[2])
    plt.legend()

elif plotWhat == 'Tbif_beta':
    #! Effect of length of the branches on the timescale
    df_cost        = pd.read_csv('results/Tbif_beta/outputCost.csv',index_col='(beta0-betaC)/betaC')
    df_DeltaEtaLin = pd.read_csv('results/Tbif_beta/outputDeltaEtaLin.csv',index_col='(beta0-betaC)/betaC')
    plt.figure(1)
    plt.plot(df_cost        ['Tbif1'],label=r'$\alpha=cost$'           ,lw=lw)
    plt.plot(df_DeltaEtaLin ['Tbif1'],label=r'$\alpha=f(\Delta \eta)$' ,lw=lw)
    #plt.title('First stage evolutionary timescale vs scaled aspect ratio',fontsize=fs)
    plt.xlabel(r'$(\beta_0-\beta_C)/\beta_C [-]$',fontsize=fs)
    plt.ylabel(r'$T \/ ^{\prime} _{BIF} [-]$ '   ,fontsize=fs)
    plt.legend(prop={'size': fs})
    plt.grid(which='both')

    plt.figure(2)
    plt.plot(df_cost       ['Tbif2'],label=r'$\alpha=cost$'           ,lw=lw)
    plt.plot(df_DeltaEtaLin['Tbif2'],label=r'$\alpha=f(\Delta \eta)$' ,lw=lw)
    #plt.title('Second stage evolutionary timescale vs scaled aspect ratio',fontsize=fs)
    plt.xlabel(r'$(\beta_0-\beta_C)/\beta_C [-]$',fontsize=fs)
    plt.ylabel(r'$T \/ ^{\prime \prime} _{BIF} [-]$',fontsize=fs)
    plt.legend(prop={'size': fs})
    plt.grid(which='both')

elif plotWhat == 'Tbif_L':
    #! Effect of length of the branches on the timescale
    df_cost        = pd.read_csv('results/Tbif_L/outputCost0.01.csv',index_col='L')
    df_DeltaEtaLin = pd.read_csv('results/Tbif_L/outputDeltaEtaLin0.01.csv',index_col='Ls')
    plt.figure(1)
    plt.plot(df_cost        ['Tbif1'] ,label=r'$\alpha=cost$'           ,lw=lw)
    plt.plot(df_DeltaEtaLin ['Tbif1'] ,label=r'$\alpha=f(\Delta \eta)$' ,lw=lw)
    #plt.title('First stage evolutionary timescale vs branches length',fontsize=fs)
    plt.xlabel(r'$L^*/D_0 [-]$',fontsize=fs)
    plt.ylabel(r'$T \/ ^{\prime} _{BIF} [-]$',fontsize=fs)
    plt.legend(prop={'size': fs})
    plt.xscale("log")
    plt.yscale("log")
    plt.grid(which='both')

    plt.figure(2)
    plt.plot(df_cost        ['Tbif2'] ,label=r'$\alpha=cost$'           ,lw=lw)
    plt.plot(df_DeltaEtaLin ['Tbif2'] ,label=r'$\alpha=f(\Delta \eta)$' ,lw=lw)
    #plt.title('Second stage evolutionary timescale vs branches length',fontsize=fs)
    plt.xlabel(r'$L^*/D_0 [-]$',fontsize=fs)
    plt.ylabel(r'$T \/ ^{\prime \prime} _{BIF} [-]$',fontsize=fs)
    plt.legend(prop={'size': fs})
    plt.xscale("log")
    plt.yscale("log")    
    plt.grid(which='both')

elif plotWhat == 'Loops':
    #! Loops effect on timescale  
    df_loops   = pd.read_csv( 'results/Loops/alpha=cost/output_loops.csv')
    df_loops.query("inStepIC == -0.01",inplace=True)
    df1 = df_loops.query("theta0 == 0.06 & L>150")
    df2 = df_loops.query("theta0 == 0.08")
    df3 = df_loops.query("theta0 == 0.1")
    
    df1.set_index('L',inplace=True)
    df2.set_index('L',inplace=True)
    df3.set_index('L',inplace=True)
    df1.sort_index(inplace=True)
    df2.sort_index(inplace=True)
    df3.sort_index(inplace=True)

    plt.figure(1)
    plt.xlabel(r'$ L^*/D_0 [-]$',fontsize=fs)
    #plt.xlabel=(r'$(\beta_0-\beta_C)/\beta_C [-]$')
    plt.ylabel(r'$(T \/ ^{\prime} _{BIF} - T \/ ^{\prime} _{BIF,min})/T \/ ^{\prime} _{BIF,min} [-]$',fontsize=fs)
    plt.plot((df1['Tbif1']-np.min(df1['Tbif1']))/np.min(df1['Tbif1']),label=r'$\theta_0=0.06$',lw=lw)
    plt.plot((df2['Tbif1']-np.min(df2['Tbif1']))/np.min(df2['Tbif1']),label=r'$\theta_0=0.08$',lw=lw)
    plt.plot((df3['Tbif1']-np.min(df3['Tbif1']))/np.min(df3['Tbif1']),label=r'$\theta_0=0.1 $',lw=lw)
    plt.legend(prop={'size': fs})
    plt.grid()

    plt.figure(2)
    plt.xlabel(r'$ L^*/D_0 [-]$',fontsize=fs)
    #plt.xlabel=(r'$(\beta_0-\beta_C)/\beta_C [-]$')
    plt.ylabel(r'$\/ \Omega = 1/T^{\prime}_{BIF}[-]$',fontsize=fs)
    plt.plot(1/df1['Tbif1'],label=r'$\theta_0=0.06$',lw=lw)
    #plt.plot(1/df2['Tbif1'],label=r'$\theta_0=0.08$',lw=lw)
    #plt.plot(1/df3['Tbif1'],label=r'$\theta_0=0.1 $',lw=lw)
    plt.legend(prop={'size': fs})
    plt.grid()

elif plotWhat == 'labData':
    #! Lab data
    df_data  = pd.read_csv('results/labData/rawData_fittings/Tbif2new_measured.csv',index_col='run_ID')
    df_model = pd.read_csv('results/labData/modelOutputs/set1_P90.csv')
    
    df_model_cost     = df_model.query("alpha_var == 'cost'")
    df_model_deltaEta = df_model.query("alpha_var == 'deltaEtaLin'")

    df_model_cost.    set_index(df_data.index,inplace=True)
    df_model_deltaEta.set_index(df_data.index,inplace=True)

    df_data['Tbif2_new[h]'] = df_data['Tbif2_new[h]'].values/(df_model_cost['Tf'].values)
    print(df_data['Tbif2_new[h]'])

    plt.plot(df_data          ['Tbif2_new[h]']     ,'o',label= 'lab data'                 )
    plt.plot(df_model_cost    ['Tbif2_new'   ][:-1],'o',label=r'$\alpha = cost$'          )
    plt.plot(df_model_deltaEta['Tbif2_new'   ][:-1],'o',label=r'$\alpha = f(\Delta \eta)$')
    plt.ylabel(r'$T^{\prime \prime}_{BIF,new}$',fontsize=fs)
    plt.legend(prop={'size': fs})
    plt.grid()
    
    """
    df_cost = pd.read_csv('results/labData/outputLabCost.csv')
    df_eta  = pd.read_csv('results/labData/outputLabDeltaEtaLin.csv')
    
    plt.figure(1)
    plt.plot(df_cost['Tbif1'],'o',label=r'$T\/^{\prime}_{BIF} \alpha=cost$' )
    plt.plot(df_eta ['Tbif1'],'o',label=r'$T\/^{\prime}_{BIF} \alpha=f(\Delta \eta)$')
    plt.legend(prop={'size': fs})
    plt.grid()

    plt.figure(2)
    plt.plot(df_cost['Tbif2'],'o',label=r'$T\/^{\prime \prime}_{BIF} \/ \alpha=cost$' )
    plt.plot(df_eta ['Tbif2'],'o',label=r'$T\/^{\prime \prime}_{BIF} \/ \alpha=f(\Delta \eta)$')
    df_measures = pd.read_csv('results/labData/Tbif2_measured.csv',index_col=0)
    plt.plot(df_measures[:4],'o',label=r'$T \/ ^{\prime \prime}_{BIF} lab data$')
    plt.xlabel('Run ID',fontsize=fs)
    plt.ylabel(r'$T \/ ^{\prime \prime} _{BIF} [-]$',fontsize=fs)
    plt.legend(prop={'size': fs},loc='best')
    plt.grid()
    """
plt.show()