#! The model assumes W_b = W_c = rW * W0

from functions import *

plt.rcParams.update({
    "mathtext.fontset": "cm" })

# -------------------------
# INPUTS AND MODEL SETTINGS
# -------------------------

# Model main settings
dsBC       = 2 # downstream boundary condition switch. 0: H=H(t) according to THd; 1: deltaH imposed by a confluence
# imposed by a confluence is computed at each iteration according to the linear momentum cons. eqns of Ragno (2021); 2: near-critical water depth
inStepIC   = -0.01 # =(inlet step at t=0)/(inlet step at equilibrium computed via BRT)
alpha_var  = 'cost' # 'deltaEtaLin': alpha varies linearly with the inlet step
THd        = 0 # =downstream BC timescale (the higher THd, the slower Hd varies over time)
HdTriangle = False

# I/O settings
numPlots    = 20
expFittings = True
fontsize    = 14

# Numerical parameters
CFL       = 0.2 # Courant-Friedrichs-Lewy parameter for timestep computation
dx        = 0.3
dt_set    = 0 # if == 0, CFL condition is used
tend      = 300
nIterEq   = int(5e2) # if ∆Q remains constant for nIterEq iterations, the simulation ends
maxIter   = int(1e5) # max number of iterations during time evolution
tol       = 1e-10 # Iterations tolerance

# Hydraulic and geometry parameters
TF        = 'P90' # sediment transport formula. Available options: 'P78' (Parker78), 'MPM', 'P90' (Parker90), 'EH' (Engelund&Hansen)
RF        = 0 # flow resistance formula switch: if=0, C=C0 is a constant; if=1, C varies according to local water depth
#beta0s  = [11.09873362  ,9.71622609   ,8.65709147   ,8.34953862   ]
#theta0s = [0.04895901   ,0.04870914   ,0.05264361   ,0.0650794    ]
#ds0s    = [0.03884557   ,0.03400679   ,0.03029982   ,0.02922339   ]
#C0s     = [13.88212779  ,14.21471212  ,14.50325831  ,14.59368962  ]
#Ls      = [451.96509678 ,395.66631816 ,352.53600262 ,340.01176704 ]
#d50     = 6.3e-4
#alpha_defs = [1.7, 4.5, 8.0, 0] # values retrieved from Bertoldi2007 (best fit on equilibrium deltaQ)
#alpha_defs = [2  , 2.8, 3.8, 0] # values retrieved from Redolfi2016  (alphaT)
#alpha_defs = [2.00288421, 5.86588959, 7.77063376, 8.18691757]
rW        = 0.24/0.36 # rW=W_b/W0=W_c/W0. Usually rW=0.5, for lab runs rW=0.24/0.36
p         = 0.6 # bed porosity
r         = 0.5 # Ikeda parameter
eps_c     = 1.1 # Chézy logarithmic formula coefficient (used only if RF=1 or C0=0)
beta0s  = [10]
theta0s = [0.05]
ds0s    = [0.035]
d50     = 6.3e-4
C0s     = [14]
Ls      = [400]
alpha_defs = [5]

# Confluence parameters
kI = 0.21
kb = 0.08
kc = kb

# Physical constants
delta = 1.63
g     = 9.81

# Setup output file
with open('output.csv', 'w') as f:
    f.writelines(['dsBC,inStepIC,dx,CFL,alpha_eq,alpha_var,THd,TF,RF,C0,L,beta0,(beta0-betaC)/betaC,theta0,ds0,d50,deltaQ_eq,deltaQ_BRT,Tf[h],Tbif1,Tbif2,THd/Tbif1,avulsionCheck\n'])

# Setup colors for figure with all deltaQ(time)
crange        = np.linspace(0,1,5)
deltaQ_colors = plt.cm.viridis(crange)

nSim = -1
for alpha_def,C0,L,beta0,theta0,ds0 in zip(alpha_defs,C0s,Ls,beta0s,theta0s,ds0s):
#for L in Ls:
    nSim += 1
    # ----------------------------------
    # IC & BC DEFINITION AND MODEL SETUP
    # ----------------------------------

    # Main channel IC
    D0    = d50/ds0
    W0    = beta0*2*D0
    S0    = theta0*delta*ds0
    if C0 == 0:
        C0 = 6+2.5*np.log(D0/(eps_c*d50))
    phi00, phiD0, phiT0 = phis_scalar(theta0, TF, D0, d50)
    Q0    = uniFlowQ(RF, W0, S0, D0, d50, g, C0, eps_c)
    Qs0   = W0*np.sqrt(g*delta*d50**3)*phi00
    Fr0   = Q0/(W0*D0*np.sqrt(g*D0))
    nc    = int(L*D0/dx)

    # Compute βR and then equilibrium value for alpha assuming betaR=betaC (Redolfi et al., 2019)
    betaR = betaR_MR(theta0,  r,phiD0,phiT0,C0)
    betaC = betaC_MR(theta0,1,r,phiD0,phiT0,C0)
    alpha_MR = betaR/betaC
    if alpha_def == 0:
        alpha_eq = alpha_MR
    else:
        alpha_eq = alpha_def

    # Compute critical aspect ratio for the channel loop
    betaC_loop = betaC_loopNR(r,alpha_MR,theta0,Fr0,S0,L,D0,C0,kI,kb,TF,RF,d50)
    
    # Compute critical aspect ratio for current simulation
    if dsBC == 0 or dsBC == 2:
        betaC = betaR
    elif dsBC == 1:
        betaC = betaC_loop

    # Equilibrium ∆Q computation through BRT method
    BRT_out       = deltaQ_BRT_2(alpha_eq,[1.2,0.8,0.9,0.9],dsBC,RF,TF,beta0,theta0,ds0,rW,L,r,g,delta,d50,C0,eps_c)
    deltaQ_eq_BRT = BRT_out[0]
    inStep_eq_BRT = abs(BRT_out[1])

    # Exner time computation
    Tf = (1-p)*W0*D0/(Qs0/W0)

    # Print model settings and parameters
    print('\nINPUT PARAMETERS:\nFlow resistance formula = %s\nSolid transport formula = %s\nAlpha variation mode = %s\nnc = %d\ndx = %4.2f m'
          '\nCFL = %3.2f \nt_end = %2.1f Tf\nL = %d\nβ0 = %4.2f\nθ0 = %4.3f\nds0 = %2.1e\nα_eq = %2.1f\nr = %2.1f'
          '\nd50 = %3.2e m\n'
          '\nRESONANT ASPECT RATIO AND EQUILIBRIUM α ACCORDING TO REDOLFI ET AL., 2019\nβR = %4.2f\n(β0-βR)/βR = %4.2f\nα_MR = %2.1f\n'
          '\nMAIN CHANNEL IC:\nW0 = %4.2f m\nS0 = %3.2e\nD0 = %3.2f m\nFr0 = %2.1f\nL* = %4.1f m\nL*/W0 = %4.1f\n'
          'Q0 = %3.2e m^3 s^-1\nQs0 = %3.2e m^3 s^-1\n\nExner time: Tf = %3.2f h\n'
          % (RF, TF, alpha_var, nc, dx, CFL, tend, L, beta0, theta0, ds0, alpha_eq, r, d50, betaR, (beta0-betaR) / betaR,
             alpha_MR, W0, S0, D0, Fr0, L*D0, L*D0/W0, Q0, Qs0, Tf/3600))
    print("BRT EQUILIBRIUM SOLUTION:")
    print("∆Q = %5.4f\n∆η = %5.4f\nθ_b = %4.3f, θ_c = %4.3f\nS_b = %2.1e, S_c = %2.1e\n" % (BRT_out))

    if dsBC==1:
        print('CHANNEL LOOP CRITICAL ASPECT RATIO (RAGNO ET AL., 2021)')
        print('βC = %4.2f' % betaC_loop)
        print('(β0-βC)/βC = %4.2f\n' % ((beta0-betaC_loop)/betaC_loop))
    # Arrays initialization to IC
    t         : list = [0]
    dts       : list = []  # list where to store the timestep values computed by means of CFL
    inStep    : list = [inStepIC*inStep_eq_BRT]
    deltaQ    : list = [0]
    eta_bn    = np.zeros(maxIter+1)
    eta_cn    = np.zeros(maxIter+1)
    eta_a     = np.zeros((maxIter+1,nc+1))
    eta_b     = np.zeros((maxIter+1,nc+1))
    eta_c     = np.zeros((maxIter+1,nc+1))
    D_a       = np.ones((maxIter+1,nc+1))*D0
    D_b       = np.ones((maxIter+1,nc+1))*(0.5/rW)**(2/3)*D0
    D_c       = np.ones((maxIter+1,nc+1))*(0.5/rW)**(2/3)*D0
    S_a       = np.ones((maxIter+1,nc))*S0
    S_b       = np.ones((maxIter+1,nc))*S0
    S_c       = np.ones((maxIter+1,nc))*S0
    Theta_a   = np.ones((maxIter+1,nc+1))*theta0
    Theta_b   = np.ones((maxIter+1,nc+1))*(0.5/rW)**(2/3)*theta0
    Theta_c   = np.ones((maxIter+1,nc+1))*(0.5/rW)**(2/3)*theta0
    Q_b       = np.ones(maxIter+1)*Q0/2
    Q_c       = np.ones(maxIter+1)*Q0/2
    Qs_a      = np.ones((maxIter+1,nc+1))*Qs0
    Qs_b      = np.ones((maxIter+1,nc+1))*Qs0/2
    Qs_c      = np.ones((maxIter+1,nc+1))*Qs0/2
    Qs_y      = np.zeros(maxIter+1)
    Hd_b      = np.zeros(maxIter+1)
    Hd_c      = np.zeros(maxIter+1)
    xi        = np.linspace(0,nc*dx,nc+1)  # cell interfaces coordinates

    # Bed elevation and width IC. Horizontal step -> elevation of first cell of each branch is equal to the corresponding node cell
    eta_bn[0]  =  inStep[0]*D0/2
    eta_cn[0]  = -inStep[0]*D0/2
    eta_a[0,:] = np.linspace(S0*dx*nc,0,num=nc+1)  # 0 is the mean bed elevation at the node, i.e., 0.5*(eta_bn+eta_cn)
    eta_b[0,:] = np.linspace(eta_bn[0], eta_bn[0]-S0*nc*dx, num=nc+1)
    eta_c[0,:] = np.linspace(eta_cn[0], eta_cn[0]-S0*nc*dx, num=nc+1)
    W_b = rW*W0
    W_c = W_b

    # Branches downstream BC: H(t)=H0
    H0 = 0.5*(eta_b[0,-1]+eta_c[0,-1])+(0.5/rW)**(2/3)*D0  # H0 is set according to the normal depth in the downstream branches,
    # that in the general case is equal to D=(0.5/rW)**(2/3)*D0 (using Chezy's resistance formula), where D0 is the normal depth of the upstream channel
    Hd_b[0] = H0
    Hd_c[0] = H0

    # alpha time variation setup
    if alpha_var == 'cost':
        alpha = alpha_eq
    else:
        alpha_eps = 0.1
        alpha = alpha_eps+(alpha_eq-alpha_eps)*abs(inStep[0]/inStep_eq_BRT)

    # Time evolution
    eqIndex  = 0  # time index at which system reaches equilibrium
    regIndex = 0  # time index at which sustem reaches a regime condition
    avulsionCheck = 0
    for n in range(1, maxIter):
        # Compute dt according to CFL condition and update time. Check if system has reached equilibrium
        if dt_set == 0:
            # Compute propagation celerity along the three branches
            Ceta_a = C_eta(Q0    ,W0 ,D_a[n,:],g,delta,d50,p,C0,D0,RF,TF,eps_c)
            Ceta_b = C_eta(Q_b[n],W_b,D_b[n,:],g,delta,d50,p,C0,D0,RF,TF,eps_c)
            Ceta_c = C_eta(Q_c[n],W_c,D_c[n,:],g,delta,d50,p,C0,D0,RF,TF,eps_c)

            # Compute mean variables at the node to compute propagation celerity at the node
            Wn     = np.asarray([0.5*(W0+0.5*(W_b+W_c))])
            Dn     = np.asarray([0.5*(D_a[n,-1]+0.5*(D_b[n,0]+D_c[n,0]))])
            Ceta_n = C_eta(Q0,Wn,Dn,g,delta,d50,p,C0,D0,RF,TF,eps_c) # propagation velocity of a bed perturbation at the node, computed considering Exner 1D applied to both cells
            Cmax   = max(max(Ceta_a),max(Ceta_b),max(Ceta_c),Ceta_n)
            dt     = CFL*dx/Cmax
        else:
            dt = dt_set
        dts.append(dt)
        t.append(t[-1]+dt)

        # Check for avulsion
        if abs(deltaQ[-1])>0.98:
            print("\nAvulsion occured\n")
            avulsionCheck = 1
            break

        # Check if equilibrium or end of simulation time have been reached
        if n>nIterEq:
            if np.all(abs((np.asarray(deltaQ[-nIterEq:])+tol)/(deltaQ[-1]+tol)-1)<np.sqrt(tol)):
                if eqIndex == 0:
                    eqIndex = n
                    print('\nEquilibrium reached\n')
                    if THd == 0:
                        break
                else:
                    print('\nRegime condition reached\n')
                    if HdTriangle and regIndex == 0:
                        regIndex = n
                        THd      = -THd
                    else:
                        break
        if t[-1] >= (tend*Tf):
            eqIndex = n
            print('\nEnd time reached\n')
            break

        # Update and impose the downstream BC
        if dsBC == 0:
            # Equal wse at downstream ends of branches b and c
            if eqIndex == 0:
                Hd_b[n] = H0
                Hd_c[n] = H0
            elif THd != 0:
                Hd_b[n] = Hd_b[n-1]+D0/THd*dt/Tf
                Hd_c[n] = Hd_c[n-1]+D0/THd*dt/Tf
        elif dsBC == 1:
            # deltaH set by a confluence
            Hd_b[n],Hd_c[n] = deltaH_confluence(Q_b[n-1],Q_c[n-1],D_b[n-1,-1],D_c[n-1,-1],W_b,W_c,Q0,D0,W0,H0,kI,kb,kc,g)
        elif dsBC == 2:
            # Critical depth
            Hd_b[n] = eta_b[n-1,-1]+criticalDepth(Q_b[n-1],g,W_b)*1.05
            Hd_c[n] = eta_c[n-1,-1]+criticalDepth(Q_c[n-1],g,W_c)*1.05
        D_b[n,-1] = Hd_b[n]-eta_b[n-1,-1]
        D_c[n,-1] = Hd_c[n]-eta_c[n-1,-1]

        # Compute the discharge partitioning at the node through profiles+(Hb=Hc)
        D_b[n,:] = buildProfile_rk4(RF,D_b[n,-1],Q_b[n-1],W_b,S_b[n-1,:],d50,dx,g,C0,eps_c)
        D_c[n,:] = buildProfile_rk4(RF,D_c[n,-1],Q_c[n-1],W_c,S_c[n-1,:],d50,dx,g,C0,eps_c)
        if abs((D_b[n,0]-D_c[n,0]+inStep[-1]*D0)/(0.5*(D_b[n,0]+D_c[n,0])))>tol:
            Q_b[n]   = newton(Q_b[n-1],fSys_rk4,(RF,D_b[n,-1],D_c[n,-1],D0,inStep[-1],Q0,W_b,W_c,S_b[n-1,:],S_c[n-1,:],d50,dx,g,C0,eps_c))
            Q_c[n]   = Q0-Q_b[n]
            D_b[n,:] = buildProfile_rk4(RF,D_b[n,-1],Q_b[n],W_b,S_b[n-1,:],d50,dx,g,C0,eps_c)
            D_c[n,:] = buildProfile_rk4(RF,D_c[n,-1],Q_c[n],W_c,S_c[n-1,:],d50,dx,g,C0,eps_c)
        else:
            Q_b[n] = Q_b[n-1]
            Q_c[n] = Q_c[n-1]

        # Computer water-surface profile along channel a
        D_a[n,-1] = 0.5*(D_b[n,0]+D_c[n,0])+0.5*(eta_bn[n-1]+eta_cn[n-1])-eta_a[n-1,-1]
        if D_a[n,-1] < 1.1*criticalDepth(Q0,g,W0):
            D_a[n,-1] = 1.1*criticalDepth(Q0,g,W0)
        D_a[n, :] = buildProfile_rk4(RF,D_a[n,-1],Q0,W0,S_a[n-1,:],d50,dx,g,C0,eps_c)

        #Shields and Qs update for the three channels
        Theta_a[n,:] = shieldsUpdate(RF,Q0    ,W0 ,D_a[n,:],d50,g,delta,C0,eps_c)
        Theta_b[n,:] = shieldsUpdate(RF,Q_b[n],W_b,D_b[n,:],d50,g,delta,C0,eps_c)
        Theta_c[n,:] = shieldsUpdate(RF,Q_c[n],W_c,D_c[n,:],d50,g,delta,C0,eps_c)
        Qs_a[n,:]    = W0 *np.sqrt(g*delta*d50**3)*phis(Theta_a[n,:],TF,D0,d50)[0]
        Qs_b[n,:]    = W_b*np.sqrt(g*delta*d50**3)*phis(Theta_b[n,:],TF,D0,d50)[0]
        Qs_c[n,:]    = W_c*np.sqrt(g*delta*d50**3)*phis(Theta_c[n,:],TF,D0,d50)[0]

        # Compute liquid and solid discharge in transverse direction at the node; update bed elevation of node cells accordingly
        Q_y       = 0.5*(Q_b[n]-Q_c[n])
        Qs_y  [n] = Qs0*(Q_y/Q0-2*alpha*r/np.sqrt(Theta_a[n,-1])*(eta_bn[n-1]-eta_cn[n-1])/(0.5*(W0+W_b+W_c)))
        eta_bn[n] = eta_bn[n-1]+dt*(Qs0/2-Qs_b[n,0]+Qs_y[n])/((1-p)*0.5*(0.5*W0+W_b)*alpha*W0)
        eta_cn[n] = eta_cn[n-1]+dt*(Qs0/2-Qs_c[n,0]-Qs_y[n])/((1-p)*0.5*(0.5*W0+W_c)*alpha*W0)

        # Update of bed elevation along the branches
        eta_a[n,0 ] = eta_a [n-1,0] -dt/dx*(Qs_a[n,1]-Qs0)/(W0*(1-p))
        eta_a[n,1:] = eta_a [n-1,1:]-dt/dx*(Qs_a[n,1:]-Qs_a[n,:-1])/(W0*(1-p))
        eta_b[n,0 ] = eta_bn[n] # horizontal step
        eta_c[n,0 ] = eta_cn[n]
        eta_b[n,1:] = eta_b [n-1,1:]-dt/dx*(Qs_b[n,1:]-Qs_b[n,:-1])/(W_b*(1-p)) # upwind exner (Fr<1, so bed level perturbations travel downstream)
        eta_c[n,1:] = eta_c [n-1,1:]-dt/dx*(Qs_c[n,1:]-Qs_c[n,:-1])/(W_c*(1-p))

        # If dsBC==2, make last cell non-erodible
        if dsBC == 2 and eta_b[n,-1] < eta_b[0,-1]:
            eta_b[n,-1] = eta_b[0,-1]
        if dsBC == 2 and eta_c[n,-1] < eta_c[0,-1]:
            eta_c[n,-1] = eta_c[0,-1]

        # Update bed slopes
        S_a[n,:] = (eta_a[n,:-1]-eta_a[n,1:])/dx
        S_b[n,:] = (eta_b[n,:-1]-eta_b[n,1:])/dx
        S_c[n,:] = (eta_c[n,:-1]-eta_c[n,1:])/dx

        # Update time-controlled lists
        deltaQ.append((Q_b[n]-Q_c[n])/Q0)
        inStep.append((eta_bn[n]-eta_cn[n])/D0)

        # alpha update
        if alpha_var == 'deltaEtaLin':
            alpha = alpha_eps+(alpha_eq-alpha_eps)*abs(inStep[-1]/inStep_eq_BRT)

        # Time print
        if n % 500 == 0:
            print("Elapsed time = %4.1f Tf, ∆Q = %5.4f" % (t[n]/Tf,deltaQ[n]))

    deltaQ = np.asarray(deltaQ)  #for plots

    # Print final ∆Q and compare it with that computed through BRT model
    print('Final ∆Q = %5.4f' % deltaQ[-1])
    print('∆Q at equilibrium according to BRT = %5.4f' % deltaQ_eq_BRT)
    print('Difference = %2.1f %%\n' % (100*abs((abs(deltaQ[-1])-deltaQ_eq_BRT)/deltaQ_eq_BRT)))
    
    # Compute first stage evolutionary timescale (Tbif1=1/Omega)
    flexIndex = flexFinder(deltaQ,(np.asarray(t[1:-2])-np.asarray(t[:-3]))/Tf)
    Tbif1,Tbif2 = [0,0]
    if expFittings:
        exp2FitStart = flexIndex
        #exp2FitEnd   = np .where(deltaQ==np.max(deltaQ))[0][0]
        exp2FitEnd   = n-1
        exp2Par,exp2Fit,exp2Error = interpolate(expLab,t[exp2FitStart:exp2FitEnd]/Tf,deltaQ[exp2FitStart:exp2FitEnd],ic=[deltaQ[exp2FitStart], deltaQ[exp2FitEnd], 0.2])
        Tbif2        = 1/exp2Par [2]
        if flexIndex != 0:
            exp1FitEnd                = int(0.8*flexIndex)
            exp1Par,exp1Fit,exp1Error = interpolate(exponential,t[:exp1FitEnd]/Tf,deltaQ[:exp1FitEnd],ic=[deltaQ[0], 0.2])
            Tbif1 = 1/exp1Par[1]

    # Save outputs on text files
    with open('output.csv','a') as f:
        f.writelines(['%d,%4.3f,%3.1f,%2.1f,%3.2f,%s,%2.1f,%s,%d,%3.1f,%d,%3.1f,%4.3f,%4.3f,%4.3f,%4.3f,%5.4f,%5.4f,%4.3f,%4.3f,%4.3f,%5.3f,%d\n'
        % (dsBC,inStepIC,dx,CFL,alpha_eq,alpha_var,THd,TF,RF,C0,L,beta0,(beta0-betaC)/betaC,theta0,ds0,d50,deltaQ[-1],deltaQ_eq_BRT,Tf/3600,Tbif1,Tbif2,THd/(Tbif1+tol),avulsionCheck)])
    deltaQtime = np.zeros((n,2))
    deltaQtime[:,0] = t[:-1]/Tf
    deltaQtime[:,1] = deltaQ
    np.savetxt('deltaQtime.txt',deltaQtime)

    # Plot ∆Q evolution over time along with exponential fittings for each simulation
    myPlot(nSim+1,t[1:-1]/Tf,deltaQ[1:]/deltaQ_eq_BRT,'Computed','Discharge asymmetry vs time',r'$t^*/T_F \/ [-]$',r'$\Delta Q/\Delta Q_{BRT} [-]$',fontsize=fontsize)
    if flexIndex != 0 and expFittings and deltaQ[n-1] != 0:
        myPlot(nSim+1,t[1:exp1FitEnd]/Tf,exp1Fit[1:]/deltaQ_eq_BRT,'First stage exponential fit')
        myPlot(nSim+1,t[exp2FitStart:exp2FitEnd]/Tf,exp2Fit/deltaQ_eq_BRT,'Second stage exponential fit')
    plt.savefig('plots/' + str(nSim)+'.pdf')
    plt.close()

    # Plot ∆Q evolutions over time for all simulations in the same figure
    myPlot(0,t[1:-1]/Tf,deltaQ[1:],nSim,'Discharge asymmetry vs time',r'$t^*/T_F \/ [-]$',r'$\Delta Q [-]$',color=deltaQ_colors[nSim],fontsize=fontsize)

plt.savefig('plots/all.pdf')

# PLOTS
# -----
nFig = nSim+2

# Bed evolution plot
plotTimeIndexes = np.linspace(0, n-1, numPlots)
crange          = np.linspace(0,   1, numPlots)
bed_colors      = plt.cm.viridis(crange)
plt.figure(nFig                            )
plt.title ('Branch A bed evolution in time')
plt.xlabel(r'$x/W_0 [-]$'                  ,fontsize=fontsize)
plt.ylabel(r'$(\eta-\eta_0)/D_0 [-]$'      ,fontsize=fontsize)
plt.figure(nFig+1                          )
plt.title ('Branch B bed evolution in time')
plt.xlabel(r'$x/W_0 [-]$'                  ,fontsize=fontsize)
plt.ylabel(r'$(\eta-\eta_0)/D_0 [-]$'      ,fontsize=fontsize)
plt.figure(nFig+2                          )
plt.title ('Branch C bed evolution in time')
plt.xlabel(r'$x/W_0 [-]$'                  ,fontsize=fontsize)
plt.ylabel(r'$(\eta-\eta_0)/D_0 [-]$'      ,fontsize=fontsize)
for i in range(numPlots):
    plotTimeIndex = int(plotTimeIndexes[i])
    myPlot(nFig,   xi/W0, (eta_a[plotTimeIndex,:]-eta_a[0,:])/D0, ('t=%3.1f Tf' % (t[plotTimeIndex]/Tf)), color=bed_colors[i])
    myPlot(nFig+1, xi/W0, (eta_b[plotTimeIndex,:]-eta_b[0,:])/D0, ('t=%3.1f Tf' % (t[plotTimeIndex]/Tf)), color=bed_colors[i])
    myPlot(nFig+2, xi/W0, (eta_c[plotTimeIndex,:]-eta_c[0,:])/D0, ('t=%3.1f Tf' % (t[plotTimeIndex]/Tf)), color=bed_colors[i])
nFig += 2

# Plot evolution of node cells over time
nFig += 1
myPlot(nFig,t[:-1]/Tf,(eta_bn[:n]-0.5*(eta_bn[0]+eta_cn[0]))/D0,'Node cell B','Evolution of node cells elevation over time', r'$t^*/T_F \/ [-]$', r'$(\eta-\eta_0)/D_0 [-]$',fontsize=fontsize)
myPlot(nFig,t[:-1]/Tf,(eta_cn[:n]-0.5*(eta_bn[0]+eta_cn[0]))/D0,'Node cell C')

# Plot bed evolution at relevant cross-sections (upstream, middle, downstream)
nFig += 1
fig, ax = plt.subplots(1, 3, num=nFig)
ax[0].plot(t[:-1]/Tf, (eta_b[:n,       1 ]-eta_b[0,       1 ])/D0, label='Branch B')
ax[0].plot(t[:-1]/Tf, (eta_c[:n,       1 ]-eta_c[0,       1 ])/D0, label='Branch C')
ax[1].plot(t[:-1]/Tf, (eta_b[:n,int(nc/2)]-eta_b[0,int(nc/2)])/D0, label='Branch B')
ax[1].plot(t[:-1]/Tf, (eta_c[:n,int(nc/2)]-eta_c[0,int(nc/2)])/D0, label='Branch C')
ax[2].plot(t[:-1]/Tf, (eta_b[:n,-      1 ]-eta_b[0,-      1 ])/D0, label='Branch B')
ax[2].plot(t[:-1]/Tf, (eta_c[:n,-      1 ]-eta_c[0,-      1 ])/D0, label='Branch C')
subplotsLayout(ax, [r'$t^*/T_F \/ [-]$', r'$t^*/T_F \/ [-]$', r'$t^*/T_F \/ [-]$'], [r'$(\eta-\eta_0)/D_0 [-]$', None, None],
               ['upstream', 'Bed elevation vs time\n\nmiddle', 'downstream'])

# Plot ∆η evolution over time
nFig += 1
myPlot(nFig,t[:-1]/Tf,abs(inStep/inStep_eq_BRT),None,'Non-dimensional inlet step vs time',r'$t^*/T_F \/ [-]$',r'$\vert \Delta \eta/\Delta \eta_{BRT}\vert [-]$')

# Plot evolution of bed slopes over time
nFig += 1
myPlot(nFig,t[:-1]/Tf,np.mean(S_b[:n],axis=1),'Branch B average slope',xlabel=r'$t^*/T_F \/ [-]$',ylabel=r'$S \/ [-]$',fontsize=fontsize)
myPlot(nFig,t[:-1]/Tf,np.mean(S_c[:n],axis=1),'Branch C average slope'                                                )

# Plot final water-surface profiles
nFig += 1
myPlot(nFig,xi/W0,D_b[n-1,:]/D0,'Branch B water depth',xlabel=r'$x/W_0 [-]$',ylabel=r'$D/D_0 \/ [-]$',fontsize=fontsize)
myPlot(nFig,xi/W0,D_c[n-1,:]/D0,'Branch C water depth')

# Plot wse over time if a confluence is set as a downstream BC
if dsBC==1:
    nFig += 1
    myPlot(nFig,t[1:-1]/Tf,(Hd_b[1:n]-H0)/D0,'Hd_b',xlabel=r'$t^*/T_F \/ [-]$',ylabel=r'$(H_d-H_0)/D_0 \/ [-]$',fontsize=fontsize)
    myPlot(nFig,t[1:-1]/Tf,(Hd_c[1:n]-H0)/D0,'Hd_c')

plt.show()
print("Done")