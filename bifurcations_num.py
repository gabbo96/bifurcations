from functions import *

# -------------------------
# INPUTS AND MODEL SETTINGS
# -------------------------

# Model main settings
dsBC           = 0 # if 0, downstream BC is H0, possibly varying according to THd; if 1, the deltaH
# imposed by a confluence is computed at each iteration according to the linear momentum cons. eqns of Ragno (2020)
inStepIC       = -0.01  # =(inlet step at t=0)/(inlet step at equilibrium computed via BRT)
alpha_var      = 'cost' # 'deltaEtaLin': alpha varies linearly with the inlet step
THd            = 0 # =downstream BC timescale (the higher THd, the slower Hd varies over time)

# I/O settings
numPlots    = 20
saveOutputs = False
expFittings = True

# Setup output file
with open('output.txt', 'w') as f:
    f.writelines(['dsBC;inStepIC;alpha_var;THd;TF;RF;C0;Ls;beta0;theta0;ds0;d50;deltaQ_eq;deltaQ_BRT;Tbif1;Tbif2\n'])

# Numerical parameters
CFL       = 0.9 # Courant-Friedrichs-Lewy parameter for timestep computation
dx        = 50 # cell length [m]
tend      = 600
tEq       = 4 # if ∆Q remains constant for tEq (non-dimensional), the simulation ends
maxIter   = int(1e6) # max number of iterations during time evolution
tol       = 1e-10 # Iterations tolerance

# Hydraulic parameters
TF     = 'P90' # sediment transport formula. Available options: 'P78' (Parker78), 'MPM', 'P90' (Parker90), 'EH' (Engelund&Hansen)
RF     = 0 # flow resistance formula switch: if=0, C=C0 is a constant; if=1, C varies according to local water depth
C0     = 12 # if =0, it's computed as a function of D0=ds0/d50 via a logarithmic formula
eps_c  = 2.5 # Chézy logarithmic formula coefficient (used only if RF=1 or C0=0)
Ls     = 750 # =L/D0, L=branches' dimensional length
beta0  = 16
theta0 = 0.08
ds0    = 0.01 # =d50/D0
d50    = 0.01 # median sediment diameter [m]
p      = 0.6 # bed porosity
alpha  = 0  # if ==0, it is computed by assuming betaR=betaC (Redolfi et al., 2019)
r      = 0.5 # Ikeda parameter

# Confluence parameters
kI = 0.21
kb = 0.08
kc = kb

# Physical constants
delta = 1.65
g     = 9.81

# ----------------------------------
# IC & BC DEFINITION AND MODEL SETUP
# ----------------------------------

# Main channel IC
S0    = theta0*delta*ds0
D0    = d50/ds0
W0    = beta0*2*D0
if C0 == 0:
    C0 = 6+2.5*np.log(D0/(eps_c*d50))
phi00, phiD0, phiT0 = phis_scalar(theta0, TF, D0, d50)
Q0    = uniFlowQ(RF, W0, S0, D0, d50, g, C0, eps_c)
Qs0   = W0*np.sqrt(g*delta*d50**3)*phi00
Fr0   = Q0/(W0*D0*np.sqrt(g*D0))
nc    = int(Ls*D0/dx)

# Compute βR and then equilibrium value for alpha assuming betaR=betaC (Redolfi et al., 2019)
betaR = betaR_MR(RF,theta0,ds0,r,phiD0,phiT0,eps_c)
betaC = betaC_MR(RF,theta0,ds0,1,r,phiD0,phiT0,eps_c)
alpha_MR = betaR/betaC
if alpha == 0:
    alpha_eq = alpha_MR
else:
    alpha_eq = alpha

# Compute critical aspect ratio for the channel loop
betaC_loop = betaC_loopNR(r,alpha_MR,theta0,Fr0,S0,Ls,D0,C0,kI,kb,TF,RF,d50)

# Equilibrium ∆Q computation through BRT method
BRT_out = deltaQ_BRT([1.1,0.9,0.9,1.1],0,RF,TF,theta0,Q0,Qs0,W0,Ls*D0,D0,S0,0.5*W0,0.5*W0,d50,alpha_eq,r,g,delta,tol,C0,eps_c)
deltaQ_eq_BRT = BRT_out[0]
inStep_eq_BRT = (BRT_out[7]-BRT_out[8])/D0

# Exner time computation
Tf = (1-p)*W0*D0/(Qs0/W0)*alpha_eq

# Print model settings and parameters
print('\nINPUT PARAMETERS:\nFlow resistance formula = %s\nSolid transport formula = %s\nnc = %d\ndx = %4.2f m'
      '\nCFL = %3.2f \nt_end = %2.1f Tf\nL* = %d\nβ0 = %4.2f\nθ0 = %4.3f\nds0 = %2.1e\nα = %2.1f\nα_eq = %2.1f\nr = %2.1f'
      '\nd50 = %3.2e m\n'
      '\nRESONANT ASPECT RATIO AND EQUILIBRIUM α ACCORDING TO REDOLFI ET AL., 2019\nβR = %4.2f\n(β0-βR)/βR = %4.2f\nα_MR = %2.1f\n'
      '\nMAIN CHANNEL IC:\nW0 = %4.2f m\nS0 = %3.2e\nD0 = %3.2f m\nFr0 = %2.1f\nL = %4.1f m\nL/W0 = %4.1f\n'
      'Q0 = %3.2e m^3 s^-1\nQs0 = %3.2e m^3 s^-1\n\nExner time: Tf = %3.2f h\n'
      % (RF, TF, nc, dx, CFL, tend, Ls, beta0, theta0, ds0, alpha, alpha_eq, r, d50, betaR, (beta0-betaR) / betaR,
         alpha_MR, W0, S0, D0, Fr0, Ls*D0, Ls*D0/W0, Q0, Qs0, Tf/3600))
print("BRT EQUILIBRIUM SOLUTION:\n∆η = %5.4f" % inStep_eq_BRT)
print("∆Q = %5.4f\nθ_b = %4.3f, θ_c = %4.3f\nFr_b = %3.2f, Fr_c = %3.2f\nS = %2.1e\n" % (BRT_out[:6]))

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
D_b       = np.ones((maxIter+1,nc+1))*D0
D_c       = np.ones((maxIter+1,nc+1))*D0
S_a       = np.ones((maxIter+1,nc))*S0
S_b       = np.ones((maxIter+1,nc))*S0
S_c       = np.ones((maxIter+1,nc))*S0
Theta_a   = np.ones((maxIter+1,nc+1))*theta0
Theta_b   = np.ones((maxIter+1,nc+1))*theta0
Theta_c   = np.ones((maxIter+1,nc+1))*theta0
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
W_b = 0.5*W0
W_c = 0.5*W0

# Branches downstream BC: H(t)=H0
H0 = 0.5*(eta_b[0,-1]+eta_c[0,-1])+D0
Hd_b[0] = H0
Hd_c[0] = H0

# alpha time variation setup
if alpha_var == 'cost':
    alpha = alpha_eq
else:
    alpha = 0.1+(alpha_eq-0.1)*abs(inStep[0]/inStep_eq_BRT)

# Time evolution
eqReached = False
for n in range(1, maxIter):
    # Compute dt according to CFL condition and update time. Check if system has reached equilibrium
    Ceta_a = C_eta(Q0    ,W0 ,D_a[n,:],g,delta,d50,p,C0,D0,RF,TF,eps_c)
    Ceta_b = C_eta(Q_b[n],W_b,D_b[n,:],g,delta,d50,p,C0,D0,RF,TF,eps_c)
    Ceta_c = C_eta(Q_c[n],W_c,D_c[n,:],g,delta,d50,p,C0,D0,RF,TF,eps_c)
    Cmax   = max(max(Ceta_a),max(Ceta_b),max(Ceta_c))
    dt     = CFL*dx/Cmax
    dts.append(dt)
    t.append(t[-1]+dt)
    
    # Check for avulsion
    if abs(deltaQ[-1])>0.98:
        print("\nAvulsion occured\n")
        break

    # Check if equilibrium or end of simulation time have been reached
    if t[-1]>tEq*Tf:
        if np.all(abs(deltaQ[-int(tEq*Tf/dt):]/(deltaQ[-1])-1)<np.sqrt(tol)):
            if THd == 0:
                print('\nEquilibrium reached\n')
                break
            elif not eqReached:
                print('\nEquilibrium reached\n')
                eqReached = True
            else:
                print('\nRegime condition reached\n')
                break
    if t[-1] >= (tend*Tf):
        print('\nEnd time reached\n')
        break

    # Update and impose the downstream BC
    if dsBC == 0:
        # Equal wse at downstream ends of branches b and c
        if not eqReached:
            Hd_b[n] = H0
            Hd_c[n] = H0
        elif eqReached and THd != 0:
            Hd_b[n] = Hd_b[n-1]+D0/THd*dt/Tf
            Hd_c[n] = Hd_c[n-1]+D0/THd*dt/Tf
    elif dsBC == 1:
        # deltaH set by a confluence
        Hd_b[n],Hd_c[n] = deltaH_confluence(Q_b[n-1],Q_c[n-1],D_b[n-1,-1],D_c[n-1,-1],W_b,W_c,Q0,D0,W0,H0,kI,kb,kc,g)

    D_b[n,-1] = Hd_b[n]-eta_b[n-1,-1]
    D_c[n,-1] = Hd_c[n]-eta_c[n-1,-1]
    
    # Compute the discharge partitioning at the node through profiles+(Hb=Hc)
    D_b[n,:] = buildProfile_rk4(RF,D_b[n,-1],Q_b[n],W_b,S_b[n-1,:],d50,dx,g,C0,eps_c)
    D_c[n,:] = buildProfile_rk4(RF,D_c[n,-1],Q_c[n],W_c,S_c[n-1,:],d50,dx,g,C0,eps_c)
    if abs((D_b[n,0]-D_c[n,0]+inStep[-1]*D0)/(0.5*(D_b[n,0]+D_c[n,0])))>tol:
        Q_b[n]   = opt.fsolve(fSys_rk4,Q_b[n],(RF,D_b[n,-1],D_c[n,-1],D0,inStep[-1],Q0,W_b,W_c,S_b[n-1,:],S_c[n-1,:],d50,dx,g,C0,eps_c),xtol=tol)
        Q_c[n]   = Q0-Q_b[n]
        D_b[n,:] = buildProfile_rk4(RF,D_b[n,-1],Q_b[n],W_b,S_b[n-1,:],d50,dx,g,C0,eps_c)
        D_c[n,:] = buildProfile_rk4(RF,D_c[n,-1],Q_c[n],W_c,S_c[n-1,:],d50,dx,g,C0,eps_c)

    # Computer water-surface profile along channel a
    D_a[n,-1] = 0.5*(D_b[n,0]+D_c[n,0])+0.5*(eta_bn[n-1]+eta_cn[n-1])-eta_a[n-1,-1]
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
    Qs_y  [n] = Qs0*(Q_y/Q0-2*alpha*r/np.sqrt(Theta_a[n,-1])*(eta_bn[n-1]-eta_cn[n-1])/W0)
    eta_bn[n] = eta_bn[n-1]+dt*(Qs0/2-Qs_b[n,0]+Qs_y[n])/((1-p)*alpha*W0*W_b)
    eta_cn[n] = eta_cn[n-1]+dt*(Qs0/2-Qs_c[n,0]-Qs_y[n])/((1-p)*alpha*W0*W_c)
    
    # Update of bed elevation along the branches
    eta_a[n,0 ] = eta_a [n-1,0] -dt/dx*(Qs_a[n,1]-Qs0)/(W0*(1-p))
    eta_a[n,1:] = eta_a [n-1,1:]-dt/dx*(Qs_a[n,1:]-Qs_a[n,:-1])/(W0*(1-p))
    eta_b[n,0 ] = eta_bn[n] # horizontal step
    eta_c[n,0 ] = eta_cn[n]
    eta_b[n,1:] = eta_b [n-1,1:]-dt/dx*(Qs_b[n,1:]-Qs_b[n,:-1])/(W_b*(1-p)) # upwind exner (Fr<1, so bed level perturbations travel downstream)
    eta_c[n,1:] = eta_c [n-1,1:]-dt/dx*(Qs_c[n,1:]-Qs_c[n,:-1])/(W_c*(1-p))    

    # Update bed slopes
    S_a[n,:] = (eta_a[n,:-1]-eta_a[n,1:])/dx
    S_b[n,:] = (eta_b[n,:-1]-eta_b[n,1:])/dx
    S_c[n,:] = (eta_c[n,:-1]-eta_c[n,1:])/dx

    # Update time-controlled lists
    deltaQ.append((Q_b[n]-Q_c[n])/Q0)
    inStep.append((eta_bn[n]-eta_cn[n])/D0)

    # alpha update
    if alpha_var == 'deltaEtaLin':
        alpha = 0.1+(alpha_eq-0.1)*abs(inStep[-1]/inStep_eq_BRT)
    
    # Time print
    if n % 500 == 0:
        print("Elapsed time = %4.1f Tf, ∆Q = %5.4f" % (t[n]/Tf,deltaQ[n]))

# Print final ∆Q and compare it with that computed through BRT model
print('Final ∆Q = %5.4f' % deltaQ[-1])
print('∆Q at equilibrium according to BRT = %5.4f' % deltaQ_eq_BRT)
print('Difference = %2.1f %%\n' % (100*abs((abs(deltaQ[-1])-deltaQ_eq_BRT)/deltaQ_eq_BRT)))

# Compute evolutionary timescales
flexIndex = flexFinder(np.asarray(deltaQ),(np.asarray(t[1:-2])-np.asarray(t[:-3]))/Tf)
Tbif1,Tbif2 = [0,0]
if flexIndex != 0 and expFittings:
    exp1FitEnd                = int(0.8*flexIndex)
    exp1Par,exp1Fit,exp1Error = interpolate(exponential,t[:exp1FitEnd]/Tf,deltaQ[:exp1FitEnd],ic=[deltaQ[0], 0.2])
    exp2Par,exp2Fit,exp2Error = interpolate(expLab,t[flexIndex:-1]/Tf,deltaQ[flexIndex:],ic=[deltaQ[flexIndex], deltaQ[-1], 0.2])
    Tbif1,Tbif2               = [1/exp1Par[1],1/exp2Par[2]]

# Save outputs on text file
with open('output.txt', 'a') as f:
    f.writelines(['%d;%4.3f;%s;%2.1f;%s;%d;%3.1f;%d;%3.1f;%4.3f;%4.3f;%4.3f;%5.4f;%5.4f;%4.3f;%4.3f\n'
    % (dsBC,inStepIC,alpha_var,THd,TF,RF,C0,Ls,beta0,theta0,ds0,d50,deltaQ[-1],deltaQ_eq_BRT,Tbif1,Tbif2)])

# PLOTS
# -----
nFig = 0

# Bed evolution plot
plotTimeIndexes = np.linspace(0, n-1, numPlots)
crange          = np.linspace(0,   1, numPlots)
bed_colors      = plt.cm.viridis(crange)
plt.figure(nFig)
plt.title('Branch A bed evolution in time')
plt.xlabel('x/W0 [-]')
plt.ylabel('(η-η0)/D0 [-]')
plt.figure(nFig+1)
plt.title('Branch B bed evolution in time')
plt.xlabel('x/W0 [-]')
plt.ylabel('(η-η0)/D0 [-]')
plt.figure(nFig+2)
plt.title('Branch C bed evolution in time')
plt.xlabel('x/W0 [-]')
plt.ylabel('(η-η0)/D0 [-]')
for i in range(numPlots):
    plotTimeIndex = int(plotTimeIndexes[i])
    myPlot(nFig,   xi/W0, (eta_a[plotTimeIndex,:]-eta_a[0,:])/D0, ('t=%3.1f Tf' % (t[plotTimeIndex]/Tf)), color=bed_colors[i])
    myPlot(nFig+1, xi/W0, (eta_b[plotTimeIndex,:]-eta_b[0,:])/D0, ('t=%3.1f Tf' % (t[plotTimeIndex]/Tf)), color=bed_colors[i])
    myPlot(nFig+2, xi/W0, (eta_c[plotTimeIndex,:]-eta_c[0,:])/D0, ('t=%3.1f Tf' % (t[plotTimeIndex]/Tf)), color=bed_colors[i])
nFig += 2

# Plot evolution of node cells over time
nFig += 1
myPlot(nFig,t[:-1]/Tf,(eta_bn[:n]-0.5*(eta_bn[0]+eta_cn[0]))/D0,'Node cell B','Evolution of node cells elevation over time', 't/Tf [-]', '(η-η0)/D0 [-]')
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
subplotsLayout(ax, ['t/Tf [-]', 't/Tf [-]', 't/Tf [-]'], ['(η-η0)/D0 [-]', None, None],
               ['upstream', 'Bed elevation vs time\n\nmiddle', 'downstream'])

# Plot ∆Q evolution over time along with exponential fittings if available
nFig += 1
myPlot(nFig,t[1:-1]/Tf,deltaQ[1:]/deltaQ_eq_BRT,'Computed','Discharge asymmetry vs time','t/Tf [-]','∆Q/∆Q_BRT [-]')
if expFittings:
    myPlot(nFig,t[1:exp1FitEnd]/Tf,exp1Fit[1:]/deltaQ_eq_BRT,'First stage exponential fit')
    myPlot(nFig,t[flexIndex:-1]/Tf,exp2Fit/deltaQ_eq_BRT,'Second stage exponential fit')

# Plot ∆η evolution over time
nFig += 1
myPlot(nFig,t[:-1]/Tf,abs(inStep/inStep_eq_BRT),None,'Non-dimensional inlet step vs time','t/Tf [-]','|∆η/∆η_BRT| [-]')

plt.show()
