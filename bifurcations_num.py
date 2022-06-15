from functions import *

# -------------------------
# INPUTS AND MODEL SETTINGS
# -------------------------

# Model settings
alpha      = 0  # if ==0, it is computed by assuming betaR=betaC
alpha_var  = 'cost'
inStepIC   = -0.1  # =(inlet step at t=0)/(inlet step at equilibrium computed via BRT)
Hcoeff     = 0 # =(Hd-H0)/D0, where H0 is the wse associated to D0 and Hd is the imposed downstream BC
THd        = 0 # =dHd/d(t/Tf), downstream BC timescale

# I/O settings
numPlots    = 20
showPlots   = True
saveOutputs = False

# Numerical parameters
CFL       = 0.9 # Courant-Friedrichs-Lewy parameter for timestep computation
dx        = 25 # cell length [m]
tend      = 50
tEq       = 5 # if ∆Q remains constant for tEq (non-dimensional), the simulation ends
maxIter   = int(1e6) # max number of iterations during time evolution
tol       = 1e-10 # Iterations tolerance

# Hydraulic parameters
RF     = 0 # flow resistance formula switch: if=0, C=C0 is a constant; if=1, C varies according to local water depth
C0     = 0 # if =0, it's computed as a function of D0=ds0/d50 via a logarithmic formula
eps_c  = 2.5 # Chézy logarithmic formula coefficient (used only if RF=1 or C0=0)
TF     = 'P78' # sediment transport formula. Available options: 'P78' (Parker78), 'MPM', 'P90' (Parker90), 'EH' (Engelund&Hansen)
Ls     = 500 # =L/D0, L=branches' dimensional length
beta0  = 15
theta0 = 0.06
ds0    = 0.02 # =d50/D0
d50    = 0.02 # median sediment diameter [m]
p      = 0.6 # bed porosity
r      = 0.5 # Ikeda parameter

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

# βR and equilibrium alpha computation
betaR = betaR_MR(theta0, ds0, r, phiD0, phiT0, eps_c)
betaC = betaC_MR(RF, theta0, ds0, 1, r, phiD0, phiT0, eps_c)
alpha_MR = betaR/betaC
if alpha == 0:
    alpha_eq = alpha_MR

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
      '\nREDOLFI ET AL. [2019] RESONANT ASPECT RATIO COMPUTATION\nβR = %4.2f\n(β0-βR)/βR = %4.2f\nα_MR = %2.1f\n'
      '\nMAIN CHANNEL IC:\nW_a = %4.2f m\nS0 = %3.2e\nD0 = %3.2f m\nFr0 = %2.1f\nL = %4.1f m\n'
      'Q0 = %3.2e m^3 s^-1\nQs0 = %3.2e m^3 s^-1\n\nExner time: Tf = %3.2f h\n'
      % (RF, TF, nc, dx, CFL, tend, Ls, beta0, theta0, ds0, alpha, alpha_eq, r, d50, betaR, (beta0-betaR) / betaR,
         alpha_MR, W0, S0, D0, Fr0, Ls * D0, Q0, Qs0, Tf / 3600))
print("BRT equilibrium solution:\n∆η = %5.4f" % inStep_eq_BRT)
print("∆Q = %5.4f\nθ_b = %4.3f, θ_c = %4.3f\nFr_b = %3.2f, Fr_c = %3.2f\nS = %2.1e\n" % (BRT_out[:6]))

# Arrays initialization
t         : list = [0]
inStep    : list = [inStepIC*inStep_eq_BRT]
deltaQ    : list = []
dts       : list = []
eta_bn    = np.zeros(maxIter+1)
eta_cn    = np.zeros(maxIter+1)
eta_b     = np.zeros((maxIter+1,nc+1))
eta_c     = np.zeros((maxIter+1,nc+1))
D_b       = np.ones(nc+1)*D0
D_c       = np.ones(nc+1)*D0
S_b       = np.ones(nc)*S0
S_c       = np.ones(nc)*S0
Theta_b   = np.ones(nc+1)*theta0
Theta_c   = np.ones(nc+1)*theta0
Qs_b      = np.ones(nc+1)*Qs0/2
Qs_c      = np.ones(nc+1)*Qs0/2
xi        = np.linspace(0,nc*dx,nc+1)  # cell interfaces coordinates

# Branches IC. Horizontal step -> elevation of first cell of each branch is equal to the corresponding node cell
eta_bn[0]  =  inStep[0]*D0/2
eta_cn[0]  = -inStep[0]*D0/2
eta_b[0,:] = np.linspace(eta_bn[0], eta_bn[0]-S0*nc*dx, num=nc+1)
eta_c[0,:] = np.linspace(eta_cn[0], eta_cn[0]-S0*nc*dx, num=nc+1)
W_b = 0.5*W0
W_c = 0.5*W0
Q_b = Q0/2
Q_c = Q0/2

# Branch B and C downstream BC: H(t)=H0
H0 = 0.5*(eta_b[0,-1]+eta_c[0,-1])+D0*(1+Hcoeff)
Hd_b: list = [H0]
Hd_c: list = [H0]

# alpha time variation setup
if alpha_var == 'cost':
    alpha = alpha_eq
else:
    alpha = alpha_eq*abs(inStep[0]/inStep_eq_BRT)

# Time evolution
eqReached = False
for n in range(0, maxIter):
    # Compute dt according to CFL condition and update time. Check if system has reached equilibrium
    Ceta_b = C_eta(Q_b,W_b,D_b,g,delta,d50,p,C0,D0,RF,TF,eps_c)
    Ceta_c = C_eta(Q_c,W_c,D_c,g,delta,d50,p,C0,D0,RF,TF,eps_c)
    Cmax = max(max(Ceta_b),max(Ceta_c))
    dt   = CFL*dx/Cmax
    dts.append(dt)
    t.append(t[-1]+dt)
    if t[-1]>tEq*Tf:
        if np.all(abs(deltaQ[-int(tEq*Tf/dt):]/deltaQ[-1]-1)<np.sqrt(tol)):
            if THd == 0:
                print('\nEquilibrium reached\n')
                break
            elif not eqReached:
                eqReached = True
            else:
                print('\nRegime condition reached\n')
                break
    if t[-1] >= (tend*Tf):
        print('\nEnd time reached\n')
        break

    # Update the downstream BC and use it to compute water depths at branches end
    if eqReached and THd != 0:
        Hd_b.append(Hd_b[-1]+D0/THd*dt/Tf)
        Hd_c.append(Hd_c[-1]+D0/THd*dt/Tf)
    D_b[-1] = Hd_b[-1]-eta_b[n,-1]
    D_c[-1] = Hd_c[-1]-eta_c[n,-1]
    
    # Compute the discharge partitioning at the node through profiles+Hb=Hc
    D_b = buildProfile_rk4(RF,D_b[-1],Q_b,W_b,S_b,d50,dx,g,C0,eps_c)
    D_c = buildProfile_rk4(RF,D_c[-1],Q_c,W_c,S_c,d50,dx,g,C0,eps_c)
    if abs((D_b[0]-D_c[0]+inStep[-1]*D0)/(0.5*(D_b[0]+D_c[0])))>tol:
        Q_b = opt.fsolve(fSys_rk4,Q_b,(RF,D_b[-1],D_c[-1],D0,inStep[-1],Q0,W_b,W_c,S_b,S_c,d50,dx,g,C0,eps_c),xtol=tol)
        Q_c = Q0-Q_b
        D_b = buildProfile_rk4(RF,D_b[-1],Q_b,W_b,S_b,d50,dx,g,C0,eps_c)
        D_c = buildProfile_rk4(RF,D_c[-1],Q_c,W_c,S_c,d50,dx,g,C0,eps_c)
    
    #Shields and Qs update for the anabranches
    Theta_b = shieldsUpdate(RF,Q_b,W_b,D_b,d50,g,delta,C0,eps_c)
    Theta_c = shieldsUpdate(RF,Q_c,W_c,D_c,d50,g,delta,C0,eps_c)
    if np.any(Theta_b<0.03) or np.any(Theta_c<0.03):
        print("Shields number below threshold")
        break
    Qs_b    = W_b*np.sqrt(g*delta*d50**3)*phis(Theta_b,TF,D0,d50)[0]
    Qs_c    = W_c*np.sqrt(g*delta*d50**3)*phis(Theta_c,TF,D0,d50)[0]

    # Compute liquid and solid discharge in transverse direction at the node; update bed elevation of node cells accordingly
    Q_y  = Q_b-Q0/2
    Qs_y = Qs0*(Q_y/Q0-2*alpha*r/np.sqrt(theta0)*(eta_bn[n]-eta_cn[n])/W0)
    eta_bn[n+1] = eta_bn[n]+dt*(Qs0/2-Qs_b[0]+Qs_y)/((1-p)*alpha*W0*W_b)
    eta_cn[n+1] = eta_cn[n]+dt*(Qs0/2-Qs_c[0]-Qs_y)/((1-p)*alpha*W0*W_c)
    
    # Horizontal step: bed elevation is constant along node cells
    eta_b[n+1,0] = eta_bn[n+1]
    eta_c[n+1,0] = eta_cn[n+1]

    # Exner equation integration via upwind method: since Fr<1, bed level perturbations propagate in the downstream direction
    eta_b[n+1,1:] = eta_b[n,1:]-dt/dx*(Qs_b[1:]-Qs_b[:-1])/(W_b*(1-p))
    eta_c[n+1,1:] = eta_c[n,1:]-dt/dx*(Qs_c[1:]-Qs_c[:-1])/(W_c*(1-p))    

    # Update bed slopes
    S_b = (eta_b[n+1,:-1]-eta_b[n+1,1:])/dx
    S_c = (eta_c[n+1,:-1]-eta_c[n+1,1:])/dx

    # Update time-controlled lists
    deltaQ.append((Q_b-Q_c)/Q0)
    inStep.append((eta_bn[n+1]-eta_cn[n+1])/D0)

    # alpha update
    if alpha_var == 'deltaEtaLin':
        alpha = alpha_eq*abs(inStep[-1]/inStep_eq_BRT)

    # Time print, update and check if the simulation ends
    if n % 500 == 0:
        print("Elapsed time = %4.1f Tf, ∆Q = %5.4f" % (t[n]/Tf,deltaQ[n]))
    

# Print final ∆Q
print('Final ∆Q = %5.4f' % deltaQ[-1])
print('∆Q at equilibrium according to BRT = %5.4f' % deltaQ_eq_BRT)
print('Difference = %2.1f %%\n' % (100 * abs((abs(deltaQ[-1]) - deltaQ_eq_BRT) / deltaQ_eq_BRT)))

# PLOTS
# -----
nFig = 0

# Bed evolution plot
plotTimeIndexes = np.linspace(0, n, numPlots)
crange          = np.linspace(0, 1, numPlots)
bed_colors      = plt.cm.viridis(crange)
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
    myPlot(nFig+1, xi/W0, (eta_b[plotTimeIndex,:]-eta_b[0,:])/D0, ('t=%3.1f Tf' % (plotTimeIndex*dt/Tf)), color=bed_colors[i])
    myPlot(nFig+2, xi/W0, (eta_c[plotTimeIndex,:]-eta_c[0,:])/D0, ('t=%3.1f Tf' % (plotTimeIndex*dt/Tf)), color=bed_colors[i])
nFig += 2
# Plot bed evolution at relevant cross-sections (upstream, middle, downstream)
nFig += 1
fig, ax = plt.subplots(1, 3, num=nFig)
ax[0].plot(t[:-1]/Tf, (eta_b[:n+1,       0 ]-eta_b[0,       0 ])/D0, label='Branch B')
ax[0].plot(t[:-1]/Tf, (eta_c[:n+1,       0 ]-eta_c[0,       0 ])/D0, label='Branch C')
ax[1].plot(t[:-1]/Tf, (eta_b[:n+1,int(nc/2)]-eta_b[0,int(nc/2)])/D0, label='Branch B')
ax[1].plot(t[:-1]/Tf, (eta_c[:n+1,int(nc/2)]-eta_c[0,int(nc/2)])/D0, label='Branch C')
ax[2].plot(t[:-1]/Tf, (eta_b[:n+1,-      1 ]-eta_b[0,-      1 ])/D0, label='Branch B')
ax[2].plot(t[:-1]/Tf, (eta_c[:n+1,-      1 ]-eta_c[0,-      1 ])/D0, label='Branch C')
subplotsLayout(ax, ['t/Tf [-]', 't/Tf [-]', 't/Tf [-]'], ['(η-η0)/D0 [-]', None, None],
               ['upstream', 'Bed elevation vs time\n\nmiddle', 'downstream'])
# Plot ∆Q evolution over time
nFig += 1
myPlot(nFig,t[1:-1]/Tf,deltaQ/deltaQ_eq_BRT,None,'Discharge asymmetry vs time','t/Tf [-]','∆Q/∆Q_BRT [-]')

# Plot ∆η evolution over time
nFig += 1
myPlot(nFig,t[:-1]/Tf,abs(inStep/inStep_eq_BRT),None,'Non-dimensional inlet step vs time','t/Tf [-]','|∆η/∆η_BRT| [-]')

# Plot computed values of timesteps
nFig += 1
myPlot(nFig,np.indices((len(dts),1))[0],dts,None,'Computed timesteps along the simulation','Iteration #','dt [s]')
plt.show()