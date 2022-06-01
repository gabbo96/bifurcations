from functions import *

# -------------------------
# INPUTS AND MODEL SETTINGS
# -------------------------

# Model settings
alpha      = 0  # if ==0, it is computed by assuming betaR=betaC
alpha_var  = 'deltaEtaLin'
inStep     :list = [-0.004]
Hcoeff     = 0 # =(Hd-H0)/D0, where H0 is the wse associated to D0 and Hd is the imposed downstream BC

# I/O settings
numPlots    = 20
showPlots   = True
saveOutputs = False

# Numerical parameters
dt        = 200 # timestep [s]
dx        = 50 # cell length [m]
tend      = 50
maxIter   = int(1e6)+1 # max number of iterations during time evolution
tol       = 1e-10 # Iterations tolerance

# Hydraulic parameters
RF     = 0 # flow resistance formula switch: if=0, C=C0 is a constant; if=1, C varies according to local water depth
C0     = 0 # if =0, it's computed as a function of D0=ds0/d50 via a logarithmic formula
eps_c  = 2.5 # Chézy logarithmic formula coefficient (used only if RF=1 or C0=0)
TF     = 'P90' # sediment transport formula. Available options: 'P78' (Parker78), 'MPM', 'P90' (Parker90), 'EH' (Engelund&Hansen)
Ls     = 900 # =L/D0, L=branches' dimensional length
beta0  = 18
theta0 = 0.08
ds0    = 0.01 # =d50/D0
d50    = 0.01 # median sediment diameter [m]
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
W_a   = beta0*2*D0
if C0 == 0:
    C0 = 6+2.5*np.log(D0/(eps_c*d50))
phi00, phiD0, phiT0 = phis_scalar(theta0, TF, D0, d50)
Q0    = uniFlowQ(RF, W_a, S0, D0, d50, g, C0, eps_c)
Qs0   = W_a*np.sqrt(g*delta*d50**3)*phi00
Fr0   = Q0/(W_a*D0*np.sqrt(g*D0))
nc    = int(Ls*D0/dx)

# Exner time computation
Tf = (1-p)*W_a*D0/(Qs0/W_a)

# βR and equilibrium alpha computation
betaR = betaR_MR(theta0, ds0, r, phiD0, phiT0, eps_c)
betaC = betaC_MR(RF, theta0, ds0, 1, r, phiD0, phiT0, eps_c)
alpha_MR = betaR/betaC
if alpha == 0:
    alpha_eq = alpha_MR

# alpha time variation setup
if alpha_var == 'cost':
    alpha = alpha_eq
else:
    alpha_eps = 0.1  # alpha's initial small value
    alpha = alpha_eps

# Arrays initialization
deltaQ    : list = [0]
nIter     = math.ceil(tend*Tf/dt)+2
eta_bn    = np.zeros(nIter)
eta_cn    = np.zeros(nIter)
eta_b     = np.zeros((nIter,nc))
eta_c     = np.zeros((nIter,nc))
D_b       = np.zeros(nc+1)
D_c       = np.zeros(nc+1)
S_b       = np.zeros(nc)
S_c       = np.zeros(nc)
Theta_b   = np.zeros(nc+1)
Theta_c   = np.zeros(nc+1)
Qs_b      = np.zeros(nc+1)
Qs_c      = np.zeros(nc+1)

# Space-time domain for the branches
xi = np.linspace(0,nc*dx,nc+1)  # cell interfaces coordinates
xc = np.linspace(dx/2,nc*dx-dx/2,nc) # cell centroids coordinates
t  :list = [0]

# Branches IC
eta_bn[0]  = inStep[0]*D0/2
eta_cn[0]  = -inStep[0]*D0/2
eta_b[0,:] = np.linspace(eta_bn[0]-S0*dx, eta_bn[0]-S0*nc*dx, num=nc)
eta_c[0,:] = np.linspace(eta_cn[0]-S0*dx, eta_cn[0]-S0*nc*dx, num=nc)
W_b = 0.5*W_a
W_c = 0.5*W_a
Q_b = Q0/2
Q_c = Q0/2

# Branch B and C downstream BC: H(t)=H0
H0 = 0.5*(eta_b[0,-1]+eta_c[0,-1]-S0*dx)+D0*(1+Hcoeff)
Hd_b: list = [H0]
Hd_c: list = [H0]

# Equilibrium deltaQ computation through BRT method
BRT_out = deltaQ_BRT([1.1,0.9,0.9,1.1],0,RF,TF,theta0,Q0,Qs0,W_a,Ls*D0,D0,S0,W_b,W_c,d50,alpha_eq,r,g,delta,tol,C0,eps_c)
deltaQ_eq_BRT = BRT_out[0]
inStep_eq_BRT = (BRT_out[7]-BRT_out[8])/D0

# Print model settings and parameters
print('INPUT PARAMETERS:\nFlow resistance formula = %s\nSolid transport formula = %s\nnc = %d\ndx = %4.2f m'
      '\ndt = %d s\nt_end = %2.1f Tf\nL* = %d\nβ0 = %4.2f\nθ0 = %4.3f\nds0 = %2.1e\nα = %2.1f\nα_eq = %2.1f\nr = %2.1f'
      '\nd50 = %3.2e m\n'
      '\nREDOLFI ET AL. [2019] RESONANT ASPECT RATIO COMPUTATION\nβR = %4.2f\n(β0-βR)/βR = %4.2f\nα_MR = %2.1f\n'
      '\nMAIN CHANNEL IC:\nW_a = %4.2f m\nS0 = %3.2e\nD0 = %3.2f m\nH0 = %2.1f m\nFr0 = %2.1f\nL = %4.1f m\n'
      'Q0 = %3.2e m^3 s^-1\nQs0 = %3.2e m^3 s^-1\n\nExner time: Tf = %3.2f h\n'
      % (RF, TF, nc, dx, dt, tend, Ls, beta0, theta0, ds0, alpha, alpha_eq, r, d50, betaR, (beta0-betaR) / betaR,
         alpha_MR, W_a, S0, D0, H0, Fr0, Ls * D0, Q0, Qs0, Tf / 3600))
print("BRT equilibrium solution:\ndeltaQ = %5.4f\nθ_b = %4.3f, θ_c ="
      " %4.3f\nFr_b = %3.2f, Fr_c = %3.2f\nS = %2.1e\n"
      % (BRT_out[:6]))

# Time evolution
for n in range(0, maxIter):
    S_b [0]    = (eta_bn[n]-eta_b[n,0])/dx
    S_c [0]    = (eta_cn[n]-eta_c[n,0])/dx
    S_b [1:-1] = (eta_b[n,:-2]-eta_b[n,2:])/(2*dx)
    S_c [1:-1] = (eta_c[n,:-2]-eta_c[n,2:])/(2*dx)
    S_b [-1]   = (eta_b[n,-2]-eta_b[n,-1])/dx
    S_c [-1]   = (eta_c[n,-2]-eta_c[n,-1])/dx       
    # Use the downstream BC to compute water depths at branches end
    D_b[-1] = Hd_b[-1]-(eta_b[n,-1]-0.5*S_b[-1]*dx)
    D_c[-1] = Hd_c[-1]-(eta_c[n,-1]-0.5*S_c[-1]*dx)
    # Compute the discharge partitioning at the node through profiles+Hb=Hc
    D_b = buildProfile(RF,D_b[-1],Q_b,W_b,S_b,d50,dx,g,C0,eps_c)
    D_c = buildProfile(RF,D_c[-1],Q_c,W_c,S_c,d50,dx,g,C0,eps_c)
    if abs((D_b[0]-D_c[0]+inStep[-1]*D0)/(0.5*(D_b[0]+D_c[0])))>tol:
        Q_b = opt.fsolve(fSys,Q_b,(RF,D_b[-1],D_c[-1],D0,inStep[-1],Q0,W_b,W_c,S_b,S_c,d50,dx,g,C0,eps_c),xtol=tol)
        Q_c = Q0-Q_b
        D_b = buildProfile(RF,D_b[-1],Q_b,W_b,S_b,d50,dx,g,C0,eps_c)
        D_c = buildProfile(RF,D_c[-1],Q_c,W_c,S_c,d50,dx,g,C0,eps_c)
    
    #Shields, Qs and eta update for the anabranches
    Theta_b      = shieldsUpdate(RF, Q_b, W_b, D_b, d50, g, delta, C0, eps_c)
    Theta_c      = shieldsUpdate(RF, Q_c, W_c, D_c, d50, g, delta, C0, eps_c)
    Qs_b         = W_b*np.sqrt(g*delta*d50**3)*phis(Theta_b, TF, D0, d50)[0]
    Qs_c         = W_c*np.sqrt(g*delta*d50**3)*phis(Theta_c, TF, D0, d50)[0]  
    eta_b[n+1,:] = eta_b[n,:]+dt*(Qs_b[:-1]-Qs_b[1:])/((1-p)*dx*W_b)
    eta_c[n+1,:] = eta_c[n,:]+dt*(Qs_c[:-1]-Qs_c[1:])/((1-p)*dx*W_c)
    
    # Node cells' transverse discharges and bed elevation update
    Q_y  = Q_b-Q0/2
    Qs_y = Qs0*(Q_y/Q0-2*alpha*r/np.sqrt(theta0)*(eta_bn[n]-eta_cn[n])/W_a)
    eta_bn[n+1] = eta_bn[n]+dt*(Qs0/2-Qs_b[0]+Qs_y)/((1-p)*alpha*W_a*W_b)
    eta_cn[n+1] = eta_cn[n]+dt*(Qs0/2-Qs_c[0]-Qs_y)/((1-p)*alpha*W_a*W_c)

    # Update time-controlled lists
    deltaQ.append((Q_b-Q_c)/Q0)
    inStep.append((eta_bn[n+1]-eta_cn[n+1])/D0)

    # alpha update
    if alpha_var == 'deltaEtaLin':
        alpha = alpha_eps+(alpha_eq-alpha_eps)*abs(inStep[-1]/inStep_eq_BRT)

    # Time print, update and check if the simulation ends
    # Print elapsed time
    if n % 2500 == 0:
        print("Elapsed time = %4.1f Tf, deltaQ = %5.4f" % (t[n]/Tf,deltaQ[n+1]))
    t.append(t[-1]+dt)
    if t[n] >= (tend * Tf):
        print('\nEnd time reached\n')
        break    

# Print final deltaQ
print('Final ∆Q = %5.4f' % deltaQ[-1])
print('∆Q at equilibrium according to BRT = %5.4f' % deltaQ_eq_BRT)
print('Difference = %2.1f %%\n' % (100 * abs((abs(deltaQ[-1]) - deltaQ_eq_BRT) / deltaQ_eq_BRT)))

# PLOTS
# -----
nFig = 0

# Bed evolution plot
crange          = np.linspace(0, 1, numPlots)
bed_colors      = plt.cm.viridis(crange)
plotTimeIndexes = np.linspace(0, n+1, numPlots)
plt.figure(nFig+1)
plt.title('Branch B bed evolution in time')
plt.xlabel('x/Wa [-]')
plt.ylabel('(η-η0)/D0 [-]')
plt.figure(nFig+2)
plt.title('Branch C bed evolution in time')
plt.xlabel('x/Wa [-]')
plt.ylabel('(η-η0)/D0 [-]')
for i in range(numPlots):
    plotTimeIndex = int(plotTimeIndexes[i])
    myPlot(nFig+1, xc/W_a, (eta_b[plotTimeIndex,:]-eta_b[0,:])/D0, ('t=%3.1f Tf' % (plotTimeIndex*dt/Tf)), color=bed_colors[i])
    myPlot(nFig+2, xc/W_a, (eta_c[plotTimeIndex,:]-eta_c[0,:])/D0, ('t=%3.1f Tf' % (plotTimeIndex*dt/Tf)), color=bed_colors[i])
nFig += 2   
# Plot deltaQ evolution over time
nFig += 1
myPlot(nFig,t/Tf,deltaQ,'deltaQ','Discharge asymmetry vs time','t/Tf [-]','deltaQ')

plt.show()