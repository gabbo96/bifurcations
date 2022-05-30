from functions import *

# -------------------------
# INPUTS AND MODEL SETTINGS
# -------------------------

# Model settings
st       = 1 # Solid transport switch: 0=fixed bed; 1=movable bed
bedIC    = 'triangular' # 'exp' adds a gradual inlet step with an exponential trend, so that deltaEta[-1 ]=inStep[0]; 'flat'; 'lastCell'; 'triangular'
ISlength = 0.8 # % of channel length influenced by inlet step; used if bedIC=='triangular' or bedIC=='exp'
inStep   :list = [-0.01]
Hcoeff   = 0 # =(Hd-H0)/D0, where H0 is the wse associated to D0 and Hd is the imposed downstream BC

# I/O settings
numPlots    = 20
showPlots   = True
saveOutputs = False

# Numerical parameters
dt        = 30 # timestep [s]
dx        = 10 # cell length [m]
tend      = 0.35
maxIter   = int(1e6)*st+1 # max number of iterations during time evolution
maxIterQy = 3
tol       = 1e-10 # Iterations tolerance

# Hydraulic parameters
RF     = 0 # flow resistance formula switch: if=0, C=C0 is a constant; if=1, C varies according to local water depth
C0     = 0 # if =0, it's computed as a function of D0=ds0/d50 via a logarithmic formula
eps_c  = 2.5 # Chézy logarithmic formula coefficient (used only if RF=1 or C0=0)
TF     = 'P90' # sediment transport formula. Available options: 'P78' (Parker78), 'MPM', 'P90' (Parker90), 'EH' (Engelund&Hansen)
Ls     = 900 # =L/D0, L=branches' dimensional length
beta0  = 20
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
phi00, phiD0, phiT0 = phis(np.array([theta0]), TF, D0, d50)
Q0    = uniFlowQ(RF, W_a, S0, D0, d50, g, C0, eps_c)
Qs0   = W_a*np.sqrt(g*delta*d50**3)*phi00
Fr0   = Q0/(W_a*D0*np.sqrt(g*D0))
nc    = int(Ls*D0/dx)

# Exner time computation
Tf = (1-p)*W_a*D0/(Qs0/W_a)

# βR alpha computation
betaR = betaR_MR(theta0, ds0, r, phiD0, phiT0, eps_c)

# Arrays initialization
deltaQ    : list = [0]
qylist    : list = []
nIter     = math.ceil(tend*Tf/dt)+1
eta_a     = np.zeros((nIter,nc))
eta_ab    = np.zeros((nIter,nc))
eta_ac    = np.zeros((nIter,nc))
eta_b     = np.zeros((nIter,nc))
eta_c     = np.zeros((nIter,nc))
eta0      = np.zeros(nc)
D_a       = np.zeros(nc+1)
D_ab      = np.zeros(nc+1)
D_ac      = np.zeros(nc+1)
D_b       = np.zeros(nc+1)
D_c       = np.zeros(nc+1)
Q_ab      = np.zeros(nc+1)
Q_ac      = np.zeros(nc+1)
Q_y       = np.zeros(nc)
S_ab      = np.zeros(nc)
S_ac      = np.zeros(nc)
S_b       = np.zeros(nc)
S_c       = np.zeros(nc)
Theta_ab  = np.zeros(nc+1)
Theta_ac  = np.zeros(nc+1)
Theta_b   = np.zeros(nc+1)
Theta_c   = np.zeros(nc+1)
Qs_ab     = np.zeros(nc+1)
Qs_ac     = np.zeros(nc+1)
Qs_y      = np.zeros(nc)
Qs_b      = np.zeros(nc+1)
Qs_c      = np.zeros(nc+1)

# Space-time domain
xi = np.linspace(0,nc*dx,nc+1)  # cell interfaces coordinates
xc = np.linspace(dx/2,nc*dx-dx/2,nc) # cell centroids coordinates
t  :list = [0]

# Channel A IC
eta_ab[0,:]  = np.linspace(0,-S0*dx*(nc-1),num=len(eta_ab[0,:]))
eta_ac[0,:]  = eta_ab[0,:]
eta0[:]      = eta_ab[0,:]
if bedIC == 'exp':
    eta_ab[0,:]  += inStep[0]/2*D0*np.exp(-(xc[-1]-xc[:])/((1-ISlength)*Ls*D0))
    eta_ac[0,:]  -= inStep[0]/2*D0*np.exp(-(xc[-1]-xc[:])/((1-ISlength)*Ls*D0))
elif bedIC == 'triangular':
    inStepcells   = int(nc*ISlength)
    eta_ab[0,-1]  += inStep[0]/2*D0
    eta_ac[0,-1]  -= inStep[0]/2*D0
    eta_ab[0,-inStepcells:] = np.linspace(eta_ab[0,-inStepcells],eta_ab[0,-1],num=inStepcells) 
    eta_ac[0,-inStepcells:] = np.linspace(eta_ac[0,-inStepcells],eta_ac[0,-1],num=inStepcells)
    S_ab[1:-1]    = (eta_ab[0,:-2]-eta_ab[0,2:])/(2*dx)
    S_ac[1:-1]    = (eta_ac[0,:-2]-eta_ac[0,2:])/(2*dx)
elif bedIC == 'lastCell':
    eta_ab[0,-1:]  += inStep[0]/2*D0
    eta_ac[0,-1:]  -= inStep[0]/2*D0
eta_a[0,:] = (eta_ab[0,:]+eta_ac[0,:])/2
Hd_a: list = [eta_a[0,-1]+D0]  # wse at the downstream end of channel A
W_ab       = W_a/2
W_ac       = W_a/2

# Anabranches IC
eta_b[0,:] = np.linspace(eta_ab[0,-1]-S0*dx, eta_ab[0,-1]-S0*nc*dx, num=len(eta_b[0,:]))
eta_c[0,:] = np.linspace(eta_ac[0,-1]-S0*dx, eta_ac[0,-1]-S0*nc*dx, num=len(eta_c[0,:]))
W_b        = W_a/2
W_c        = W_a/2
Q_b        = Q0/2
Q_c        = Q0/2

#? Branch B and C downstream BC: H(t)=H0
H0 = 0.5*(eta_b[0,-1]+eta_c[0,-1]-S0*dx)+D0*(1+Hcoeff)
Hd_b: list = [H0]
Hd_c: list = [H0]

# Print model parameters
print('\nINPUT PARAMETERS:\nHcoeff = %3.2f\nSolid transport formula = %s\nnc = %d\ndx = %d m'
      '\ndt = %d s\nt_end = %2.1f Tf\nL* = %d\nβ0 = %4.2f\nθ0 = %4.3f\nds0 = %2.1e\nr = %2.1f'
      '\nd50 = %3.2e m\n'
      '\nMAIN CHANNEL IC:\nW_a = %4.2f m\nS0 = %3.2e\nD0 = %3.2f m\nFr0 = %2.1f\nL = %4.1f m\n'
      'Q0 = %3.2e m^3 s^-1\nQs0 = %3.2e m^3 s^-1\n\nExner time: Tf = %3.2f h'
      % (Hcoeff, TF, nc, dx, dt, tend, Ls, beta0, theta0, ds0, r, d50, W_a, S0, D0, Fr0, Ls*D0, Q0, Qs0, Tf/3600))

print('\nINLET STEP SETTINGS:\nbedIC = %s\nISlength = %3.2f\ninStep = %3.2f' 
      % (bedIC,ISlength,inStep[0]))
if bedIC == 'triangular':
    print('S_ab/S_ac = %3.2f\n' % (S_ab[-3]/S_ac[-3]))
print('\n')

# Plot bed elevation IC
if bedIC != 'flat':
    myPlot(0,xc,eta_ab[0,:],'eta_ab','Semichannels bed profiles IC','x[m]','η [m]')
    myPlot(0,xc,eta_ac[0,:],'eta_ac')
    if showPlots:
        plt.show()

for n in range(0, maxIter):
    # Channel cells slope update through central difference approximation
    S_ab[0]    = (eta_ab[n,0]-eta_ab[n,1])/dx
    S_ac[0]    = (eta_ac[n,0]-eta_ac[n,1])/dx    
    S_ab[1:-1] = (eta_ab[n,:-2]-eta_ab[n,2:])/(2*dx)
    S_ac[1:-1] = (eta_ac[n,:-2]-eta_ac[n,2:])/(2*dx)
    S_ab[-1]   = (eta_ab[n,-2]-eta_b[n,0])/(2*dx)
    S_ac[-1]   = (eta_ac[n,-2]-eta_c[n,0])/(2*dx)
    S_b [0]    = (eta_ab[n,-1]-eta_b[n,1])/(2*dx)
    S_c [0]    = (eta_ac[n,-1]-eta_c[n,1])/(2*dx)
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
    Q_ab[-1] = Q_b
    Q_ac[-1] = Q_c
    D_ab[-1] = D_b[0]
    D_ac[-1] = D_c[0]
    D_a[-1]  = 0.5*(D_ab[-1]+D_ac[-1])

    #? Compute water depths and discharge along the semichannels
    for i in range(nc-1,0,-1):
        # 'True' bed elevations at section M
        eta_abM = 0.5*(eta_ab[n,i]+eta_ab[n,i-1])
        eta_acM = 0.5*(eta_ac[n,i]+eta_ac[n,i-1])
        inStepM = (eta_abM-eta_acM)/D0
        # First guess
        Q_abM = 0.5*(Q0/2+Q_ab[i+1])
        Q_acM = 0.5*(Q0/2+Q_ac[i+1])
        Q_yMV = 0.5*(Q_ab[i+1]-Q_ac[i+1])
        D_aM  = 0.5*(D0+D_a[i+1])
        D_abM = D_aM-0.5*inStepM*D0
        D_acM = D_aM+0.5*inStepM*D0
        D_abM_control: list = [D_abM/D0]
        D_aM_control : list = [D_aM/D0]
        QabM_control : list = [Q_abM/Q0/2]
        Qy_control   : list = [Q_yMV/(0.5*(Q_ab[i+1]-Q_ac[i+1]))]
        
        # Iterations
        for k in range(maxIterQy):
            # Average quantities
            Q_abK = 0.5*(Q_abM+Q_ab[i+1])
            Q_acK = 0.5*(Q_acM+Q_ac[i+1])
            D_abK = 0.5*(D_abM+D_ab[i+1])
            D_acK = 0.5*(D_acM+D_ac[i+1])
            # Update water depths at section M
            dDadx = dDadxExpl(Q_yMV,D_abK,D_acK,Q_abK,Q_acK,S_ab[i],S_ac[i],W_ab,W_ac,g,d50,dx,C0,RF,eps_c)
            D_aM  = D_a[i+1]-dDadx*dx
            D_abM = D_aM-0.5*inStepM*D0
            D_acM = D_aM+0.5*inStepM*D0
            # Update water discharges at section M
            Q_yMV = QyExpl(D_abK,D_acK,Q_abK,Q_acK,S_ab[i],S_ac[i],W_ab,W_ac,g,d50,dx,C0,RF,eps_c)
            Q_abM = Q_ab[i+1]-Q_yMV
            Q_acM = Q_ac[i+1]+Q_yMV
            # Debug lists
            D_abM_control.append(D_abM/D0)
            D_aM_control .append(D_aM/D0)
            QabM_control .append(Q_abM/Q0/2)
            Qy_control   .append(Q_yMV/(0.5*(Q_ab[i+1]-Q_ac[i+1])))

        D_ab[i] = D_abM
        D_ac[i] = D_acM
        D_a [i] = 0.5*(D_ab[i]+D_ac[i])
        Q_ab[i] = Q_abM
        Q_ac[i] = Q_acM
        Q_y[i]  = Q_yMV


    # Upstream BC (water depths at the semichannels upstream end do not determine solid discharge)
    Q_ab[0] = Q0/2
    Q_ac[0] = Q0/2
    Q_y [0] = 0.5*(Q_ab[1]-Q_ac[1])
    dDadxMV = dDadxExpl(Q_y[0],D_ab[1],D_ac[1],0.5*(Q_ab[0]+Q_ab[1]),0.5*(Q_ac[0]+Q_ac[1]),S_ab[0],S_ac[0],W_ab,W_ac,g,d50,dx,C0,RF,eps_c)
    D_ab[0] = D_ab[1]-(dDadxMV+0.5*(S_ab[0]-S_ac[0]))*dx
    D_ac[0] = D_ac[1]-(dDadxMV-0.5*(S_ab[0]-S_ac[0]))*dx
    D_a [0] = (D_ab[0]+D_ac[0])*0.5

    # Shields parameter update
    Theta_ab = shieldsUpdate(RF, Q_ab, W_ab, D_ab, d50, g, delta, C0, eps_c)
    Theta_ac = shieldsUpdate(RF, Q_ac, W_ac, D_ac, d50, g, delta, C0, eps_c)
    Theta_b  = shieldsUpdate(RF, Q_b, W_b, D_b, d50, g, delta, C0, eps_c)
    Theta_c  = shieldsUpdate(RF, Q_c, W_c, D_c, d50, g, delta, C0, eps_c)

    # Solid discharge computation along the three channels + second upstream BC
    Qs_ab    = W_ab*np.sqrt(g*delta*d50**3)*phis(Theta_ab, TF, D0, d50)[0]*st
    Qs_ac    = W_ac*np.sqrt(g*delta*d50**3)*phis(Theta_ac, TF, D0, d50)[0]*st
    Qs_ab[0] = Qs0/2*st
    Qs_ac[0] = Qs0/2*st
    Qs_b     = W_b*np.sqrt(g*delta*d50**3)*phis(Theta_b, TF, D0, d50)[0]*st
    Qs_c     = W_c*np.sqrt(g*delta*d50**3)*phis(Theta_c, TF, D0, d50)[0]*st

    # Transverse solid discharge along channel a
    Theta_a = shieldsUpdate(RF, Q0, W_a, D_a, d50, g, delta, C0, eps_c)
    Qs_y[:] = (Qs_ab[:-1]+Qs_ac[:-1])*(Q_y/Q0-2*r*dx/(W_a*Theta_a[:-1]**0.5)*(eta_ab[n,:]-eta_ac[n,:])/W_a)*st
    
    # Apply Exner equation to update node cells elevation
    eta_ab[n+1,:] = eta_ab[n,:]+dt*(Qs_ab[:-1]-Qs_ab[1:]+Qs_y[:])/((1-p)*W_ab*dx)*st
    eta_ac[n+1,:] = eta_ac[n,:]+dt*(Qs_ac[:-1]-Qs_ac[1:]-Qs_y[:])/((1-p)*W_ac*dx)*st
    eta_a [n+1,:] = (eta_ab[n+1,:]+eta_ac[n+1,:])*0.5
    eta_b [n+1,:] = eta_b[n,:]+dt*(Qs_b[:-1]-Qs_b[1:])/((1-p)*dx*W_b)*st
    eta_c [n+1,:] = eta_c[n,:]+dt*(Qs_c[:-1]-Qs_c[1:])/((1-p)*dx*W_c)*st

    # Update time-controlled lists
    deltaQ.append((Q_b-Q_c)/Q0)
    inStep.append((0.5*(eta_ab[n+1,-1]+eta_b[n+1,0])-0.5*(eta_ac[n+1,-1]+eta_c[n+1,0]))/D0)

    # Time update + end-time condition for the simulation's end
    t.append(t[-1]+dt)
    if t[-1] >= (tend * Tf):
        print('\nEnd time reached\n')
        break

    # Print elapsed time
    if n % 100 == 0:
        print("Elapsed time = %4.1f Tf" % (t[n] / Tf))
        print("Q_b = %3.2f, Q_c = %3.2f, D_b[0]/D0 = %4.3f, D_c[0]/D0 = %4.3f" % (Q_b,Q_c,D_b[0]/D0,D_c[0]/D0))

if n == maxIter:
    print("\nMax number of iterations reached\n")

# -----
# PLOTS
# -----
nFig = 0

# Bed evolution plot
if st == 1:
    # Plot evolution of bed elevation for channels Ab,Ac,B and C
    crange          = np.linspace(0, 1, numPlots)
    bed_colors      = plt.cm.viridis(crange)
    plotTimeIndexes = np.linspace(0, n+1, numPlots)
    nFig += 1
    plt.figure(nFig)
    plt.title('Average bed elevation evolution in time')
    plt.xlabel('x [m]')
    plt.ylabel('η - η0 [m]')
    plt.figure(nFig+1)
    plt.title('Semichannel ab bed elevation evolution in time')
    plt.xlabel('x [m]')
    plt.ylabel('η - η_0 [m]')
    plt.figure(nFig+2)
    plt.title('Semichannel ac bed elevation evolution in time')
    plt.xlabel('x [m]')
    plt.ylabel('η - η_0 [m]')
    plt.figure(nFig+3)
    plt.title('Branch B bed evolution in time')
    plt.xlabel('x [m]')
    plt.ylabel('η - η_0 [m]')
    plt.figure(nFig+4)
    plt.title('Branch C bed evolution in time')
    plt.xlabel('x [m]')
    plt.ylabel('η - η_0 [m]')
    for i in range(numPlots):
        plotTimeIndex = int(plotTimeIndexes[i])
        myPlot(nFig, xc, eta_a[plotTimeIndex,:]-eta0[:], ('t=%3.1f Tf' % (plotTimeIndex*dt/Tf)), color=bed_colors[i])
        myPlot(nFig+1, xc, eta_ab[plotTimeIndex,:]-eta0, ('t=%3.1f Tf' % (plotTimeIndex*dt/Tf)), color=bed_colors[i])
        myPlot(nFig+2, xc, eta_ac[plotTimeIndex,:]-eta0, ('t=%3.1f Tf' % (plotTimeIndex*dt/Tf)), color=bed_colors[i])
        myPlot(nFig+3, xc, eta_b[plotTimeIndex,:]-eta_b[0,:], ('t=%3.1f Tf' % (plotTimeIndex*dt/Tf)), color=bed_colors[i])
        myPlot(nFig+4, xc, eta_c[plotTimeIndex,:]-eta_c[0,:], ('t=%3.1f Tf' % (plotTimeIndex*dt/Tf)), color=bed_colors[i])    
    nFig += 4
    # Plot semichannels bed elevation values
    eta_a_plot = np.vstack([eta_ac[n+1,:]-eta_ac[0,:], eta_ab[n+1,:]-eta_ab[0,:]])
    etamin     = min(np.amin(eta_a_plot), -np.amax(eta_a_plot))
    etamax     = max(np.amax(eta_a_plot), -np.amin(eta_a_plot))
    nFig       += 1
    plt.figure(nFig)
    plt.imshow(eta_a_plot, vmin=etamin, vmax=etamax, cmap='coolwarm', aspect='auto')
    plt.xticks(range(0, nc,int(nc/10)), range(0,nc,int(nc/10)))
    plt.yticks(range(2), ['ac', 'ab'])
    plt.title('Bed elevation difference wrt IC')
    plt.colorbar()
    
# Plot water depths in 2D
nFig    += 1
plt.figure(nFig)
Da_disp = np.vstack([D_ac, D_ab])
Dmin    = np.amin(Da_disp)
Dmax    = np.amax(Da_disp)
if Dmax > Dmin+tol:
    plt.imshow(Da_disp, cmap='Blues', vmin=Dmin-1/2*(Dmax-Dmin), vmax=Dmax, aspect='auto')
else:
    plt.imshow(Da_disp, cmap='Blues', aspect='auto')
plt.xticks(range(0,nc,int(nc/10)), range(0,nc,int(nc/10)))
plt.yticks(range(2), ['ac', 'ab'])
plt.title('Water depth [m]')
plt.colorbar()

# Plot mean water depth longitudinal profile
nFig += 1
myPlot(nFig,xi,D_a,'Mean water depth','Cross-section-averaged water depth along the channel','x [m]','D_a [m]')
myPlot(nFig+1,xi,D_ab,'ab water depth','ab depth','x [m]','D_ab [m]')
myPlot(nFig+2,xi,D_ac,'ac water depth','ac depth','x [m]','D_ac [m]')
nFig += 2

# Plot water discharges in 2D
Qa_disp = np.vstack([Q_ac, Q_ab])
nFig += 1
plt.figure(nFig)
plt.imshow(Qa_disp, cmap='Blues', vmin=np.amin(Qa_disp), vmax=np.amax(Qa_disp), aspect='auto')
plt.xticks(range(0,nc,int(nc/10)), range(0,nc,int(nc/10)))
plt.yticks(range(2), ['ac', 'ab'])
plt.title('Water discharge [m^3/s]')
plt.colorbar()

# Plot Qy along the channel
nFig += 1
myPlot(nFig,xc,Q_y/dx,'q_y','Transverse discharge per unit length q_y=Q_y/dx','x [m]','q_y [m^2 s^-1]')
nFig_qy = nFig

print('\nQ_y[1]/dx = %3.2e' % (Q_y[1]/dx))
if bedIC == 'triangular' and st == 0:
    print('dq_y/dx = %3.2e' % ((Q_y[-1]-Q_y[-inStepcells])/(xc[-1]-xc[-inStepcells])/dx))

# Plot deltaQ evolution over time
nFig += 1
myPlot(nFig,t[1:],deltaQ,'deltaQ','Discharge asymmetry vs time',)

# Plot export
if saveOutputs:
    filename = 'IS' + str(inStep[0]) + '_' + bedIC + '_ISL' + str(ISlength) + '_dx' + str(dx)
    output_dir = 'plots_Qy/Ls' + str(Ls)
    mkdir_p(output_dir)
    plt.figure(nFig_qy)
    plt.savefig(output_dir+'/'+filename+'.pdf')

if showPlots:
    plt.show()

print('\nDone')
