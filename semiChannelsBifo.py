from functions import *

# -------------------------
# INPUTS AND MODEL SETTINGS
# -------------------------

# Model settings
procedure  = 1 # 0: simplified procedure, i.e., first compute water depths along semichannels in upstream direction, then compute water discharges.
# 1: complete procedure: starting from the known discharge value 
bifoSwitch = 1 # if ==1, two anabranches are attached at the downstream end of the semichannels
st         = 0 # Solid transport switch: 0=fixed bed; 1=movable bed
bedIC      = 'triangular' # 'exp' adds a gradual inlet step with an exponential trend, so that deltaEta[-1 ]=inStep[0]; 'flat'; 'lastCell'; 'triangular'
ISlength   = 1 # % of channel length influenced by inlet step; used if bedIC=='triangular' or bedIC=='exp'
inStep     :list = [-0.004]
Hcoeff     = 0 # =(Hd-H0)/D0, where H0 is the wse associated to D0 and Hd is the imposed downstream BC

# I/O settings
numPlots    = 20
showPlots   = True
saveOutputs = False

# Numerical parameters
dt        = 600 # timestep [s]
dx        = 5 # cell length [m]
tend      = 3
maxIter   = int(1e6)*st+1 # max number of iterations during time evolution
maxIterQy = 10
tol       = 1e-10 # Iterations tolerance

# Hydraulic parameters
RF     = 0 # flow resistance formula switch: if=0, C=C0 is a constant; if=1, C varies according to local water depth
C0     = 0 # if =0, it's computed as a function of D0=ds0/d50 via a logarithmic formula
eps_c  = 2.5 # Chézy logarithmic formula coefficient (used only if RF=1 or C0=0)
TF     = 'P90' # sediment transport formula. Available options: 'P78' (Parker78), 'MPM', 'P90' (Parker90), 'EH' (Engelund&Hansen)
Ls     = 500 # =L/D0, L=branches' dimensional length
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
W0   = beta0*2*D0
if C0 == 0:
    C0 = 6+2.5*np.log(D0/(eps_c*d50))
phi00, phiD0, phiT0 = phis_scalar(theta0, TF, D0, d50)
Q0    = uniFlowQ(RF, W0, S0, D0, d50, g, C0, eps_c)
Qs0   = W0*np.sqrt(g*delta*d50**3)*phi00
Fr0   = Q0/(W0*D0*np.sqrt(g*D0))
nc    = int(Ls*D0/dx)

# Exner time computation
Tf = (1-p)*W0*D0/(Qs0/W0)

# βR alpha computation
betaR = betaR_MR(theta0, ds0, r, phiD0, phiT0, eps_c)

# Arrays initialization
t         : list = [0]
deltaQ    : list = [0]
qylist    : list = []
k_control : list = []
eta_ab    = np.zeros((maxIter+1,nc+1))
eta_ac    = np.zeros((maxIter+1,nc+1))
eta_b     = np.zeros((maxIter+1,nc))
eta_c     = np.zeros((maxIter+1,nc))
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
S_b       = np.ones(nc)*S0
S_c       = np.ones(nc)*S0
Theta_ab  = np.zeros(nc+1)
Theta_ac  = np.zeros(nc+1)
Theta_b   = np.zeros(nc+1)
Theta_c   = np.zeros(nc+1)
Qs_ab     = np.zeros(nc+1)
Qs_ac     = np.zeros(nc+1)
Qs_y      = np.zeros(nc+1)
Qs_b      = np.zeros(nc+1)
Qs_c      = np.zeros(nc+1)

# Space domain for the semichannels
xi = np.linspace(0,nc*dx,nc+1)  # cell interfaces coordinates
xc = np.linspace(dx/2,nc*dx-dx/2,nc) # cell centroids coordinates

# Channel A IC
eta_ab[0,:]  = np.linspace(0,-S0*dx*nc,num=len(eta_ab[0,:]))
eta_ac[0,:]  = eta_ab[0,:]
if bedIC == 'triangular':
    inStepcells   = int(nc*ISlength)
    eta_ab[0,-1]  += inStep[0]/2*D0
    eta_ac[0,-1]  -= inStep[0]/2*D0
    eta_ab[0,-inStepcells:] = np.linspace(eta_ab[0,-inStepcells],eta_ab[0,-1],num=inStepcells) 
    eta_ac[0,-inStepcells:] = np.linspace(eta_ac[0,-inStepcells],eta_ac[0,-1],num=inStepcells)
elif bedIC == 'lastCell':
    eta_ab[0,-1:]  += inStep[0]/2*D0
    eta_ac[0,-1:]  -= inStep[0]/2*D0
elif bedIC == 'shifted':
    eta_ab[0,:] += inStep[0]/2*D0
    eta_ac[0,:] -= inStep[0]/2*D0
S_ab = (eta_ab[0,:-1]-eta_ab[0,1:])/dx
S_ac = (eta_ac[0,:-1]-eta_ac[0,1:])/dx
W_ab = W0/2
W_ac = W0/2

if bifoSwitch == 1:
    # Anabranches IC
    eta_b[0,:] = np.linspace(eta_ab[0,-1]-S0*dx, eta_ab[0,-1]-S0*nc*dx, num=len(eta_b[0,:]))
    eta_c[0,:] = np.linspace(eta_ac[0,-1]-S0*dx, eta_ac[0,-1]-S0*nc*dx, num=len(eta_c[0,:])) 
    W_b        = W0/2
    W_c        = W0/2
    Q_b        = Q0/2
    Q_c        = Q0/2
    # Branch B and C downstream BC: H(t)=H0
    H0 = 0.5*(eta_b[0,-1]+eta_c[0,-1])+D0*(1+Hcoeff)
    Hd_b: list = [H0]
    Hd_c: list = [H0]
else:
    H0 = 0.5*(eta_ab[0,-1]+eta_ac[0,-1])+D0*(1+Hcoeff)

# Print model parameters
print('\nINPUT PARAMETERS:\nHcoeff = %3.2f\nSolid transport formula = %s\nnc = %d\ndx = %d m'
      '\ndt = %d s\nt_end = %2.1f Tf\nL* = %d\nβ0 = %4.2f\nθ0 = %4.3f\nds0 = %2.1e\nr = %2.1f'
      '\nd50 = %3.2e m\n'
      '\nMAIN CHANNEL IC:\nW0 = %4.2f m\nS0 = %3.2e\nD0 = %3.2f m\nFr0 = %2.1f\nL = %4.1f m\n'
      'Q0 = %3.2e m^3 s^-1\nQs0 = %3.2e m^3 s^-1\n\nExner time: Tf = %3.2f h'
      % (Hcoeff, TF, nc, dx, dt, tend, Ls, beta0, theta0, ds0, r, d50, W0, S0, D0, Fr0, Ls*D0, Q0, Qs0, Tf/3600))

print('\nINLET STEP SETTINGS:\nbedIC = %s\nISlength = %3.2f\ninStep = %3.2f' 
      % (bedIC,ISlength,inStep[0]))

if bedIC == 'triangular':
    print('S_ab/S_ac = %3.2f\n' % (S_ab[-3]/S_ac[-3]))
print('\n')

# Plot bed elevation IC
if bedIC != 'flat':
    myPlot(0,xi/W0,eta_ab[0,:],'eta_ab','Semichannels bed profiles IC','x[m]','η [m]')
    myPlot(0,xi/W0,eta_ac[0,:],'eta_ac')
    if showPlots:
        plt.show()

for n in range(0, maxIter):
    #? Downstream BC update: if there is no bifurcation, downstream depth is given
    #? by the known value of wse. If there is, first the water discharge partitioning at the
    #? node is computed, and the resulting average wse is used as downstream BC for the semichannels
    if bifoSwitch==0:
        D_a [-1]   = H0-0.5*(eta_ab[n,-1]+eta_ac[n,-1])
        D_ab[-1]   = D_a[-1]-0.5*(eta_ab[n,-1]-eta_ac[n,-1])
        D_ac[-1]   = D_a[-1]+0.5*(eta_ab[n,-1]-eta_ac[n,-1])
    else:
        # Compute water depths at branches end
        D_b[-1] = Hd_b[-1]-eta_b[n,-1]
        D_c[-1] = Hd_c[-1]-eta_c[n,-1]
        # Compute the discharge partitioning at the node through profiles + (Hb=Hc)
        D_b = buildProfile_rk4(RF,D_b[-1],Q_b,W_b,S_b,d50,dx,g,C0,eps_c)
        D_c = buildProfile_rk4(RF,D_c[-1],Q_c,W_c,S_c,d50,dx,g,C0,eps_c)
        if abs((D_b[0]-D_c[0]+inStep[-1]*D0)/(0.5*(D_b[0]+D_c[0])))>tol:
            Q_b = opt.fsolve(fSys_rk4,Q_b,(RF,D_b[-1],D_c[-1],D0,inStep[-1],Q0,W_b,W_c,S_b,S_c,d50,dx,g,C0,eps_c),xtol=tol)
            Q_c = Q0-Q_b
            D_b = buildProfile_rk4(RF,D_b[-1],Q_b,W_b,S_b,d50,dx,g,C0,eps_c)
            D_c = buildProfile_rk4(RF,D_c[-1],Q_c,W_c,S_c,d50,dx,g,C0,eps_c)
            Fr = np.hstack([Q_b/(W_b*D_b*np.sqrt(g*D_b)),(Q_c/(W_c*D_c*np.sqrt(g*D_c)))])
            if np.any(Fr>0.95):
                print("Supercritical flow")
                break
        Q_ab[-1] = Q_b
        Q_ac[-1] = Q_c
        D_ab[-1] = D_b[0]
        D_ac[-1] = D_c[0]
        D_a [-1] = 0.5*(D_ab[-1]+D_ac[-1])

    if procedure == 0:
        #? 'Simplified' procedure
        # First solve the 1D profile in channel a, then compute Dab and Dac exploiting dH/dy=0, then compute Qy,Qab,Qac
        D_a        = buildProfile_rk4(RF,D_a[-1],Q0,W0,0.5*(S_ab+S_ac),d50,dx,g,C0,eps_c)
        D_ab [:-1] = D_a[:-1]-0.5*(eta_ab[n,:-1]-eta_ac[n,:-1])
        D_ac [:-1] = D_a[:-1]+0.5*(eta_ab[n,:-1]-eta_ac[n,:-1])
        Q_ab[0]    = Q0/2
        Q_ac[0]    = Q0/2

        for i in range(0,nc):
            # Compute Qy explicitly in downstream direction, computing Qab and Qac using mass continuity 
            Q_y[i] = QyExpl(D_ab[i],D_ac[i],Q_ab[i],Q_ac[i],S_ab[i],S_ac[i],W_ab,W_ac,g,d50,dx,C0,RF,eps_c)
            Q_ab[i+1] = Q_ab[i]+Q_y[i]
            Q_ac[i+1] = Q_ac[i]-Q_y[i]
        print('(Q_ab[-1]-Q_ac[-1])/(Q_b-Q_c) = %4.3f' % ((Q_ab[-1]-Q_ac[-1])/(Q_b-Q_c)))

    else:
        #? 'Complete' procedure
        # MT procedure (19.05.2022) for semichannels with bifo: start from Qab[-1] and Qac[-1], then
        # solve linear momentum equation in upstream direction to compute Da and Qy. Force Qy to stay>0
        for i in range(nc-1,-1,-1):
            # First guess
            Q_abM = 0.5*(Q0/2+Q_ab[i+1])
            Q_acM = 0.5*(Q0/2+Q_ac[i+1])
            Q_yMV = 0.5*(Q_ab[i+1]-Q_ac[i+1])
            D_aM  = 0.5*(D0+D_a[i+1])
            D_abM = D_aM-0.5*(eta_ab[n,i]-eta_ac[n,i])
            D_acM = D_aM+0.5*(eta_ab[n,i]-eta_ac[n,i])
            D_abM_control : list = [D_abM/D0]
            D_aM_control  : list = [D_aM/D0]
            Q_abM_control : list = [Q_abM/(Q0/2)]
            Q_yMV_control : list = [Q_yMV/(0.5*(Q_ab[i+1]-Q_ac[i+1]))]

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
                D_abM = D_aM-0.5*(eta_ab[n,i]-eta_ac[n,i])
                D_acM = D_aM+0.5*(eta_ab[n,i]-eta_ac[n,i])
                # Update water discharges at section M
                Q_yMV_new = QyExpl(D_abK,D_acK,Q_abK,Q_acK,S_ab[i],S_ac[i],W_ab,W_ac,g,d50,dx,C0,RF,eps_c)
                if Q_yMV_new < 0:
                    Q_yMV_new = 0
                Q_abM = Q_ab[i+1]-Q_yMV_new
                Q_acM = Q_ac[i+1]+Q_yMV_new
                # Debug lists
                D_abM_control .append(D_abM/D0)
                D_aM_control  .append(D_aM/D0)
                Q_abM_control .append(Q_abM/(Q0/2))
                Q_yMV_control .append(Q_yMV_new/(0.5*(Q_ab[i+1]-Q_ac[i+1])))
                # Check for convergence
                if abs((Q_yMV_new-Q_yMV)/Q0)<np.sqrt(tol):
                    k_control.append(k)
                    break
                Q_yMV = Q_yMV_new

            D_ab[i] = D_abM
            D_ac[i] = D_acM
            D_a [i] = D_aM
            Q_ab[i] = Q_abM
            Q_ac[i] = Q_acM
            Q_y [i] = Q_yMV_new
        
    """
    # Upstream BC
    Q_ab[0] = Q0/2
    Q_ac[0] = Q0/2
    Q_y [0] = 0.5*(Q_ab[1]-Q_ac[1])
    dDadxMV = dDadxExpl(Q_y[0],D_ab[1],D_ac[1],0.5*(Q_ab[0]+Q_ab[1]),0.5*(Q_ac[0]+Q_ac[1]),S_ab[0],S_ac[0],W_ab,W_ac,g,d50,dx,C0,RF,eps_c)
    D_ab[0] = D_ab[1]-(dDadxMV+0.5*(S_ab[0]-S_ac[0]))*dx
    D_ac[0] = D_ac[1]-(dDadxMV-0.5*(S_ab[0]-S_ac[0]))*dx
    D_a [0] = (D_ab[0]+D_ac[0])*0.5
    """

    # Shields and Qs update for the semchannels
    Theta_a  = shieldsUpdate(RF, Q0, W0, D_a, d50, g, delta, C0, eps_c)
    Theta_ab = shieldsUpdate(RF, Q_ab, W_ab, D_ab, d50, g, delta, C0, eps_c)
    Theta_ac = shieldsUpdate(RF, Q_ac, W_ac, D_ac, d50, g, delta, C0, eps_c)
    Qs_ab    = W_ab*np.sqrt(g*delta*d50**3)*phis(Theta_ab, TF, D0, d50)[0]*st
    Qs_ac    = W_ac*np.sqrt(g*delta*d50**3)*phis(Theta_ac, TF, D0, d50)[0]*st
    Qs_y[1:] = (Qs_ab[1:]+Qs_ac[1:])*(Q_y/Q0-2*r*dx/(W0*Theta_a[1:]**0.5)*(eta_ab[n,1:]-eta_ac[n,1:])/W0)*st
    Qs_y[0]  = Qs0*((Q_ab[0]-Q_ac[0])/(2*Q0)-2*r*dx/(W0*theta0**0.5)*(eta_ab[n,0]-eta_ac[n,0])/W0)*st

    # Bed elevation and slope update
    eta_ab[n+1,1:] = eta_ab[n,1:]+dt*(Qs_ab[:-1]-Qs_ab[1:]+Qs_y[1:])/((1-p)*W_ab*dx)*st
    eta_ac[n+1,1:] = eta_ac[n,1:]+dt*(Qs_ac[:-1]-Qs_ac[1:]-Qs_y[1:])/((1-p)*W_ac*dx)*st
    eta_ab[n+1,0]  = eta_ab[n,0]+dt*(Qs0/2-Qs_ab[0]+Qs_y[0])/((1-p)*W_ab*dx)*st
    eta_ab[n+1,0]  = eta_ac[n,0]+dt*(Qs0/2-Qs_ac[0]-Qs_y[0])/((1-p)*W_ac*dx)*st
    S_ab = (eta_ab[n+1,:-1]-eta_ab[n+1,1:])/dx
    S_ac = (eta_ac[n+1,:-1]-eta_ac[n+1,1:])/dx

    if bifoSwitch==1:
        #Shields, Qs and eta update for the anabranches
        Theta_b  = shieldsUpdate(RF,Q_b,W_b,D_b,d50,g,delta,C0,eps_c)
        Theta_c  = shieldsUpdate(RF,Q_c,W_c,D_c,d50,g,delta,C0,eps_c)
        Qs_b     = W_b*np.sqrt(g*delta*d50**3)*phis(Theta_b, TF, D0, d50)[0]*st
        Qs_c     = W_c*np.sqrt(g*delta*d50**3)*phis(Theta_c, TF, D0, d50)[0]*st
        Qs_b[0]  = Qs_ab[-1]
        Qs_c[0]  = Qs_ac[-1]
        eta_b [n+1,:] = eta_b[n,:]+dt*(Qs_b[:-1]-Qs_b[1:])/((1-p)*dx*W_b)*st
        eta_c [n+1,:] = eta_c[n,:]+dt*(Qs_c[:-1]-Qs_c[1:])/((1-p)*dx*W_c)*st

        # Bed slope update
        S_b[0]  = (eta_ab[n+1,-1]-eta_b[n+1,0])/dx
        S_c[0]  = (eta_ac[n+1,-1]-eta_c[n+1,0])/dx
        S_b[1:] = (eta_b[n+1,:-1]-eta_b[n+1,1:])/dx
        S_c[1:] = (eta_c[n+1,:-1]-eta_c[n+1,1:])/dx

        # Update time-controlled lists
        deltaQ.append((Q_b-Q_c)/Q0)
        inStep.append((eta_ab[n+1,-1]-eta_ac[n+1,-1])/D0)

    # Time update + end-time condition for the simulation's end
    t.append(t[-1]+dt)
    if t[-1] >= (tend * Tf):
        print('\nEnd time reached\n')
        break

    # Print elapsed time
    if n % 100 == 0:
        print("\nElapsed time = %4.1f Tf" % (t[n] / Tf))
        #print("Q_b/Q0 = %3.2f, Q_c/Q0 = %3.2f, D_b[0]/D0 = %4.3f, D_c[0]/D0 = %4.3f" % (Q_b/Q0,Q_c/Q0,D_b[0]/D0,D_c[0]/D0))
        #print("(Q_ab[-1]-Q_b)/Q0=%3.2e, (Q_ac[-1]-Q_c)/Q0=%3.2e" % ((Q_ab[-1]-Q_b)/Q0,(Q_ac[-1]-Q_c)/Q0))

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
    plt.xlabel('x/Wa [-]')
    plt.ylabel('(η-η0)/D0 [-]')
    plt.figure(nFig+1)
    plt.title('Semichannel ab bed elevation evolution in time')
    plt.xlabel('x/Wa [-]')
    plt.ylabel('(η-η0)/D0 [-]')
    plt.figure(nFig+2)
    plt.title('Semichannel ac bed elevation evolution in time')
    plt.xlabel('x/Wa [-]')
    plt.ylabel('(η-η0)/D0 [-]')
    
    for i in range(numPlots):
        plotTimeIndex = int(plotTimeIndexes[i])
        myPlot(nFig,   xi/W0, (0.5*(eta_ab[plotTimeIndex,:]+eta_ac[plotTimeIndex,:])-0.5*(eta_ab[0,:]+eta_ac[0,:]))/D0, 
            ('t=%3.1f Tf' % (plotTimeIndex*dt/Tf)), color=bed_colors[i])
        myPlot(nFig+1, xi/W0, (eta_ab[plotTimeIndex,:]-eta_ab[0,:])/D0, ('t=%3.1f Tf' % (plotTimeIndex*dt/Tf)), color=bed_colors[i])
        myPlot(nFig+2, xi/W0, (eta_ac[plotTimeIndex,:]-eta_ac[0,:])/D0, ('t=%3.1f Tf' % (plotTimeIndex*dt/Tf)), color=bed_colors[i]) 
    nFig += 2

    # Plot semichannels bed elevation values
    eta_a_plot = np.vstack([(eta_ac[n+1,:]-eta_ac[0,:])/D0, (eta_ab[n+1,:]-eta_ab[0,:])/D0])
    etamin     = min(np.amin(eta_a_plot), -np.amax(eta_a_plot))
    etamax     = max(np.amax(eta_a_plot), -np.amin(eta_a_plot))
    nFig       += 1
    plt.figure(nFig)
    plt.imshow(eta_a_plot, vmin=etamin, vmax=etamax, cmap='coolwarm', aspect='auto')
    plt.xticks(range(0, nc,int(nc/10)), range(0,nc,int(nc/10)))
    plt.yticks(range(2), ['ac', 'ab'])
    plt.title('Bed elevation difference wrt IC, scaled with D0')
    plt.colorbar()
    
# Plot water depths in 2D
nFig    += 1
plt.figure(nFig)
Da_disp = np.vstack([D_ac/D0, D_ab/D0])
Dmin    = np.amin(Da_disp)
Dmax    = np.amax(Da_disp)
if Dmax > Dmin+tol:
    plt.imshow(Da_disp, cmap='Blues', vmin=Dmin-1/2*(Dmax-Dmin), vmax=Dmax, aspect='auto')
else:
    plt.imshow(Da_disp, cmap='Blues', aspect='auto')
plt.xticks(range(0,nc,int(nc/10)), range(0,nc,int(nc/10)))
plt.yticks(range(2), ['ac', 'ab'])
plt.title('Water depth scaled with D0')
plt.colorbar()

# Plot mean water depth longitudinal profile
nFig += 1
myPlot(nFig,xi/W0,D_a/D0,'Mean water depth','Cross-section-averaged water depth along the channel','x/W0 [-]','D_a/D0 [-]')
myPlot(nFig+1,xi/W0,D_ab/D0,'ab water depth','ab depth','x/W0 [-]','D_ab/D0 [-]')
myPlot(nFig+2,xi/W0,D_ac/D0,'ac water depth','ac depth','x/W0 [-]','D_ac/D0 [-]')
nFig += 2

# Plot water discharges in 2D
Qa_disp = np.vstack([Q_ac/(Q0/2), Q_ab/(Q0/2)])
nFig += 1
plt.figure(nFig)
plt.imshow(Qa_disp, cmap='Blues', vmin=np.amin(Qa_disp), vmax=np.amax(Qa_disp), aspect='auto')
plt.xticks(range(0,nc,int(nc/10)), range(0,nc,int(nc/10)))
plt.yticks(range(2), ['ac', 'ab'])
plt.title('Water discharge scaled with Q0/2 [-]')
plt.colorbar()

# Plot Qy along the channel
nFig += 1
myPlot(nFig,xc/W0,Q_y/Q0,None,'Transverse discharge Qy scaled with Q0','x/W0 [-]','Q_y/Q_0 [-]')

if bifoSwitch==1 and st==1:
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
        myPlot(nFig+1, xc/W0, (eta_b[plotTimeIndex,:]-eta_b[0,:])/D0, ('t=%3.1f Tf' % (plotTimeIndex*dt/Tf)), color=bed_colors[i])
        myPlot(nFig+2, xc/W0, (eta_c[plotTimeIndex,:]-eta_c[0,:])/D0, ('t=%3.1f Tf' % (plotTimeIndex*dt/Tf)), color=bed_colors[i])
    nFig += 2   
    # Plot deltaQ evolution over time
    nFig += 1
    myPlot(nFig,t/Tf,deltaQ,'deltaQ','Discharge asymmetry vs time','t/Tf [-]','deltaQ')

if showPlots:
    plt.show()

print('\nDone')
