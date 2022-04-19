from functions import *

# -------------------------
# INPUTS AND MODEL SETTINGS
# -------------------------

# Model settings
st          = 0 # Solid transport switch: 0=fixed bed; 1=movable bed
Hcoeff      = 0 # =(Hd-H0)/D0, where H0 is the wse associated to D0 and Hd is the imposed downstream BC
bedIC       = 'triangularIS' # 'exp' adds a gradual inlet step with an exponential trend, so that deltaEta[-1]=inStep; 'flat'; 'lastCell'; 'triangularIS'
ISlength    = 0.66 # % of channel length influenced by inlet step; used if bedIC=='triangularIS'
inStep      = -0.2
D_b_eq      = 0  # alternative way to set a Hd != H0

# I/O settings
numPlots    = 20
saveOutputs = False

# Numerical parameters
dt        = 50 # timestep [s]
dx        = 2 # cell length [m]
tend      = 2
maxIter   = int(1e6)*st+1 # max number of iterations during time evolution
maxIterQy = 20
tol       = 1e-10 # Newton method tolerance

# Hydraulic parameters
RF     = 0 # flow resistance formula switch: if=0, C=C0 is a constant; if=1, C varies according to local water depth
C0     = 10 # if =0, it is computed using a logarithmic formula; otherwise it's considered a constant
eps_c  = 2.5 # Chézy logarithmic formula coefficient
TF     = 'P90' # sediment transport formula. Available options: 'P78' (Parker78), 'MPM', 'P90' (Parker90), 'EH' (Engelund&Hansen)
Ls     = 1000 # =L/D0, L=branches' dimensional length
beta0  = 16
theta0 = 0.08
ds0    = 0.01 # =d50/D0
rW     = 0.5 # =Wb/W_a, where Wb=Wc and W_a=upstream channel width
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
phi00, phiD0, phiT0 = phis(np.array([theta0]), TF, D0, d50)
Q0    = uniFlowQ(RF, W_a, S0, D0, d50, g, C0, eps_c)
Qs0   = W_a*np.sqrt(g*delta*d50**3)*phi00
Fr0   = Q0/(W_a*D0*np.sqrt(g*D0))
nc    = int(Ls*D0/dx)

# Exner time computation
Tf = (1-p)*W_a*D0/(Qs0/W_a)

# Print model paramters
print('\nINPUT PARAMETERS:\nSolid transport formula = %s\nnc = %d\ndx = %d m'
      '\ndt = %d s\nt_end = %2.1f Tf\nL* = %d\nβ0 = %4.2f\nθ0 = %4.3f\nds0 = %2.1e\nr = %2.1f'
      '\nd50 = %3.2e m\n'
      '\nMAIN CHANNEL IC:\nW_a = %4.2f m\nS0 = %3.2e\nD0 = %3.2f m\nFr0 = %2.1f\nL = %4.1f m\n'
      'Q0 = %3.2e m^3 s^-1\nQs0 = %3.2e m^3 s^-1\n\nExner time: Tf = %3.2f h\n'
      % (TF, nc, dx, dt, tend, Ls, beta0, theta0, ds0, r, d50, W_a, S0, D0, Fr0, Ls * D0, Q0, Qs0, Tf / 3600))

# Arrays initialization
nIter     = math.ceil(tend*Tf/dt)+1
eta_a     = np.zeros((nIter,nc))
eta_ab    = np.zeros((nIter,nc))
eta_ac    = np.zeros((nIter,nc))
eta_ab_i  = np.zeros(nc+1)
eta_ac_i  = np.zeros(nc+1)
D_a       = np.zeros(nc+1)
D_ab      = np.zeros(nc+1)
D_ac      = np.zeros(nc+1)
Q_ab      = np.zeros(nc+1)
Q_ac      = np.zeros(nc+1)
Q_y       = np.zeros(nc)
S_ab      = np.zeros(nc)
S_ac      = np.zeros(nc)
Theta_ab  = np.zeros(nc+1)
Theta_ac  = np.zeros(nc+1)
Qs_ab     = np.zeros(nc+1)
Qs_ac     = np.zeros(nc+1)
Qs_y      = np.zeros(nc)

# Space-time domain
xi = np.linspace(0,nc*dx,nc+1)  # cell interfaces coordinates
xc = np.linspace(dx/2,nc*dx-dx/2,nc) # cell centroids coordinates
t:list = [0]

# IC
eta_ab[0,:]  = np.linspace(0,-S0*dx*(nc-1),num=len(eta_ab[0,:]))
eta_ac[0,:]  = eta_ab[0,:]
if bedIC == 'exp':
    eta_ab[0,:]  += inStep/2*D0*np.exp(-(xc[-1]-xc[:])/(0.25*Ls*D0))
    eta_ac[0,:]  -= inStep/2*D0*np.exp(-(xc[-1]-xc[:])/(0.25*Ls*D0))
elif bedIC == 'triangularIS':
    inStepcells   = int(nc*ISlength)
    eta_ab[0,-1]  += inStep/2*D0
    eta_ac[0,-1]  -= inStep/2*D0
    eta_ab[0,-inStepcells:] = np.linspace(eta_ab[0,-inStepcells],eta_ab[0,-1],num=inStepcells) 
    eta_ac[0,-inStepcells:] = np.linspace(eta_ac[0,-inStepcells],eta_ac[0,-1],num=inStepcells)
elif bedIC == 'lastCell':
    eta_ab[0,-1]  += inStep/2*D0
    eta_ac[0,-1]  -= inStep/2*D0
eta_a[0,:] = (eta_ab[0,:]+eta_ac[0,:])/2
W_ab       = W_a/2
W_ac       = W_a/2

#? Downstream BC: H(t)=H0
if Hcoeff != 0 or D_b_eq == 0:
    D_b_eq = D0
H0 = eta_a[0,-1]+D_b_eq+D0*Hcoeff

# Plot bed elevation IC
if bedIC != 'flat':
    myPlot(0,xc,eta_ab[0,:],'eta_ab','Semichannels bed profiles IC','x[m]','η [m]')
    myPlot(0,xc,eta_ac[0,:],'eta_ac')
    plt.show()

for n in range(0, maxIter):
    # Channel cells slope update through central difference approximation
    S_ab[1:-1]    = (eta_ab[n,:-2]-eta_ab[n,2:])/(2*dx)
    S_ac[1:-1]    = (eta_ac[n,:-2]-eta_ac[n,2:])/(2*dx)
    S_ab[0]       = (eta_ab[n,0]-eta_ab[n,1])/dx
    S_ab[-1]      = (eta_ab[n,-2]-eta_ab[n,-1])/dx
    S_ac[0]       = (eta_ac[n,0]-eta_ac[n,1])/dx
    S_ac[-1]      = (eta_ac[n,-2]-eta_ac[n,-1])/dx

    # Upstream BC: even water discharge partitioning
    Q_ab[0] = Q0/2
    Q_ac[0] = Q0/2

    # Downstream BC: D=H0-eta
    D_ab[-1] = H0-eta_ab[n,-1]
    D_ac[-1] = H0-eta_ac[n,-1]
    D_a[-1]  = (D_ab[-1]+D_ac[-1])/2

    # Solve the governing system to compute the unknowns D_ab[i], D_ac[i], Q_ab[i], Q_ac[i] and Qy[i] iteratively in upstream direction
    for i in range(nc-1, -1, -1):
        # First attempt, assuming D_aM=D0
        Q_abM    = Q0/2
        Q_acM    = Q0/2
        D_ab0    = D0-0.5*(eta_ab[n,i]-eta_ac[n,i])
        D_ac0    = D0+0.5*(eta_ab[n,i]-eta_ac[n,i])
        Q_y_I    = QyExpl(D_ab0,D_ac0,Q_abM,Q_acM,S_ab[i],S_ac[i],W_ab,W_ac,g,d50,dx,C0,RF,eps_c)
        Q_abV    = Q_abM+Q_y_I
        Q_acV    = Q_acM-Q_y_I
        dDadx_I  = dDadxExpl(Q_y_I/dx,D_ab[i+1],D_ac[i+1],Q_abV,Q_acV,S_ab[i],S_ac[i],W_ab,W_ac,g,d50,C0,RF,eps_c)
        dDabdx_I = dDadx_I+0.5*(S_ab[i]-S_ac[i])
        dDacdx_I = dDadx_I-0.5*(S_ab[i]-S_ac[i])
        D_abM    = D_ab[i+1]-dDabdx_I*dx
        D_acM    = D_ac[i+1]-dDacdx_I*dx
        # Further iterations, updating upstream water depths and discharges
        for k in range(maxIterQy):
            D_ab_k    = 0.5*(D_abM+D_ab[i+1])
            D_ac_k    = 0.5*(D_acM+D_ac[i+1])
            Q_ab_k    = 0.5*(Q_abM+Q_abV)
            Q_ac_k    = 0.5*(Q_acM+Q_acV)
            Q_y_k     = QyExpl(D_ab_k,D_ac_k,Q_ab_k,Q_ac_k,S_ab[i],S_ac[i],W_ab,W_ac,g,d50,dx,C0,RF,eps_c)
            dDadx_k   = dDadxExpl(Q_y_k/dx,D_ab_k,D_ac_k,Q_ab_k,Q_ac_k,S_ab[i],S_ac[i],W_ab,W_ac,g,d50,C0,RF,eps_c)
            dDabdx_k  = dDadx_k+0.5*(S_ab[i]-S_ac[i])
            dDacdx_k  = dDadx_k-0.5*(S_ab[i]-S_ac[i])
            D_abM     = D_ab[i+1]-dDabdx_k*dx
            D_acM     = D_ac[i+1]-dDacdx_k*dx
            Q_abV_new = Q_abM+Q_y_k
            Q_acV_new = Q_acM-Q_y_k
            #Q_abM     += Q_abV-Q_abV_new
            #Q_acM     += Q_acV-Q_acV_new
            err       = np.array([Q_abV-Q_abV_new,Q_acV-Q_acV_new])
            Q_abV     = Q_abV_new
            Q_acV     = Q_acV_new
            if np.dot(err,err)<tol:
                break
        D_ab[i]   = D_abM
        D_ac[i]   = D_acM
        D_a [i]   = 0.5*(D_ab[i]+D_ac[i])
        Q_ab[i+1] = Q_abV
        Q_ac[i+1] = Q_acV
        Q_y [i]   = Q_y_k

    # Shields parameter update
    Theta_ab = shieldsUpdate(RF, Q_ab, W_ab, D_ab, d50, g, delta, C0, eps_c)
    Theta_ac = shieldsUpdate(RF, Q_ac, W_ac, D_ac, d50, g, delta, C0, eps_c)

    # Solid discharge computation along the semichannels + second upstream BC
    Qs_ab    = W_ab*np.sqrt(g*delta*d50**3)*phis(Theta_ab, TF, D0, d50)[0]*st
    Qs_ac    = W_ac*np.sqrt(g*delta*d50**3)*phis(Theta_ac, TF, D0, d50)[0]*st
    Qs_ab[0] = Qs0/2*st
    Qs_ac[0] = Qs0/2*st

    # Transverse solid discharge along channel a
    Theta_a = shieldsUpdate(RF, Q0, W_a, D_a, d50, g, delta, C0, eps_c)
    Qs_y[:] = (Qs_ab[:-1]+Qs_ac[:-1])*(Q_y/Q0-2*r*dx/(W_a*Theta_a[:-1]**0.5)*(eta_ab[n,:]-eta_ac[n,:])/W_a)
    
    # Apply Exner equation to update node cells elevation
    eta_ab[n+1,:] = eta_ab[n,:] + dt*(Qs_ab[:-1]-Qs_ab[1:]+Qs_y[:])/((1-p)*W_ab*dx)*st
    eta_ac[n+1,:] = eta_ac[n,:] + dt*(Qs_ac[:-1]-Qs_ac[1:]-Qs_y[:])/((1-p)*W_ac*dx)*st
    eta_a[n+1,:]  = (eta_ab[n+1,:]+eta_ac[n+1,:])/2

    # Time update + end-time condition for the simulation's end
    t.append(t[-1]+dt)
    if t[-1] >= (tend * Tf):
        print('\nEnd time reached\n')
        break

    # Print elapsed time
    if n % 100 == 0:
        print("Elapsed time = %4.1f Tf" % (t[n] / Tf))

    if np.any(np.isnan(D_a)):
        break

if n == maxIter:
    print("\nMax number of iterations reached\n")
# -----
# PLOTS
# -----
nFig = 0

# Bed evolution plot
if st==1:
    # Plot evolution of average elevation eta_a
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
    plt.ylabel('η_ab - η_ab0 [m]')
    plt.figure(nFig+2)
    plt.title('Semichannel ac bed elevation evolution in time')
    plt.xlabel('x [m]')
    plt.ylabel('η_ac - η_ac0 [m]')
    for i in range(numPlots):
        plotTimeIndex = int(plotTimeIndexes[i])
        myPlot(nFig, xc, eta_a[plotTimeIndex,:]-eta_a[0,:], ('t=%3.1f Tf' % (plotTimeIndex*dt/Tf)), color=bed_colors[i])
        myPlot(nFig+1, xc, eta_ab[plotTimeIndex,:]-eta_ab[0,:], ('t=%3.1f Tf' % (plotTimeIndex*dt/Tf)), color=bed_colors[i])
        myPlot(nFig+2, xc, eta_ac[plotTimeIndex,:]-eta_ac[0,:], ('t=%3.1f Tf' % (plotTimeIndex*dt/Tf)), color=bed_colors[i])
    nFig += 2
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

print(Q_y[1]/dx)
print('res= ',(Q_y[-1]-Q_y[-inStepcells])/(xc[-1]-xc[-inStepcells])/dx)

# Plot export
if saveOutputs:
    filename = 'IS' + str(inStep) + '_' + bedIC + '_ISL' + str(ISlength) + '_dx' + str(dx)
    output_dir = 'plots_Qy/Ls' + str(Ls)
    mkdir_p(output_dir)
    plt.figure(nFig_qy)
    plt.savefig(output_dir+'/'+filename+'.pdf')

plt.show()
print('temp')
