from functions import *

# -------------------------
# INPUTS AND MODEL SETTINGS
# -------------------------

# Model settings
st       = 0  # Solid transport switch: 0=fixed bed; 1=movable bed
Hcoeff   = 0  # =(Hd-H0)/D0, where H0 is the wse associated to D0 and Hd is the imposed downstream BC
bedBC    = 'exp'  # available options: 'exp' adds a gradual inlet step with an exponential trend, that causes
# deltaEta[-1]=inStep; 'flat'
numPlots = 20

# Numerical parameters
dt      = 100  # timestep [s]
dx      = 25  # cell length [m]
tend    = 8
maxIter = int(1e6)*st+1 # max number of iterations during time evolution
tol     = 1e-6 # Newton method tolerance

# Hydraulic parameters
RF     = 'ks' # flow resistance formula. Available options: 'ks' (Gauckler&Strickler), 'C' (Chézy)
ks0    = 0 # if =0, it is computed using Gauckler&Strickler formula; otherwise it's considered a constant
C0     = 0 # if =0, it is computed using a logarithmic formula; otherwise it's considered a constant
eps_c  = 2.5 # Chézy logarithmic formula coefficient
TF     = 'P90' # sediment transport formula. Available options: 'P78' (Parker78), 'MPM', 'P90' (Parker90), 'EH' (Engelund&Hansen)
Ls     = 900 # =L/D0, L=branches' dimensional length
beta0  = 20
theta0 = 0.08
ds0    = 0.01 # =d50/D0
rW     = 0.5 # =Wb/W_a, where Wb=Wc and W_a=upstream channel width
d50    = 0.01 # median sediment diameter [m]
p      = 0.6 # bed porosity
r      = 0.5 # Ikeda parameter
inStep = d50

# Physical constants
delta = 1.65
g     = 9.81

# ----------------------------------
# IC & BC DEFINITION AND MODEL SETUP
# ----------------------------------

# Main channel IC
if ks0 == 0:
    ks0 = 21.1/(d50**(1/6))
S0    = theta0*delta*ds0
D0    = d50/ds0
W_a   = beta0*2*D0
phi00, phiD0, phiT0 = phis_scalar(theta0, TF, D0, d50)
Q0    = uniFlowQ(RF, W_a, S0, D0, d50, g, ks0, C0, eps_c)
Qs0   = W_a*np.sqrt(g*delta*d50 ** 3)*phi00
Fr0   = Q0/(W_a*D0*np.sqrt(g*D0))
nc    = int(Ls*D0/dx)

# Exner time computation
Tf = (1-p)*W_a*D0/(Qs0/W_a)

# Print model paramters
print('\nINPUT PARAMETERS:\nFlow resistance formula = %s\nSolid transport formula = %s\nnc = %d\ndx = %d m'
      '\ndt = %d s\nt_end = %2.1f Tf\nL* = %d\nβ0 = %4.2f\nθ0 = %4.3f\nds0 = %2.1e\nr = %2.1f'
      '\nd50 = %3.2e m\n'
      '\nMAIN CHANNEL IC:\nW_a = %4.2f m\nS0 = %3.2e\nD0 = %3.2f m\nFr0 = %2.1f\nL = %4.1f m\n'
      'Q0 = %3.2e m^3 s^-1\nQs0 = %3.2e m^3 s^-1\n\nExner time: Tf = %3.2f h\n'
      % (RF, TF, nc, dx, dt, tend, Ls, beta0, theta0, ds0, r, d50, W_a, S0, D0, Fr0, Ls * D0, Q0, Qs0, Tf / 3600))

# Arrays initialization
nIter     = math.ceil(tend*Tf/dt)+1
eta_a     = np.zeros((nIter,nc))
eta_ab    = np.zeros((nIter,nc))
eta_ac    = np.zeros((nIter,nc))
eta_ab_i  = np.zeros(nc+1)
eta_ac_i  = np.zeros(nc+1)
eta_a_ic  = np.zeros(nc)
eta_ab_ic = np.zeros(nc)
eta_ac_ic = np.zeros(nc)
deltaEta  = np.zeros((nIter,nc+1))
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
if bedBC == 'exp':
    eta_ab[0,:]  -= inStep*np.exp(-(xc[-1]-xc[:])/(0.25*Ls*D0))
    eta_ac[0,:]  += inStep*np.exp(-(xc[-1]-xc[:])/(0.25*Ls*D0))
eta_a[0,:]   = (eta_ab[0,:]+eta_ac[0,:])/2
eta_ab_ic[:] = eta_ab[0,:]
eta_ac_ic[:] = eta_ac[0,:]
eta_a_ic[:]  = eta_a[0,:]
W_ab         = W_a/2
W_ac         = W_a/2
Q_ab[:]      += Q0/2
Q_ac[:]      += Q0/2

# Plot bed elevation IC
eta_a_plot = np.vstack([eta_ac[0,:], eta_ab[0,:]])
etamin     = min(np.amin(eta_a_plot), -np.amax(eta_a_plot))
etamax     = max(np.amax(eta_a_plot), -np.amin(eta_a_plot))

if bedBC=='exp':
    myPlot(0,xc,eta_ab[0,:],'eta_ab')
    myPlot(0,xc,eta_ac[0,:],'eta_ac')
    plt.show()

for n in range(0, maxIter):
    # Channel cells slope and inlet step update
    S_ab[1:-1]      = ((eta_ab[n,:-2]-eta_ab[n,1:-1])/dx+(eta_ab[n,1:-1]-eta_ab[n,2:])/dx)/2
    S_ab[0]         = (eta_ab[n,0]- eta_ab[n,1])/dx
    S_ab[-1]        = (eta_ab[n,-2]-eta_ab[n,-1])/dx
    S_ac[1:-1]      = ((eta_ac[n,:-2]-eta_ac[n,1:-1])/dx+(eta_ac[n,1:-1]-eta_ac[n,2:])/dx)/2
    S_ac[0]         = (eta_ac[n,0]- eta_ac[n,1])/dx
    S_ac[-1]        = (eta_ac[n,-2]-eta_ac[n,-1])/dx
    eta_ab_i[:-1]   = eta_ab[n,:]+S_ab[:]*dx/2
    eta_ac_i[:-1]   = eta_ac[n,:]+S_ac[:]*dx/2
    eta_ab_i[-1]    = eta_ab[n,-1]-S_ab[-1]*dx/2
    eta_ac_i[-1]    = eta_ac[n,-1]-S_ac[-1]*dx/2
    deltaEta[n,:]   = (eta_ab_i[:]-eta_ac_i[:])/D0

    # Downstream BC: H(t)=H0
    Q_ab[-1] = opt.fsolve(fSysLocalUnsteady, Q_ab[-1], (eta_ab_i[-1], eta_ac_i[-1], S_ab[-1],
         S_ac[-1], W_ab, W_ac, Q0, g, d50, D0, RF, ks0, C0, eps_c), xtol=tol)[0]
    Q_ac[-1] = Q0-Q_ab[-1]
    D_ab[-1] = uniFlowD(RF, Q_ab[-1], W_ab, S_ab[-1], d50, g, ks0, C0, eps_c, D0)
    D_ac[-1] = uniFlowD(RF, Q_ac[-1], W_ac, S_ac[-1], d50, g, ks0, C0, eps_c, D0)
    D_a[-1]  = (D_ab[-1]+D_ac[-1])/2

    # Solve the governing system to compute the unknowns D_ab[i], D_ac[i], Q_ab[i], Q_ac[i] and Qy[i]
    for i in range(nc-1, -1, -1):
        Q_y[i], D_a[i]  = QyDaUpdate(D0, Q0, D_ab[i+1], D_ac[i+1], Q_ab[i+1], Q_ac[i+1], S_ab[i], S_ac[i], 
            deltaEta[n,i], deltaEta[n,i+1], W_ab, W_ac, g, d50, dx, ks0, C0, RF, eps_c)
        D_ab[i] = D_a[i]-deltaEta[n,i]*D0/2
        D_ac[i] = D_a[i]+deltaEta[n,i]*D0/2
        Q_ab[i] = Q_ab[i+1]-Q_y[i]
        Q_ac[i] = Q_ac[i+1]+Q_y[i]

    # Shields parameter update
    Theta_ab = shieldsUpdate(RF, Q_ab, W_ab, D_ab, d50, g, delta, ks0, C0, eps_c)
    Theta_ac = shieldsUpdate(RF, Q_ac, W_ac, D_ac, d50, g, delta, ks0, C0, eps_c)

    # Solid discharge computation along the semichannels
    Qs_ab    = W_ab*np.sqrt(g*delta*d50**3)*phis(Theta_ab, TF, D0, d50)[0]*st
    Qs_ac    = W_ac*np.sqrt(g*delta*d50**3)*phis(Theta_ac, TF, D0, d50)[0]*st
    Qs_ab[0] = Qs0/2*st
    Qs_ac[0] = Qs0/2*st

    # Transverse solid discharge along channel a
    Theta_a = shieldsUpdate(RF, Q0, W_a, D_a, d50, g, delta, ks0, C0, eps_c)
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
    if n % 2000 == 0:
        print("Elapsed time = %4.1f Tf" % (t[n] / Tf))

    if np.any(np.isnan(D_a)):
        break

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
    for i in range(numPlots):
        plotTimeIndex = int(plotTimeIndexes[i])
        myPlot(nFig, xc, eta_a[plotTimeIndex,:]-eta_a_ic[:], ('t=%3.1f Tf' % (plotTimeIndex*dt/Tf)), color=bed_colors[i])
    # Plot semichannels bed elevation values
    eta_a_plot = np.vstack([eta_ac[n+1,:]-eta_ac_ic[:], eta_ab[n+1,:]-eta_ab_ic[:]])
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
Qa_plot = np.vstack([Q_ac, Q_ab])
nFig += 1
plt.figure(nFig)
plt.imshow(Qa_plot, cmap='Purples', vmin=Q0/4, vmax=2/3*Q0, aspect='auto')
plt.xticks(range(0,nc,int(nc/10)), range(0,nc,int(nc/10)))
plt.yticks(range(2), ['ac', 'ab'])
plt.title('Water discharge [m^3/s]')
plt.colorbar()

plt.show()