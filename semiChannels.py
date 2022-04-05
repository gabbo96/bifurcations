from functions import *

# -------------------------
# INPUTS AND MODEL SETTINGS
# -------------------------

# Model settings
st        = 1  # Solid transport switch: 0=fixed bed; 1=movable bed
H0coeff   = -0.1  # =(Hd-H0)/D0, where H0 is the wse associated to D0 and Hd is the imposed downstream BC

# Numerical parameters
dt      = 100 # timestep [s]
dx      = 50  # cell length [m]
tend    = 200
maxIter = 12000000*st+1 # max number of iterations during time evolution
tol     = 1e-6 # Newton method tolerance

# Hydraulic parameters
RF     = 'ks' # flow resistance formula. Available options: 'ks' (Gauckler&Strickler), 'C' (Chézy)
ks0    = 30 # if =0, it is computed using Gauckler&Strickler formula; otherwise it's considered a constant
C0     = 0 # if =0, it is computed using a logarithmic formula; otherwise it's considered a constant
eps_c  = 2.5 # Chézy logarithmic formula coefficient
TF     = 'P90' # sediment transport formula. Available options: 'P78' (Parker78), 'MPM', 'P90' (Parker90), 'EH' (Engelund&Hansen)
Ls     = 3000 # =L/D0, L=branches' dimensional length
beta0  = 20
theta0 = 0.08
ds0    = 0.01 # =d50/D0
rW     = 0.5 # =Wb/W_a, where Wb=Wc and W_a=upstream channel width
d50    = 0.01 # median sediment diameter [m]
p      = 0.6 # bed porosity
r      = 0.5 # Ikeda parameter

# Bed profile plotting parameters & setup
plotBedEvolution  = True
iterationPlotStep = 2*int(4e2)
numMaxPlots       = 30
crange            = np.linspace(0, 1, numMaxPlots)
bed_colors        = plt.cm.viridis(crange)
cindex            = 0

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
phi00, phiD0, phiT0 = phis_scalar(theta0, TF, D0, d50)
Q0    = uniFlowQ(RF, W_a, S0, D0, d50, g, ks0, C0, eps_c)
Qs0   = W_a*np.sqrt(g*delta*d50 ** 3)*phi00
Fr0   = Q0/(W_a*D0*np.sqrt(g*D0))
nc    = int(Ls*D0/dx)

# Exner time computation
Tf = (1-p)*W_a*D0/(Qs0/W_a)

# Arrays initialization
eta_a     = np.zeros(nc)
eta_ab    = np.zeros(nc)
eta_ac    = np.zeros(nc)
eta_a_ic  = np.zeros(nc)
eta_ab_ic = np.zeros(nc)
eta_ac_ic = np.zeros(nc)
deltaEta  = np.zeros(nc+1)
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
eta_ab[:]    = np.linspace(0,-S0*dx*(nc-1),num=len(eta_ab))
eta_ac[:]    = eta_ab[:]
eta_ab_ic[:] = eta_ab[:]
eta_ac_ic[:] = eta_ac[:]
eta_a_ic[:]  = (eta_ab_ic[:]+eta_ac_ic[:])/2
W_ab         = W_a/2
W_ac         = W_a/2
Q_ab[:]      += Q0/2
Q_ac[:]      += Q0/2

# Downstream BC
H0 = (eta_ab[-1]-S0*dx/2+D0)+D0*H0coeff

for n in range(0, maxIter):
    # Bed profile plot
    if plotBedEvolution and n % iterationPlotStep == 0 and cindex < numMaxPlots:
        myPlot(4, xc, (eta_ab+eta_ac)/2-eta_a_ic, None, color=bed_colors[cindex])
        cindex = cindex + 1

    # Channel cells slope and inlet step update
    S_ab[1:-1] = ((eta_ab[:-2]-eta_ab[1:-1])/dx+(eta_ab[1:-1]-eta_ab[2:])/dx)/2
    S_ab[0]    = (eta_ab[0]- eta_ab[1])/dx
    S_ab[-1]   = (eta_ab[-2]-eta_ab[-1])/dx
    S_ac[1:-1] = ((eta_ac[:-2]-eta_ac[1:-1])/dx+(eta_ac[1:-1]-eta_ac[2:])/dx)/2
    S_ac[0]    = (eta_ac[0]- eta_ac[1])/dx
    S_ac[-1]   = (eta_ac[-2]-eta_ac[-1])/dx
    deltaEta[:-1] = (eta_ab[:]-eta_ac[:])/D0
    deltaEta[-1]  = deltaEta[-2]

    # Downstream BC: H(t)=H0
    D_ab[-1] = H0-eta_ab[-1]+S_ab[-1]*dx/2
    D_ac[-1] = H0-eta_ac[-1]+S_ac[-1]*dx/2
    D_a[-1]  = (D_ab[-1]+D_ac[-1])/2

    # Solve the governing system to compute the unknowns D_ab[i], D_ac[i], Q_ab[i], Q_ac[i] and Qy[i]
    for i in range(nc-1, -1, -1):
        Q_y[i], D_a[i]  = QyDaUpdate(D0, Q0, D_ab[i+1], D_ac[i+1], Q_ab[i+1], Q_ac[i+1], S_ab[i], S_ac[i], 
            deltaEta[i], deltaEta[i+1], W_ab, W_ac, g, d50, dx, ks0, C0, RF, eps_c)
        D_ab[i] = D_a[i]-deltaEta[i]*D0/2
        D_ac[i] = D_a[i]+deltaEta[i]*D0/2
        Q_ab[i] = Q_ab[i+1]-Q_y[i]
        Q_ac[i] = Q_ac[i+1]+Q_y[i]

    # Shields parameter update
    Theta_ab = shieldsUpdate(RF, Q_ab, W_ab, D_ab, d50, g, delta, ks0, C0, eps_c)
    Theta_ac = shieldsUpdate(RF, Q_ac, W_ac, D_ac, d50, g, delta, ks0, C0, eps_c)

    # Solid discharge computation + exner
    if st==1:
        # Qsx along the semichannels
        Qs_ab = W_ab*np.sqrt(g*delta*d50**3)*phis(Theta_ab, TF, D0, d50)[0]
        Qs_ac = W_ac*np.sqrt(g*delta*d50**3)*phis(Theta_ac, TF, D0, d50)[0]
        
        Qs_ab[0] = Qs0/2
        Qs_ac[0] = Qs0/2
        # Transverse solid discharge along channel a
        Theta_a = shieldsUpdate(RF, Q0, W_a, D_a, d50, g, delta, ks0, C0, eps_c)
        Qs_y[:] = (Qs_ab[:-1]+Qs_ac[:-1])*(Q_y/Q0-2*r*dx/(W_a*Theta_a[:-1]**0.5)*(eta_ab-eta_ac)/W_a)
        # Apply Exner equation to update node cells elevation
        eta_ab[:] += dt*(Qs_ab[:-1]-Qs_ab[1:]+Qs_y[:])/((1-p)*W_ab*dx)
        eta_ac[:] += dt*(Qs_ac[:-1]-Qs_ac[1:]-Qs_y[:])/((1-p)*W_ac*dx)

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


# Plot final bed elevation profile
myPlot(4, xc, (eta_ab+eta_ac)/2-eta_a_ic, 'final', title='Branch A evolution in time', xlabel='x [m]',
           ylabel='η - η0 [m]', color='red')

# Plot semichannels bed elevation values
if st==1:
    plt.figure(0)
    eta_a = np.vstack([eta_ac-eta_ac_ic, eta_ab-eta_ab_ic])
    etamin = min(np.amin(eta_a),-np.amax(eta_a))
    etamax = max(np.amax(eta_a),-np.amin(eta_a))
    plt.imshow(eta_a, vmin=etamin, vmax=etamax, cmap='coolwarm', aspect='auto')
    plt.xticks(range(0,nc,int(nc/10)), range(0,nc,int(nc/10)))
    plt.yticks(range(2), ['ac', 'ab'])
    plt.title('Bed elevation difference wrt IC')
    plt.colorbar()
    
# Plot water depths in 2D
plt.figure(1)
Da_disp = np.vstack([D_ac, D_ab])
Dmin = np.amin(Da_disp)
Dmax = np.amax(Da_disp)
if Dmax > Dmin+tol:
    plt.imshow(Da_disp, cmap='Blues', vmin=Dmin-1/2*(Dmax-Dmin), vmax=Dmax, aspect='auto')
else:
    plt.imshow(Da_disp, cmap='Blues', aspect='auto')
plt.xticks(range(0,nc,int(nc/10)), range(0,nc,int(nc/10)))
plt.yticks(range(2), ['ac', 'ab'])
plt.title('Water depth [m]')
plt.colorbar()

# Plot mean water depth longitudinal profile
myPlot(2,xi,D_a,'Mean water depth','Cross-section-averaged water depth along the channel','x [m]','D_a [m]')

# Plot water discharges in 2D
plt.figure(3)
Qa_disp = np.vstack([Q_ac, Q_ab])
plt.imshow(Qa_disp, cmap='Purples', vmin=Q0/4, vmax=2/3*Q0, aspect='auto')
plt.xticks(range(0,nc,int(nc/10)), range(0,nc,int(nc/10)))
plt.yticks(range(2), ['ac', 'ab'])
plt.title('Water discharge [m^3/s]')
plt.colorbar()

plt.show()