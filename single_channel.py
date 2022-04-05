from functions import *

# -------------------------
# INPUTS AND MODEL SETTINGS
# -------------------------

# Model main settings
st        = 1  # Solid transport switch: 0=fixed bed; 1=movable bed
H0coeff   = 0  # =(Hd-H0)/D0, where H0 is the wse associated to D0 and Hd is the imposed downstream BC

# Numerical parameters
dt      = 150 # timestep [s]
dx      = 50  # cell length [m]
tend    = 35
maxIter = 120000*st+1 # max number of iterations during time evolution
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
iterationPlotStep = int(4e2)
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
eta_a    = np.zeros(nc)
eta_a_ic = np.zeros(nc)
D_a      = np.zeros(nc+1)
S_a      = np.zeros(nc) 
Theta_a  = np.zeros(nc+1)
Qs_a     = np.zeros(nc+1)

# Space-time domain
xi = np.linspace(0,nc*dx,nc+1)  # cell interfaces coordinates
xc = np.linspace(dx/2,nc*dx-dx/2,nc) # cell centroids coordinates
t:list = [0]

# IC
eta_a[:]    = np.linspace(0,-S0*dx*(nc-1),num=len(eta_a))
eta_a_ic[:] = eta_a[:]

# Downstream BC
H0 = (eta_a[-1]-S0*dx/2+D0)+D0*H0coeff

for n in range(0, maxIter):
    # Bed profile plot
    if plotBedEvolution and n % iterationPlotStep == 0 and cindex < numMaxPlots:
        myPlot(4, xc, eta_a - eta_a_ic, None, color=bed_colors[cindex])
        cindex = cindex + 1

    # Channel cells slope update
    #S_a[1:-1] = minmod((eta_a[:-2]-eta_a[1:-1])/dx,(eta_a[1:-1]-eta_a[2:])/dx)
    S_a[1:-1] = ((eta_a[:-2]-eta_a[1:-1])/dx+(eta_a[1:-1]-eta_a[2:])/dx)/2
    S_a[0]    = (eta_a[0]-eta_a[1])/dx
    S_a[-1]   = (eta_a[-2]-eta_a[-1])/dx

    # Downstream BC: H(t)=H0
    D_a[-1] = H0-eta_a[-1]+S_a[-2]*dx/2

    # Wse profile update
    D_a = buildProfile_minmod(RF,D_a[-1],Q0,W_a,S_a,d50,dx,g,ks0,C0,eps_c)

    # Shields parameter update
    Theta_a = shieldsUpdate(RF, Q0, W_a, D_a, d50, g, delta, ks0, C0, eps_c)
    
    # Solid discharge computation + exner
    if st==1:
        # Qsx along the semichannels
        Qs_a = W_a*np.sqrt(g*delta*d50**3)*phis(Theta_a, TF, D0, d50)[0]
        Qs_a[0] = Qs0
        # Apply Exner equation to update node cells elevation
        eta_a[:] += dt*(Qs_a[:-1]-Qs_a[1:])/((1-p)*W_a*dx)

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
myPlot(4, xc, eta_a - eta_a_ic, 'final', title='Branch A evolution in time', xlabel='x [m]',
           ylabel='η - η0 [m]', color='red')

# Plot semichannels bed elevation values
if st==1:
    plt.figure(0)
    eta_a = np.vstack([eta_a-eta_a_ic, eta_a-eta_a_ic])
    etamin = min(np.amin(eta_a),-np.amax(eta_a))
    etamax = max(np.amax(eta_a),-np.amin(eta_a))
    plt.imshow(eta_a, vmin=etamin, vmax=etamax, cmap='coolwarm', aspect='auto')
    plt.xticks(range(0,nc,int(nc/10)), range(0,nc,int(nc/10)))
    plt.yticks(range(2), ['ac', 'ab'])
    plt.title('Bed elevation difference wrt IC')
    plt.colorbar()
    
# Plot water depths in 2D
plt.figure(1)
Da_disp = np.vstack([D_a, D_a])
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

plt.show()