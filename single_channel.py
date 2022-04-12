from functions import *

# -------------------------
# INPUTS AND MODEL SETTINGS
# -------------------------

# Model main settings
st          = 1  # Solid transport switch: 0=fixed bed; 1=movable bed
H0coeff     = 0.1  # =(Hd-H0)/D0, where H0 is the wse associated to D0 and Hd is the imposed downstream BC
numPlots    = 30
slopeMethod = 'central'  # available options: 'central', 'downstream', 'minmod'

# Numerical parameters
dt      = 50 # timestep [s]
dx      = 50  # cell length [m]
tend    = 2
maxIter = 120000*st+1 # max number of iterations during time evolution
tol     = 1e-6 # Newton method tolerance

# Hydraulic parameters
RF     = 'ks' # flow resistance formula. Available options: 'ks' (Gauckler&Strickler), 'C' (Chézy)
ks0    = 30 # if =0, it is computed using Gauckler&Strickler formula; otherwise it's considered a constant
C0     = 0 # if =0, it is computed using a logarithmic formula; otherwise it's considered a constant
eps_c  = 2.5 # Chézy logarithmic formula coefficient
TF     = 'P90' # sediment transport formula. Available options: 'P78' (Parker78), 'MPM', 'P90' (Parker90), 'EH' (Engelund&Hansen)
Ls     = 900 # =L/D0, L=branches' dimensional length
beta0  = 15
theta0 = 0.07
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
if ks0 == 0:
    ks0 = 21.1/(d50**(1/6))
S0    = theta0*delta*ds0
D0    = d50/ds0
W_a   = beta0*2*D0
phi00, phiD0, phiT0 = phis(np.array([theta0]), TF, D0, d50)
Q0    = uniFlowQ(RF, W_a, S0, D0, d50, g, ks0, C0, eps_c)
Qs0   = W_a*np.sqrt(g*delta*d50 ** 3)*phi00
Fr0   = Q0/(W_a*D0*np.sqrt(g*D0))
nc    = int(Ls*D0/dx)

# Exner time computation
Tf = (1-p)*W_a*D0/(Qs0/W_a)

# Arrays initialization
nIter    = math.ceil(tend*Tf/dt)+1
eta_a    = np.zeros((nIter,nc))
eta_a_ic = np.zeros(nc)
S_a      = np.zeros((nIter,nc))
S_a_temp = np.zeros(nc+1)
eta_a_i = np.zeros(nc+1)
D_a      = np.zeros(nc+1) 
Theta_a  = np.zeros(nc+1)
Qs_a     = np.zeros(nc+1)

# Space-time domain
xi = np.linspace(0,nc*dx,nc+1)  # cell interfaces coordinates
xc = np.linspace(dx/2,nc*dx-dx/2,nc) # cell centroids coordinates
t:list = [0]

# IC
eta_a[0,:]  = np.linspace(0,-S0*dx*(nc-1),num=len(eta_a[0,:]))
eta_a_ic[:] = eta_a[0,:]

# Downstream BC
H0 = (eta_a[0,-1]-S0*dx/2+D0)+D0*H0coeff

for n in range(0, maxIter):
    # Channel cells slope update
    if slopeMethod == 'central':
        S_a[n,1:-1] = ((eta_a[n,:-2]-eta_a[n,1:-1])/dx+(eta_a[n,1:-1]-eta_a[n,2:])/dx)/2
        S_a[n,0]    = (eta_a[n,0]-eta_a[n,1])/dx
        S_a[n,-1]   = (eta_a[n,-2]-eta_a[n,-1])/dx
    elif slopeMethod == 'downstream':
        S_a [n,:-1] = (eta_a[n,:-1]-eta_a[n,1:])/dx
        S_a [n,-1]  = (eta_a[n,-2]-eta_a[n,-1])/dx
    elif slopeMethod == 'minmod':
        S_a[n,1:-1] = -minmod((eta_a[n,1:-1]-eta_a[n,:-2])/dx,(eta_a[n,2:]-eta_a[n,1:-1])/dx)
        S_a[n,0]    = (eta_a[n,0]-eta_a[n,1])/dx
        S_a[n,-1]   = (eta_a[n,-2]-eta_a[n,-1])/dx
    elif slopeMethod == 'maxmod':
        S_a[n,1:-1] = -maxmod((eta_a[n,1:-1]-eta_a[n,:-2])/dx,(eta_a[n,2:]-eta_a[n,1:-1])/dx)
        S_a[n,0]    = (eta_a[n,0]-eta_a[n,1])/dx
        S_a[n,-1]   = (eta_a[n,-2]-eta_a[n,-1])/dx
    elif slopeMethod == 'MCslope':
        S_a[n,1:-1] = -MCslope((eta_a[n,1:-1]-eta_a[n,:-2])/dx,(eta_a[n,2:]-eta_a[n,1:-1])/dx)
        S_a[n,0]    = (eta_a[n,0]-eta_a[n,1])/dx
        S_a[n,-1]   = (eta_a[n,-2]-eta_a[n,-1])/dx
    elif slopeMethod == 'chiara':
        S_a_temp[1:-1] = (eta_a[n,:-1]-eta_a[n,1:])/dx
        eta_a_i[1:-1]  = eta_a[n,:-1]-S_a_temp[1:-1]*dx/2
        S_a[n,1:-1]    = (eta_a_i[1:-2]-eta_a_i[2:-1])/dx
        S_a[n,0] = S_a[n,1]
        S_a[n,-1] = S_a[n,-2]
        if np.any(S_a_temp<0):
            print('errore')

    # Downstream BC: H(t)=H0
    D_a[-1] = H0-eta_a[n,-1]+S_a[n,-1]*dx/2

    # Wse profile and Shields parameter update
    D_a     = buildProfile(RF, D_a[-1], Q0, W_a, S_a[n,:], d50, dx, g, ks0, C0, eps_c)
    # D_a     = buildProfile_rk4(RF, D_a[-1], Q0, W_a, S_a[n,:], d50, dx, g, ks0, C0, eps_c)
    Theta_a = shieldsUpdate(RF, Q0, W_a, D_a, d50, g, delta, ks0, C0, eps_c)
    
    # Solid discharge computation + exner
    Qs_a         = W_a*np.sqrt(g*delta*d50**3)*phis(Theta_a, TF, D0, d50)[0]*st
    Qs_a[0]      = Qs0*st
    eta_a[n+1,:] = eta_a[n,:]+dt*(Qs_a[:-1]-Qs_a[1:])/((1-p)*W_a*dx)

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

# Plot evolution of eta_a
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
    
    for i in range(numPlots-1):
        plotTimeIndex = int(plotTimeIndexes[i])
        myPlot(nFig+1, xc, S_a[plotTimeIndex,:], ('t=%3.1f Tf' % (plotTimeIndex*dt/Tf)), color=bed_colors[i])

plt.show()