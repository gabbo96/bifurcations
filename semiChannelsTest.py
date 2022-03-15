import time
import numpy as np
from functions import *

# -------------------------
# INPUTS AND MODEL SETTINGS
# -------------------------

# I/O settings
numMaxPlots = 10000  # maximum number of branches' bed evolution plots during time evolution
iterationPlotStep = 10

# Model settings
dsBC = 0
tend = 2
eq_it_max = int(1e4)

# Hydraulic parameters
RF     = 'ks' # flow resistance formula. Available options: 'ks' (Gauckler&Strickler), 'C' (Chézy)
ks0    = 0 # if =0, it is computed using Gauckler&Strickler formula; otherwise it's considered a constant
C0     = 0 # if =0, it is computed using a logarithmic formula; otherwise it's considered a constant
eps_c  = 2.5 # Chézy logarithmic formula coefficient
TF     = 'P90' # sediment transport formula. Available options: 'P78' (Parker, 1978), 'MPM' (Meyer-Peter&Mueller, 1948),
# 'P90' (Parker, 1990), 'EH' (Engelund&Hansen)
Ls     = 3000 # =L/D0, L=branches' dimensional length
beta0  = 25
theta0 = 0.08
ds0    = 0.01 # =d50/D0
rW     = 0.5 # =Wb/W_a, where Wb=Wc and W_a=upstream channel width
d50    = 0.01 # median sediment diameter [m]
p      = 0.6 # bed porosity
r      = 0.5 # Ikeda parameter
inStep = -5e-4 # imposed initial inlet step = (eta_bn - eta_cn)/D0

# Numerical parameters
dt      = 100 # timestep [s] (complete model only)
nc      = 30 # branches' number of cells
maxIter = int(1e5) # max number of iterations during time evolution
tol     = 1e-6 # Newton method tolerance

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

# Exner time computation
Tf = (1 - p)*W_a*D0/(Qs0/W_a)

# Arrays initialization
eta_ab    = np.zeros(nc)
eta_ac    = np.zeros(nc)
eta_ab_ic = np.zeros(nc)
eta_ac_ic = np.zeros(nc)
D_ab      = np.zeros(nc+1)
D_ac      = np.zeros(nc+1)
Q_ab      = np.zeros(nc+1)
Q_ac      = np.zeros(nc+1)
Q_y       = np.zeros(nc)
S_ab      = np.zeros(nc+1)
S_ac      = np.zeros(nc+1)
Theta_ab  = np.zeros(nc+1)
Theta_ac  = np.zeros(nc+1)
Qs_ab     = np.zeros(nc+1)
Qs_ac     = np.zeros(nc+1)
Qs_y      = np.zeros(nc)

# IC
t        :list = [0]
dx        = Ls*D0/nc
eta_ab   [:] = np.linspace(0, -S0*dx*nc, num=len(eta_ab))
eta_ac   [:] = eta_ab[:]
eta_ab_ic[:] = eta_ab[:]
eta_ac_ic[:] = eta_ac[:]
W_ab      = W_a/2
W_ac      = W_a/2
Q_ab      += Q0/2
Q_ac      += Q0/2
x = np.array([1,1,0.5,0.5,0.001])  # initial guess for first iteration of the system to be solved for channel a
eta_ab[-1] += 2*d50

# Downstream BC
H0 = eta_ab[-1] + D0
Hd_ab: list = [H0]
Hd_ac: list = [H0]

# Plot semichannels bed elevation values
eta_a = np.vstack([eta_ac-eta_ac_ic, eta_ab-eta_ab_ic])
plt.imshow(eta_a, cmap='RdYlBu', aspect='auto')
plt.xticks(range(0,nc,5), range(0,nc,5))
plt.yticks(range(2), ['ac', 'ab'])
plt.show()

for n in range(0, maxIter):
    # Channel cells slope update
    S_ab[1:-1] = (eta_ab[:-1]-eta_ab[1:])/dx
    S_ac[1:-1] = (eta_ac[:-1]-eta_ac[1:])/dx
    S_ab[-1] = S_ab[-2]
    S_ac[-1] = S_ac[-2]

    # Downstream BC
    if dsBC == 0:
        D_ab[-1] = Hd_ab[-1] - eta_ab[-1]
        D_ac[-1] = Hd_ac[-1] - eta_ac[-1]

    # Solve the governing system to compute the unknowns D_ab[i], D_ac[i], Q_ab[i], Q_ac[i] and Qy[i]
    # along the portion of semichannels ab and ac influenced by the bar
    for i in range(nc, 1, -1):
        x = opt.fsolve(fSysSC, x, (D0, Q0, D_ab[i], D_ac[i], Q_ab[i], Q_ac[i], (S_ab[i]+S_ab[i-1])/2, 
            (S_ac[i]+S_ac[i-1])/2, W_ab, W_ac, (eta_ab[i-1]+eta_ab[i-2])/2, (eta_ac[i-1]+eta_ac[i-2])/2,
             g, d50, dx, ks0, C0, RF), xtol=tol)[:5]
        D_ab[i-1], D_ac[i-1], Q_ab[i-1], Q_ac[i-1], Q_y[i-1] = (x[0]*D0, x[1]*D0, x[2]*Q0, x[3]*Q0, x[4]*Q0)
        if eta_ab[i-1] == eta_ac[i-1] and D_ab[i-1] == D_ac[i-1] and Q_ab[i-1] == Q_ac[i-1]:
            break 
    
    # Water depth update in channel a, upstream the bar    
    D_ab[:i-1] = buildProfile(RF, (D_ab[i-1]+D_ac[i-1])/2, Q0, W_a, S_ab[:i-1], d50, dx, g, ks0, C0, eps_c)
    D_ac[:i-1] = D_ab[:i-1]
    Q_ab[:i-1] = Q0/2
    Q_ac[:i-1] = Q0/2

    # Shields parameter update
    Theta_ab = shieldsUpdate(RF, Q_ab, W_ab, D_ab, d50, g, delta, ks0, C0, eps_c)
    Theta_ac = shieldsUpdate(RF, Q_ac, W_ac, D_ac, d50, g, delta, ks0, C0, eps_c)

    # Solid discharge computation
    Qs_ab = W_ab*np.sqrt(g*delta*d50**3)*phis(Theta_ab, TF, D0, d50)[0]
    Qs_ac = W_ac*np.sqrt(g*delta*d50**3)*phis(Theta_ac, TF, D0, d50)[0]

    # Impose the second upstream BC
    Qs_ab[0] = Qs0/2
    Qs_ac[0] = Qs0/2

    # Compute transverse solid discharge along channel a
    Theta_a_avg = shieldsUpdate(RF, Q0, W_a, (D_ab+D_ac)/2, d50, g, delta, ks0, C0, eps_c)
    Qs_y[:] = (Qs_ab[:-1]+Qs_ac[:-1])*(Q_y/Q0-2*r*dx/(W_a*Theta_a_avg[:-1]**0.5)*(eta_ab-eta_ac)/W_a)

    # Apply Exner equation to update node cells elevation
    eta_ab += dt*(Qs_ab[:-1]-Qs_ab[1:]+Qs_y)/((1-p)*W_ab*dx)
    eta_ac += dt*(Qs_ac[:-1]-Qs_ac[1:]-Qs_y)/((1-p)*W_ac*dx)

    # Time update + end-time condition for the simulation's end
    t.append(t[-1]+dt)
    if t[-1] >= (tend * Tf):
        print('\nEnd time reached\n')
        break

    # Print elapsed time
    if n % 50 == 0:
        print("Elapsed time = %4.1f Tf, i = %d" % (t[n] / Tf, i))

    
# Plot semichannels bed elevation values
eta_a = np.vstack([eta_ac-eta_ac_ic, eta_ab-eta_ab_ic])
plt.imshow(eta_a, cmap='RdYlBu', aspect='auto')
plt.xticks(range(0,nc,5), range(0,nc,5))
plt.yticks(range(2), ['ac', 'ab'])
plt.show()