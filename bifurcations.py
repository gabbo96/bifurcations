import time

import numpy as np

from functions import *

# -------------------------
# INPUTS AND MODEL SETTINGS
# -------------------------

# I/O settings
numMaxPlots = 75  # maximum number of branches' bed evolution plots during time evolution
iterationPlotStep = 2500

# Model settings
dsBC = 0
tend = 1500
eq_it_max = int(1e4)

# Hydraulic parameters
RF = 'ks'  # flow resistance formula. Available options: 'ks' (Gauckler&Strickler), 'C' (Chézy)
ks0 = 0  # if =0, it is computed using Gauckler&Strickler formula; otherwise it's considered a constant
C0 = 0  # if =0, it is computed using a logarithmic formula; otherwise it's considered a constant
eps_c = 2.5  # Chézy logarithmic formula coefficient
TF = 'P90'  # sediment transport formula. Available options: 'P78' (Parker, 1978), 'MPM' (Meyer-Peter&Mueller, 1948),
# 'P90' (Parker, 1990), 'EH' (Engelund&Hansen)
Ls = 3000  # =L/D0, L=branches' dimensional length
beta0 = 25
theta0 = 0.1
ds0 = 0.01  # =d50/D0
rW = 0.5  # =Wb/W_a, where Wb=Wc and W_a=upstream channel width
d50 = 0.01  # median sediment diameter [m]
p = 0.6  # bed porosity
r = 0.5  # Ikeda parameter
inStep = -5e-4  # imposed initial inlet step = (eta_bn - eta_cn) / D0

# Numerical parameters
dt = 100  # timestep [s] (complete model only)
nc = 30  # branches' number of cells
maxIter = int(1e8)  # max number of iterations during time evolution
tol = 1e-6  # Newton method tolerance

# Physical constants
delta = 1.65
g = 9.81

# ----------------------------------
# IC & BC DEFINITION AND MODEL SETUP
# ----------------------------------

# Main channel IC
S0 = theta0 * delta * ds0
D0 = d50 / ds0
W_a = beta0 * 2 * D0
phi00, phiD0, phiT0 = phis_scalar(theta0, TF, D0, d50)
Q0 = uniFlowQ(RF, W_a, S0, D0, d50, g, ks0, C0, eps_c)
Qs0 = W_a * np.sqrt(g * delta * d50 ** 3) * phi00
Fr0 = Q0 / (W_a * D0 * np.sqrt(g * D0))

# βR and alpha computation
betaR = betaR_MR(RF, theta0, ds0, r, phiD0, phiT0, eps_c)
betaC = betaC_MR(RF, theta0, ds0, 1, r, phiD0, phiT0, eps_c)
alpha_MR = betaR / betaC

# Exner time computation
Tf = (1 - p) * W_a * D0 / (Qs0 / W_a)

# Arrays initialization
eta_ab = np.zeros(nc)
eta_ac = np.zeros(nc)
eta_b = np.zeros(nc)
eta_c = np.zeros(nc)
D_ab = np.zeros(nc)
D_ac = np.zeros(nc)
Q_ab = np.zeros(nc)
Q_ac = np.zeros(nc)
D_b = np.zeros(nc + 1)
D_c = np.zeros(nc + 1)
S_a = np.zeros(nc + 1)
S_b = np.zeros(nc + 1)
S_c = np.zeros(nc + 1)
Theta_ab = np.zeros(nc)
Theta_ac = np.zeros(nc)
Theta_b = np.zeros(nc + 1)
Theta_c = np.zeros(nc + 1)
Phi_ab = np.zeros(nc)
Phi_ac = np.zeros(nc)
Phi_b = np.zeros(nc + 1)
Phi_c = np.zeros(nc + 1)
Qs_ab = np.zeros(nc)
Qs_ac = np.zeros(nc)
Qs_y = np.zeros(nc)
Qs_b = np.zeros(nc + 1)
Qs_c = np.zeros(nc + 1)

# Branches' IC
dx = Ls * D0 / nc
eta_ab[0] = 10000
for i in range(1, nc):
    eta_ab[i] = eta_ab[i - 1] - S0 * dx
eta_ac = eta_ab

eta_b[0] = eta_ab[-1] - S0 * dx
for i in range(1, nc):
    eta_b[i] = eta_b[i - 1] - S0 * dx
eta_c = eta_b

eta_ab_ic = eta_ab[:]
eta_ac_ic = eta_ac[:]
eta_b_ic = eta_b[:]
eta_c_ic = eta_c[:]
W_b = rW * W_a
W_c = W_b
Q_b = Q0 / 2
Q_c = Q0 / 2
Q_ab += Q0 / 2
Q_ac += Q0 / 2

# Time-tracking lists definition
deltaQ: list = [0]

# Downstream IC
H0 = eta_b[-1] + D0
Hd_b: list = [H0]
Hd_c: list = [H0]

# Equilibrium deltaQ computation through BRT method
BRT_out = deltaQ_BRT([1.1, 0.9, 0.9, 1.1], dsBC, RF, TF, theta0, Q0, Qs0, W_a, Ls * D0, D0, S0, W_b, W_c, d50,
                     alpha_MR, r, g, delta, tol, ks0, C0, eps_c)
deltaQ_eq_BRT = BRT_out[0]
inStep_eq_BRT = (BRT_out[7] - BRT_out[8]) / D0

# Print model parameters
print('INPUT PARAMETERS:\nFlow resistance formula = %s\nSolid transport formula = %s\nnc = %d\ndx = %4.2f m'
      '\ndt = %d s\nt_end = %2.1f Tf\nL* = %d\nβ0 = %4.2f\nθ0 = %4.3f\nds0 = %2.1e\nr = %2.1f'
      '\nd50 = %3.2e m\n'
      '\nREDOLFI ET AL. [2019] RESONANT ASPECT RATIO COMPUTATION\nβR = %4.2f\n(β0-βR)/βR = %4.2f\nα_MR = %2.1f\n'
      '\nMAIN CHANNEL IC:\nW_a = %4.2f m\nS0 = %3.2e\nD0 = %3.2f m\nH0 = %2.1f m\nFr0 = %2.1f\nL = %4.1f m\n'
      'Q0 = %3.2e m^3 s^-1\nQs0 = %3.2e m^3 s^-1\n\nExner time: Tf = %3.2f h\n'
      % (RF, TF, nc, dx, dt, tend, Ls, beta0, theta0, ds0, r, d50, betaR, (beta0-betaR) / betaR,
         alpha_MR, W_a, S0, D0, H0, Fr0, Ls * D0, Q0, Qs0, Tf / 3600))
print("BRT equilibrium solution:\ndeltaQ = %5.4f\nθ_b = %4.3f, θ_c ="
      " %4.3f\nFr_b = %3.2f, Fr_c = %3.2f\nS = %2.1e\n"
      % (BRT_out[:6]))

# --------------
# TIME EVOLUTION
# --------------

# Bed elevation profiles' plotting setup
x_plot = np.linspace(-dx / 2, Ls * D0 - dx / 2, num=nc + 1)
crange = np.linspace(0, 1, numMaxPlots)
bed_colors = plt.cm.viridis(crange)
cindex = 0

# Time-control variables definition
t: list = [0]  # time array
total_time = time.time()  # total simulation time
eq_it = 0  # progressive number of iterations in which deltaQ is constant. When eq_it=eq_it_max, iterations stop
eqIndex = 0

# Node perturbation
eta_ab[-1] += inStep * D0 / 2
eta_ac[-1] -= inStep * D0 / 2

for n in range(0, maxIter):
    # Finds the index of the upstream end of the steady bar
    barIndex = np.where(eta_ab != eta_ac)[0][0]

    # Channel cells slope update
    eta_a_mean = (eta_ab + eta_ac) / 2
    S_a[1:] = (eta_a_mean[:-1] - eta_a_mean[1:]) / dx
    S_b[0] = (eta_ab[-1] - eta_b[0]) / dx
    S_c[0] = (eta_ac[-1] - eta_c[0]) / dx
    S_b[1:-1] = (eta_b[:-1] - eta_b[1:]) / dx
    S_c[1:-1] = (eta_c[:-1] - eta_c[1:]) / dx

    # Downstream BC
    if dsBC == 0:
        D_b[-1] = Hd_b[-1] - eta_b[-1]
        D_c[-1] = Hd_c[-1] - eta_c[-1]
    elif dsBC == 1:
        D_b[-1] = uniFlowD(RF, Q_b, W_b, S_b[-2], d50, g, ks0, C0, eps_c, D0)
        D_c[-1] = uniFlowD(RF, Q_c, W_c, S_c[-2], d50, g, ks0, C0, eps_c, D0)

    # Channel cells' D computation through water profiles
    D_b = buildProfile(RF, D_b[-1], Q_b, W_b, S_b, d50, dx, g, ks0, C0, eps_c)
    D_c = buildProfile(RF, D_c[-1], Q_c, W_c, S_c, d50, dx, g, ks0, C0, eps_c)

    # If the wse is different at the inlets, re-compute discharge partitioning
    if abs(D_b[0] - D_c[0] + inStep * D0) / ((D_b[0] + D_c[0]) / 2) > 1e-10:
        Q_b = opt.fsolve(fSys, Q_b, (RF, D_b[-1], D_c[-1], D0, inStep, Q0, W_b, W_c, S_b, S_c, d50, dx,
                                     g, ks0, C0, eps_c), maxfev=1000, xtol=tol)[0]
        Q_c = Q0 - Q_b
        D_b = buildProfile(RF, D_b[-1], Q_b, W_b, S_b, d50, dx, g, ks0, C0, eps_c)
        D_c = buildProfile(RF, D_c[-1], Q_c, W_c, S_c, d50, dx, g, ks0, C0, eps_c)

    # Compute discharge at each bar cell: each half channel varies its discharge linearly from Q0/2 to the corresponding
    # branch discharge linearly in space along the bar
    Q_ab[barIndex:] = np.linspace(Q0 / 2, Q_b, num=(nc - barIndex + 1))[:-1]
    Q_ac[barIndex:] = np.linspace(Q0 / 2, Q_c, num=(nc - barIndex + 1))[:-1]

    # Water depth update in channel a
    D_ab[-1] = D_b[0] + (eta_b[0] - eta_ab[-2]) / 2
    D_ac[-1] = D_c[0] + (eta_c[0] - eta_ac[-2]) / 2
    for i in range(nc - 1, barIndex - 1, -1):
        D_ab[i - 1] = D_ab[i] + (eta_ab[i] - eta_ab[i - 2]) / 2
        D_ac[i - 1] = D_ac[i] + (eta_ac[i] - eta_ac[i - 2]) / 2
    D_ab[:barIndex - 1] = buildProfile(RF, (D_ab[barIndex - 1] + D_ac[barIndex - 1]) / 2, Q0, W_a, S_a, d50, dx, g, ks0,
                                       C0, eps_c)
    D_ac[:barIndex - 1] = D_ab[:barIndex - 1]

    # Shields parameter update
    Theta_ab = shieldsUpdate(RF, Q_ab, W_a / 2, D_ab, d50, g, delta, ks0, C0, eps_c)
    Theta_ac = shieldsUpdate(RF, Q_ac, W_a / 2, D_ac, d50, g, delta, ks0, C0, eps_c)
    Theta_b = shieldsUpdate(RF, Q_b, W_b, D_b, d50, g, delta, ks0, C0, eps_c)
    Theta_c = shieldsUpdate(RF, Q_c, W_c, D_c, d50, g, delta, ks0, C0, eps_c)

    # Solid discharge computation
    Qs_ab = W_a / 2 * np.sqrt(g * delta * d50 ** 3) * phis(Theta_ab, TF, D0, d50)[0]
    Qs_ac = W_a / 2 * np.sqrt(g * delta * d50 ** 3) * phis(Theta_ac, TF, D0, d50)[0]
    Qs_b = W_b * np.sqrt(g * delta * d50 ** 3) * phis(Theta_b, TF, D0, d50)[0]
    Qs_c = W_c * np.sqrt(g * delta * d50 ** 3) * phis(Theta_c, TF, D0, d50)[0]

    # Apply the second upstream BC
    Qs_ab[0] = Qs0
    Qs_ac[0] = Qs0

    # Node cells' transverse solid discharges and bed elevation update
    Q_y = Q_b - Q0 / 2  # considered constant along the bar

    # End time condition for the simulation's end
    if t[n] >= (tend * Tf):
        print('\nEnd time reached\n')
        eqIndex = n
        break

    # Print elapsed time
    if n % 2500 == 0:
        print("Elapsed time = %4.1f Tf, deltaQ = %5.4f" % (t[n] / Tf, deltaQ[n]))

    # Bed profile plot
    if n % iterationPlotStep == 0 and cindex < numMaxPlots:
        myPlot(1, x_plot[1:], eta_ab - eta_ab_ic, None, color=bed_colors[cindex])
        myPlot(1, x_plot[1:], eta_ac - eta_ac_ic, None, color=bed_colors[cindex])
        myPlot(2, x_plot[1:], eta_b - eta_b_ic, None, color=bed_colors[cindex])
        myPlot(3, x_plot[1:], eta_c - eta_c_ic, None, color=bed_colors[cindex])
        cindex = cindex + 1
