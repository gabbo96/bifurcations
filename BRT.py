import numpy as np
from scipy import optimize as opt
from matplotlib import pyplot as plt
from functions import *

# Hydraulic model parameters
RF     = 1
TF     = 'EH'
theta0 = 0.8
ds0    = 0.005 # d50/D0
d50    = 0.001 # median sediment diameter [m]
alpha  = 3
C0     = 10
r      = 0.5

# Newton method parameters
maxIter = 10000
tol     = 1e-6

# Physical constants
delta = 1.65
g     = 9.81

# Prints model parameters
print("\nInput parameters:\nTransport formula = %s\ntheta0 = %4.3f\nds0 = %3.2f\nd50 = %4.3f m\n"
      % (TF, theta0, ds0, d50))

# Main channel IC
D0 = d50 / ds0
S0 = theta0 * delta * d50 / D0
qs0 = np.sqrt(g * delta * d50 ** 3) * phis(np.array([theta0]), TF, D0, d50)[0]
print("Main channel IC:\nD0 = %3.2f\nS0 = %2.1e\nqs0 = %3.2e m^2s^-1\n" % (D0, S0, qs0))

# Generation of deltaQ(beta)
nb       = 10000 # number of generated cases
beta_max = 30
beta0    = np.linspace(0.5, beta_max, nb)
rq       = np.zeros(nb)

# Initial guess
Db = 1.8 * D0
Dc = 0.6 * D0
S  = 0.5 * S0
X  = np.array([[Db], [Dc], [S], [S]])

# Iterations
beta_temp  = 1
eps        = 1e-15
beta_R     = -1
betaRcheck = False
for i in range(0, nb, 1):
    # Derived flow conditions for the main channel
    B   = 2*beta0[i]*D0
    Q0  = uniFlowQ(RF,B,S0,D0,d50,g,C0,2.5)
    Qs0 = B*qs0

    # BRT equilibrium solution
    BRT_out = deltaQ_BRT((X[0]/D0,X[1]/D0,X[2]/S0,X[3]/S0),0,RF,TF,theta0,Q0,Qs0,B,0,D0,S0,B/2,B/2,d50,alpha,r,g,delta,tol,C0,2.5)
    rq[i]   = BRT_out[0]
    X       = np.array([[BRT_out[-2]], [BRT_out[-1]], [BRT_out[-3]], [BRT_out[-3]]])

    # Prints model output
    if beta0[i]/(beta_temp + eps) - 1 < eps and beta_temp != 0:
        print('beta = ' + str(int(round(beta0[i]))) + ' deltaQ = %5.4f' % rq[i])
        beta_temp = beta_temp + 1

    # Finds resonant aspect ratio
    if i < nb - 1 and not betaRcheck:
        if rq[i] < eps < rq[i + 1]:
            beta_R = beta0[i]
            betaRcheck = True

print("\nbeta_R = %4.2f" % beta_R)

myPlot(1, beta0, rq, 'deltaQ', title='Discharge asymmetry vs beta0', xlabel='beta0 [-]', ylabel='deltaQ [-]')
plt.show()
