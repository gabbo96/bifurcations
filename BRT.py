import numpy as np
from scipy import optimize as opt
from matplotlib import pyplot as plt
from functions import *

# Hydraulic model parameters
#dsBCs   = [0,2,0,2]  #dowsntream boundary condition switch. 0: H=H0; 1: confluence (not implemented); 2: critical depth
dsBCs = [2]
RF     = 0
TF     = 'P90'
theta0 = 0.07
ds0    = 0.01 # d50/D0
d50    = 0.01 # median sediment diameter [m]
L      = 2000 # =L*/D0, L*=dimensional length of the branches. Used only if dsBC==2.
alpha  = 3
#rWs    = [0.5,0.67,0.67,0.5] # width ratio rW=W_b/W0=W_c/W0
rWs = [0.5]
C0     = 14.50325831
eps_c  = 1.1
r      = 0.5

# Newton method parameters
maxIter = 10000
tol     = 1e-6

# Physical constants
delta = 1.63
g     = 9.81

for rW,dsBC in zip(rWs,dsBCs):
    # Prints model parameters
    print("\nInput parameters:\nTransport formula = %s\ntheta0 = %3.2f\nds0 = %3.2f\nd50 = %4.3f m\n"
          % (TF, theta0, ds0, d50))

    # Main channel IC
    D0  = d50/ds0
    S0  = theta0*delta*d50/D0
    qs0 = np.sqrt(g*delta*d50**3)*phis(np.array([theta0]),TF,D0,d50)[0]
    print("Main channel IC:\nD0 = %3.2f m\nS0 = %2.1e\nqs0 = %3.2e m^2s^-1\n" % (D0, S0, qs0))

    # Generation of deltaQ(beta)
    nb       = 2000 # number of generated cases
    beta_max = 20
    beta0    = np.linspace(5, beta_max, nb)
    deltaQ   = np.zeros(nb)
    inStep   = np.zeros(nb)

    # Initial guess
    Db = 1.4*D0
    Dc = 0.6*D0
    Sb  = 0.8*S0
    Sc  = 1.2*S0
    X  = np.array([[Db],[Dc],[Sb],[Sc]])

    # Iterations
    eps        = 1e-6
    beta_R     = -1
    betaRcheck = False
    for i in range(nb-1,-1,-1):
        # Derived flow conditions for the main channel
        W   = 2*beta0[i]*D0
        Q0  = uniFlowQ(RF,W,S0,D0,d50,g,C0,eps_c)
        Qs0 = W*qs0

        # BRT equilibrium solution
        BRT_out = deltaQ_BRT_2(alpha,(X[0]/D0,X[1]/D0,X[2]/S0,X[3]/S0),dsBC,RF,TF,beta0[i],theta0,ds0,rW,L,r,g,delta,d50,C0,eps_c)
        deltaQ  [i] = BRT_out [0]
        theta_b,theta_c = BRT_out [2:4]
        Sb,Sc  = BRT_out [4:]
        Db     = theta_b *delta*d50/Sb
        Dc     = theta_c *delta*d50/Sc
        if deltaQ[i] > 0:
            X = np.array([[Db],[Dc],[Sb],[Sc]])

        # Finds resonant aspect ratio
        if i < nb-1 and not betaRcheck:
            if deltaQ[i] < eps < deltaQ[i + 1]:
                beta_R = beta0[i]
                betaRcheck = True
        
        #if i % int(nb/20) == 0:
        #    print("beta = %3.1f, deltaQ = %5.4f" % (beta0[i],deltaQ[i]))

    print("\nNumerical beta_R = %4.2f" % beta_R)

    # Compute resonant aspect ratio according to the simplified two-order solution of Camporeale et al. (2007)
    phiD, phiT  = phis(np.array([theta0]),TF,D0,d50)[1:]
    betaR_simpl = betaR_MR(theta0,r,phiD,phiT,C0)
    print("beta_R according to Camporeale et al. (2007) = %4.2f" % betaR_simpl)

    myPlot(1, beta0, deltaQ, ("dsBC = %d, rW = %3.2f" % (dsBC,rW)), title='Discharge asymmetry vs aspect ratio', xlabel=r'$\beta_0 [-]$', ylabel=r'$\Delta Q [-]$')

plt.show()
