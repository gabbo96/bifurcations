from functions import *

setNum = 1

TF = 'P90'
arl = True

# Measured data
g      = 9.81
delta  = 1.63
d50    = 6.3e-4
W      = 0.36
rW     = 0.24/0.36
S      = np.array([0.0031,0.0027,0.0026,0.0031]) # measured longitudinal bed slope along the upstream channel
Q      = np.array([1.80E-03,2.10E-03,2.50E-03,2.90E-03]) # measured water discharge
Qs     = np.array([1.06E-07,1.66E-07,2.68E-07,4.20e-07]) # solid discharge values estimated from the rpm of the engine of the sediment feeder
deltaQ = np.array([0.5503876,0.28205128,0.21212121,0.11111111]) # measured value of discharge asymmetry at equilibrium
eps_c  = 1.1

# Empty arrays of unknowns
D     = np.zeros((len(Qs)))
theta = np.zeros((len(Qs)))
alpha = np.zeros((len(Qs)))

# Compute water depth or shields according to chosen set
for i in range(len(Q)):
    if setNum == 1:
        #! Compute D using stage-discharge relationship and logarithmic formula for C
        D[i]   = uniFlowD(1,Q[i],W,S[i],d50,g,0,eps_c,0.02,arl=arl)
    elif setNum == 2:
        #! Compute theta by reversing the transport formula using the known values of Qs
        phi_target = Qs[i]/(W*np.sqrt(g*delta*d50**3))
        theta[i]   = newton(0.05,labNewton,[TF,1,d50,phi_target])

# Compute beta, theta, ds and C
if setNum == 1:
    if arl:
        theta = S*D/(delta*d50)
    else:
        Rh    = (W*D)/(W+2*D)
        theta = S*Rh/(delta*d50)
elif setNum == 2: #? we always assume a wide channel
    D = theta*delta*d50/S
        
C = 6+2.5*np.log(D/(eps_c*d50))
beta  = W/2/D
ds    = d50/D

# compute resonant aspect ratio
phi0,phiD,phiT = phis(theta,TF,D,d50)
betaR = betaR_MR(theta,0.5,phiD,phiT,C)

# compute best-fit value of alpha that matches deltaQ_BRT with measured deltaQ
for i in range(np.size(Q)):
    #alpha[i] = newton(6,falpha_bestfit,(deltaQ[i],[1.1,0.9,0.9,0.9],0,0,TF,beta[i],theta[i],ds[i],rW,7.33/(d50/ds[i]),0.5,g,delta,d50,C[i],eps_c))
    alpha[i] = opt.fsolve(falpha_bestfit,6,(deltaQ[i],[1.1,0.9,0.9,0.9],0,0,TF,beta[i],theta[i],ds[i],rW,7.33/(d50/ds[i]),0.5,g,delta,d50,C[i],eps_c))
    print(deltaQ_BRT_2(alpha[i],[1.1,0.9,0.9,0.9],0,0,TF,beta[i],theta[i],ds[i],rW,7.33/(d50/ds[i]),0.5,g,delta,d50,C[i],eps_c))
# Print outputs
print("beta0s  = ",beta)
print("theta0s = ",theta)
print("ds0s    = ",d50/D)
print("C0s     = ",C)
print("Ls      = ",7.33/D)
print("betaR   = ",betaR)
print("(beta-betaR)/betaR = ", (beta-betaR)/betaR)
print("D       = ",D)
print("alpha   = ",alpha)


# C0s = ks/9.81**0.5*D**(1/6)
# nRun   = -1
"""
for S,Q,Q in zip(S,Q,Qs):
    nRun           += 1
    phi_target     = Q/(W*np.sqrt(g*delta*d50**3))
    theta[nRun]    = newton(0.05,labNewton,[TF,1,d50,phi_target])
    D[nRun] = (Q/(W*ks*S**0.5))**(3/5)
    #D[nRun]      = theta[nRun]*delta*d50/S
    D0s2[nRun]     = (Q**2/(delta*W**2*ks**2*d50*theta[nRun]))**(3/7)
    beta[nRun]    = W/2/D[nRun]
    phi0,phiD,phiT = phis_scalar(theta[nRun],TF,D[nRun],d50)
"""

