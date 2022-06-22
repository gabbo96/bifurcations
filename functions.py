from cmath import sqrt
from matplotlib import pyplot as plt
from pyrsistent import b
from scipy import optimize as opt
import numpy as np
import math


def uniFlowQ(rf, w, s, d, d50, g, c, eps_c):
    # Uniform flow discharge computation
    c = (6+2.5*np.log(d/(eps_c*d50)))*rf+c*(1-rf)
    return w*c*(g*s)**0.5*d**1.5


def chezySys(d, q, w, s, d50, g, eps_c):
    # function used to solve the system of 2 equations defined by the Chézy uniform flow formula and the logarithmic
    # expression for the roughness coefficient
    return q/(w*(6+2.5*np.log(d/(eps_c*d50)))*np.sqrt(g*s)*d**1.5)-1


def uniFlowD(rf, q, w, s, d50, g, c, eps_c, d0):
    # Uniform/gradually varying flow water depth computation
    if rf == 1:
        return opt.fsolve(chezySys, np.array([d0]), (q, w, s, d50, g, eps_c))[0]
    else:
        return (q/(w*c*np.sqrt(g*s)))**(2/3)

def uniFlowS(rf, q, w, d, d50, g, c, eps_c):
    # Uniform/gradually varying flow bed/energy slope computation (wide rectangular cross-section hypothesis is made)
    c = (6+2.5*np.log(d/(eps_c*d50)))*rf+c*(1-rf)
    return (q/(w*c*g**0.5*d**1.5))**2
    

def fSys(q_b, rf, ddb, ddc, d0, inStep, qu, w_b, w_c, s_b, s_c, d50, dx, g, c, eps_c):
    # fSys = 0 is solved to solve the flow partition problem at the node for each time step
    q_c = qu-q_b
    d_b = buildProfile(rf,ddb,q_b,w_b,s_b,d50,dx,g,c,eps_c)
    d_c = buildProfile(rf,ddc,q_c,w_c,s_c,d50,dx,g,c,eps_c)
    return (d_b[0]-d_c[0]+inStep*d0)/((d_b[0]+d_c[0])/2)

def fSys_rk4(q_b, rf, ddb, ddc, d0, inStep, qu, w_b, w_c, s_b, s_c, d50, dx, g, c, eps_c):
    # fSys = 0 is solved to solve the flow partition problem at the node for each time step
    q_c = qu-q_b
    d_b = buildProfile_rk4(rf,ddb,q_b,w_b,s_b,d50,dx,g,c,eps_c)
    d_c = buildProfile_rk4(rf,ddc,q_c,w_c,s_c,d50,dx,g,c,eps_c)
    return (d_b[0]-d_c[0]+inStep*d0)/((d_b[0]+d_c[0])/2)

def phis_scalar(theta, tf, d0, d50):
    phi0 = 0
    phiD = 0
    phiT = 0
    if tf == 'MPM':
        theta_cr = 0.047
        if theta > theta_cr:
            phi0 = 8 * (theta - theta_cr) ** 1.5
            phiT = 1.5 * theta / (theta - theta_cr)
    elif tf == 'EH':
        c = 6 + 2.5 * np.log(d0 / (2.5 * d50))
        cD = 2.5 / c
        phi0 = 0.05 * c ** 2 * theta ** 2.5
        phiD = 2 * cD
        phiT = 2.5
    elif tf == 'P90':  # Parker (1990)
        A = 0.0386
        B = 0.853
        C = 5474
        D = 0.00218
        x = theta / A
        if x > 1.59:
            phi0 = C * D * theta ** 1.5 * (1 - B / x) ** 4.5
            Phi_der = 1.5 * theta ** 0.5 * (1 - B / x) ** 4.5 * C * D + 4.5 * A * B * (1. - B / x) ** 3.5 * C * D / (
                        theta ** 0.5)
        elif x >= 1:
            phi0 = D * theta ** 1.5 * (np.exp(14.2 * (x - 1) - 9.28 * (x - 1) ** 2))
            Phi_der = 1. / A * phi0 * (14.2 - 9.28 * 2. * (x - 1.)) + 1.5 * phi0 / theta
        else:
            phi0 = D * theta ** 1.5 * x ** 14.2
            Phi_der = 14.2 / A * D * theta ** 1.5 * x ** 13.2 + D * x ** 14.2 * 1.5 * theta ** 0.5
        phiT = theta / phi0 * Phi_der
    elif tf == 'P78':  # Parker (1978)
        theta_cr = 0.03
        phi0 = 11.2 * theta ** 1.5 * (1 - theta_cr / theta) ** 4.5
        phiT = 1.5 + 4.5 * theta_cr / (theta - theta_cr)
    elif tf =='WP':  # Wong&Parker (2006)
        theta_cr = 0.0495
        phi0 = 3.97 * (theta - theta_cr) ** 1.5
        phiT = 1.5 * theta / (theta - theta_cr)
    else:
        print('error: unknown transport formula')
        phi0 = None
    return phi0, phiD, phiT

# Computes phi and phiT given array values of theta and D
def phis(theta, tf, d0, d50):
    phi  = np.zeros(np.size(theta))
    phiD = np.zeros(np.size(theta))
    phiT = np.zeros(np.size(theta))
    if tf == 'MPM':  # Meyer-Peter and Muller (1948)
        theta_cr = 0.047
        nst = theta < theta_cr
        phi[~nst] = 8 * (theta[~nst] - theta_cr) ** 1.5
        phiT[nst] = None
        phiT[~nst] = 1.5 * theta[~nst] / (theta[~nst] - theta_cr)
    elif tf == 'EH':  # Engelund & Hansen
        c = 6+2.5*np.log(d0/(2.5*d50))
        phi = 0.05*c**2*theta**2.5
        phiD = 2*2.5/c
        phiT = 2.5
    elif tf == 'P90':  # Parker (1990)
        a = 0.0386
        b = 0.853
        c = 5474
        d = 0.00218
        x = theta / a
        phi[x >= 1] = d * (theta[x >= 1] ** 1.5) * (np.exp(14.2 * (x[x >= 1] - 1) - 9.28 * (x[x >= 1] - 1) ** 2))
        phiT[x >= 1] = 1.5 + (-18.56 * x[x >= 1] ** 2 + 32.76 * x[x >= 1]) / a
        phi[x > 1.59] = c * d * theta[x > 1.59] ** 1.5 * (1 - b / x[x > 1.59]) ** 4.5
        phiT[x > 1.59] = 1.5 + 4.5 / (((x[x > 1.59] / b) - 1) * a)
        phi[x < 1] = d * theta[x < 1] ** 1.5 * x[x < 1] ** 14.2
        phiT[x < 1] = 1.5 + 14.2 / a
    elif tf == 'P78':  # Parker (1978)
        theta_cr = 0.03
        phi = 11.2 * theta ** 1.5 * (1 - theta_cr / theta) ** 4.5
        phiT = 1.5 + 4.5 * theta_cr / (theta - theta_cr)
    elif tf == 'WP':  # Wong&Parker (2006)
        theta_cr = 0.0495
        nst = theta < theta_cr
        phi[~nst] = 3.97 * (theta[~nst] - theta_cr) ** 1.5
        phiT[nst] = None
        phiT[~nst] = 1.5 * theta[~nst] / (theta[~nst] - theta_cr)
    else:
        print('error: unknown transport formula')
        phi = None
        phiT = None
    return phi, phiD, phiT


def betaR_MR(rf,theta, ds, r, phiD, phiT, eps_c):
    c0 = 6+2.5*np.log(1/(eps_c*ds))
    cD = 2.5/c0*rf
    betaR = np.pi/(2*np.sqrt(2))*c0*np.sqrt(r)/(theta**0.25*np.sqrt(phiD+phiT-(1.5+cD)))
    return betaR


def betaC_MR(rf, theta, ds, alpha, r, phiD, phiT, eps_c):
    c0 = 6 + 2.5 * np.log(1 / (eps_c * ds))
    cD = 2.5/c0*rf
    betaC = r * alpha * 4 / (theta ** 0.5) * 1 / (-(1.5 + cD) + phiT + phiD)
    return betaC


def landau(time_landau, rq_0, k2, omega):
    return (-k2 / omega + (rq_0 ** (-2) + k2 / omega) * np.exp((-2 * omega) * time_landau)) ** (-0.5)


def buildProfile(rf, dd, q, w, s, d50, dx, g, c, eps_c):
    d = np.zeros(len(s)+1)
    d[-1] = dd
    for i in range(len(s), 0, -1):
        j = uniFlowS(rf, q, w, d[i], d50, g, c, eps_c)
        Fr = q / (w * d[i] * np.sqrt(g * d[i]))
        d[i-1] = d[i]-dx*(s[i-1]-j)/(1-Fr**2)
    Fr = q/(w*d*np.sqrt(g*d))
    if np.any(Fr>0.95):
        print("Warning: supercritical flow")
    return d

def profilesF(s, rf, q, w, d, d50, g, c, eps_c):
    j = uniFlowS(rf, q, w, d, d50, g, c, eps_c)
    Fr = q/(w*d*np.sqrt(g*d))
    return (s-j)/(1-Fr**2)


def buildProfile_rk4(rf, dd, q, w, s, d50, dx, g, c, eps_c):
    d = np.zeros(len(s)+1)
    d[-1] = dd
    for i in range(len(s), 0, -1):
        k1 = profilesF(s[i-1], rf, q, w, d[i], d50, g, c, eps_c)
        k2 = profilesF(s[i-1], rf, q, w, d[i]-dx/2*k1, d50, g, c, eps_c)
        k3 = profilesF(s[i-1], rf, q, w, d[i]-dx/2*k2, d50, g, c, eps_c)
        k4 = profilesF(s[i-1], rf, q, w, d[i]-dx*k3, d50, g, c, eps_c)
        d[i-1] = d[i]-dx/6*(k1+2*k2+2*k3+k4)
    return d


def shieldsUpdate(rf, q, w, d, d50, g, delta, c, eps_c):
    j = uniFlowS(rf, q, w, d, d50, g, c, eps_c)
    theta = j * d / (delta * d50)
    return theta


def exponential(time_exp, rq_0, omega):
    return rq_0 * np.exp(omega * time_exp)


def expLab(time_exp, deltaQ_0, deltaQ_eq, omega):
    return (deltaQ_0 - deltaQ_eq) * np.exp(- omega * (time_exp - time_exp[0])) + deltaQ_eq


def fSys_BRT(x, dsBC, rf, tf, theta0, Q0, Qs0, w, l, d0, s0, w_b, w_c, d50, alpha, r, g, delta, c, eps_c):
    # returns the residuals of the system of equation given by liquid and solid mass balance applied to the node and
    # one cell
    q0 = Q0 / w
    qs0 = Qs0 / w
    if dsBC == 0:
        # When the downstream BC is H=cost, the unknowns are the 2 water depths and the slope S=S_b=S_c
        res = np.zeros((3,))
        theta_b = x[2] * s0 * x[0] / (delta * d50)
        theta_c = x[2] * s0 * x[1] / (delta * d50)
        phi_b = phis_scalar(theta_b, tf, d0, d50)[0]
        phi_c = phis_scalar(theta_c, tf, d0, d50)[0]
        qs_b = np.sqrt(g * delta * d50 ** 3) * phi_b
        qs_c = np.sqrt(g * delta * d50 ** 3) * phi_c
        q_b = uniFlowQ(rf, w_b, x[2] * s0, x[0], d50, g, c, eps_c) / w_b
        q_c = uniFlowQ(rf, w_c, x[2] * s0, x[1], d50, g, c, eps_c) / w_c
        qs_y = (qs_b - qs_c) / (2 * alpha) * w_b / w
        q_y = (q_b - q_c) / (2 * alpha) * w_b / w
        res[0] = qs_y / qs0 - q_y / q0 + 2 * r * (x[1] - x[0]) / (theta0 ** 0.5 * (0.5 * (w + w_b + w_c)))
        res[1] = (qs_b + qs_c) / (w / w_b * qs0) - 1
        res[2] = (q_b + q_c) / (w / w_b * q0) - 1
        return res
    if dsBC == 1:
        # When the downstream BC is uniform flow, S_b != S_c. Therefore an unknown is added to the problem. The missing
        # equation is derived from eta_b[-1]=eta_c[-1], which correlates the inlet step at equilibrium with the two
        # slopes, given the branches length
        res = np.zeros((4,))
        theta_b = x[2] * s0 * x[0] / (delta * d50)
        theta_c = x[3] * s0 * x[1] / (delta * d50)
        phi_b = phis(np.array([theta_b]), tf, d0, d50)[0]
        phi_c = phis(np.array([theta_c]), tf, d0, d50)[0]
        qs_b = np.sqrt(g * delta * d50 ** 3) * phi_b
        qs_c = np.sqrt(g * delta * d50 ** 3) * phi_c
        q_b = uniFlowQ(rf, w_b, x[2] * s0, x[0], d50, g, c, eps_c) / w_b
        q_c = uniFlowQ(rf, w_c, x[3] * s0, x[1], d50, g, c, eps_c) / w_c
        qs_y = (qs_b - qs_c) / (2 * alpha) * w_b / w
        q_y = (q_b - q_c) / (2 * alpha) * w_b / w
        res[0] = qs_y / qs0 - q_y / q0 + 2 * r * (x[1] - x[0]) / (theta0 ** 0.5 * (0.5 * (w + w_b + w_c)))
        res[1] = (qs_b + qs_c) / (w / w_b * qs0) - 1
        res[2] = (q_b + q_c) / (w / w_b * q0) - 1
        res[3] = (x[1] - x[0]) / ((x[2] - x[3]) * s0 * l) - 1
        return res


def deltaQ_BRT(ic, dsBC, rf, tf, theta_u, Qu, Qsu, w, l, d0, s0, w_b, w_c, d50, alpha, r, g, delta, tol, c, eps_c):
    # Computes the equilibrium solution of a bifurcation, given geometry and boundary conditions, by solving
    # numerically the nonlinear system of equations described in Bolla Pittaluga et al. (2003). Returns the discharge
    # asymmetry and the branches' main hydraulic parameters.
    x = np.array([ic[0] * d0, ic[1] * d0, ic[2], ic[3]])
    if dsBC == 0:
        x = opt.fsolve(fSys_BRT, x[:3], (dsBC, rf, tf, theta_u, Qu, Qsu, w, l, d0, s0, w_b, w_c, d50, alpha, r, g, delta,
                                     c, eps_c), xtol=tol, maxfev=100000)[:3]
        d_b, d_c = x[:2]
        s_b = x[2] * s0
        s_c = s_b
    elif dsBC == 1:
        x = opt.fsolve(fSys_BRT, x, (dsBC, rf, tf, theta_u, Qu, Qsu, w, l, d0, s0, w_b, w_c, d50, alpha, r, g, delta,
                                     c, eps_c), xtol=tol, maxfev=100000)[:4]
        d_b, d_c = x[:2]
        s_b = x[2] * s0
        s_c = x[3] * s0
    else:
        d_b, d_c, s_b, s_c = [None, None, None, None]
    q_b = uniFlowQ(rf, w_b, s_b, d_b, d50, g, c, eps_c)
    q_c = uniFlowQ(rf, w_c, s_c, d_c, d50, g, c, eps_c)
    theta_b = s_b * x[0] / (delta * d50)
    theta_c = s_c * x[1] / (delta * d50)
    fr_b = q_b / (w_b * x[0] * np.sqrt(g * x[0]))
    fr_c = q_c / (w_c * x[1] * np.sqrt(g * x[1]))
    return (q_b - q_c) / Qu, theta_b, theta_c, fr_b, fr_c, s_b, s_c, d_b, d_c


def betaC_BRT(nb, beta_max, dsBC, rf, tf, theta0, ds0, w_b, w_c, Ls, d50, alpha, r, g, delta, c, eps_c):
    # Hydraulic parameters
    d0 = d50 / ds0
    s0 = theta0 * delta * d50 / d0
    phi0 = phis(np.array([theta0]), tf, d0, d50)[0]
    qs0 = np.sqrt(g * delta * d50 ** 3) * phi0
    beta0 = np.linspace(0.5, beta_max, nb)
    rq = np.zeros(nb)
    db = 1.8 * d0
    dc = 0.6 * d0
    s = 0.5 * s0
    x = np.array([[db], [dc], [s]])

    # Numeric and iterations parameters
    tol = 1e-6
    eps = 1e-15
    betaC = -1
    betaRcheck = False
    for i in range(nb - 1, -1, -1):
        # Derived flow conditions for the main channel
        w = 2 * beta0[i] * d0
        Q0 = uniFlowQ(rf, w, s0, d0, d50, g, c, eps_c)
        Qs0 = w * qs0

        # BRT equilibrium solution
        BRT_out = deltaQ_BRT((x[0] / d0, x[1] / d0, x[2] / s0), dsBC, rf, tf, theta0, Q0, Qs0, w, Ls * d0, d0, s0, w_b,
                             w_c, d50, alpha, r, g, delta, tol, c, eps_c)
        rq[i] = BRT_out[0]
        x = np.array([[BRT_out[6]], [BRT_out[7]], [BRT_out[5]]])

        if i < nb - 1 and not betaRcheck:
            if rq[i] < eps < rq[i + 1]:
                betaC = beta0[i]
                betaRcheck = True
    return betaC


def myPlot(fig_number, x, y, label, title=None, xlabel=None, ylabel=None, color=None, fontsize=12):
    plt.figure(fig_number)
    if title is not None:
        plt.title(title, fontsize=fontsize)
    if xlabel is not None:
        plt.xlabel(xlabel)
    if ylabel is not None:
        plt.ylabel(ylabel)
    plt.plot(x, y, label=label, color=color)
    plt.grid(True)
    if label is not None:
        plt.legend(loc='best')


def subplotsLayout(axes, x_labels, y_labels, titles):
    ncols = len(axes)
    for j in range(ncols):
        axes[j].set_xlabel(x_labels[j])
        axes[j].set_ylabel(y_labels[j])
        axes[j].set_title(titles[j])
        axes[j].grid()
        axes[j].legend(loc='best')
    plt.tight_layout()


def flexFinder(f, dx):
    # Finds the inflection point of a given 1d array, by setting the numerical second derivative equal to 0
    # NB: dx must be an array
    df2dx2 = (f[2:]-2*f[1:-1]+f[:-2])/dx
    for i in range(int(0.01 * len(df2dx2)),len(df2dx2)-1):
        if df2dx2[i]*df2dx2[i+1]<0:
            print("Inflection point found at x = %2.1f\n" % (np.sum(dx[:i+1])))
            return i+1
    print("Inflection point not found. flexIndex is set to 0. \n")
    return 0


def interpolate(func, xData, yData, ic=None, bounds=(-np.inf, np.inf)):
    # Interpolate data by fitting a given function, then returns the interpolated curve as a 1d array.
    par, covar = opt.curve_fit(func, xData, yData, p0=ic, maxfev=8000, bounds=bounds)
    if len(par) == 2:
        intCurve = func(xData, par[0], par[1])
    elif len(par) == 3:
        intCurve = func(xData, par[0], par[1], par[2])
    elif len(par) == 4:
        intCurve = func(xData, par[0], par[1], par[2], par[3])
    else:
        print("Interpolation failed. The interpolation function must have less than 5 parameters")
        intCurve = -1 * np.ones(len(xData))
    return par, intCurve, covar


def fSysLocalUnsteady(q_b, eta_b, eta_c, s_b, s_c, w_b, w_c, q_u, g, d50, d0, rf, c0, eps_c):
    # Returns the residual of the system of equations composed by continuity (q_b+q_c=q_u), uniform
    # flow stage-discharge relationships and dH/dy=0
    q_c = q_u-q_b
    d_b = uniFlowD(rf,q_b,w_b,s_b,d50,g,c0,eps_c,d0)
    d_c = uniFlowD(rf,q_c,w_c,s_c,d50,g,c0,eps_c,d0)
    return ((d_b+eta_b)-(d_c+eta_c))/((d_b+d_c)/2)


def eqSin(t, a, f, t0, b):
    """
    Used to interpolate the statistically stationary stage of deltaQ vs time, in order to find the amplitude and
    frequency of its oscillations
    """
    return a*np.sin(2*f*np.pi*(t-t0))+b


def mkdir_p(mypath):
    """Creates a directory. equivalent to using mkdir -p on the command line"""

    from errno import EEXIST
    from os import makedirs, path

    try:
        makedirs(mypath)
    except OSError as exc:
        if exc.errno == EEXIST and path.isdir(mypath):
            pass
        else:
            raise

def fSysSC(x, D0, Q0, D_abV, D_acV, Q_abV, Q_acV, S_ab, S_ac, W_ab, W_ac, eta_ab, eta_ac, g, d50, dx, c0, rf):
    # Function to be used to iteratively solve the system governing water partitioning along the semichannels.
    # The unknown array x is equal to x=[D_ab[i]/D0, D_ac[i]/D0, Q_ab[i]/Q0, Q_ac[i]/Q0, Qy[i]/Q0]
    # Change in notation: i -> M, i+1 -> V
    res=np.ones((5))
    D_abM, D_acM, Q_abM, Q_acM, Q_y = (x[0]*D0, x[1]*D0, x[2]*Q0, x[3]*Q0, x[4]*Q0)
    j_ab  = uniFlowS(rf, Q_abV, W_ab, D_abV, d50, g, c0, 2.5)
    j_ac  = uniFlowS(rf, Q_acV, W_ac, D_acV, d50, g, c0, 2.5)
    Fr_ab = Q_abV/(W_ab*D_abV)/sqrt(g*D_abV)    
    Fr_ac = Q_acV/(W_ac*D_acV)/sqrt(g*D_acV)
    res[0] = (D_abV-D_abM)/dx*(1-Fr_ab**2)/(S_ab-j_ab-Q_y/dx*Q_abV/(g*W_ab**2*D_abV**2))-1
    res[1] = (D_acV-D_acM)/dx*(1-Fr_ac**2)/(S_ac-j_ac+Q_y/dx*Q_acV/(g*W_ac**2*D_acV**2))-1
    res[2] = (Q_abM+Q_y)/Q_abV-1
    res[3] = (Q_acM-Q_y)/Q_acV-1
    res[4] = (D_abM+eta_ab)/(D_acM+eta_ac)-1
    return res

def coeffSysSC(D_abV, D_acV, Q_abV, Q_acV, S_ab, S_ac, W_ab, W_ac, eta_ab, eta_ac, g, d50, dx, c0, rf):
    # Computes the coefficient matrix A and the array B for the semichannels system to compute
    # the unknowns x=[D_ab[i]/D0, D_ac[i]/D0, Q_ab[i]/Q0, Q_ac[i]/Q0, Qy[i]/Q0], that is linear
    # as long as the equations discretization is explicits
    j_ab  = uniFlowS(rf, Q_abV, W_ab, D_abV, d50, g, c0, 2.5)
    j_ac  = uniFlowS(rf, Q_acV, W_ac, D_acV, d50, g, c0, 2.5)
    Fr_ab = Q_abV/(W_ab*D_abV)/sqrt(g*D_abV)    
    Fr_ac = Q_acV/(W_ac*D_acV)/sqrt(g*D_acV)
    A = np.array([[Q_abV/(dx*g*W_ab**2*D_abV**2*(1-Fr_ab**2)),-1/dx, 0, 0, 0],
                  [-Q_acV/(dx*g*W_ac**2*D_acV**2*(1-Fr_ac**2)),0,-1/dx, 0, 0],
                  [1,0,0,1,0],
                  [-1,0,0,0,1],
                  [0,1,-1,0,0]])
    B = np.array([-D_abV/dx+(S_ab-j_ab)/(1-Fr_ab**2), -D_acV/dx+(S_ac-j_ac)/(1-Fr_ac**2), Q_abV, Q_acV, -eta_ab+eta_ac])
    return A, B

def QyExpl(D_ab,D_ac,Q_ab,Q_ac,S_ab,S_ac,W_ab,W_ac,g,d50,dx,c0,rf,eps_c):
    j_ab   = uniFlowS(rf,Q_ab,W_ab,D_ab,d50,g,c0,eps_c)
    j_ac   = uniFlowS(rf,Q_ac,W_ac,D_ac,d50,g,c0,eps_c)
    Fr_ab  = Q_ab/(W_ab*D_ab*np.sqrt(g*D_ab))
    Fr_ac  = Q_ac/(W_ac*D_ac*np.sqrt(g*D_ac))
    deltaJ = -j_ab/(1-Fr_ab**2)+j_ac/(1-Fr_ac**2)
    deltaS = S_ab*Fr_ab**2/(1-Fr_ab**2)-S_ac*Fr_ac**2/(1-Fr_ac**2)
    a1     = (S_ab*Fr_ab**2-j_ab)/(1-Fr_ab**2)
    a2     = (S_ac*Fr_ac**2-j_ac)/(1-Fr_ac**2)
    b1     = 1/(1-Fr_ab**2)*Q_ab/(g*W_ab**2*D_ab**2)
    b2     = 1/(1-Fr_ac**2)*Q_ac/(g*W_ac**2*D_ac**2)
    Q_y    = (a1-a2)/(b1+b2)*dx
    return Q_y

def dDadxExpl(Q_y,D_ab,D_ac,Q_ab,Q_ac,S_ab,S_ac,W_ab,W_ac,g,d50,dx,c0,rf,eps_c):
    j_ab  = uniFlowS(rf,Q_ab,W_ab,D_ab,d50,g,c0,eps_c)
    j_ac  = uniFlowS(rf,Q_ac,W_ac,D_ac,d50,g,c0,eps_c)
    Fr_ab = Q_ab/(W_ab*D_ab*np.sqrt(g*D_ab))
    Fr_ac = Q_ac/(W_ac*D_ac*np.sqrt(g*D_ac))
    a1    = (S_ab-j_ab)/(1-Fr_ab**2)
    a2    = (S_ac-j_ac)/(1-Fr_ac**2)
    b1    = 1/(1-Fr_ab**2)*Q_ab/(g*W_ab**2*D_ab**2)
    b2    = 1/(1-Fr_ac**2)*Q_ac/(g*W_ac**2*D_ac**2)
    dDadx = 0.5*(a1+a2)-0.5*Q_y/dx*(b1-b2)
    return dDadx

def ddeltaQdxF(D_a,deltaQ,inStep,S_ab,S_ac,W_ab,W_ac,Q0,D0,g,d50,dx,c0,rf,eps_c):
    D_ab = D_a-0.5*inStep*D0
    D_ac = D_a+0.5*inStep*D0
    Q_ab = Q0/2*(1+deltaQ)
    Q_ac = Q0/2*(1-deltaQ)
    Q_y = QyExpl(D_ab,D_ac,Q_ab,Q_ac,S_ab,S_ac,W_ab,W_ac,g,d50,dx,c0,rf,eps_c)
    ddeltaQdx = 2*Q_y/(Q0*dx)
    return ddeltaQdx

def ddeltaQ_rk4(deltaQ_M,D_aM,D_aV,inStepM,inStepV,S_ab,S_ac,W_ab,W_ac,Q0,D0,g,d50,dx,c0,rf,eps_c):
    k1 = ddeltaQdxF(D_aM,deltaQ_M,inStepM,S_ab,S_ac,W_ab,W_ac,Q0,D0,g,d50,dx,c0,rf,eps_c)
    k2 = ddeltaQdxF(0.5*(D_aM+D_aV),deltaQ_M+0.5*dx*k1,0.5*(inStepM+inStepV),S_ab,S_ac,W_ab,W_ac,Q0,D0,g,d50,dx,c0,rf,eps_c)
    k3 = ddeltaQdxF(0.5*(D_aM+D_aV),deltaQ_M+0.5*dx*k2,0.5*(inStepM+inStepV),S_ab,S_ac,W_ab,W_ac,Q0,D0,g,d50,dx,c0,rf,eps_c)
    k4 = ddeltaQdxF(D_aV,deltaQ_M+dx*k3,inStepV,S_ab,S_ac,W_ab,W_ac,Q0,D0,g,d50,dx,c0,rf,eps_c)
    deltaQ_V = deltaQ_M+dx/6*(k1+2*k2+2*k3+k4)
    return deltaQ_V

def scF(inStep):
    dDadx = ...
    ddeltaQdx = ...
    return dDadx, ddeltaQdx

def sc_rk4():
    # Solves the linear momentum equation for the semichannels in a coupled fashion
    # taking as unkwnons the average depth Da and the discahrge asymmetry deltaQ
    return None

def minmod(slope1,slope2):
    omega    = abs(slope1)<abs(slope2)
    limslope = omega*slope1+(1-omega)*slope2
    signch   = slope1*slope2>0
    limslope = limslope*signch
    return limslope

def maxmod(slope1,slope2):
    omega    = abs(slope1)>abs(slope2)
    limslope = omega*slope1+(1-omega)*slope2
    signch   = slope1*slope2>0
    limslope = limslope*signch
    return limslope

def MCslope(slope1,slope2):
    slopeC   = (slope1+slope2)/2
    slopeabs = np.vstack([np.abs(slope1),np.abs(slope2),np.abs(slopeC)])
    omega    = slopeabs==np.min(slopeabs,0)
    limslope = np.diag(np.dot(omega.T,np.vstack((slope1,slope2,slopeC))))
    signch   = (slope1*slope2>0)*(slopeC*slope2>0)
    limslope = limslope*signch
    return limslope

def C_eta(Q,W,D,g,delta,d50,p,C0,D0,rf,tf,eps_c,eps=1e-6):
    # Computes the celerity of the propagation of a perturbation of the bed elvel (kinematic wave approximation)
    theta_epsp = shieldsUpdate(rf,Q+eps,W,D,d50,g,delta,C0,eps_c)
    theta_epsm = shieldsUpdate(rf,Q-eps,W,D,d50,g,delta,C0,eps_c)
    Qs_epsp    = W*np.sqrt(g*delta*d50**3)*phis(theta_epsp,tf,D0,d50)[0]
    Qs_epsm    = W*np.sqrt(g*delta*d50**3)*phis(theta_epsm,tf,D0,d50)[0]
    dQsdQ      = (Qs_epsp-Qs_epsm)/(2*eps)
    Fr         = Q/(W*D*np.sqrt(g*D))
    C_eta      = dQsdQ*(Q/(W*D))/(1-Fr**2)/(1-p)
    return C_eta

def deltaH_confluence(Q_b,Q_c,D_b,D_c,W_b,W_c,Q0,D0,W0,H0,kI,kb,kc,g):
    U_b      = Q_b/(W_b*D_b)
    U_c      = Q_c/(W_c*D_c)
    U0       = Q0/(W0*D0)
    deltaQ   = (Q_b-Q_c)/Q0
    deltaM_b = Q_b*(U0-U_b)
    deltaM_c = Q_c*(U0-U_c)
    FI       = 0.5*W0*kI*(U_b**2-U_c**2)*(D_b+D_c)*(1-deltaQ**2)
    FS_b     = 0.5*kb*U_b**2*D_b*W0*(1+deltaQ)
    FS_c     = 0.5*kc*U_c**2*D_c*W0*(1-deltaQ)
    sumP_b   = deltaM_b+FI+FS_b
    sumP_c   = deltaM_c-FI+FS_c
    H_b      = sumP_b/(0.5*g*(D_b+D0)*W0*0.5*(1+deltaQ))+H0
    H_c      = sumP_c/(0.5*g*(D_c+D0)*W0*0.5*(1-deltaQ))+H0
    return H_b,H_c

def betaC_loopNR(r,alpha,theta0,Fr0,S0,Ls,D0,C0,kI,kb,tf,rf,d50):
    Lambda         = Fr0**2/(S0*Ls)
    csi            = 8*kI+2*kb-1
    phi0,phiD,phiT = phis_scalar(theta0,tf,D0,d50)
    cD             = 2.5/C0*rf
    phi            = csi*(0.5*(phiT+phiD)-phiT*(1.5+cD))-(0.5*kb-csi)*(phiT-0.5)
    return 2*r*alpha/np.sqrt(theta0)*(2+Lambda*csi)/(phiT+phiD-cD-1.5+Lambda*phi)

def newton(xn,f,args,tol=1e-10):
    eps = 1e-7
    maxIterNewton = 100
    for n in range(maxIterNewton):
        res = f(xn,*args)
        if abs(res)<tol:
            break
        df = (f(xn+eps,*args)-f(xn-eps,*args))/(2*eps)
        xn = xn-res/df
    return xn