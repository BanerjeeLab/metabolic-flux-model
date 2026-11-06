"""Drop-in replacement for the closed-form balanced solutions (PhiRBalanced, PhiFBalanced,
PhiOBalanced, JOutBalanced) in methods22.py, methodsFig6.py, methodsRodCrowding*.py.

Solves the three steady-state balances directly as a linear system.  Identical to the sympy
closed forms when CF = 1; with CF = 1e6/12.011 the biomass carbon drain J_B = f_C dM/dt
[MDa/h] is converted to carbon atoms/h, consistent with J_in and J_G/e_G.

Balances (chi = beta/T from the host module's Crowding; y = phiR - phiRMin; g = kt*M*chi):
  carbon :  J_in - J_G/e_G - CF*f_C*g*y = 0
  acCoA  :  J_G/(3 e_G) - J_O/e_O - J_P - J_out = 0
  ATP    :  J_G + J_O + J_out - (eBtotal*f_C)*g*y - e_P J_P - a M = 0
with J_G = eps_G phiG M chi, J_O = eps_O phiO M chi, J_P = 16 rho_P phiE M kt chi y/(rho_E ln2).
Normal regime: unknowns (y, phiG, phiO), J_out = 0.   Overflow: phiO = phiOMaxFrac*phiE (as passed
in args, i.e. gamma_eff), unknowns (y, phiG, J_out).

Usage:  import fluxbalance_linear as fb;  fb.install(methods22)          # args = [M, phiE, ...]
        fb.install(methodsRodCrowding, has_V=True)                        # args = [M, V, phiE, ...]
"""
import numpy as np

CF_ATOMS_PER_MDA = 1e6/12.011   # carbon atoms per MDa of carbon
CF = CF_ATOMS_PER_MDA           # set to 1.0 to reproduce the closed forms without the unit conversion

def _solve(host, M, phiE, chi, G0, a, epsG, epsO, eG, eO, eP, rhoE, rhoP, fC, kt, phiRMin, gmax_frac, overflow):
    M=float(M); phiE=float(phiE); chi=float(chi); G0=float(G0)
    JIn = 6.0*host.Pprime*(phiE*M/rhoE)*(G0/(G0+host.K_ptsg))
    lam = 16*rhoP*phiE*M*kt*chi/(rhoE*np.log(2))
    g   = kt*M*chi
    cB  = host.eBtotal*fC                       # ATP per unit dM/dt
    if not overflow:
        A = np.array([[-CF*fC*g,        -epsG*M*chi/eG,      0.0          ],
                      [-lam,             epsG*M*chi/(3*eG),  -epsO*M*chi/eO],
                      [-cB*g - eP*lam,   epsG*M*chi,          epsO*M*chi   ]])
        b = np.array([-JIn, 0.0, a*M])
        y, phiG, phiO = np.linalg.solve(A, b)
        return y+phiRMin, phiG, phiO, 0.0
    phiO = gmax_frac*phiE
    A = np.array([[-CF*fC*g,        -epsG*M*chi/eG,      0.0],
                  [-lam,             epsG*M*chi/(3*eG),  -1.0],
                  [-cB*g - eP*lam,   epsG*M*chi,          1.0]])
    b = np.array([-JIn, epsO*phiO*M*chi/eO, a*M - epsO*phiO*M*chi])
    y, phiG, JOut = np.linalg.solve(A, b)
    return y+phiRMin, phiG, phiO, JOut

def install(host, has_V=False):
    def unpack(args):
        if has_V:
            M,V,phiE,phiRMin,phi0,a,G0,P,epsF,epsO,eF,eO,eB,eP,beta,rhoE,rhoP,d,fC,kt,gfrac,switch = args
            chi = host.Crowding(M,V,phiE,beta,rhoE,d,switch)
        else:
            M,phiE,phiRMin,phi0,a,G0,P,epsF,epsO,eF,eO,eB,eP,beta,rhoE,rhoP,d,fC,kt,gfrac,switch = args
            chi = host.Crowding(M,phiE,beta,rhoE,d,switch)
        return dict(M=M,phiE=phiE,chi=chi,G0=G0,a=a,epsG=epsF,epsO=epsO,eG=eF,eO=eO,eP=eP,
                    rhoE=rhoE,rhoP=rhoP,fC=fC,kt=kt,phiRMin=phiRMin,gmax_frac=gfrac)
    def PhiRBalanced(args, overflow):
        return _solve(host, overflow=overflow, **unpack(args))[0]
    def PhiFBalanced(args, overflow):
        return _solve(host, overflow=overflow, **unpack(args))[1]
    def PhiOBalanced(args):
        return _solve(host, overflow=False, **unpack(args))[2]
    def JOutBalanced(args):
        return _solve(host, overflow=True, **unpack(args))[3]
    host.PhiRBalanced = PhiRBalanced; host.PhiFBalanced = PhiFBalanced
    host.PhiOBalanced = PhiOBalanced; host.JOutBalanced = JOutBalanced
    host._fluxbalance_linear_installed = True
    return host
