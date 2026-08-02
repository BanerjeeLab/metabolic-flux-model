import numpy as np
from scipy.optimize import minimize
from scipy.optimize import fsolve
from scipy.optimize import Bounds
from scipy.optimize import least_squares
from scipy.optimize import brute
import matplotlib.pyplot as plt

f_in = 0.1 #fraction of phiE that pertains to glucose transporters phi_in
CONV_G0_TO_uM = 0.001660539067 #gives uM scaling factor that corresponds to 1 molecule per cubic micron
fC=0.47

def set_f_in(val):
    global f_in
    f_in = val

# eB, eGAM absorb the factor f_C * m_a so the flux-balance expressions can use
# them as bare coefficients (m_a = 1.1e-4). Values below are the physical
# 23.5 and 8 ATP/aa divided by f_C*m_a. Multiply by f_C*m_a to recover ATP/aa
eB = 23.5/(fC*1.1*10**-4)
eGAM = 8/(fC*1.1*10**-4)
eBtotal = eB + eGAM 

def set_eGAM(val):
    """Update eGAM and the dependent eBtotal at module level."""
    global eGAM, eBtotal
    eGAM = val
    eBtotal = eB + eGAM

CONV_num_per_um3_from_uM = 1 / CONV_G0_TO_uM

K_ptsg_uM = 10.0 #in uM 
K_C_uM = 30 #in uM
# converting saturation constants from uM to molecules/um^3 to match G0 units
K_ptsg = K_ptsg_uM*CONV_num_per_um3_from_uM  # molecules/um^3
K_C = K_C_uM*CONV_num_per_um3_from_uM  # molecules/um^3

P=12000
Pprime = P*K_C  #(in the same units as G0) (also defined in Jin function)

def set_K_C(val_uM):
    """Update K_C and the dependent Pprime at module level. val_uM in uM."""
    global K_C_uM, K_C, Pprime
    K_C_uM = val_uM
    K_C = val_uM * CONV_num_per_um3_from_uM
    Pprime = P * K_C


# Crowding: diffusion penalty on metabolic fluxes. 
#For Fig. 2, 3, 4, and S1, crowding is switched off (the caller passes switch=False), so it returns 1 and has no
# effect; beta (beta' in the paper) is read only in the switch=True branch.
# The diffusion-limited form is used only in Fig. 5 and 6. See README.
def Crowding(M,phiE,beta,rhoE,d,switch):
    return 1


###############
def JIn(M, phiE, G0, P, rhoE, K_ptsg):
    Pprime = P * K_C  # 
    S = phiE * M / rhoE
    return 6.0 * Pprime * S * (G0 / (G0 + K_ptsg))


def JF(M,phiE,phiF,beta,rhoE,d,epsilonF,switch):
    return epsilonF*phiF*M*Crowding(M,phiE,beta,rhoE,d,switch)

def JO(M,phiE,phiO,beta,rhoE,d,epsilonO,switch):
    return epsilonO*phiO*M*Crowding(M,phiE,beta,rhoE,d,switch)

def JB(M,phiE,phiR,phiRMin,beta,rhoE,d,fC,kt,switch):
    return fC*kt*(phiR-phiRMin)*M*Crowding(M,phiE,beta,rhoE,d,switch)

def JP(M,phiE,phiR,phiRMin,beta,rhoE,rhoP,d,kt,switch):
    return 16*rhoP*phiE*M*kt*(phiR-phiRMin)*Crowding(M,phiE,beta,rhoE,d,switch)/(np.log(2)*rhoE)

def JMait(M,a):
    return a*M

def FluxBalanceGlucose(M,phiE,phiR,phiRMin,phiF,phiO,JOut,a,G0,P,epsilonF,epsilonO,eF,eO,eB,eP,beta,rhoE,rhoP,d,fC,kt,switch):
    return JIn(M,phiE,G0,P,rhoE,K_ptsg)-JF(M,phiE,phiF,beta,rhoE,d,epsilonF,switch)/eF-JB(M,phiE,phiR,phiRMin,beta,rhoE,d,fC,kt,switch)

def FluxBalanceACA(M,phiE,phiR,phiRMin,phiF,phiO,JOut,a,G0,P,epsilonF,epsilonO,eF,eO,eB,eP,beta,rhoE,rhoP,d,fC,kt,switch):
    return 2*JF(M,phiE,phiF,beta,rhoE,d,epsilonF,switch)/(6*eF)-JO(M,phiE,phiO,beta,rhoE,d,epsilonO,switch)/eO-JP(M,phiE,phiR,phiRMin,beta,rhoE,rhoP,d,kt,switch)-JOut

######## added eGAM to ATP flux balance eqn
def FluxBalanceATP(M,phiE,phiR,phiRMin,phiF,phiO,JOut,a,G0,P,epsilonF,epsilonO,eF,eO,eB,eP,beta,rhoE,rhoP,d,fC,kt,switch):
    return JF(M,phiE,phiF,beta,rhoE,d,epsilonF,switch)+JO(M,phiE,phiO,beta,rhoE,d,epsilonO,switch)+JOut- eBtotal*JB(M,phiE,phiR,phiRMin,beta,rhoE,d,fC,kt,switch)-eP*JP(M,phiE,phiR,phiRMin,beta,rhoE,rhoP,d,kt,switch)-JMait(M,a)


def efficiency(M,phiE,phiR,phiRMin,phiF,phiO,JOut,a,G0,P,epsilonF,epsilonO,eF,eO,eB,eP,beta,rhoE,rhoP,d,fC,kt,switch):
    num=eB*JB(M,phiE,phiR,phiRMin,beta,rhoE,d,fC,kt,switch)+eP*JP(M,phiE,phiR,phiRMin,beta,rhoE,rhoP,d,kt,switch)
    #num=eBtotal*JB(M,phiE,phiR,phiRMin,beta,rhoE,d,fC,kt,switch)+eP*JP(M,phiE,phiR,phiRMin,beta,rhoE,rhoP,d,kt,switch)
    waste=JOut*8
    denom=JF(M,phiE,phiF,beta,rhoE,d,epsilonF,switch)+JO(M,phiE,phiO,beta,rhoE,d,epsilonO,switch)+waste
    return num/denom



def PhiRBalanced(args,overflow):
    #replacing eB with eBtotal
    M,phiE,phiRMin,phi0,a,G0,P,epsilonF,epsilonO,eF,eO,eB,eP,beta,rhoE,rhoP,d,fC,kt,phiOMaxFrac,switch=args
    betaT=Crowding(M,phiE,beta,rhoE,d,switch)
    if overflow:
        num=betaT*(-epsilonO*phiE*phiOMaxFrac*rhoE*np.log(8)+eO*epsilonO*phiE*phiOMaxFrac*rhoE*np.log(8)+eO*kt*phiRMin*(48*(1+eP)*phiE*rhoP+eBtotal*fC*rhoE*np.log(8)+fC*rhoE*(np.log(2)+eF*np.log(8))))+eO*(18*eF*G0/(G0 + K_ptsg)*Pprime*phiE*np.log(2)-a*rhoE*np.log(8)+G0/(G0 + K_ptsg)*Pprime*phiE*np.log(64))
        denom=betaT*eO*kt*(48*(1+eP)*phiE*rhoP+eBtotal*fC*rhoE*np.log(8)+fC*rhoE*(np.log(2)+eF*np.log(8)))
        return num/denom
    else:
        num=18*eF*G0/(G0 + K_ptsg)*Pprime*phiE*np.log(2)-a*rhoE*np.log(8)+betaT*kt*phiRMin*(48*eP*phiE*rhoP+eO*(48*phiE*rhoP+fC*rhoE*np.log(2))+(eBtotal+eF)*fC*rhoE*np.log(8))+eO*G0/(G0 + K_ptsg)*Pprime*phiE*np.log(64)
        denom=betaT*kt*(48*eP*phiE*rhoP+eO*(48*phiE*rhoP+fC*rhoE*np.log(2))+(eBtotal+eF)*fC*rhoE*np.log(8))
        return num/denom
    
def PhiFBalanced(args,overflow):
    #replacing eB with eBtotal
    M,phiE,phiRMin,phi0,a,G0,P,epsilonF,epsilonO,eF,eO,eB,eP,beta,rhoE,rhoP,d,fC,kt,phiOMaxFrac,switch=args
    betaT=Crowding(M,phiE,beta,rhoE,d,switch)
    if overflow:
        num=3*eF*(betaT*epsilonO*fC*phiE*phiOMaxFrac*rhoE**2*np.log(2)+eO*fC*(a-betaT*epsilonO*phiE*phiOMaxFrac)*rhoE**2*np.log(2)+6*eO*G0/(G0 + K_ptsg)*Pprime*phiE*(16*(1+eP)*phiE*rhoP+eBtotal*fC*rhoE*np.log(2)))
        denom=betaT*eO*epsilonF*rhoE*(48*(1+eP)*phiE*rhoP+eBtotal*fC*rhoE*np.log(8)+fC*rhoE*(np.log(2)+eF*np.log(8)))
        return num/denom
    else:
        num=3*eF*(96*eO*G0/(G0 + K_ptsg)*Pprime*phiE**2*rhoP+96*eP*G0/(G0 + K_ptsg)*Pprime*phiE**2*rhoP+fC*rhoE*(6*eBtotal*G0/(G0 + K_ptsg)*Pprime*phiE+a*rhoE)*np.log(2))
        denom=betaT*epsilonF*rhoE*(48*eP*phiE*rhoP+eO*(48*phiE*rhoP+fC*rhoE*np.log(2))+(eBtotal+eF)*fC*rhoE*np.log(8))
        return num/denom
    
def JOutBalanced(args):
    #replacing eB with eBtotal
    M,phiE,phiRMin,phi0,a,G0,P,epsilonF,epsilonO,eF,eO,eB,eP,beta,rhoE,rhoP,d,fC,kt,phiOMaxFrac,switch=args
    betaT=Crowding(M,phiE,beta,rhoE,d,switch)
    return -((M*(betaT*epsilonO*phiE*phiOMaxFrac*rhoE*(48*eP*phiE*rhoP + 
          eBtotal*fC*rhoE*np.log(8)) + 
       eF*phiE*(288*eO*G0/(G0 + K_ptsg)*Pprime*phiE*rhoP + 
          betaT*epsilonO*fC*phiOMaxFrac*rhoE**2*np.log(8)) - 
       eO*(96*eP*G0/(G0 + K_ptsg)*Pprime*phiE**2*rhoP + 
          rhoE*(a*(48*phiE*rhoP + fC*rhoE*np.log(2)) - 
             betaT*epsilonO*phiE*phiOMaxFrac*(48*phiE*rhoP + 
                fC*rhoE*np.log(2)) + 
             eBtotal*fC*G0/(G0 + K_ptsg)*Pprime*phiE*np.log(64)))))/(eO*rhoE*(48*(1 + 
          eP)*phiE*rhoP + fC*eBtotal*rhoE*np.log(8) + 
       fC*rhoE*(np.log(2) + eF*np.log(8)))))

    
def PhiOBalanced(args):
    #replacing eB with eBtotal
    M,phiE,phiRMin,phi0,a,G0,P,epsilonF,epsilonO,eF,eO,eB,eP,beta,rhoE,rhoP,d,fC,kt,phiOMaxFrac,switch=args
    betaT=Crowding(M,phiE,beta,rhoE,d,switch)
    num=eO*(-288*eF*G0/(G0 + K_ptsg)*Pprime*phiE**2*rhoP+96*eP*G0/(G0 + K_ptsg)*Pprime*phiE**2*rhoP+rhoE*(48*a*phiE*rhoP+a*fC*rhoE*np.log(2)+eBtotal*fC*G0/(G0 + K_ptsg)*Pprime*phiE*np.log(64)))
    denom=betaT*epsilonO*rhoE*(48*eP*phiE*rhoP+eO*(48*phiE*rhoP+fC*rhoE*np.log(2))+(eBtotal+eF)*fC*rhoE*np.log(8))
    return num/denom

def optimizedNormal(x,phiRMin,phi0,a,G0,P,epsilonF,epsilonO,eF,eO,eB,eP,beta,rhoE,rhoP,d,fC,kt,phiOMaxFrac,switch):
    M,phiE=x
    #phiRMin,phi0,a,G0,P,epsilonF,epsilonO,eF,eO,eB,eP,beta,rhoE,rhoP,d,fC,kt,phiOMaxFrac = args
    argz = [M,phiE,phiRMin,phi0,a,G0,P,epsilonF,epsilonO,eF,eO,eB,eP,beta,rhoE,rhoP,d,fC,kt,phiOMaxFrac,switch]

    phiR=PhiRBalanced(argz,False)
    phiF=PhiFBalanced(argz,False)
    phiO=PhiOBalanced(argz)
    if phiR+phiF+phiE+phi0>1 or phiR<0 or phiF<0 or phiO<0:
        return 10000
    return -kt*(phiR-phiRMin)*Crowding(M,phiE,beta,rhoE,d)

def optimizedOverflow(x,phiRMin,phi0,a,G0,P,epsilonF,epsilonO,eF,eO,eB,eP,beta,rhoE,rhoP,d,fC,kt,phiOMaxFrac,switch):
    M,phiE=x
    gamma_eff = phiOMaxFrac*(1 - f_in)
    argz = [M,phiE,phiRMin,phi0,a,G0,P,epsilonF,epsilonO,eF,eO,eB,eP,beta,rhoE,rhoP,d,fC,kt,gamma_eff,switch]

    phiR=PhiRBalanced(argz,True)
    phiF=PhiFBalanced(argz,True)
    phiO=gamma_eff*phiE
    JOut=JOutBalanced(argz)
    if phiR+phiF+phiE+phi0>1 or phiR<0 or phiF<0 or JOut<0:
        return 10000
    return -kt*(phiR-phiRMin)*Crowding(M,phiE,beta,rhoE,d)

def G0Optimizer(args,res,G0min,G0max,phiEmin,phiEmax,Mmin,Mmax):       
    phiRMin,phi0,a,P,epsilonF,epsilonO,eF,eO,eB,eP,beta,rhoE,rhoP,d,fC,kt,phiOMaxFrac,switch = args
    #bounds=Bounds([Mmin,Mmax],[phiEmin,phiEmax])
    sols=[]
    for G0 in np.linspace(start=G0min, stop=G0max, num=res):
        argz=(phiRMin,phi0,a,G0,P,epsilonF,epsilonO,eF,eO,eB,eP,beta,rhoE,rhoP,d,fC,kt,phiOMaxFrac)

        solNormal=brute(optimizedNormal,((Mmin,Mmax),(phiEmin,phiEmax)), args=argz,Ns=800,finish=None)
        print(solNormal)
        #if PhiOBalanced([solNormal[0],solNormal[1],phiRMin,phi0,a,G0,P,epsilonF,epsilonO,eF,eO,eB,eP,beta,rhoE,rhoP,d,fC,kt,phiOMaxFrac])<phiOMaxFrac*solNormal[1]:
        
        ##########
        #where f_in is the fraction of envelope protein sector that are glucose transporters
        gamma_eff = phiOMaxFrac*(1 - f_in)
        if PhiOBalanced([solNormal[0],solNormal[1],phiRMin,phi0,a,G0,P,epsilonF,epsilonO,eF,eO,eB,eP,beta,rhoE,rhoP,d,fC,kt,phiOMaxFrac]) < gamma_eff * solNormal[1]:

        #optimizedNormal(solNormal,phiRMin,phi0,a,G0,P,epsilonF,epsilonO,eF,eO,eB,eP,beta,rhoE,rhoP,d,fC,kt,phiOMaxFrac)<optimizedOverflow(solOverflow,phiRMin,phi0,a,G0,P,epsilonF,epsilonO,eF,eO,eB,eP,beta,rhoE,rhoP,d,fC,kt,phiOMaxFrac):
            argzz=[solNormal[0],solNormal[1],phiRMin,phi0,a,G0,P,epsilonF,epsilonO,eF,eO,eB,eP,beta,rhoE,rhoP,d,fC,kt,phiOMaxFrac,switch]
            phiR=PhiRBalanced(argzz,False)
            phiF=PhiFBalanced(argzz,False)
            phiO=PhiOBalanced(argzz)
            sols.append([G0,kt*(phiR-phiRMin)*Crowding(solNormal[0],solNormal[1],beta,rhoE,d),solNormal[0],solNormal[1],phiR,phiF,phiO,0,Crowding(solNormal[0],solNormal[1],beta,rhoE,d)])
        else:            
            solOverflow=brute(optimizedOverflow,((Mmin,Mmax),(phiEmin,phiEmax)), args=argz,Ns=800,finish=None)

            argzz_ov=[solOverflow[0],solOverflow[1],phiRMin,phi0,a,G0,P,epsilonF,epsilonO,eF,eO,eB,eP,beta,rhoE,rhoP,d,fC,kt,gamma_eff,switch]
            phiR=PhiRBalanced(argzz_ov,True)
            phiF=PhiFBalanced(argzz_ov,True)
            phiO = gamma_eff * solOverflow[1]
            JOut=JOutBalanced(argzz_ov)
            sols.append([G0,kt*(phiR-phiRMin)*Crowding(solOverflow[0],solNormal[1],beta,rhoE,d),solOverflow[0],solNormal[1],phiR,phiF,phiO,JOut,Crowding(solOverflow[0],solNormal[1],beta,rhoE,d)])
    return sols

def surface(volume,scaleFactor):
    return scaleFactor[1]*6.24*volume**(scaleFactor[0]*2/3)

def volume(kappa,scaleFactor):
    return scaleFactor[3]*0.17*np.exp(1.16*scaleFactor[2]*kappa)

def MSolveCrowding(x,phiRMin,phi0,a,P,epsilonF,epsilonO,eF,eO,eB,eP,beta,rhoE,rhoP,d,fC,kt,phiOMaxFrac,rhoM,switch,phiR,scaleFactor):
    M,phiE=x
    #phiRMin,phi0,a,P,epsilonF,epsilonO,eF,eO,eB,eP,beta,rhoE,rhoP,d,fC,kt,phiOMaxFrac,rhoM,switch,phiR=args
    x1=((M-rhoM*volume(kt*(phiR-phiRMin)*Crowding(M,phiE,beta,rhoE,d,switch),scaleFactor))/rhoM*volume(kt*(phiR-phiRMin)*Crowding(M,phiE,beta,rhoE,d,switch),scaleFactor))**2
    x2=((phiE-rhoE*surface(volume(kt*(phiR-phiRMin)*Crowding(M,phiE,beta,rhoE,d,switch),scaleFactor),scaleFactor)/M)/rhoE*surface(volume(kt*(phiR-phiRMin)*Crowding(M,phiE,beta,rhoE,d,switch),scaleFactor),scaleFactor)/M)**2
    return x1**2+x2**2

def G0Determined(args,res,G0min,G0max,phiRmax,scaleFactor):       
    phiRMin,phi0,a,P,epsilonF,epsilonO,eF,eO,eB,eP,beta,rhoE,rhoP,d,fC,kt,phiOMaxFrac,rhoM,switch = args
    #bounds=Bounds([Mmin,Mmax],[phiEmin,phiEmax])  
    solFinal=[] 
    for G0 in np.linspace(start=G0min, stop=G0max, num=res):
        solsTemp=[] 
        for phiR in np.linspace(start=0.01, stop=phiRmax, num=res):
            if switch:
                argz=(phiRMin,phi0,a,P,epsilonF,epsilonO,eF,eO,eB,eP,beta,rhoE,rhoP,d,fC,kt,phiOMaxFrac,rhoM,switch,phiR,scaleFactor)
                sol= brute(MSolveCrowding,((15000,70000),(0.1,0.35)),args=argz,Ns=800,finish=None)
                #fsolve(MSolveCrowding,[1.7*10**5,0.2],args=argz,full_output=True)
                #print(bb)
                M=sol[0]
                phiE=sol[1]
            else:            
                M=rhoM*volume(kt*(phiR-phiRMin),scaleFactor)
                phiE=rhoE*surface(volume(kt*(phiR-phiRMin),scaleFactor),scaleFactor)/M
            sols=[]

            gamma_eff = phiOMaxFrac*(1 - f_in)
            argz=[M,phiE,phiRMin,phi0,a,G0,P,epsilonF,epsilonO,eF,eO,eB,eP,beta,rhoE,rhoP,d,fC,kt,phiOMaxFrac,switch]

            if PhiOBalanced(argz) < gamma_eff * phiE:
                phiR2=PhiRBalanced(argz,False)
                phiF=PhiFBalanced(argz,False)
                phiO=PhiOBalanced(argz)
                sols=[G0,kt*(phiR2-phiRMin)*Crowding(M,phiE,beta,rhoE,d,switch),M,phiE,phiR2,phiF,phiO,0,1,efficiency(M,phiE,phiR2,phiRMin,phiF,phiO,0,a,G0,P,epsilonF,epsilonO,eF,eO,eB,eP,beta,rhoE,rhoP,d,fC,kt,switch)]
            else:
                argz_ov=[M,phiE,phiRMin,phi0,a,G0,P,epsilonF,epsilonO,eF,eO,eB,eP,beta,rhoE,rhoP,d,fC,kt,gamma_eff,switch]
                phiR2=PhiRBalanced(argz_ov,True)
                phiF=PhiFBalanced(argz_ov,True)
                phiO = gamma_eff * phiE
                JOut=JOutBalanced(argz_ov)
                sols=[G0,kt*(phiR2-phiRMin)*Crowding(M,phiE,beta,rhoE,d,switch),M,phiE,phiR2,phiF,phiO,JOut,1,efficiency(M,phiE,phiR2,phiRMin,phiF,phiO,JOut,a,G0,P,epsilonF,epsilonO,eF,eO,eB,eP,beta,rhoE,rhoP,d,fC,kt,switch)]
            if len(solsTemp)>=1:
                if (sols[4]-phiR)**2<=(solsTemp[0][4]-phiR)**2:
                    solsTemp[0]=sols
            else:
                solsTemp.append(sols)
        solFinal.append(solsTemp[0])
    return solFinal
        

def optimizedNormalV(x,phiRMin,phi0,a,G0,P,epsilonF,epsilonO,eF,eO,eB,eP,beta,rhoE,rhoP,d,fC,kt,phiOMaxFrac,switch):
    M,phiE=x
    #phiRMin,phi0,a,G0,P,epsilonF,epsilonO,eF,eO,eB,eP,beta,rhoE,rhoP,d,fC,kt,phiOMaxFrac = args

    argz = [M,phiE,phiRMin,phi0,a,G0,P,epsilonF,epsilonO,eF,eO,eB,eP,beta,rhoE,rhoP,d,fC,kt,phiOMaxFrac,switch]
    phiR=PhiRBalanced(argz,False)
    phiF=PhiFBalanced(argz,False)
    phiO=PhiOBalanced(argz)

    if phiR+phiF+phiE+phi0>1 or phiR<0 or phiF<0 or phiO<0 or kt*(phiR-phiRMin)*Crowding(M,phiE,beta,rhoE,d,switch)<0.10:
        return 10000
    return -((M*phiE/rhoE)**3/(36*np.pi))**0.5

def optimizedOverflowV(x,phiRMin,phi0,a,G0,P,epsilonF,epsilonO,eF,eO,eB,eP,beta,rhoE,rhoP,d,fC,kt,phiOMaxFrac,switch):
    M,phiE=x

    gamma_eff = phiOMaxFrac*(1 - f_in)
    argz = [M,phiE,phiRMin,phi0,a,G0,P,epsilonF,epsilonO,eF,eO,eB,eP,beta,rhoE,rhoP,d,fC,kt,gamma_eff,switch]
    phiR=PhiRBalanced(argz,True)
    phiF=PhiFBalanced(argz,True)
    phiO=gamma_eff*phiE
    JOut=JOutBalanced(argz)
    if phiR+phiF+phiE+phi0>1 or phiR<0 or phiF<0 or JOut<0 or kt*(phiR-phiRMin)*Crowding(M,phiE,beta,rhoE,d,switch)<0.10:
        return 10000
    return -((M*phiE/rhoE)**3/(36*np.pi))**0.5  

def G0OptimizerV(args,res,G0min,G0max,phiEmin,phiEmax,Mmin,Mmax):       
    phiRMin,phi0,a,P,epsilonF,epsilonO,eF,eO,eB,eP,beta,rhoE,rhoP,d,fC,kt,phiOMaxFrac,switch = args
    #bounds=Bounds([Mmin,Mmax],[phiEmin,phiEmax])
    sols=[]
    for G0 in np.linspace(start=G0min, stop=G0max, num=res):
        argz=(phiRMin,phi0,a,G0,P,epsilonF,epsilonO,eF,eO,eB,eP,beta,rhoE,rhoP,d,fC,kt,phiOMaxFrac,switch)
        #solNormal=minimize(optimizedNormal, [1.7*10**5*0.55,(phiEmin+phiEmax)/2], args=argz,
        #                   method='BFGS', bounds=((Mmin,Mmax),(phiEmin,phiEmax)))
        #solOverflow=minimize(optimizedOverflow, [1.7*10**5*0.55,(phiEmin+phiEmax)/2], args=argz,
        #                   method='BFGS', bounds=((Mmin,Mmax),(phiEmin,phiEmax)))
        solNormal=brute(optimizedNormalV,((Mmin,Mmax),(phiEmin,phiEmax)), args=argz,Ns=2500,finish=None)
        #print(solNormal)
        gamma_eff = phiOMaxFrac*(1 - f_in)
        if PhiOBalanced([solNormal[0],solNormal[1],phiRMin,phi0,a,G0,P,epsilonF,epsilonO,eF,eO,eB,eP,beta,rhoE,rhoP,d,fC,kt,phiOMaxFrac,switch])<gamma_eff*solNormal[1]:
        #optimizedNormal(solNormal,phiRMin,phi0,a,G0,P,epsilonF,epsilonO,eF,eO,eB,eP,beta,rhoE,rhoP,d,fC,kt,phiOMaxFrac)<optimizedOverflow(solOverflow,phiRMin,phi0,a,G0,P,epsilonF,epsilonO,eF,eO,eB,eP,beta,rhoE,rhoP,d,fC,kt,phiOMaxFrac):
            
            argzz=[solNormal[0],solNormal[1],phiRMin,phi0,a,G0,P,epsilonF,epsilonO,eF,eO,eB,eP,beta,rhoE,rhoP,d,fC,kt,phiOMaxFrac,switch]
            phiR=PhiRBalanced(argzz,False)
            phiF=PhiFBalanced(argzz,False)
            phiO=PhiOBalanced(argzz)
            sols.append([G0,kt*(phiR-phiRMin)*Crowding(solNormal[0],solNormal[1],beta,rhoE,d,switch),solNormal[0],solNormal[1],phiR,phiF,phiO,0,Crowding(solNormal[0],solNormal[1],beta,rhoE,d,switch)])
        else:            
            solOverflow=brute(optimizedOverflowV,((Mmin,Mmax),(phiEmin,phiEmax)), args=argz,Ns=2500,finish=None)

            argzz_ov=[solOverflow[0],solOverflow[1],phiRMin,phi0,a,G0,P,epsilonF,epsilonO,eF,eO,eB,eP,beta,rhoE,rhoP,d,fC,kt,gamma_eff,switch]
            phiR=PhiRBalanced(argzz_ov,True)
            phiF=PhiFBalanced(argzz_ov,True)
            phiO=gamma_eff*solOverflow[1]
            JOut=JOutBalanced(argzz_ov)
            sols.append([G0,kt*(phiR-phiRMin)*Crowding(solOverflow[0],solNormal[1],beta,rhoE,d,switch),solOverflow[0],solNormal[1],phiR,phiF,phiO,JOut,Crowding(solOverflow[0],solNormal[1],beta,rhoE,d,switch)])
    return sols

def optimizedNormalOnlyV(x,phiRMin,phi0,a,kappa,P,epsilonF,epsilonO,eF,eO,eB,eP,beta,rhoE,rhoP,d,fC,kt,phiOMaxFrac):
    M,phiE,G0=x
    #phiRMin,phi0,a,G0,P,epsilonF,epsilonO,eF,eO,eB,eP,beta,rhoE,rhoP,d,fC,kt,phiOMaxFrac = args
    argz = [M,phiE,phiRMin,phi0,a,G0,P,epsilonF,epsilonO,eF,eO,eB,eP,beta,rhoE,rhoP,d,fC,kt,phiOMaxFrac]
    phiR=PhiRBalanced(argz,False)
    phiF=PhiFBalanced(argz,False)
    phiO=PhiOBalanced(argz)
    if phiR+phiF+phiE+phi0>1 or phiR<0 or phiF<0 or phiO<0 or phiO>phiOMaxFrac*(1-f_in)*phiE or kt*(phiR-phiRMin)*Crowding(M,phiE,beta,rhoE,d)<kappa:
        return 10000
    return -((M*phiE/rhoE)**3/(36*np.pi))**0.5

def G0OptimizerVNormal(args,res,G0min,G0max,phiEmin,phiEmax,Mmin,Mmax,kappa):       
    phiRMin,phi0,a,P,epsilonF,epsilonO,eF,eO,eB,eP,beta,rhoE,rhoP,d,fC,kt,phiOMaxFrac = args
    #bounds=Bounds([Mmin,Mmax],[phiEmin,phiEmax])
    sols=[]
    for k in kappa:
        argz=(phiRMin,phi0,a,k,P,epsilonF,epsilonO,eF,eO,eB,eP,beta,rhoE,rhoP,d,fC,kt,phiOMaxFrac)
        solNormal=brute(optimizedNormalOnlyV,((Mmin,Mmax),(phiEmin,phiEmax),(G0min,G0max)), args=argz,Ns=500,finish=None)
        G0=solNormal[2] 

        argzz=[solNormal[0],solNormal[1],phiRMin,phi0,a,G0,P,epsilonF,epsilonO,eF,eO,eB,eP,beta,rhoE,rhoP,d,fC,kt,phiOMaxFrac]
        phiR=PhiRBalanced(argzz,False)
        phiF=PhiFBalanced(argzz,False)
        phiO=PhiOBalanced(argzz)
        sols.append([G0,kt*(phiR-phiRMin)*Crowding(solNormal[0],solNormal[1],beta,rhoE,d),solNormal[0],solNormal[1],phiR,phiF,phiO,0,Crowding(solNormal[0],solNormal[1],beta,rhoE,d)])
    return sols

def optimizedNormalMito(x,phiRMin,phi0,a,kappa,P,epsilonF,epsilonO,eF,eO,eB,eP,beta,rhoE,rhoP,d,fC,kt,phiOMaxFrac):
    M,phiE,G0=x
    #phiRMin,phi0,a,G0,P,epsilonF,epsilonO,eF,eO,eB,eP,beta,rhoE,rhoP,d,fC,kt,phiOMaxFrac = args

    argz = [M,phiE,phiRMin,phi0,a,G0,P,epsilonF,epsilonO,eF,eO,eB,eP,beta,rhoE,rhoP,d,fC,kt,phiOMaxFrac]
    phiR=PhiRBalanced(argz,False)
    phiF=PhiFBalanced(argz,False)
    phiO=PhiOBalanced(argz)
    if phiR+phiF+phiE+phi0>1 or phiR<0 or phiF<0 or phiO<0 or kt*(phiR-phiRMin)*Crowding(M,phiE,beta,rhoE,d)<kappa:
        return 10000
    return -((M*phiE/rhoE)**3/(36*np.pi))**0.5

def OptimizerVNormalMito(args,res,G0min,G0max,phiEmin,phiEmax,Mmin,Mmax,kappa):       
    phiRMin,phi0,a,P,epsilonF,epsilonO,eF,eO,eB,eP,beta,rhoE,rhoP,d,fC,kt,phiOMaxFrac = args
    #bounds=Bounds([Mmin,Mmax],[phiEmin,phiEmax])
    sols=[]
    for k in kappa:
        argz=(phiRMin,phi0,a,k,P,epsilonF,epsilonO,eF,eO,eB,eP,beta,rhoE,rhoP,d,fC,kt,phiOMaxFrac)
        solNormal=brute(optimizedNormalMito,((Mmin,Mmax),(phiEmin,phiEmax),(G0min,G0max)), args=argz,Ns=400,finish=None)
        G0=solNormal[2] 

        argzz=[solNormal[0],solNormal[1],phiRMin,phi0,a,G0,P,epsilonF,epsilonO,eF,eO,eB,eP,beta,rhoE,rhoP,d,fC,kt,phiOMaxFrac]
        phiR=PhiRBalanced(argzz,False)
        phiF=PhiFBalanced(argzz,False)
        phiO=PhiOBalanced(argzz)
        sols.append([G0,kt*(phiR-phiRMin)*Crowding(solNormal[0],solNormal[1],beta,rhoE,d),solNormal[0],solNormal[1],phiR,phiF,phiO,0,Crowding(solNormal[0],solNormal[1],beta,rhoE,d)])
    return sols

def optimizedOverflowMito(x,phiRMin,phi0,a,kappa,P,epsilonF,epsilonO,eF,eO,eB,eP,beta,rhoE,rhoP,d,fC,kt,phiOMaxFrac):
    M,phiE,G0=x

    gamma_eff = phiOMaxFrac*(1 - f_in)
    #phiRMin,phi0,a,G0,P,epsilonF,epsilonO,eF,eO,eB,eP,beta,rhoE,rhoP,d,fC,kt,phiOMaxFrac = args
    argz = [M,phiE,phiRMin,phi0,a,G0,P,epsilonF,epsilonO,eF,eO,eB,eP,beta,rhoE,rhoP,d,fC,kt,gamma_eff]
    phiR=PhiRBalanced(argz,True)
    phiF=PhiFBalanced(argz,True)
    phiO=gamma_eff*phiE
    JOut=JOutBalanced(argz)
    if phiR+phiF+phiE+phi0>1 or phiR<0 or phiF<0 or JOut<0 or kt*(phiR-phiRMin)*Crowding(M,phiE,beta,rhoE,d)<kappa:
        return 10000
    return -((M*phiE/rhoE)**3/(36*np.pi))**0.5

def OptimizerVOverflowMito(args,res,G0min,G0max,phiEmin,phiEmax,Mmin,Mmax,kappa):       
    phiRMin,phi0,a,P,epsilonF,epsilonO,eF,eO,eB,eP,beta,rhoE,rhoP,d,fC,kt,phiOMaxFrac = args
    #bounds=Bounds([Mmin,Mmax],[phiEmin,phiEmax])
    sols=[]
    for k in kappa:
        argz=(phiRMin,phi0,a,k,P,epsilonF,epsilonO,eF,eO,eB,eP,beta,rhoE,rhoP,d,fC,kt,phiOMaxFrac)
        solNormal=brute(optimizedOverflowMito,((Mmin,Mmax),(phiEmin,phiEmax),(G0min,G0max)), args=argz,Ns=200,finish=None)
        G0=solNormal[2] 

        argzz=[solNormal[0],solNormal[1],phiRMin,phi0,a,G0,P,epsilonF,epsilonO,eF,eO,eB,eP,beta,rhoE,rhoP,d,fC,kt,phiOMaxFrac]
        phiR=PhiRBalanced(argzz,False)
        phiF=PhiFBalanced(argzz,False)
        phiO=PhiOBalanced(argzz)
        sols.append([G0,kt*(phiR-phiRMin)*Crowding(solNormal[0],solNormal[1],beta,rhoE,d),solNormal[0],solNormal[1],phiR,phiF,phiO,0,Crowding(solNormal[0],solNormal[1],beta,rhoE,d)])
    return sols


###############################################################################

def ProteomePlot(sols,runName):
    G0List=[]
    phiRList=[]
    phiEList=[]
    phiFList=[]
    phiOList=[]
    phi0List=[]
    for i in range(len(sols)):
        G0List.append(sols[i][0] * CONV_G0_TO_uM)
        phiRList.append(sols[i][4])
        phiEList.append(sols[i][3])
        phiFList.append(sols[i][5])
        phiOList.append(sols[i][6])
        phi0List.append(1-sols[i][3]-sols[i][5]-sols[i][4])
    proteomePlot=plt.figure(figsize=(5, 5))
    plt.plot(G0List,phiRList,color='red')
    plt.plot(G0List,phiEList,color='green')
    plt.plot(G0List,phiFList,color='blue')
    plt.plot(G0List,phiOList,color='green',linestyle='dashed')
    plt.plot(G0List,phi0List,color='black')
    threshold = max(sols[j][7] for j in range(len(sols))) * 0.01
    overflow_G0 = next((G0List[j] for j in range(len(G0List)) if sols[j][7] > threshold), None)
    if overflow_G0:
        plt.axvline(x=overflow_G0, color='gray', linestyle='--', linewidth=1)
    plt.xlabel(r'Ext. glucose conc. $[G_0]$ [$\mu$M]',fontsize=18)
    plt.ylabel('Proteome mass fraction',fontsize=18)   
    plt.margins(x=0,y=0.05)        
    plt.xticks(fontsize=12)
    plt.yticks(fontsize=12)
    #plt.axvspan(8806.86 * CONV_G0_TO_uM, np.max(G0List), facecolor='0.05', alpha=0.1)
    plt.legend([r'Ribosomes ($\phi_R$)',r'Envelope ($\phi_E$)',r'Glycolytic ($\phi_G$)',r'Respiration ($\phi_O$)',r'Other ($\phi_0$)'],loc='upper right', fontsize=12)
    plt.savefig(runName+'Proteome.svg',bbox_inches='tight')
    
def ProteomePlotKappa(sols,runName):
    G0List=[]
    phiRList=[]
    phiEList=[]
    phiFList=[]
    phiOList=[]
    phi0List=[]
    kappaList=[]
    for i in range(len(sols)):
        G0List.append(sols[i][0] * CONV_G0_TO_uM)
        phiRList.append(sols[i][4])
        phiEList.append(sols[i][3])
        phiFList.append(sols[i][5])
        phiOList.append(sols[i][6])
        phi0List.append(1-sols[i][3]-sols[i][5]-sols[i][4])       
        kappaList.append(sols[i][1])
    proteomePlot=plt.figure(figsize=(5, 5))
    plt.plot(kappaList,phiRList,color='red')
    plt.plot(kappaList,phiEList,color='green')
    plt.plot(kappaList,phiFList,color='blue')
    plt.plot(kappaList,phiOList,color='green',linestyle='dashed')
    plt.plot(kappaList,phi0List,color='black')
    plt.xlabel(r'$\kappa$ $[h^{-1}] $',fontsize=14)
    plt.ylabel('Proteome Fraction',fontsize=14)   
    plt.margins(x=0,y=0.05)
    plt.legend([r'$\phi_R$',r'$\phi_E$',r'$\phi_F$',r'$\phi_O$',r'$\phi_0$'],loc='upper right')
    plt.axvspan(0.756, np.max(kappaList), facecolor='0.05', alpha=0.1)
    plt.savefig(runName+'ProteomeKappa.svg',bbox_inches='tight')

def KappaPlot(sols,runName):
    G0List=[]
    kappaList=[]
    for i in range(len(sols)):
        G0List.append(sols[i][0] * CONV_G0_TO_uM)
        kappaList.append(sols[i][1])
    kappaPlot=plt.figure(figsize=(5, 5))
    plt.plot(G0List,kappaList,color='black')
    threshold = max(sols[j][7] for j in range(len(sols))) * 0.01
    overflow_G0 = next((G0List[j] for j in range(len(G0List)) if sols[j][7] > threshold), None)
    if overflow_G0:
        plt.axvline(x=overflow_G0, color='gray', linestyle='--', linewidth=1)
    plt.xlabel(r'Ext. glucose conc. $[G_0]$ [$\mu$M]',fontsize=18)
    plt.ylabel(r'Growth rate $\kappa$ $[h^{-1}] $',fontsize=18)   
    plt.margins(x=0,y=0.05)  
    #plt.axvspan(8806.86 * CONV_G0_TO_uM, np.max(G0List), facecolor='0.05', alpha=0.1)
    plt.savefig(runName+'GrowthRate.svg',bbox_inches='tight')
    
def MPlot(sols,runName):
    G0List=[]
    MList=[]
    for i in range(len(sols)):
        G0List.append(sols[i][0] * CONV_G0_TO_uM)
        MList.append(sols[i][2])
    mPlot=plt.figure(figsize=(5, 5))
    plt.plot(G0List,MList,color='black')
    plt.xlabel(r'$[G_0]$ $[\mu m^{-3}] $',fontsize=14)
    plt.ylabel('M [MDa]',fontsize=14)   
    plt.savefig(runName+'Mass.svg',bbox_inches='tight')
    
def OutPlot(sols,runName):
    G0List=[]
    JList=[]
    for i in range(len(sols)):
        G0List.append(sols[i][0] * CONV_G0_TO_uM)
        JList.append(sols[i][7])
    jPlot=plt.figure(figsize=(5, 5))
    plt.plot(G0List,JList,color='black')
    threshold = max(JList) * 0.01
    overflow_G0 = next((G0List[j] for j in range(len(JList)) if JList[j] > threshold), None)
    if overflow_G0:
        plt.axvline(x=overflow_G0, color='gray', linestyle='--', linewidth=1)
    plt.xlabel(r'Ext. glucose conc. $[G_0]$ [$\mu$M]',fontsize=18)
    plt.ylabel(r'$J_{out}$ [Acetate/h]',fontsize=18)   
    plt.margins(x=0,y=0.05)    
    #plt.axvspan(8806.86 * CONV_G0_TO_uM, np.max(G0List), facecolor='0.05', alpha=0.1)
    plt.xticks(fontsize=12)
    plt.yticks(fontsize=12)
    plt.ticklabel_format(axis='y', style='sci', scilimits=(0, 0), useMathText=True)
    plt.savefig(runName+'OutRate.svg',bbox_inches='tight')

def EfficiencyPlot(sols,runName):
    G0List=[]
    eList=[]
    for i in range(len(sols)):
        G0List.append(sols[i][0] * CONV_G0_TO_uM)
        eList.append(sols[i][9])
    ePlot=plt.figure(figsize=(5, 5))
    plt.plot(G0List,eList,color='black')
    plt.xlabel(r'$[G_0]$ $[\mu m^{-3}] $',fontsize=18)
    plt.ylabel(r'Efficiency',fontsize=18)   
    plt.xticks(fontsize=12)
    plt.yticks(fontsize=12)
    plt.savefig(runName+'Efficiency.svg',bbox_inches='tight')
    
def CrowdPlot(sols,runName):
    G0List=[]
    crowdList=[]
    for i in range(len(sols)):
        G0List.append(sols[i][0] * CONV_G0_TO_uM)
        crowdList.append(sols[i][8])
    jPlot=plt.figure(figsize=(5, 5))
    plt.plot(G0List,crowdList,color='black')
    plt.xlabel(r'$[G_0]$ $[\mu m^{-3}] $',fontsize=14)
    plt.ylabel(r'$\beta/T$',fontsize=14)   
    plt.savefig(runName+'Crowding.svg',bbox_inches='tight')
        
def EnergyPlot(sols,args,runName):
    phiRMin,phi0,a,P,epsilonF,epsilonO,eF,eO,eB,eP,beta,rhoE,rhoP,d,fC,kt,phiOMaxFrac,rhoM,switch=args
    G0List=[]
    JFList=[]
    JOList=[]
    JOutList=[]
    JBList=[]
    JPList=[]
    maitList=[]
    for i in range(len(sols)):
        G0List.append(sols[i][0] * CONV_G0_TO_uM)
        M=sols[i][2]
        phiE=sols[i][3]
        phiR=sols[i][4]       
        phiF=sols[i][5]
        phiO=sols[i][6]
        JFList.append(JF(M,phiE,phiF,beta,rhoE,d,epsilonF,switch))
        JOList.append(JO(M,phiE,phiO,beta,rhoE,d,epsilonO,switch))
        JOutList.append(sols[i][7])
        JBList.append(-eBtotal*JB(M,phiE,phiR,phiRMin,beta,rhoE,d,fC,kt,switch))
        JPList.append(-eP*JP(M,phiE,phiR,phiRMin,beta,rhoE,rhoP,d,kt,switch))
        maitList.append(-JMait(M,a))
    energyPlot=plt.figure(figsize=(5, 5))
    overflow_G0 = next((G0List[j] for j in range(len(JOutList)) if JOutList[j] > 0), None)
    if overflow_G0:
        plt.axvline(x=overflow_G0, color='gray', linestyle='--', linewidth=1, label='_nolegend_')
    plt.plot(G0List,JFList,color='blue')
    plt.plot(G0List,JOList,color='green')
    plt.plot(G0List,JOutList,color='cyan') 
    plt.plot(G0List,JPList,color='orange')  
    plt.plot(G0List,maitList,color='purple')
    plt.plot(G0List,JBList,color='red')
    plt.plot(G0List,[0 for i in G0List],color='black',linestyle='dashed',zorder=0)
    #plt.legend(['Glycolysis → acetyl-CoA','Respiration TCA','Acetate Excretion','Lipid Synthesis','Maintenance','Translation'],loc='lower left',fontsize=12)
    plt.legend([
    r'Glycolysis $\rightarrow$ acetyl-CoA $(J_G)$',
    r'Respiration TCA $(J_O)$',
    r'Acetate Excretion $(J_\mathrm{out})$',
    r'Lipid Synthesis $(J_P)$',
    r'Maintenance $(J_\mathrm{mait})$',
    r'Translation $(J_B)$'], loc='lower left', fontsize=12)
    plt.xticks(fontsize=12)
    plt.yticks(fontsize=12)
    plt.ticklabel_format(axis='y', style='sci', scilimits=(0, 0), useMathText=True)
    plt.xlabel(r'Ext. glucose conc. $[G_0]$ [$\mu$M]',fontsize=18)
    plt.ylabel('Flux [ATP/h]',fontsize=18)   
    plt.margins(x=0,y=0.05)
    #plt.axvspan(8806.86 * CONV_G0_TO_uM, np.max(G0List), facecolor='0.05', alpha=0.1)
    plt.tight_layout()
    plt.savefig(runName+'Energy.svg')#,bbox_inches='tight'
            
def CarbonPlot(sols,args,runName):
    phiRMin,phi0,a,P,epsilonF,epsilonO,eF,eO,eB,eP,beta,rhoE,rhoP,d,fC,kt,phiOMaxFrac=args
    G0List=[]
    JInList=[]
    JFList=[]
    JBList=[]
    for i in range(len(sols)):
        G0List.append(sols[i][0] * CONV_G0_TO_uM)
        M=sols[i][2]
        phiE=sols[i][3]
        phiR=sols[i][4]       
        phiF=sols[i][5]
        phiO=sols[i][6]
        JBList.append(-JB(M,phiE,phiR,phiRMin,beta,rhoE,d,fC,kt))
        JFList.append(-JF(M,phiE,phiF,beta,rhoE,d,epsilonF)/eF)
        JInList.append(JIn(M,phiE,sols[i][0],P,rhoE,K_ptsg))
    carbonPlot=plt.figure(figsize=(5, 5))
    plt.plot(G0List,JBList,color='red')
    plt.plot(G0List,JFList,color='blue')
    plt.plot(G0List,JInList,color='black')
    plt.xlabel(r'$[G_0]$ $[\mu m^{-3}] $',fontsize=14)
    plt.ylabel('C/h',fontsize=14)   
    plt.savefig(runName+'Carbon.svg',bbox_inches='tight')    
    
def VPlot(sols,args,runName):
    phiRMin,phi0,a,P,epsilonF,epsilonO,eF,eO,eB,eP,beta,rhoE,rhoP,d,fC,kt,phiOMaxFrac,switch=args
    G0List=[]
    VList=[]
    for i in range(len(sols)):
        G0List.append(sols[i][0] * CONV_G0_TO_uM)
        VList.append(((sols[i][3]*sols[i][2]/rhoE)**3/(36*np.pi))**0.5)
    VPlot=plt.figure(figsize=(5, 5))
    plt.plot(G0List,VList,color='black')
    plt.xlabel(r'Ext. glucose conc. $[G_0]$ [$\mu$M]',fontsize=18)
    plt.ylabel(r'Volume V $[\mu m^3]$',fontsize=18)   
    plt.margins(x=0,y=0.05)  
    plt.xticks(fontsize=12)
    plt.yticks(fontsize=12)
    plt.locator_params(axis='x', nbins=7)
    plt.savefig(runName+'Volume.svg',bbox_inches='tight')
    
def rhoPlot(sols,args,runName,density):
    phiRMin,phi0,a,P,epsilonF,epsilonO,eF,eO,eB,eP,beta,rhoE,rhoP,d,fC,kt,phiOMaxFrac=args
    G0List=[]
    rhoList=[]
    rhoCalList=[]
    for i in range(len(sols)):
        G0List.append(sols[i][0] * CONV_G0_TO_uM)
        rhoList.append(sols[i][2]/(((sols[i][3]*sols[i][2]/rhoE)**3/(36*np.pi))**0.5))
        rhoCalList.append(density)
    rhoPlot=plt.figure(figsize=(5, 5))
    plt.plot(G0List,rhoList,color='black')
    plt.plot(G0List,rhoCalList,color='black',linestyle='dashed')
    plt.xlabel(r'$[G_0]$ $[\mu m^{-3}] $',fontsize=14)
    plt.ylabel(r'$\rho$ $[MDa\mu$ $m^{-3}]$',fontsize=14)   
    plt.savefig(runName+'Mass.svg',bbox_inches='tight')
    

def KappaPlot2(sols,runName):
    G0List=[]
    kappaList=[]
    for i in range(len(sols)):
        G0List.append(sols[i][0] * CONV_G0_TO_uM)
        kappaList.append(sols[i][1])
    kappaPlot=plt.figure(figsize=(5, 5))
    plt.plot(G0List,kappaList,color='black')
    plt.xlabel(r'$[G_0]$ $[\mu m^{-3}] $',fontsize=14)
    plt.ylabel(r'$\kappa$ $[h^{-1}] $',fontsize=14)   
    plt.margins(x=0,y=0.05)  
    plt.savefig(runName+'GrowthRate.svg',bbox_inches='tight')
    
def ProteomePlot2(sols,runName):
    G0List=[]
    phiRList=[]
    phiEList=[]
    phiFList=[]
    phiOList=[]
    phi0List=[]
    for i in range(len(sols)):
        G0List.append(sols[i][0] * CONV_G0_TO_uM)
        phiRList.append(sols[i][4])
        phiEList.append(sols[i][3])
        phiFList.append(sols[i][5])
        phiOList.append(sols[i][6])
        phi0List.append(1-sols[i][3]-sols[i][5]-sols[i][4])
    proteomePlot=plt.figure(figsize=(5, 5))
    plt.plot(G0List,phiRList,color='red')
    plt.plot(G0List,phiEList,color='green')
    plt.plot(G0List,phiFList,color='blue')
    plt.plot(G0List,phiOList,color='green',linestyle='dashed')
    plt.plot(G0List,phi0List,color='black')
    plt.xlabel(r'Ext. glucose conc. $[G_0]$ [$\mu$M]',fontsize=18)
    plt.ylabel('Proteome mass fraction',fontsize=18)   
    plt.margins(x=0,y=0.05)        
    plt.xticks(fontsize=12)
    plt.yticks(fontsize=12)
    plt.legend([r'Ribosomes ($\phi_R$)',r'Envelope ($\phi_E$)',r'Glycolysis → Acetyl-CoA ($\phi_G$)',r'Respiration TCA ($\phi_O$)',r'Other ($\phi_0$)'],loc='upper right', fontsize=12)
    plt.locator_params(axis='x', nbins=7)
    plt.savefig(runName+'Proteome.svg',bbox_inches='tight')
    