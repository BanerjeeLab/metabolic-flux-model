import numpy as np
from scipy.optimize import minimize
from scipy.optimize import fsolve
from scipy.optimize import Bounds
from scipy.optimize import least_squares
from scipy.optimize import brute
import matplotlib.pyplot as plt
from sympy import var, Eq, solve, re, im

####################### model with internal diffusion constraint ####################

rhoM_value = 1.8*10**5*0.55

f_in = 0.1 #0.2 #fraction of phiE that pertains to glucose transporters phi_in
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


def Crowding(M,V,phiE,beta,rhoE,d,switch):
    if switch:
        x = var('x')
        sol = solve(Eq(-(2/3)*np.pi*x**3 + (phiE*M/(2*rhoE))*x - V, 0), x) 
        realSols=[re(i) for i in sol]
        realSols.remove(max(realSols))
        realSols.remove(min(realSols))
        '''
        for y in sol:
           if re(y)>0:
               if re(y)>=re(r):
                   if np.abs(im(y))<=np.abs(im(r)):
                       r=y
        '''
        r=realSols[0]
        length= V/(np.pi*r**2) + 2*r/3

        #beta here corresponds to betaprime in the manuscript; betaT represents beta/T in the manuscript
        num=np.longdouble(beta*np.exp(-(M/V)/rhoM_value))
        denom=np.longdouble((length-2*d)**2)
        if denom==0:
            return 0
            print('Division by 0')
        else:
            #print(num/denom)
            return np.longdouble(num/denom)
    else:
        return 1



###############
def JIn(M, phiE, G0, P, rhoE, K_ptsg):
    Pprime = P * K_C  
    S = phiE * M / rhoE
    return 6.0 * Pprime * S * (G0 / (G0 + K_ptsg))



def JF(M,V,phiE,phiF,beta,rhoE,d,epsilonF,switch):
    return epsilonF*phiF*M*Crowding(M,V,phiE,beta,rhoE,d,switch)

def JO(M,V,phiE,phiO,beta,rhoE,d,epsilonO,switch):
    return epsilonO*phiO*M*Crowding(M,V,phiE,beta,rhoE,d,switch)

def JB(M,V,phiE,phiR,phiRMin,beta,rhoE,d,fC,kt,switch):
    return fC*kt*(phiR-phiRMin)*M*Crowding(M,V,phiE,beta,rhoE,d,switch)

def JP(M,V,phiE,phiR,phiRMin,beta,rhoE,rhoP,d,kt,switch):
    return 16*rhoP*phiE*M*kt*(phiR-phiRMin)*Crowding(M,V,phiE,beta,rhoE,d,switch)/(np.log(2)*rhoE)

def JMait(M,a):
    return a*M

def FluxBalanceGlucose(M,V,phiE,phiR,phiRMin,phiF,phiO,JOut,a,G0,P,epsilonF,epsilonO,eF,eO,eB,eP,beta,rhoE,rhoP,d,fC,kt,switch):
    return JIn(M,phiE,G0,P,rhoE)-JF(M,V,phiE,phiF,beta,rhoE,d,epsilonF,switch)/eF-JB(M,V,phiE,phiR,phiRMin,beta,rhoE,d,fC,kt,switch)

def FluxBalanceACA(M,V,phiE,phiR,phiRMin,phiF,phiO,JOut,a,G0,P,epsilonF,epsilonO,eF,eO,eB,eP,beta,rhoE,rhoP,d,fC,kt,switch):
    return 2*JF(M,V,phiE,phiF,beta,rhoE,d,epsilonF,switch)/(6*eF)-JO(M,V,phiE,phiO,beta,rhoE,d,epsilonO,switch)/eO-JP(M,V,phiE,phiR,phiRMin,beta,rhoE,rhoP,d,kt,switch)-JOut

def FluxBalanceATP(M,V,phiE,phiR,phiRMin,phiF,phiO,JOut,a,G0,P,epsilonF,epsilonO,eF,eO,eB,eP,beta,rhoE,rhoP,d,fC,kt,switch):
    return JF(M,V,phiE,phiF,beta,rhoE,d,epsilonF,switch)+JO(M,V,phiE,phiO,beta,rhoE,d,epsilonO,switch)+JOut-eBtotal*JB(M,V,phiE,phiR,phiRMin,beta,rhoE,d,fC,kt,switch)-eP*JP(M,V,phiE,phiR,phiRMin,beta,rhoE,rhoP,d,kt,switch)-JMait(M,a)

def efficiency(M,V,phiE,phiR,phiRMin,phiF,phiO,JOut,a,G0,P,epsilonF,epsilonO,eF,eO,eB,eP,beta,rhoE,rhoP,d,fC,kt,switch):
    num=eB*JB(M,V,phiE,phiR,phiRMin,beta,rhoE,d,fC,kt,switch)+eP*JP(M,V,phiE,phiR,phiRMin,beta,rhoE,rhoP,d,kt,switch)
    waste=JOut*8
    denom=JF(M,V,phiE,phiF,beta,rhoE,d,epsilonF,switch)+JO(M,V,phiE,phiO,beta,rhoE,d,epsilonO,switch)+waste
    return num/denom

def PhiRBalanced(args,overflow):
    M,V,phiE,phiRMin,phi0,a,G0,P,epsilonF,epsilonO,eF,eO,eB,eP,beta,rhoE,rhoP,d,fC,kt,phiOMaxFrac,switch=args
    betaT=Crowding(M,V,phiE,beta,rhoE,d,switch)
    if overflow:
        num=betaT*(-epsilonO*phiE*phiOMaxFrac*rhoE*np.log(8)+eO*epsilonO*phiE*phiOMaxFrac*rhoE*np.log(8)+eO*kt*phiRMin*(48*(1+eP)*phiE*rhoP+eBtotal*fC*rhoE*np.log(8)+fC*rhoE*(np.log(2)+eF*np.log(8))))+eO*(18*eF*G0/(G0 + K_ptsg)*Pprime*phiE*np.log(2)-a*rhoE*np.log(8)+G0/(G0 + K_ptsg)*Pprime*phiE*np.log(64))
        denom=betaT*eO*kt*(48*(1+eP)*phiE*rhoP+eBtotal*fC*rhoE*np.log(8)+fC*rhoE*(np.log(2)+eF*np.log(8)))
        return num/denom
    else:
        num=18*eF*G0/(G0 + K_ptsg)*Pprime*phiE*np.log(2)-a*rhoE*np.log(8)+betaT*kt*phiRMin*(48*eP*phiE*rhoP+eO*(48*phiE*rhoP+fC*rhoE*np.log(2))+(eBtotal+eF)*fC*rhoE*np.log(8))+eO*G0/(G0 + K_ptsg)*Pprime*phiE*np.log(64)
        denom=betaT*kt*(48*eP*phiE*rhoP+eO*(48*phiE*rhoP+fC*rhoE*np.log(2))+(eBtotal+eF)*fC*rhoE*np.log(8))
        return num/denom
    
def PhiFBalanced(args,overflow):
    M,V,phiE,phiRMin,phi0,a,G0,P,epsilonF,epsilonO,eF,eO,eB,eP,beta,rhoE,rhoP,d,fC,kt,phiOMaxFrac,switch=args
    betaT=Crowding(M,V,phiE,beta,rhoE,d,switch)
    if overflow:
        num=3*eF*(betaT*epsilonO*fC*phiE*phiOMaxFrac*rhoE**2*np.log(2)+eO*fC*(a-betaT*epsilonO*phiE*phiOMaxFrac)*rhoE**2*np.log(2)+6*eO*G0/(G0 + K_ptsg)*Pprime*phiE*(16*(1+eP)*phiE*rhoP+eBtotal*fC*rhoE*np.log(2)))
        denom=betaT*eO*epsilonF*rhoE*(48*(1+eP)*phiE*rhoP+eBtotal*fC*rhoE*np.log(8)+fC*rhoE*(np.log(2)+eF*np.log(8)))
        return num/denom
    else:
        num=3*eF*(96*eO*G0/(G0 + K_ptsg)*Pprime*phiE**2*rhoP+96*eP*G0/(G0 + K_ptsg)*Pprime*phiE**2*rhoP+fC*rhoE*(6*eBtotal*G0/(G0 + K_ptsg)*Pprime*phiE+a*rhoE)*np.log(2))
        denom=betaT*epsilonF*rhoE*(48*eP*phiE*rhoP+eO*(48*phiE*rhoP+fC*rhoE*np.log(2))+(eBtotal+eF)*fC*rhoE*np.log(8))
        return num/denom
    
def JOutBalanced(args):
    M,V,phiE,phiRMin,phi0,a,G0,P,epsilonF,epsilonO,eF,eO,eB,eP,beta,rhoE,rhoP,d,fC,kt,phiOMaxFrac,switch=args
    betaT=Crowding(M,V,phiE,beta,rhoE,d,switch)
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
    M,V,phiE,phiRMin,phi0,a,G0,P,epsilonF,epsilonO,eF,eO,eB,eP,beta,rhoE,rhoP,d,fC,kt,phiOMaxFrac,switch=args
    betaT=Crowding(M,V,phiE,beta,rhoE,d,switch)
    num=eO*(-288*eF*G0/(G0 + K_ptsg)*Pprime*phiE**2*rhoP+96*eP*G0/(G0 + K_ptsg)*Pprime*phiE**2*rhoP+rhoE*(48*a*phiE*rhoP+a*fC*rhoE*np.log(2)+eBtotal*fC*G0/(G0 + K_ptsg)*Pprime*phiE*np.log(64)))
    denom=betaT*epsilonO*rhoE*(48*eP*phiE*rhoP+eO*(48*phiE*rhoP+fC*rhoE*np.log(2))+(eBtotal+eF)*fC*rhoE*np.log(8))
    return num/denom

def surface(volume,scaleFactor):
    return scaleFactor[1]*6.24*volume**(scaleFactor[0]*2/3)

def volume(kappa,scaleFactor):
    return scaleFactor[3]*0.17*np.exp(1.16*scaleFactor[2]*kappa)

def G0Determined(args,res,G0min,G0max,phiRmax,scaleFactor):       
    phiRMin,phi0,a,P,epsilonF,epsilonO,eF,eO,eB,eP,beta,rhoE,rhoP,d,fC,kt,phiOMaxFrac,rhoM,switch = args
    #bounds=Bounds([Mmin,Mmax],[phiEmin,phiEmax])  
    solFinal=[] 
    for G0 in np.linspace(start=G0min, stop=G0max, num=res):
        solsTemp=[] 
        for phiR in np.linspace(start=0.01, stop=phiRmax, num=res):
            if switch:
                print("Diffusion limitation is not permitted here")
            else:            
                M=rhoM*volume(kt*(phiR-phiRMin),scaleFactor)
                phiE=rhoE*surface(volume(kt*(phiR-phiRMin),scaleFactor),scaleFactor)/M
            sols=[]
            gamma_eff = phiOMaxFrac*(1 - f_in)
            argz=[M,M/rhoM,phiE,phiRMin,phi0,a,G0,P,epsilonF,epsilonO,eF,eO,eB,eP,beta,rhoE,rhoP,d,fC,kt,phiOMaxFrac,switch]
            if PhiOBalanced(argz)<gamma_eff*phiE:
                phiR2=PhiRBalanced(argz,False)
                phiF=PhiFBalanced(argz,False)
                phiO=PhiOBalanced(argz)
                #########################*Crowding(solNormal[0],solNormal[1],beta,rhoE,d),Crowding(solNormal[0],solNormal[1],beta,rhoE,d)
                sols=[G0,kt*(phiR2-phiRMin)*Crowding(M,M/rhoM,phiE,beta,rhoE,d,switch),M,phiE,phiR2,phiF,phiO,0,1,efficiency(M,M/rhoM,phiE,phiR2,phiRMin,phiF,phiO,0,a,G0,P,epsilonF,epsilonO,eF,eO,eB,eP,beta,rhoE,rhoP,d,fC,kt,switch)]
            else:
                argz_ov=[M,M/rhoM,phiE,phiRMin,phi0,a,G0,P,epsilonF,epsilonO,eF,eO,eB,eP,beta,rhoE,rhoP,d,fC,kt,gamma_eff,switch]
                phiR2=PhiRBalanced(argz_ov,True)
                phiF=PhiFBalanced(argz_ov,True)
                phiO=gamma_eff*phiE
                JOut=JOutBalanced(argz_ov)
                sols=[G0,kt*(phiR2-phiRMin)*Crowding(M,M/rhoM,phiE,beta,rhoE,d,switch),M,phiE,phiR2,phiF,phiO,JOut,1,efficiency(M,M/rhoM,phiE,phiR2,phiRMin,phiF,phiO,JOut,a,G0,P,epsilonF,epsilonO,eF,eO,eB,eP,beta,rhoE,rhoP,d,fC,kt,switch)]
            if len(solsTemp)>=1:
                if (sols[4]-phiR)**2<=(solsTemp[0][4]-phiR)**2:
                    solsTemp[0]=sols
            else:
                solsTemp.append(sols)
        solFinal.append(solsTemp[0])
    return solFinal

def perturbFromGrowthLaw(args,G0,M,V,phiE):       
    phiRMin,phi0,a,P,epsilonF,epsilonO,eF,eO,eB,eP,beta,rhoE,rhoP,d,fC,kt,phiOMaxFrac,rhoM,switch = args
    #bounds=Bounds([Mmin,Mmax],[phiEmin,phiEmax])  
    gamma_eff = phiOMaxFrac*(1 - f_in)
    argz=[M,V,phiE,phiRMin,phi0,a,G0,P,epsilonF,epsilonO,eF,eO,eB,eP,beta,rhoE,rhoP,d,fC,kt,phiOMaxFrac,switch]
    if PhiOBalanced(argz)<gamma_eff*phiE:
        phiR2=PhiRBalanced(argz,False)
        phiF=PhiFBalanced(argz,False)
        phiO=PhiOBalanced(argz)
        #########################*Crowding(solNormal[0],solNormal[1],beta,rhoE,d),Crowding(solNormal[0],solNormal[1],beta,rhoE,d)
        sols=[G0,kt*(phiR2-phiRMin)*Crowding(M,V,phiE,beta,rhoE,d,switch),M,phiE,phiR2,phiF,phiO,0,1,efficiency(M,V,phiE,phiR2,phiRMin,phiF,phiO,0,a,G0,P,epsilonF,epsilonO,eF,eO,eB,eP,beta,rhoE,rhoP,d,fC,kt,switch)]
    else:
        argz_ov=[M,V,phiE,phiRMin,phi0,a,G0,P,epsilonF,epsilonO,eF,eO,eB,eP,beta,rhoE,rhoP,d,fC,kt,gamma_eff,switch]
        phiR2=PhiRBalanced(argz_ov,True)
        phiF=PhiFBalanced(argz_ov,True)
        phiO=gamma_eff*phiE
        JOut=JOutBalanced(argz_ov)
        sols=[G0,kt*(phiR2-phiRMin)*Crowding(M,V,phiE,beta,rhoE,d,switch),M,phiE,phiR2,phiF,phiO,JOut,1,efficiency(M,V,phiE,phiR2,phiRMin,phiF,phiO,JOut,a,G0,P,epsilonF,epsilonO,eF,eO,eB,eP,beta,rhoE,rhoP,d,fC,kt,switch)]
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
    plt.xlabel(r'$[G_0]$ $[\mu\mathrm{M}]$',fontsize=14)
    plt.ylabel('Proteome Fraction',fontsize=14)   
    plt.savefig(runName+'Proteome.svg',bbox_inches='tight')
    
def KappaPlot(sols,runName):
    G0List=[]
    kappaList=[]
    for i in range(len(sols)):
        G0List.append(sols[i][0] * CONV_G0_TO_uM)
        kappaList.append(sols[i][1])
    kappaPlot=plt.figure(figsize=(5, 5))
    plt.plot(G0List,kappaList,color='black')
    plt.xlabel(r'$[G_0]$ $[\mu\mathrm{M}]$',fontsize=14)
    plt.ylabel(r'$\kappa$ $[h^{-1}] $',fontsize=14)   
    plt.savefig(runName+'GrowthRate.svg',bbox_inches='tight')
    
def MPlot(sols,runName):
    G0List=[]
    MList=[]
    for i in range(len(sols)):
        G0List.append(sols[i][0] * CONV_G0_TO_uM)
        MList.append(sols[i][2])
    mPlot=plt.figure(figsize=(5, 5))
    plt.plot(G0List,MList,color='black')
    plt.xlabel(r'$[G_0]$ $[\mu\mathrm{M}]$',fontsize=14)
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
    plt.xlabel(r'$[G_0]$ $[\mu\mathrm{M}]$',fontsize=14)
    plt.ylabel(r'$J_{out}$ [Acetate/h]',fontsize=14)   
    plt.savefig(runName+'OutRate.svg',bbox_inches='tight')

def EfficiencyPlot(sols,runName):
    G0List=[]
    eList=[]
    for i in range(len(sols)):
        G0List.append(sols[i][0] * CONV_G0_TO_uM)
        eList.append(sols[i][9])
    ePlot=plt.figure(figsize=(5, 5))
    plt.plot(G0List,eList,color='black')
    plt.xlabel(r'$[G_0]$ $[\mu\mathrm{M}]$',fontsize=14)
    plt.ylabel(r'Efficiency',fontsize=14)   
    plt.savefig(runName+'Efficiency.svg',bbox_inches='tight')
    
def CrowdPlot(sols,runName):
    G0List=[]
    crowdList=[]
    for i in range(len(sols)):
        G0List.append(sols[i][0] * CONV_G0_TO_uM)
        crowdList.append(sols[i][8])
    jPlot=plt.figure(figsize=(5, 5))
    plt.plot(G0List,crowdList,color='black')
    plt.xlabel(r'$[G_0]$ $[\mu\mathrm{M}]$',fontsize=14)
    plt.ylabel(r'$\beta/T$',fontsize=14)   
    plt.savefig(runName+'Crowding.svg',bbox_inches='tight')
        
def EnergyPlot(sols,args,runName):
    phiRMin,phi0,a,P,epsilonF,epsilonO,eF,eO,eB,eP,beta,rhoE,rhoP,d,fC,kt,phiOMaxFrac=args
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
        JFList.append(JF(M,phiE,phiF,beta,rhoE,d,epsilonF))
        JOList.append(JO(M,phiE,phiO,beta,rhoE,d,epsilonO))
        JOutList.append(sols[i][7])
        JBList.append(-eBtotal*JB(M,phiE,phiR,phiRMin,beta,rhoE,d,fC,kt))
        JPList.append(-eP*JP(M,phiE,phiR,phiRMin,beta,rhoE,rhoP,d,kt))
        maitList.append(-JMait(M,a))
    energyPlot=plt.figure(figsize=(5, 5))
    plt.plot(G0List,JBList,color='red')
    plt.plot(G0List,JOList,color='green')
    plt.plot(G0List,JFList,color='blue')
    plt.plot(G0List,JPList,color='yellow')
    plt.plot(G0List,JOutList,color='black')
    plt.plot(G0List,maitList,color='purple')
    plt.xlabel(r'$[G_0]$ $[\mu\mathrm{M}]$',fontsize=14)
    plt.ylabel('ATP/h',fontsize=14)   
    plt.savefig(runName+'Energy.svg',bbox_inches='tight')
            
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
        JInList.append(JIn(M,phiE,sols[i][0],P,rhoE))
    carbonPlot=plt.figure(figsize=(5, 5))
    plt.plot(G0List,JBList,color='red')
    plt.plot(G0List,JFList,color='blue')
    plt.plot(G0List,JInList,color='black')
    plt.xlabel(r'$[G_0]$ $[\mu\mathrm{M}]$',fontsize=14)
    plt.ylabel('C/h',fontsize=14)   
    plt.savefig(runName+'Carbon.svg',bbox_inches='tight')    
    
def VPlot(sols,args,runName):
    phiRMin,phi0,a,P,epsilonF,epsilonO,eF,eO,eB,eP,beta,rhoE,rhoP,d,fC,kt,phiOMaxFrac=args
    G0List=[]
    VList=[]
    for i in range(len(sols)):
        G0List.append(sols[i][0] * CONV_G0_TO_uM)
        VList.append(((sols[i][3]*sols[i][2]/rhoE)**3/(36*np.pi))**0.5)
    VPlot=plt.figure(figsize=(5, 5))
    plt.plot(G0List,VList,color='black')
    plt.xlabel(r'$[G_0]$ $[\mu\mathrm{M}]$',fontsize=14)
    plt.ylabel(r'V $[\mu m^3]$',fontsize=14)   
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
    plt.xlabel(r'$[G_0]$ $[\mu\mathrm{M}]$',fontsize=14)
    plt.ylabel(r'$\rho$ $[MDa\mu$ $m^{-3}]$',fontsize=14)   
    plt.savefig(runName+'Mass.svg',bbox_inches='tight')