import numpy as np
import methodsRodCrowding as m
import methodsRodCrowding2 as m2
import matplotlib.pyplot as plt

#################### Initialize parameters ###########################
phi0=0.45;
epsilonO=1.8*10**6#
epsilonF=0.85 * 1.92*1.8*10**6
fC=0.47
kt=5.9
P=12000
eF=5/3
eO=8;
eB=23.5/(fC*1.1*10**-4)
rhoE=2400#4900
phiRMin=0.07
eP=21/8;
rhoP=4.5*10**6
a=8390
rhoM=1.8*10**5*0.55
d=28.7*10**-3
phiOMaxFrac=0.39

CONV_G0_TO_uM = 0.001660539067 #gives uM scaling factor that corresponds to 1 molecule per cubic micron


##################### Run optimization ############################
beta=1
args=phiRMin,phi0,a,P,epsilonF,epsilonO,eF,eO,eB,eP,beta,rhoE,rhoP,d,fC,kt,phiOMaxFrac,rhoM,False


#colors=["black","green","blue","orange","red"]
colors=["black","green","red"]

G0List=[]
eList=[]
kappaList=[]
MList=[]
phiRList=[]
phiEList=[]


G0ListT=[]
eListT=[]
kappaListT=[]
MListT=[]
phiRListT=[]
phiEListT=[]
                      #args,res,G0min,G0max,phiRmax,scaleFactor
results=m.G0Determined(args,2000,1000,50000,0.55,[1,1,1,1])#400
for j in range(len(results)):
    G0ListT.append(results[j][0])
    eListT.append(results[j][9])
    kappaListT.append(results[j][1])
    MListT.append(results[j][2])
    phiRListT.append(results[j][4])
    phiEListT.append(results[j][3])
G0List.append(G0ListT)
eList.append(eListT)
kappaList.append(kappaListT)
MList.append(MListT)
phiRList.append(phiRListT)
phiEList.append(phiEListT)

#change V only
for perturb in [0.8,1.2]:
    G0ListT=[]
    eListT=[]
    kappaListT=[]
    MListT=[]
    phiRListT=[]
    phiEListT=[]
    for j in range(len(G0List[0])):
        beta=1/m.Crowding(MList[0][j], MList[0][j]/rhoM, phiEList[0][j], 1, rhoE, d, True)
        args=phiRMin,phi0,a,P,epsilonF,epsilonO,eF,eO,eB,eP,beta,rhoE,rhoP,d,fC,kt,phiOMaxFrac,rhoM,True
        #                                                     M             V               S
        results=m.perturbFromGrowthLaw(args,G0List[0][j],MList[0][j]*perturb, MList[0][j]/rhoM,phiEList[0][j]/perturb)
        #print(results)
        G0ListT.append(results[0])
        eListT.append(results[9])
        kappaListT.append(results[1])
        MListT.append(results[2])
        phiRListT.append(results[4])
        phiEListT.append(results[3])
    G0List.append(G0ListT)
    eList.append(eListT)
    kappaList.append(kappaListT)
    MList.append(MListT)
    phiRList.append(phiRListT)
    phiEList.append(phiEListT)

#change S only 
G0List2=[]
eList2=[]
kappaList2=[]
MList2=[]
phiRList2=[]
phiEList2=[]


G0ListT2=[]
eListT2=[]
kappaListT2=[]
MListT2=[]
phiRListT2=[]
phiEListT2=[]

args=phiRMin,phi0,a,P,epsilonF,epsilonO,eF,eO,eB,eP,beta,rhoE,rhoP,d,fC,kt,phiOMaxFrac,rhoM,False            
          #args,res,G0min,G0max,phiRmax,scaleFactor
results=m.G0Determined(args,2000,1000,50000,0.55,[1,1,1,1])#400
for j in range(len(results)):
    G0ListT2.append(results[j][0])
    eListT2.append(results[j][9])
    kappaListT2.append(results[j][1])
    MListT2.append(results[j][2])
    phiRListT2.append(results[j][4])
    phiEListT2.append(results[j][3])
G0List2.append(G0ListT2)
eList2.append(eListT2)
kappaList2.append(kappaListT2)
MList2.append(MListT2)
phiRList2.append(phiRListT2)
phiEList2.append(phiEListT2)

#change V only
for perturb in [0.8,1.2]:
    G0ListT2=[]
    eListT2=[]
    kappaListT2=[]
    MListT2=[]
    phiRListT2=[]
    phiEListT2=[]
    for j in range(len(G0List[0])):
        beta=1/m2.Crowding(MList[0][j], MList[0][j]/rhoM, phiEList[0][j], 1, rhoE, d, True)
        args=phiRMin,phi0,a,P,epsilonF,epsilonO,eF,eO,eB,eP,beta,rhoE,rhoP,d,fC,kt,phiOMaxFrac,rhoM,True
        #                                                     M             V               S
        results=m2.perturbFromGrowthLaw(args,G0List[0][j],MList[0][j]*perturb, MList[0][j]/rhoM,phiEList[0][j]/perturb)
        #print(results)
        G0ListT2.append(results[0])
        eListT2.append(results[9])
        kappaListT2.append(results[1])
        MListT2.append(results[2])
        phiRListT2.append(results[4])
        phiEListT2.append(results[3])
    G0List2.append(G0ListT2)
    eList2.append(eListT2)
    kappaList2.append(kappaListT2)
    MList2.append(MListT2)
    phiRList2.append(phiRListT2)
    phiEList2.append(phiEListT2)

    

kappaPlot=plt.figure(figsize=(5, 5))
for i in [1,0,2]:
    plt.plot(np.array(G0List[i])*CONV_G0_TO_uM,kappaList[i],color=colors[i],linestyle='-')
for i in [1,0,2]:
    plt.plot(np.array(G0List2[i])*CONV_G0_TO_uM,kappaList2[i],color=colors[i],linestyle='dashed')
plt.legend([r'$\rho=0.8 \rho_c$',r'$\rho=\rho_c$',r'$\rho=1.2 \rho_c$'],loc='lower right',fontsize=12)
#plt.text(4000,1.3,r'$S\propto V^{\alpha 2/3}$',fontsize=14)
#plt.text(4000,1.3,r'$S\propto \alpha V^{2/3}$',fontsize=14)
#plt.text(4000,1.3,r'$V\propto e^{1.16\alpha \kappa}$',fontsize=14)
#plt.text(4000,1.3,r'$V\propto \alpha e^{1.16 \kappa}$',fontsize=14)
plt.xlabel(r'Ext. glucose conc. $[G_0]$ $[\mu \mathrm{M}]$',fontsize=18)
plt.ylabel(r'Growth rate $\kappa$ $[h^{-1}] $',fontsize=18)   
plt.margins(x=0,y=0.05)
plt.xticks(fontsize=12)
plt.yticks(fontsize=12)
plt.savefig('5a.svg',bbox_inches='tight')



kappaEPlot=plt.figure(figsize=(5, 5))
for i in [1,0,2]:
    plt.plot(kappaList[i],eList[i],color=colors[i],linestyle='-')
for i in [1,0,2]:
    plt.plot(kappaList2[i],eList2[i],color=colors[i],linestyle='dashed')
plt.legend([r'$\rho=0.8 \rho_c$',r'$\rho=\rho_c$',r'$\rho=1.2 \rho_c$'],loc='lower right',fontsize=12)
#plt.text(1.2,0.9,r'$S\propto V^{\alpha 2/3}$',fontsize=14)
#plt.text(1.2,0.9,r'$S\propto \alpha V^{ 2/3}$',fontsize=14)
#plt.text(1.2,0.9,r'$V\propto e^{1.16\alpha \kappa}$',fontsize=14)
#plt.text(1.2,0.9,r'$V\propto \alpha e^{1.16 \kappa}$',fontsize=14)
plt.xlabel(r'Growth rate $\kappa$ $[h^{-1}] $',fontsize=18)
plt.ylabel(r'Efficiency $\mathcal{E}$',fontsize=18)   
plt.margins(x=0,y=0.05)
plt.xticks(fontsize=12)
plt.yticks(fontsize=12)
plt.savefig('5b.svg',bbox_inches='tight')

