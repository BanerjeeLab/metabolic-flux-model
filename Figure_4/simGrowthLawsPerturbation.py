import numpy as np
import methods22 as m
import matplotlib.pyplot as plt

##################### Initialize parameters ########################
phi0=0.45;
epsilonO=1.8*10**6#0.14*1.8*10**6
epsilonF= 0.85* 1.92*1.8*10**6
fC=0.47
kt=5.9
P=12000
eF=5/3
eO=8;
eB=23.5/(fC*1.1*10**-4)
#eGAM = 8/(fC*1.1*10**-4)
rhoE=2400#4900
phiRMin=0.07
eP=21/8;
rhoP=4.5*10**6
a=8390#19700#5595
rhoM=1.8*10**5*0.55#1.7*10**5
d=28.7*10**-3
phiOMaxFrac=0.39



### Conversion scheme ###
CONV_G0_TO_uM = 0.001660539067 #gives uM scaling factor that corresponds to 1 molecule per cubic micron
CONV_num_per_um3_from_uM = 1 / CONV_G0_TO_uM


################## Run optimization #####################################

## beta unused: crowding is off (switch=False) for Fig. 4, so beta isn't used here. See README.
beta=None

#crowding function switched off with False

args=phiRMin,phi0,a,P,epsilonF,epsilonO,eF,eO,eB,eP,beta,rhoE,rhoP,d,fC,kt,phiOMaxFrac,rhoM,False

scaleList1=[1,1,1,1,1]#[0.5,0.75,1,1.2,1.4]
scaleList2=[0.5,0.75,1,1.25,1.5]#[1,1,1,1,1]
scaleList=[[scaleList1[i],scaleList2[i],1,1] for i in range(len(scaleList1))]

colors=["green","blue","black","orange","red"]

results=[]
G0List=[]
eList=[]
kappaList=[]
MList=[]
phiRList=[]
for i in range(len(scaleList)):
    results.append(m.G0Determined(args,1500,2000,30000,0.55,scaleList[i]))#1000
    #print(results[0])
#OptimizerVNormalMito(args,5,3000,25000,0.01,2*phiEcal,0.1*Mcal,10*Mcal,[0.05,0.1,0.15,0.2])
#results=m.OptimizerVOverflowMito(args,5,3000,25000,0.01,2*phiEcal,0.1*Mcal,10*Mcal,[0.1,0.15,0.2])
    G0ListTemp=[]
    eListTemp=[]
    kappaListTemp=[]
    MListTemp=[]
    phiRListTemp=[]
    for j in range(len(results[i])):
        G0ListTemp.append(results[i][j][0]*CONV_G0_TO_uM)
        eListTemp.append(results[i][j][9])
        kappaListTemp.append(results[i][j][1])
        MListTemp.append(results[i][j][2])
        phiRListTemp.append(results[i][j][4])
    G0List.append(G0ListTemp)
    eList.append(eListTemp)
    kappaList.append(kappaListTemp)
    MList.append(MListTemp)
    phiRList.append(phiRListTemp)
    m.ProteomePlot(results[i],"i")
    

kappaPlot=plt.figure(figsize=(5, 5))
for i in range(len(scaleList)):
    plt.plot(G0List[i],kappaList[i],color=colors[i])
plt.legend([r'$\alpha=0.5$',r'$\alpha=0.75$',r'$\alpha=1$',r'$\alpha=1.25$',r'$\alpha=1.5$'],loc='lower right',fontsize=12)
#plt.text(4000,1.3,r'$S\propto V^{\alpha 2/3}$',fontsize=14)
#plt.text(4000,1.3,r'$S=6.24 \alpha V^{2/3}$',fontsize=14)
plt.xlabel(r'Ext. glucose conc. $[G_0]$ [$\mu$M]',fontsize=18)
plt.ylabel(r'Growth rate $\kappa$ $[h^{-1}] $',fontsize=18)   
plt.margins(x=0,y=0.05)
plt.xticks(fontsize=12)
plt.yticks(fontsize=12)
plt.savefig('4C.svg',bbox_inches='tight')



kappaEPlot=plt.figure(figsize=(5, 5))
for i in range(len(scaleList)):
    plt.plot(kappaList[i],eList[i],color=colors[i])
plt.legend([r'$\alpha=0.5$',r'$\alpha=0.75$',r'$\alpha=1$',r'$\alpha=1.25$',r'$\alpha=1.5$'],loc='lower left',fontsize=12)
#plt.text(1.2,0.9,r'$S\propto V^{\alpha 2/3}$',fontsize=14)
#plt.text(1.2,0.85,r'$S=6.24 \alpha V^{ 2/3}$',fontsize=14)
plt.xlabel(r'Growth rate $\kappa$ $[h^{-1}] $',fontsize=18)
plt.ylabel(r'Efficiency $\mathcal{E}$',fontsize=18)   
plt.margins(x=0,y=0.05)
plt.xticks(fontsize=12)
plt.yticks(fontsize=12)
plt.savefig('4D.svg',bbox_inches='tight')

