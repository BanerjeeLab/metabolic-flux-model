import numpy as np
import methods22 as m
import matplotlib.pyplot as plt

########################## Initialize parameters #############################
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
rhoM=1.8*10**5*0.55
d=28.7*10**-3
phiOMaxFrac=0.39





########################## Run optimization ##########################################

## beta unused: crowding is off (switch=False) here. See README.
beta=None

#crowding function switched off with False

args=phiRMin,phi0,a,P,epsilonF,epsilonO,eF,eO,eB,eP,beta,rhoE,rhoP,d,fC,kt,phiOMaxFrac,rhoM,False

# eGAM fixed at default; varying K_C instead
m.set_eGAM(8/(fC*1.1*10**-4))

K_C_values_uM = [20, 30, 40, 50]   # in uM
colors         = ['green','blue', 'black', 'red']
linestyles     = ['-','-', '-', '-']

kappaEPlot = plt.figure(figsize=(5, 5))
ax2A = kappaEPlot.add_subplot(111)
ax2A.set_xlabel(r'Growth rate $\kappa$ $[h^{-1}] $', fontsize=18)
ax2A.set_ylabel(r'Efficiency $\mathcal{E}$', fontsize=18)
ax2A.margins(x=0, y=0.05)
ax2A.tick_params(labelsize=12)

WVEPlot = plt.figure(figsize=(5, 5))
ax2D = WVEPlot.add_subplot(111)
ax2D.set_xlabel(r'$S/V$ $[\mu m^{-1}] $', fontsize=18)
ax2D.set_ylabel(r'Efficiency $\mathcal{E}$', fontsize=18)
ax2D.margins(x=0, y=0.05)
ax2D.tick_params(labelsize=12)

outRatePlot = plt.figure(figsize=(5, 5))
axOut = outRatePlot.add_subplot(111)
axOut.set_xlabel(r'Ext. glucose conc. $[G_0]$ [$\mu$M]', fontsize=18)
axOut.set_ylabel(r'$J_{out}$ [Acetate/h]', fontsize=18)
axOut.margins(x=0, y=0.05)
axOut.tick_params(labelsize=12)

growthRatePlot = plt.figure(figsize=(5, 5))
axKappa = growthRatePlot.add_subplot(111)
axKappa.set_xlabel(r'Ext. glucose conc. $[G_0]$ [$\mu$M]', fontsize=18)
axKappa.set_ylabel(r'Growth rate $\kappa$ $[h^{-1}]$', fontsize=18)
axKappa.margins(x=0, y=0.05)
axKappa.tick_params(labelsize=12)

for K_C_uM, color, linestyle in zip(K_C_values_uM, colors, linestyles):

    m.set_K_C(K_C_uM)

    print(f"K_C={m.K_C_uM} uM, Pprime={m.Pprime:.2f}") 
    
    K_C_label = f'$K_C$ = {K_C_uM} $\mu$M'
    name = f'growthLawsNoCrowd_KC{K_C_uM}uM'

    G0ListT=[]
    G0ListT_uM=[]
    eListT=[]
    kappaListT=[]
    MListT=[]
    phiRListT=[]
    phiEListT=[]
    VListT=[]
    SListT=[]
    SVListT=[]
    JOutListT=[]

    results=m.G0Determined(args,2500,1000,25000,0.5,[1,1,1,1])
    for j in range(len(results)):
        G0ListT.append(results[j][0])
        G0ListT_uM.append(results[j][0] * m.CONV_G0_TO_uM)
        eListT.append(results[j][9])
        kappaListT.append(results[j][1])
        MListT.append(results[j][2])
        phiRListT.append(results[j][4])
        phiEListT.append(results[j][3])
        SListT.append(results[j][3]*results[j][2]/rhoE)
        SVListT.append(results[j][3]*rhoM/rhoE)
        VListT.append(results[j][2]/rhoM)
        JOutListT.append(results[j][7])

    m.ProteomePlot(results, name)
    m.KappaPlot(results, name)
    m.OutPlot(results, name)
    m.EnergyPlot(results, args, name)

    ax2A.plot(kappaListT,    eListT,     color=color, linestyle=linestyle, label=K_C_label)
    ax2D.plot(SVListT,       eListT,     color=color, linestyle=linestyle, label=K_C_label)
    axOut.plot(G0ListT_uM,   JOutListT,  color=color, linestyle=linestyle, label=K_C_label)
    axKappa.plot(G0ListT_uM, kappaListT, color=color, linestyle=linestyle, label=K_C_label)

ax2A.legend(fontsize=11)
kappaEPlot.savefig('EfficiencyGrowth_KCvary.svg', bbox_inches='tight')

ax2D.legend(fontsize=11)
WVEPlot.savefig('EfficiencySurfaceVolume_KCvary.svg', bbox_inches='tight')

axOut.legend(fontsize=11)
outRatePlot.savefig('JOut_KCvary.svg', bbox_inches='tight')

axKappa.legend(fontsize=11)
growthRatePlot.savefig('GrowthRate_KCvary.svg', bbox_inches='tight')
