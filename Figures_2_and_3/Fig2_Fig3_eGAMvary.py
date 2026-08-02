import numpy as np
import methods22 as m
import matplotlib.pyplot as plt

#####################Initialize parameters######################
phi0=0.45;
epsilonO=1.8*10**6#1.8*10**6
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


####################### Run optimization ####################


## beta unused: crowding is off (switch=False) for Fig. 2 and 3, so beta isn't used here. See README.
beta=None

#crowding function switched off with False
args=phiRMin,phi0,a,P,epsilonF,epsilonO,eF,eO,eB,eP,beta,rhoE,rhoP,d,fC,kt,phiOMaxFrac,rhoM,False


# eB, eGAM absorb the factor f_C * m_a as implemented in the code,
# so the flux-balance expressions can use them as bare coefficients (m_a = 1.1e-4).
# Values below are the physical ATP/aa divided by f_C*m_a. Multiply by f_C*m_a to recover ATP/aa

eGAM_values = [
    8/(fC*1.1*10**-4) * 0.5,   # 0.5x default
    8/(fC*1.1*10**-4),          # experimentally motivated (Feist et al)
    8/(fC*1.1*10**-4) * 2.0,   # 2x default
]
colors     = ['gray', 'blue', 'gray']
linestyles = [':', '-', '--']
linewidths = [2, 2, 2]

kappaEPlot = plt.figure(figsize=(5, 5))
ax3A = kappaEPlot.add_subplot(111)
ax3A.set_xlabel(r'Growth rate $\kappa$ $[h^{-1}] $', fontsize=18)
ax3A.set_ylabel(r'Efficiency $\mathcal{E}$', fontsize=18)
ax3A.margins(x=0, y=0.05)
ax3A.tick_params(labelsize=12)

WVEPlot = plt.figure(figsize=(5, 5))
ax3C = WVEPlot.add_subplot(111)
ax3C.set_xlabel(r'$S/V$ $[\mu m^{-1}] $', fontsize=18)
ax3C.set_ylabel(r'Efficiency $\mathcal{E}$', fontsize=18)
ax3C.margins(x=0, y=0.05)
ax3C.tick_params(labelsize=12)

outRatePlot = plt.figure(figsize=(5, 5))
axOut = outRatePlot.add_subplot(111)
axOut.set_xlabel(r'Ext. glucose conc. $[G_0]$ [$\mu$M]', fontsize=18)
axOut.set_ylabel(r'$J_{out}$ [Acetate/h]', fontsize=18)
axOut.margins(x=0, y=0.05)
axOut.tick_params(labelsize=12)
axOut.ticklabel_format(axis='y', style='sci', scilimits=(0, 0), useMathText=True)

growthRatePlot = plt.figure(figsize=(5, 5))
axKappa = growthRatePlot.add_subplot(111)
axKappa.set_xlabel(r'Ext. glucose conc. $[G_0]$ [$\mu$M]', fontsize=18)
axKappa.set_ylabel(r'Growth rate $\kappa$ $[h^{-1}]$', fontsize=18)
axKappa.margins(x=0, y=0.05)
axKappa.tick_params(labelsize=12)

for eGAM_val, color, linestyle, linewidth in zip(eGAM_values, colors, linestyles, linewidths):

    m.set_eGAM(eGAM_val)

    #eGAM_label = f'e_{{GAM}} = {eGAM_val*(fC*1.1*10**-4):.2f} ATP/aa'
    eGAM_label = rf'$e_{{\mathrm{{GAM}}}}$ = {eGAM_val*(fC*1.1*10**-4):.2f} ATP/aa'
    name = f'growthLawsNoCrowd_eGAM{eGAM_val*(fC*1.1*10**-4):.2f}'

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


    # res=2500 is the scan resolution.
    # Lower it (e.g. 500) for faster, coarser simulations
    results=m.G0Determined(args,2500,1000,25000,0.55,[1,1,1,1])
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

    '''Per-eGAM versions of the function-based plots'''
    m.ProteomePlot(results, name)
    m.KappaPlot(results, name)
    m.OutPlot(results, name)
    m.EnergyPlot(results, args, name)

    ax3A.plot(kappaListT,   eListT,    color=color, linestyle=linestyle, linewidth=linewidth, label=eGAM_label)
    ax3C.plot(SVListT,      eListT,    color=color, linestyle=linestyle, linewidth=linewidth, label=eGAM_label)
    axOut.plot(G0ListT_uM,  JOutListT, color=color, linestyle=linestyle, linewidth=linewidth, label=eGAM_label)
    axKappa.plot(G0ListT_uM, kappaListT, color=color, linestyle=linestyle, linewidth=linewidth, label=eGAM_label)

ax3A.legend(fontsize=11, loc='lower left')
kappaEPlot.savefig('3A.svg', bbox_inches='tight')

ax3C.legend(fontsize=11, loc='lower left')
WVEPlot.savefig('3C.svg', bbox_inches='tight')

axOut.legend(fontsize=11, loc='best')
axOut.ticklabel_format(axis='y', style='sci', scilimits=(0, 0), useMathText=True)
outRatePlot.savefig('JOut_eGAMvary.svg', bbox_inches='tight')

axKappa.legend(fontsize=11, loc='best')
growthRatePlot.savefig('GrowthRate_eGAMvary.svg', bbox_inches='tight')