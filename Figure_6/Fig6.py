import numpy as np
import methodsFig6 as m
import matplotlib.pyplot as plt

######################## Initialize parameters ##############################
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

Mcal=1.8*10**5*0.55
phiEcal=(36*np.pi*rhoE**3/(Mcal*rhoM**2))**(1/3)

#beta here refers to beta' in manuscript
#betaT in methods refers to beta/T
beta=1/m.Crowding(Mcal, phiEcal, 1, rhoE, d,True) #using beta' = 1 in argument to allow normalization

###################### Run optimization ##################################

G0min=6000
G0max=25000#


allResults=[]

kappaTarget=0.0001
args=phiRMin,phi0,a,P,epsilonF,epsilonO,eF,eO,eB,eP,beta,rhoE,rhoP,d,fC,kt,phiOMaxFrac,False,kappaTarget
results=m.G0OptimizerV(args,30,G0min,G0max,0.01*Mcal,20000*Mcal)#20000#3000,[0.05,0.1,0.15,0.2]
allResults.append(results)
#results=m.OptimizerVOverflowMito(args,5,3000,25000,0.01,2*phiEcal,0.1*Mcal,10*Mcal,[0.1,0.15,0.2])
m.ProteomePlot2(results,'kappaTarget'+str(kappaTarget))

kappaTarget=0.1
args=phiRMin,phi0,a,P,epsilonF,epsilonO,eF,eO,eB,eP,beta,rhoE,rhoP,d,fC,kt,phiOMaxFrac,False,kappaTarget
results=m.G0OptimizerV(args,30,G0min,G0max,0.01*Mcal,500*Mcal)#20000#3000,[0.05,0.1,0.15,0.2]
allResults.append(results)
#results=m.OptimizerVOverflowMito(args,5,3000,25000,0.01,2*phiEcal,0.1*Mcal,10*Mcal,[0.1,0.15,0.2])
m.ProteomePlot2(results,'kappaTarget'+str(kappaTarget))


kappaTarget=0.5
args=phiRMin,phi0,a,P,epsilonF,epsilonO,eF,eO,eB,eP,beta,rhoE,rhoP,d,fC,kt,phiOMaxFrac,False,kappaTarget
results=m.G0OptimizerV(args,30,G0min,G0max,0.01*Mcal,100*Mcal)#20000#3000,[0.05,0.1,0.15,0.2]
allResults.append(results)
#results=m.OptimizerVOverflowMito(args,5,3000,25000,0.01,2*phiEcal,0.1*Mcal,10*Mcal,[0.1,0.15,0.2])
m.ProteomePlot2(results,'kappaTarget'+str(kappaTarget))
m.EfficiencyPlot(results,'kappaTargetE'+str(kappaTarget))


allResults2=[]

kappaTarget=0.0001
args=phiRMin,phi0,a,P,epsilonF,epsilonO,eF,eO,eB,eP,beta,rhoE,rhoP,d,fC,kt,phiOMaxFrac,True,kappaTarget
results=m.G0OptimizerV(args,30,G0min,G0max,0.01*Mcal,20000*Mcal)#20000#3000,[0.05,0.1,0.15,0.2]
allResults2.append(results)
#results=m.OptimizerVOverflowMito(args,5,3000,25000,0.01,2*phiEcal,0.1*Mcal,10*Mcal,[0.1,0.15,0.2])
m.ProteomePlot2(results,'kappaTarget2'+str(kappaTarget))

kappaTarget=0.1
args=phiRMin,phi0,a,P,epsilonF,epsilonO,eF,eO,eB,eP,beta,rhoE,rhoP,d,fC,kt,phiOMaxFrac,True,kappaTarget
results=m.G0OptimizerV(args,30,G0min,G0max,0.01*Mcal,500*Mcal)#20000#3000,[0.05,0.1,0.15,0.2]
allResults2.append(results)
#results=m.OptimizerVOverflowMito(args,5,3000,25000,0.01,2*phiEcal,0.1*Mcal,10*Mcal,[0.1,0.15,0.2])
m.ProteomePlot2(results,'kappaTarget2'+str(kappaTarget))


kappaTarget=0.5
args=phiRMin,phi0,a,P,epsilonF,epsilonO,eF,eO,eB,eP,beta,rhoE,rhoP,d,fC,kt,phiOMaxFrac,True,kappaTarget
results=m.G0OptimizerV(args,30,G0min,G0max,0.01*Mcal,100*Mcal)#20000#3000,[0.05,0.1,0.15,0.2]
allResults2.append(results)
#results=m.OptimizerVOverflowMito(args,5,3000,25000,0.01,2*phiEcal,0.1*Mcal,10*Mcal,[0.1,0.15,0.2])
m.ProteomePlot2(results,'kappaTarget2'+str(kappaTarget))
m.EfficiencyPlot(results,'kappaTargetE2'+str(kappaTarget))

colors=['saddlebrown','navy','darkviolet']
VPlot=plt.figure(figsize=(5, 5))
for j in range(3):
    sols=allResults[j]
    G0List=[]
    VList=[]
    for i in range(len(sols)):
        G0List.append(sols[i][0]*np.array(CONV_G0_TO_uM))
        VList.append(((sols[i][3]*sols[i][2]/rhoE)**3/(36*np.pi))**0.5)
    plt.plot(G0List,VList,color=colors[j])
   
for j in range(3):
    sols=allResults2[j]
    G0List=[]
    VList=[]
    for i in range(len(sols)):
        G0List.append(sols[i][0]*np.array(CONV_G0_TO_uM))
        VList.append(((sols[i][3]*sols[i][2]/rhoE)**3/(36*np.pi))**0.5)
    plt.plot(G0List,VList,color=colors[j],linestyle='dashed')

plt.legend([r'$\kappa=0.0001$ h$^{-1}$',r'$\kappa=0.1$ h$^{-1}$',r'$\kappa=0.5$ h$^{-1}$'],loc='lower right',fontsize=12)
plt.xlabel(r'Ext. glucose conc. $[G_0]$ $[\mu \mathrm{M}]$',fontsize=18)
plt.ylabel(r'Volume V $[\mu m^3]$',fontsize=18)   
plt.margins(x=0,y=0.15)  
plt.yscale('log')
plt.xticks(fontsize=12)
plt.yticks(fontsize=12)
plt.locator_params(axis='x', nbins=7)
plt.savefig('Fig6A.svg',bbox_inches='tight')





