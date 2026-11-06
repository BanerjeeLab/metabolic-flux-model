"""Fig. S4: emergent cell size, shape and growth rate (Serbanescu et al. 2020 dynamics) coupled to the
flux balance model.  Diffusion penalty off.

Dynamics (per cell):  dV/dt = kappa V,  dX/dt = k_p V (division at X = X0, then X -> 0, V -> V/2, S -> S/2),
                      dS/dt = beta V with beta = nu kappa V_b^{-1/3}, V_b the cell's birth volume (aspect-ratio preservation,
                      Ojkic et al. 2019; beta is constant within a cycle as in Serbanescu et al. 2020),
with kappa = kappa_FBA(phi_E, G0) from the flux balance at the cell's current phi_E = rho_E S/(rho_M V).
Steady state: V_b = kappa X0/k_p (adder), S/V = nu V^{-1/3}, kappa the fixed point of
kappa = kappa_FBA(phi_E = rho_E nu V_b^{-1/3}/rho_M, G0).
nu(eta) = eta pi ((eta pi/4) - pi/12)^(-2/3): nu(2) = 5.25, nu(4) = 6.21 (E. coli), nu(8) = 7.59; eta is the length-to-width aspect ratio.
"""
import sys, json, numpy as np
import pathlib
HERE = pathlib.Path(__file__).resolve().parent; OUT = HERE/"images"; OUT.mkdir(exist_ok=True)
sys.path.insert(0, str(HERE.parent/"Figures_2_and_3"))   # model functions shared with Figs. 2 and 3
import methods22 as m, fluxbalance_linear as fb
from scipy.optimize import brentq
import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt

phi0=0.45; epsO=1.8e6; epsG=0.85*1.92*1.8e6; fC=0.47; kt=5.9; P=12000.
eG=5/3; eO=8.0; ma=1.1e-4; rhoE=2400.; phiRMin=0.07; eP=21/8; rhoP=4.5e6
a=8390.; rhoM=1.8e5*0.55; d=28.7e-3; gamma=0.39; uM=m.CONV_G0_TO_uM
eB=23.5/(fC*ma); m.set_K_C(52)
nu=lambda eta: eta*np.pi*((eta*np.pi/4)-np.pi/12)**(-2/3)
Vemp=lambda k: 0.17*np.exp(1.16*k); SVemp=lambda k: 6.24*Vemp(k)**(2/3)/Vemp(k)

def fba(M, phiE, G0):
    """growth rate, efficiency, J_out, phiO for a cell of protein mass M and envelope fraction phiE at glucose G0"""
    argz=[M,phiE,phiRMin,phi0,a,G0,P,epsG,epsO,eG,eO,eB,eP,None,rhoE,rhoP,d,fC,kt,gamma,False]
    g=gamma*(1-m.f_in); phiO=m.PhiOBalanced(argz)
    if phiO < g*phiE:
        phiR=m.PhiRBalanced(argz,False); phiG=m.PhiFBalanced(argz,False); JOut=0.0
    else:
        argz[-2]=g; phiR=m.PhiRBalanced(argz,True); phiG=m.PhiFBalanced(argz,True); phiO=g*phiE; JOut=m.JOutBalanced(argz)
    kap=kt*(phiR-phiRMin)
    E=m.efficiency(M,phiE,phiR,phiRMin,phiG,phiO,JOut,a,G0,P,epsG,epsO,eG,eO,eB,eP,None,rhoE,rhoP,d,fC,kt,False)
    return dict(kappa=kap,E=E,JOut=JOut,phiO=phiO,phiR=phiR,phiG=phiG,phiE=phiE)

def steady(G0, eta, X0_over_kp):
    """fixed point kappa = kappa_FBA(phi_E(V_b(kappa)))"""
    n=nu(eta)
    def resid(k):
        Vb=k*X0_over_kp; M=rhoM*Vb; phiE=rhoE*n*Vb**(2/3)/M
        return fba(M,phiE,G0)['kappa']-k
    k=brentq(resid,0.02,3.0,xtol=1e-9)
    Vb=k*X0_over_kp; M=rhoM*Vb; phiE=rhoE*n*Vb**(2/3)/M
    r=fba(M,phiE,G0); r.update(Vb=Vb,SV=n*Vb**(-1/3),G0_uM=G0*uM); return r

def simulate(G0, eta, X0_over_kp, V0, SV0, T=12.0, dt=2e-3):
    """single-cell trajectory: V, S, X with kappa from the flux balance at the current phi_E"""
    n=nu(eta); kp=1.0; X0=X0_over_kp*kp
    V=V0; S=SV0*V0; X=0.0; t=0.0; out=[]; Vb=V0
    while t<T:
        M=rhoM*V; phiE=rhoE*S/M
        k=fba(M,phiE,G0)['kappa']; beta=n*k*Vb**(-1/3)
        V+=k*V*dt; S+=beta*V*dt; X+=kp*V*dt; t+=dt
        out.append((t,V,S/V,k,Vb))
        if X>=X0: V/=2; S/=2; X=0.0; Vb=V
    return np.array(out)

if __name__=="__main__":
    # calibrate X0/kp so that V_b = Vemp(kappa_crit) at kappa_crit for eta = 4
    kc=0.757; X0kp=Vemp(kc)/kc
    print(f"nu(2)={nu(2):.3f} nu(4)={nu(4):.3f} nu(8)={nu(8):.3f};  X0/kp = {X0kp:.3f} um^3 h")
    G0s=np.logspace(np.log10(1.0/uM), np.log10(60.0/uM), 160)
    res={}
    for eta in [2,4,8]:
        rows=[steady(G,eta,X0kp) for G in G0s]
        k=np.array([r['kappa'] for r in rows]); E=np.array([r['E'] for r in rows]); J=np.array([r['JOut'] for r in rows])
        on=k[J>0].min() if (J>0).any() else np.nan
        res[eta]=dict(kappa=k.tolist(),E=E.tolist(),JOut=J.tolist(),Vb=[r['Vb'] for r in rows],SV=[r['SV'] for r in rows],G0_uM=[r['G0_uM'] for r in rows],onset=float(on))
        print(f"eta={eta}: onset kappa = {on:.3f} h^-1, E at onset = {E[J>0][0] if (J>0).any() else np.nan:.3f}, kappa range {k.min():.2f}-{k.max():.2f}, S/V at onset = {np.interp(on,k,[r['SV'] for r in rows]):.2f}")
    # Eq. 6 prediction for the onset at the emergent S/V (phi_E) of each eta
    # trajectories
    traj={}
    for eta,V0,SV0 in [(4,0.15,12.0),(4,0.9,6.0),(2,0.4,8.4),(8,0.4,8.4)]:
        traj[(eta,V0)]=simulate(10.5/uM,eta,X0kp,V0,SV0)
    json.dump({'steady':{str(k):v for k,v in res.items()}}, open(OUT/'FigS4_data.json','w'))

    plt.rcParams.update({"font.size":10,"axes.labelsize":11,"legend.fontsize":8.5,"axes.spines.top":False,"axes.spines.right":False,"lines.linewidth":1.8})
    fig,axs=plt.subplots(1,3,figsize=(13.5,4.0))
    ax=axs[0]; k=np.array(res[4]['kappa'])
    ax.plot(k,res[4]['Vb'],color="#08519c",label=r"emergent $V_b$ ($\eta=4$)"); ax.plot(k,Vemp(k),color="#08519c",ls="--",label=r"empirical $V=0.17e^{1.16\kappa}$")
    ax.set_xlabel(r"Growth rate $\kappa$ (h$^{-1}$)"); ax.set_ylabel(r"Newborn volume $V_b$ ($\mu$m$^3$)",color="#08519c"); ax.set_ylim(0,1.35)
    ax2=ax.twinx(); ax2.plot(k,res[4]['SV'],color="#a63603",label=r"emergent $S/V$"); ax2.plot(k,SVemp(k),color="#a63603",ls="--",label=r"empirical $S/V$")
    ax2.set_ylabel(r"$S/V$ ($\mu$m$^{-1}$)",color="#a63603"); ax2.set_ylim(5,17); ax2.spines["right"].set_visible(True)
    h1,l1=ax.get_legend_handles_labels(); h2,l2=ax2.get_legend_handles_labels(); ax.legend(h1+h2,l1+l2,frameon=False,loc="upper left",ncol=2,columnspacing=1.0)
    ax.set_title("A",loc="left",fontweight="bold")
    ax=axs[1]
    for eta,c in [(2,"#fdae6b"),(4,"#a63603"),(8,"#7f2704")]:
        ax.plot(res[eta]['kappa'],res[eta]['E'],color=c,label=rf"$\eta={eta}$ (onset {res[eta]['onset']:.2f} h$^{{-1}}$)")
    ax.set_xlabel(r"Growth rate $\kappa$ (h$^{-1}$)"); ax.set_ylabel(r"Efficiency $\mathcal{E}$"); ax.legend(frameon=False,title="aspect ratio"); ax.set_title("B",loc="left",fontweight="bold")
    ax=axs[2]
    for (eta,V0),tr in traj.items():
        ax.plot(tr[:,0],tr[:,2],lw=1.2,label=rf"$\eta={eta}$, $V_0={V0}$ $\mu$m$^3$")
    ax.set_xlabel("Time (h)"); ax.set_ylabel(r"$S/V$ ($\mu$m$^{-1}$)"); ax.legend(frameon=False); ax.set_title("C",loc="left",fontweight="bold")
    fig.tight_layout(); fig.savefig(OUT/"FigS4.png",dpi=250); fig.savefig(OUT/"FigS4.pdf"); print("saved")
