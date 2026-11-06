"""Fig. S3: robustness of the efficiency maximum to the sharpness of respiratory saturation.

The main-text model caps respiration at phi_O <= gamma_eff phi_E and switches to an overflow regime
when the cap binds.  Here the cap is replaced by a smooth saturation of respiratory throughput in
the acetyl-CoA supply: with D the acetyl-CoA flux available for respiration and C the respiratory
capacity (C = eps_O gamma_eff phi_E M / e_O, the same capacity as the hard cap), the respired flux
is r = C h(D/C) with h(x) = x (1 + x^n)^(-1/n).  n = 1 is Michaelis-Menten saturation, n -> inf
recovers the hard cap min(D, C).  Acetate excretion is J_out = D - r, which is positive at every
glucose level for finite n.  No regime switch and no feasibility check anywhere.

Growth-law cell exactly as in Fig. 3A (Fig2_Fig3_eGAMvary.py): M = rho_M V(kappa), phi_E = rho_E S/M,
V = 0.17 e^{1.16 kappa}, S = 6.24 V^{2/3}, solved self-consistently for each [G0].  Units fix (CF)
and K_C = 52 uM as in the main-text figures.  Self-contained; run from this folder: python3 FigS3.py.
"""
import pathlib
OUT = pathlib.Path(__file__).resolve().parent/"images"; OUT.mkdir(exist_ok=True)
import numpy as np, json
from scipy.optimize import brentq
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt

# ---- parameters (Fig2_Fig3_eGAMvary.py / methods22.py) ----
phi0=0.45; epsO=1.8e6; epsG=0.85*1.92*1.8e6; fC=0.47; kt=5.9; P=12000; eG=5/3; eO=8.0
ma=1.1e-4; eB=23.5/(fC*ma); eGAM=8/(fC*ma); eBtot=eB+eGAM; rhoE=2400; phiRMin=0.07; eP=21/8
rhoP=4.5e6; a=8390; rhoM=1.8e5*0.55; f_in=0.1; gamma=0.39; gamma_eff=gamma*(1-f_in)
CONV=0.001660539067; K_ptsg=10.0/CONV; K_C_uM=52.0; K_C=K_C_uM/CONV; Pprime=P*K_C
CF=1e6/12.011
Vemp=lambda k: 0.17*np.exp(1.16*k); Semp=lambda V: 6.24*V**(2/3)

def h(x, n):
    if n is None: return np.minimum(x, 1.0)
    return x/(1.0+x**n)**(1.0/n)

def solve_cell(M, phiE, G0, n, cap=True):
    """Return dict for the cell (M, phiE) at glucose G0 with saturation sharpness n (None = hard cap;
    cap=False removes the respiratory limit altogether)."""
    S=phiE*M/rhoE; JIn=6.0*Pprime*S*G0/(G0+K_ptsg)
    lam=16*rhoP*phiE*M*kt/(rhoE*np.log(2)); g=kt*M; C=epsO*gamma_eff*phiE*M/eO
    def parts(y):
        JG=epsG*M*(eG*(JIn-CF*fC*g*y)/(epsG*M))        # carbon balance -> phiG -> J_G
        phiG=JG/(epsG*M); JP=lam*y; D=JG/(3*eG)-JP
        r = D if not cap else (C*h(D/C, n) if D>0 else D)
        JO=eO*r; JOut=D-r
        return JG, phiG, JP, D, JO, JOut
    def resid(y):
        JG,phiG,JP,D,JO,JOut=parts(y)
        return JG+JO+JOut-eBtot*fC*g*y-eP*lam*y-a*M     # ATP balance
    ymax=JIn/(CF*fC*g)*0.999999                           # phiG >= 0
    y=brentq(resid, 1e-9, ymax, xtol=1e-14, maxiter=500)
    JG,phiG,JP,D,JO,JOut=parts(y)
    E=(eB*fC*g*y+eP*lam*y)/(JG+JO+8*JOut)
    return dict(kappa=kt*y, phiR=y+phiRMin, phiG=phiG, phiO=JO/(epsO*M), JOut=JOut, JIn=JIn, JG=JG, JO=JO, E=E,
                sat=(D/C if C>0 else np.nan), M=M, phiE=phiE)

def growth_law_cell(G0, n, cap=True):
    """phi_R such that the cell on the growth laws at kappa = kt (phi_R - phi_R^min) realizes that kappa."""
    def mismatch(k):
        M=rhoM*Vemp(k); phiE=rhoE*Semp(Vemp(k))/M
        return solve_cell(M,phiE,G0,n,cap)['kappa']-k
    k=brentq(mismatch, 0.05, 2.5, xtol=1e-10)
    M=rhoM*Vemp(k); phiE=rhoE*Semp(Vemp(k))/M
    r=solve_cell(M,phiE,G0,n,cap); r['G0_uM']=G0*CONV; return r

if __name__=="__main__":
    G0s=np.linspace(1000,25000,600)
    cases=[(None,"hard cap","k","-"),(8,"n = 8","#a63603","-"),(4,"n = 4","#e6550d","-"),(2,"n = 2","#fd8d3c","-"),(1,"n = 1 (Michaelis-Menten)","#fdae6b","-")]
    out={}
    for n,lab,c,ls in cases:
        rows=[growth_law_cell(G,n) for G in G0s]
        out[str(n)]=dict(label=lab,G0_uM=[r['G0_uM'] for r in rows],kappa=[r['kappa'] for r in rows],E=[r['E'] for r in rows],
                         JOut=[r['JOut'] for r in rows],sat=[r['sat'] for r in rows],phiO=[r['phiO'] for r in rows])
        k=np.array(out[str(n)]['kappa']); E=np.array(out[str(n)]['E']); i=int(np.argmax(E))
        out[str(n)]['kpeak']=float(k[i]); out[str(n)]['Epeak']=float(E[i]); out[str(n)]['G0peak_uM']=float(out[str(n)]['G0_uM'][i])
        print(f"{lab:32s} peak E={E[i]:.4f} at kappa={k[i]:.3f} h^-1, [G0]={out[str(n)]['G0peak_uM']:.1f} uM;  E(0.5)={np.interp(0.5,k,E):.3f}  E(0.9)={np.interp(0.9,k,E):.3f}")
    rows=[growth_law_cell(G,None,cap=False) for G in G0s]
    out['nocap']=dict(label="no respiratory limit",kappa=[r['kappa'] for r in rows],E=[r['E'] for r in rows],G0_uM=[r['G0_uM'] for r in rows],JOut=[r['JOut'] for r in rows])
    print(f"no respiratory limit: E rises from {out['nocap']['E'][0]:.3f} to {out['nocap']['E'][-1]:.3f}, kappa to {out['nocap']['kappa'][-1]:.3f}")
    # excreted carbon fraction at kappa = 0.5 (where E. coli excretes no acetate) and the 99% plateau
    for n,lab,c,ls in cases:
        k=np.array(out[str(n)]['kappa']); E=np.array(out[str(n)]['E']); JO=np.array(out[str(n)]['JOut'])
        rows=[growth_law_cell(G,n) for G in G0s[::20]]
        frac=np.array([2*r['JOut']/r['JIn'] for r in rows]); kk=np.array([r['kappa'] for r in rows])   # 2 carbons per acetate
        f05=float(np.interp(0.5,kk,frac)); plateau=k[E>=0.99*E.max()]
        out[str(n)]['excreted_carbon_frac_at_0.5']=f05; out[str(n)]['plateau99']=[float(plateau.min()),float(plateau.max())]
        print(f"{lab:32s} excreted carbon fraction at kappa=0.5: {100*f05:.2f}%   99% plateau: {plateau.min():.2f} to {plateau.max():.2f}")
    ns=[1,1.5,2,3,4,6,8,12,16,24,32,64]; kp=[]; lo=[]; hi=[]
    for n in ns:
        rr=[growth_law_cell(G,n) for G in G0s[::3]]
        kk=np.array([r['kappa'] for r in rr]); EE=np.array([r['E'] for r in rr]); kp.append(float(kk[np.argmax(EE)]))
        pl=kk[EE>=0.99*EE.max()]; lo.append(float(pl.min())); hi.append(float(pl.max()))
    out['peak_vs_n']=dict(n=ns,kpeak=kp,plateau_lo=lo,plateau_hi=hi); print("kappa at peak vs n:", list(zip(ns,[round(x,3) for x in kp])))
    print("99% plateau upper edge vs n:", list(zip(ns,[round(x,3) for x in hi])))
    json.dump(out,open(OUT/"FigS3_data.json","w"),indent=1)

    fig,axs=plt.subplots(1,3,figsize=(13.5,4.0))
    ax=axs[0]
    ax.plot(out['nocap']['kappa'],out['nocap']['E'],color="#999999",ls=":",lw=1.8,label="no respiratory limit")
    for n,lab,c,ls in cases: ax.plot(out[str(n)]['kappa'],out[str(n)]['E'],color=c,ls=ls,lw=2.0 if n is None else 1.6,label=lab)
    ax.axvline(0.757,color="k",lw=0.8,ls="--"); ax.text(0.765,ax.get_ylim()[0]+0.01,r"$\kappa_\mathrm{crit}$",fontsize=9)
    ax.set_xlabel(r"Growth rate $\kappa$ (h$^{-1}$)"); ax.set_ylabel(r"Efficiency $\mathcal{E}$"); ax.legend(frameon=False,fontsize=8); ax.set_title("A",loc="left",fontweight="bold")
    ax=axs[1]
    for n,lab,c,ls in cases: ax.plot(out[str(n)]['kappa'],np.array(out[str(n)]['JOut'])/1e8,color=c,lw=2.0 if n is None else 1.6,label=lab)
    ax.axvline(0.757,color="k",lw=0.8,ls="--")
    ax.set_xlabel(r"Growth rate $\kappa$ (h$^{-1}$)"); ax.set_ylabel(r"Acetate excretion $J_\mathrm{out}$ ($10^8$ h$^{-1}$)"); ax.set_title("B",loc="left",fontweight="bold")
    ax=axs[2]
    ax.fill_between(ns,lo,hi,color="#fdae6b",alpha=0.5,label=r"$\mathcal{E}\geq 0.99\,\mathcal{E}_{\max}$")
    ax.semilogx(ns,kp,"o-",color="#a63603",ms=4,label="maximum")
    hp=out['None']['plateau99']; ax.axhline(out['None']['kpeak'],color="k",ls="--",lw=0.9); ax.axhspan(hp[0],hp[1],color="#cccccc",alpha=0.4)
    ax.text(1.1,out['None']['kpeak']+0.01,"hard cap: maximum (dashed), 99% plateau (grey)",fontsize=8)
    ax.set_xlabel(r"Saturation sharpness $n$"); ax.set_ylabel(r"Growth rate (h$^{-1}$)"); ax.set_title("C",loc="left",fontweight="bold"); ax.legend(frameon=False,fontsize=8,loc="lower right")
    fig.tight_layout(); fig.savefig(OUT/"FigS3.png",dpi=300); fig.savefig(OUT/"FigS3.pdf"); print("saved")
