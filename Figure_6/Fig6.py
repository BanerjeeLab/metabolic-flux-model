"""Fig. 6: maximum sustainable cell volume against growth rate (A) and the proteome at the overflow transition (B, C).
Rod geometry (GEOM below); density fixed at rho_M; beta/T = 1 at the empirical cell of the same kappa;
glucose throttled by bisection so kappa_realized = kappa_target; V_max = largest feasible M/rho_M.
Run from this folder: python3 Fig6.py.  Panels and the data file fig6_data.json are written to images/.
The sensitivity panels of Fig. S2 are produced by ../SupplementaryFig2/FigS2.py from the functions defined here.
"""
import os, sys, json, pathlib, numpy as np
HERE = pathlib.Path(__file__).resolve().parent; OUT = HERE/"images"; OUT.mkdir(exist_ok=True)
sys.path.insert(0, str(HERE))
import methodsFig6 as m, fluxbalance_linear as fb
from scipy.optimize import brentq
import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt

phi0=0.45; epsO=1.8e6; epsG=0.85*1.92*1.8e6; fC=0.47; kt=5.9; P=12000.
eG=5/3; eO=8.0; ma=1.1e-4; rhoE=2400.; phiRMin=0.07; eP=21/8; rhoP=4.5e6
a=8390.; rhoM=1.8e5*0.55; d=28.7e-3; gamma=0.39; uM=m.CONV_G0_TO_uM
eB_default = 23.5/(fC*ma)
Vemp = lambda k: 0.17*np.exp(1.16*k)
# Rod geometry.  'spherocylinder': cylinder of radius R with hemispherical caps and total length 8R (length =
# 4 x diameter, the E. coli aspect ratio): V = 22 pi R^3/3, S = 16 pi R^2, so S = (9216 pi/121)^{1/3} V^{2/3}
# = 6.208 V^{2/3} and phi_E = (9216 pi rho_E^3/(121 M rho_M^2))^{1/3} = (337.5/M)^{1/3}; diffusion length
# 8R - 2d.  'emp': the empirical law S = 6.24 V^{2/3} with diffusion length
# 3V/S - d.  The two differ by 0.5% in S and by the envelope-thickness term in the diffusion length.
GEOM = 'spherocylinder'
S_PREF = (9216*np.pi/121)**(1/3) if GEOM=='spherocylinder' else 6.24
def phiE_rod(M): V=M/rhoM; return rhoE*S_PREF*V**(2/3)/M
if GEOM=='spherocylinder':
    def _crowding_sc(M,phiE,beta,rhoE_,d_,switch):
        if not switch: return 1
        V=M/rhoM; R=(3*V/(22*np.pi))**(1/3); L=8*R-2*d_
        return beta*np.exp(-(M/V)/rhoM)/L**2
    m._crowding_sphere=m.Crowding   # 3V/S - d = R - d for the sphere (comparison curve)
    m.Crowding=_crowding_sc
def phiE_sph(M): return (36*np.pi*rhoE**3/(M*rhoM**2))**(1/3)

def beta_at(phiE_fn, Vref): Mr=rhoM*Vref; return 1.0/m.Crowding(Mr, phiE_fn(Mr), 1.0, rhoE, d, True)

def solve(M, phiE, G0, beta, switch, eB):
    chi=m.Crowding(M,phiE,beta,rhoE,d,switch)
    argz=[M,phiE,phiRMin,phi0,a,G0,P,epsG,epsO,eG,eO,eB,eP,beta,rhoE,rhoP,d,fC,kt,gamma,switch]
    g=gamma*(1-m.f_in); phiO=m.PhiOBalanced(argz)
    if phiO < g*phiE:
        phiR=m.PhiRBalanced(argz,False); phiG=m.PhiFBalanced(argz,False); JOut=0.0; reg='N'
        ok=(phiR>=0)&(phiG>=0)&(phiO>=0)
    else:
        argz[-2]=g; phiR=m.PhiRBalanced(argz,True); phiG=m.PhiFBalanced(argz,True); phiO=g*phiE
        JOut=m.JOutBalanced(argz); reg='O'; ok=(phiR>=0)&(phiG>=0)&(JOut>=0)
    kap=kt*(phiR-phiRMin)*chi; bud=phiR+phiG+phiE+phi0
    return dict(ok=bool(ok and bud<=1.0), kappa=kap, phiR=phiR, phiG=phiG, phiO=phiO, phiE=phiE, JOut=JOut, reg=reg, chi=chi, budget=bud)

G0lo, G0hi = 0.05/uM, 20000./uM
def feasible(M, phiE, beta, switch, kt_, eB):
    khi=solve(M,phiE,G0hi,beta,switch,eB)['kappa']
    if not np.isfinite(khi) or khi<kt_: return False, None
    klo=solve(M,phiE,G0lo,beta,switch,eB)['kappa']
    G0s=G0lo if klo>=kt_ else brentq(lambda g: solve(M,phiE,g,beta,switch,eB)['kappa']-kt_, G0lo, G0hi, xtol=1e-7*G0hi)
    r=solve(M,phiE,G0s,beta,switch,eB); r['G0_uM']=G0s*uM
    return r['ok'], r

def vmax(phiE_fn, switch, kappas, Mgrid, norm='emp', eB=eB_default):
    out={}
    for k in kappas:
        beta = (beta_at(phiE_fn, 1.0 if norm=='ref1' else Vemp(k)) if switch else 1.0)
        best=np.nan; info=None
        for M in Mgrid:
            ok,r=feasible(M,phiE_fn(M),beta,switch,k,eB)
            if ok: best=M/rhoM; info=r
        out[k]=(best,info)
    return out

def kappa_sat(eB=eB_default):
    f=lambda k: solve(rhoM*Vemp(k), phiE_rod(rhoM*Vemp(k)), G0hi, 1.0, False, eB)['kappa']-k
    return brentq(f, 0.05, 4.0)

plt.rcParams.update({"font.size":11,"axes.labelsize":13,"legend.fontsize":9.5,"axes.spines.top":False,"axes.spines.right":False,"lines.linewidth":2.0,"figure.dpi":140})

if __name__=="__main__":
    Mgrid=np.logspace(np.log10(0.003*rhoM), np.log10(3e4*rhoM), 900)
    kap=np.round(np.sort(np.append(np.arange(0.1,1.301,0.05),[0.757,0.76])),3)
    print("K_C =", m.K_C_uM, " CF =", fb.CF)
    kmax=kappa_sat(); print(f"kappa_max (growth-law cell, saturating glucose) = {kmax:.3f}")
    imp=vmax(phiE_rod,False,kap,Mgrid); dif=vmax(phiE_rod,True,kap,Mgrid,'emp')
    _cr=m.Crowding; m.Crowding=getattr(m,'_crowding_sphere',_cr)   # sphere uses its own radius R - d
    sph_i=vmax(phiE_sph,False,kap,Mgrid); sph_d=vmax(phiE_sph,True,kap,Mgrid,'ref1'); sph_de=vmax(phiE_sph,True,kap,Mgrid,'emp')
    m.Crowding=_cr
    V=lambda D:[D[k][0] for k in kap]
    data={'kappa':list(map(float,kap)),'kappa_max':kmax,'rod_import':V(imp),'rod_diff':V(dif),'sphere_import':V(sph_i),'sphere_diff_ref1':V(sph_d),'sphere_diff_emp':V(sph_de),
          'budget_rod_diff':[dif[k][1]['budget'] if dif[k][1] else None for k in kap]}
    for k in [0.2,0.5,0.75,0.9,1.0,1.1]:
        print(f"  k={k:.2f}  import={imp[k][0]:.3f}  diffusion={dif[k][0]:.3f}  (budget {dif[k][1]['budget']:.3f})  observed={Vemp(k):.3f}")

    # ---- Fig 6A ----
    fig,ax=plt.subplots(figsize=(6.4,5.0))
    ax.axvspan(0.2,kmax,color="#f2f2f2",zorder=0); ax.text((0.2+kmax)/2,0.075,"growth on a single carbon source",ha="center",fontsize=9,color="#555555")
    ax.semilogy(kap,V(sph_i),color="#bbbbbb",lw=1.3,label="sphere, import-limited")
    ax.semilogy(kap,V(sph_d),color="#bbbbbb",lw=1.3,ls="--",label="sphere, diffusion-limited")
    ax.semilogy(kap,V(imp),color="#08519c",label="import-limited ceiling")
    ax.semilogy(kap,V(dif),color="#a63603",ls="--",label="diffusion-limited ceiling")
    ax.semilogy(kap,Vemp(kap),color="k",lw=2.4,label=r"observed, $V=0.17\,e^{1.16\kappa}$")
    ax.axvline(0.757,color="#999999",lw=0.9,ls=":"); ax.text(0.762,3.0e2,r"$\kappa_{\rm crit}$",fontsize=9,color="#666666")
    ax.axvline(kmax,color="#999999",lw=0.9,ls=":"); ax.text(kmax+0.005,3.0e2,r"$\kappa_{\max}$",fontsize=9,color="#666666")
    ax.set_xlabel(r"Growth rate $\kappa$ (h$^{-1}$)"); ax.set_ylabel(r"Maximum sustainable volume $V_{\max}$ ($\mu$m$^3$)")
    ax.set_xlim(0.1,1.3); ax.set_ylim(0.05,5e2); ax.legend(loc="lower left",frameon=False,bbox_to_anchor=(0.0,0.08))
    fig.tight_layout(); fig.savefig(OUT/"Fig6A.png",dpi=300); fig.savefig(OUT/"Fig6A.pdf")

    # ---- Fig 6B,C: proteome vs V at kappa = 0.76 ----
    def prof(switch,k=0.757,n=70):
        beta=beta_at(phiE_rod,Vemp(k)) if switch else 1.0; Vs=np.logspace(np.log10(0.1),np.log10(5.0),n); rows=[]
        for Vv in Vs:
            M=rhoM*Vv; ok,r=feasible(M,phiE_rod(M),beta,switch,k,eB_default)
            rows.append((Vv,r['phiR'],r['phiG'],r['phiE'],r['phiO'],r['budget'],ok) if r else (Vv,)+(np.nan,)*5+(False,))
        return np.array(rows,dtype=float)
    fig,axs=plt.subplots(1,2,figsize=(10,4.3),sharey=True)
    for ax,sw,ttl in [(axs[0],False,"Not diffusion-limited"),(axs[1],True,"Diffusion-limited")]:
        r=prof(sw); Vv=r[:,0]; ok=r[:,6].astype(bool)
        for j,(lab,c,ls) in enumerate([(r"$\phi_R$","#d62728","-"),(r"$\phi_G$","#1f77b4","-"),(r"$\phi_E$","#2ca02c","-"),(r"$\phi_O$","#2ca02c","--")]):
            ax.semilogx(Vv,r[:,1+j],color=c,ls=ls,label=lab)
        ax.semilogx(Vv,r[:,5]-phi0,color="k",lw=1.2,label=r"$\phi_R+\phi_G+\phi_E$")
        ax.axhline(1-phi0,color="#888888",lw=0.9,ls=":"); ax.text(0.105,1-phi0+0.012,r"budget $1-\phi_0$",fontsize=8.5,color="#666666")
        if ok.any(): Vm=Vv[ok][-1]; ax.axvline(Vm,color="#a63603",lw=1.0,ls="--"); ax.text(Vm*1.06,0.025,r"$V_{\max}$",color="#a63603",fontsize=9)
        ax.axvline(Vemp(0.757),color="k",lw=0.9,ls=":"); ax.text(Vemp(0.757)*0.95,0.025,"observed",fontsize=8.5,ha="right")
        ax.set_title(ttl+r", $\kappa=\kappa_{\rm crit}=0.76$ h$^{-1}$",fontsize=11); ax.set_xlabel(r"Cell volume $V$ ($\mu$m$^3$)"); ax.xaxis.set_minor_formatter(matplotlib.ticker.NullFormatter()); ax.set_ylim(0,0.62)
    axs[0].set_ylabel("Proteome mass fraction"); axs[0].legend(loc="center right",frameon=False,ncol=1)
    fig.tight_layout(); fig.savefig(OUT/"Fig6BC.png",dpi=300); fig.savefig(OUT/"Fig6BC.pdf")

    json.dump(data, open(OUT/'fig6_data.json','w'), indent=1, default=float)
    print("saved")
