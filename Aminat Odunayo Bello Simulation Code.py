"""
SIMULATION CODE FOR:
Human-Induced Disturbance in Aquatic Predator-Prey Systems:
A Comparative Study of Integer-Order and Fractional-Order Models

Author: Aminat Odunayo Bello
Department of Mathematics, FUNAAB, Nigeria
Email: aminatodunayobello@gmail.com
ORCID: 0009-0008-0731-309X

This code numerically solves both the integer-order and
Caputo fractional-order predator-prey systems and generates
all 9 figures in the paper.

Numerical methods:
  - Integer-order: Forward Euler scheme (Section 4.1)
  - Fractional-order: Adams-type predictor-corrector
    (Diethelm et al. 2010, Section 4.2)
  - GL scheme: used only for manual verification (Section 4.3)
"""

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from math import gamma

# SECTION 1: NUMERICAL SOLVERS

def solve_integer(r, K, a, b, m, hp, hd, x0, y0, T, dt):
    """
    Solve the integer-order predator-prey system using
    the Forward Euler scheme (equation 7 in the paper):

        x[n+1] = x[n] + h * [rx[n](1-x[n]/K) - ax[n]y[n] - hp*x[n]]
        y[n+1] = y[n] + h * [bax[n]y[n] - my[n] - hd*y[n]]

    Parameters
    ----------
    r, K, a, b, m : floats - ecological parameters (Table 1)
    hp, hd        : floats - human-induced mortality rates
    x0, y0        : floats - initial conditions
    T             : float  - total simulation time
    dt            : float  - time step h

    Returns
    -------
    x, y : numpy arrays - prey and predator populations over time
    """
    N = int(T / dt)
    x = np.zeros(N + 1)
    y = np.zeros(N + 1)
    x[0] = x0
    y[0] = y0

    for n in range(N):
        x[n+1] = x[n] + dt * (r*x[n]*(1-x[n]/K) - a*x[n]*y[n] - hp*x[n])
        y[n+1] = y[n] + dt * (b*a*x[n]*y[n] - m*y[n] - hd*y[n])
        x[n+1] = max(x[n+1], 0)
        y[n+1] = max(y[n+1], 0)
    return x, y

def solve_fractional_pc(f_rhs, g_rhs, x0, y0, alpha, T, dt):
    """
    Solve the Caputo fractional-order predator-prey system using
    the Adams-type predictor-corrector method of Diethelm et al. (2010).

    This is the standard validated method for Caputo FDEs and offers
    superior numerical stability over explicit GL for long-time
    nonlinear simulations.

    Parameters
    ----------
    f_rhs  : function f(x,y) - right-hand side for prey equation
    g_rhs  : function g(x,y) - right-hand side for predator equation
    x0, y0 : floats - initial conditions
    alpha  : float  - fractional order in (0,1]
    T      : float  - total simulation time
    dt     : float  - time step h

    Returns
    -------
    x, y : numpy arrays - prey and predator populations over time
    """
    N = int(T / dt)
    x = np.zeros(N + 1)
    y = np.zeros(N + 1)
    x[0] = x0
    y[0] = y0
    ga1 = gamma(alpha + 1)
    ga2 = gamma(alpha + 2)

    for n in range(N):
        # Corrector weights
        def ac(j):
            if j == 0:
                return (n**(alpha+1) - (n-alpha)*(n+1)**alpha) / ga2
            elif j == n + 1:
                return 1.0 / ga2
            else:
                return ((n-j+2)**(alpha+1) + (n-j)**(alpha+1)
                        - 2*(n-j+1)**(alpha+1)) / ga2

        # Predictor weights
        bc = lambda j: dt**alpha / ga1 * ((n+1-j)**alpha - (n-j)**alpha)

        # Predictor step
        xp = max(x0 + sum(bc(j)*f_rhs(x[j], y[j]) for j in range(n+1)), 0)
        yp = max(y0 + sum(bc(j)*g_rhs(x[j], y[j]) for j in range(n+1)), 0)

        # Corrector step
        sx = sum(ac(j)*f_rhs(x[j], y[j]) for j in range(n+1))
        sy = sum(ac(j)*g_rhs(x[j], y[j]) for j in range(n+1))

        x[n+1] = max(x0 + dt**alpha * (f_rhs(xp, yp)/ga1 + sx), 0)
        y[n+1] = max(y0 + dt**alpha * (g_rhs(xp, yp)/ga1 + sy), 0)

    return x, y

def gl_one_step(x0, y0, alpha, h, f0, g0):
    """
    Single GL step for manual verification (Section 4.3).
    x(h) = x0 + h^alpha * f(x0, y0)
    y(h) = y0 + h^alpha * g(x0, y0)
    """
    ha = h**alpha
    return x0 + ha*f0, y0 + ha*g0

# SECTION 2: MANUAL VERIFICATION (Section 4.3)

print("=== MANUAL VERIFICATION ===")

# Euler
r,K,a,b,m,hp,hd,h = 0.1,1000,0.01,0.02,0.1,0.01,0.02,0.1
x0,y0 = 100,50
dx = r*x0*(1-x0/K) - a*x0*y0 - hp*x0
dy = b*a*x0*y0 - m*y0 - hd*y0
print(f"Euler x(0.1) = {x0+h*dx:.4f}  (paper: 95.8)")
print(f"Euler y(0.1) = {y0+h*dy:.4f}  (paper: 49.5)")

# GL
p1,p2,p3,p4 = 0.5,0.02,0.4,0.01; alpha=0.9; h=0.1; x0,y0=100,50
f0 = x0*(p1-p2*y0)
g0 = y0*(-p3+p4*x0)
ha = h**alpha
print(f"GL    x(0.1) = {x0+ha*f0:.4f}  (paper: 93.71)")
print(f"GL    y(0.1) = {y0+ha*g0:.4f}  (paper: 53.78)")

dt_i = 0.05   # integer-order step size
dt_f = 0.5    # fractional-order step size

# SECTION 3: FIGURE 2 - SCENARIO 1

print("\nGenerating Figure 2: Scenario 1...")
r,K,a,b,m,hp,hd = 1.2,100,0.01,0.1,0.05,0.02,0
x0,y0 = 40,9; T = 250
f = lambda x,y: r*x*(1-x/K) - a*x*y - hp*x
g = lambda x,y: b*a*x*y - m*y - hd*y

t_i = np.linspace(0,T,int(T/dt_i)+1)
t_f = np.linspace(0,T,int(T/dt_f)+1)

xi,yi   = solve_integer(r,K,a,b,m,hp,hd,x0,y0,T,dt_i)
xf9,yf9 = solve_fractional_pc(f,g,x0,y0,0.9,T,dt_f)
xf8,yf8 = solve_fractional_pc(f,g,x0,y0,0.8,T,dt_f)
xf7,yf7 = solve_fractional_pc(f,g,x0,y0,0.7,T,dt_f)

fig,axes = plt.subplots(1,2,figsize=(12,5))
axes[0].plot(t_i,xi,'b-',lw=2,label=r'Integer ($\alpha=1$)')
axes[0].plot(t_f,xf9,'r--',lw=2,label=r'Fractional ($\alpha=0.9$)')
axes[0].plot(t_f,xf8,'g-.',lw=2,label=r'Fractional ($\alpha=0.8$)')
axes[0].plot(t_f,xf7,'m:',lw=2,label=r'Fractional ($\alpha=0.7$)')
axes[0].axhline(50,color='gray',ls=':',lw=1,label=r'$x^*=50$')
axes[0].set_xlabel('Time $t$',fontsize=12); axes[0].set_ylabel('Prey $x(t)$',fontsize=12)
axes[0].set_title('Scenario 1: Prey dynamics',fontsize=13)
axes[0].legend(fontsize=9); axes[0].grid(True,alpha=0.3)

axes[1].plot(t_i,yi,'b-',lw=2,label=r'Integer ($\alpha=1$)')
axes[1].plot(t_f,yf9,'r--',lw=2,label=r'Fractional ($\alpha=0.9$)')
axes[1].plot(t_f,yf8,'g-.',lw=2,label=r'Fractional ($\alpha=0.8$)')
axes[1].plot(t_f,yf7,'m:',lw=2,label=r'Fractional ($\alpha=0.7$)')
axes[1].axhline(58,color='gray',ls=':',lw=1,label=r'$y^*=58$')
axes[1].set_xlabel('Time $t$',fontsize=12); axes[1].set_ylabel('Predator $y(t)$',fontsize=12)
axes[1].set_title('Scenario 1: Predator dynamics',fontsize=13)
axes[1].legend(fontsize=9); axes[1].grid(True,alpha=0.3)
plt.tight_layout()
plt.savefig('fig_scenario1.png',dpi=150,bbox_inches='tight')
plt.savefig('fig_scenario1.pdf',dpi=150,bbox_inches='tight')
plt.close(); print("  -> fig_scenario1 saved")

# SECTION 4: FIGURE 3 - SCENARIO 2

print("Generating Figure 3: Scenario 2...")
r,K,a,b,m,hp,hd = 1.2,100,0.1,0.02,0.3,0.05,0.05
x0,y0 = 30,10; T2 = 80
f2 = lambda x,y: r*x*(1-x/K) - a*x*y - hp*x
g2 = lambda x,y: b*a*x*y - m*y - hd*y
E1 = K*(r-hp)/r  # = 95.8333

t_i2 = np.linspace(0,T2,int(T2/dt_i)+1)
t_f2 = np.linspace(0,T2,int(T2/dt_f)+1)

xi2,yi2    = solve_integer(r,K,a,b,m,hp,hd,x0,y0,T2,dt_i)
xf2_9,yf2_9 = solve_fractional_pc(f2,g2,x0,y0,0.9,T2,dt_f)
xf2_8,yf2_8 = solve_fractional_pc(f2,g2,x0,y0,0.8,T2,dt_f)

fig,axes = plt.subplots(1,2,figsize=(12,5))
axes[0].plot(t_i2,xi2,'b-',lw=2,label=r'Integer ($\alpha=1$)')
axes[0].plot(t_f2,xf2_9,'r--',lw=2,label=r'Fractional ($\alpha=0.9$)')
axes[0].plot(t_f2,xf2_8,'g-.',lw=2,label=r'Fractional ($\alpha=0.8$)')
axes[0].axhline(E1,color='gray',ls=':',lw=1,label=r'$E_1\approx95.8$')
axes[0].set_xlabel('Time $t$',fontsize=12); axes[0].set_ylabel('Prey $x(t)$',fontsize=12)
axes[0].set_title('Scenario 2: Prey dynamics',fontsize=13)
axes[0].legend(fontsize=9); axes[0].grid(True,alpha=0.3)

axes[1].plot(t_i2,yi2,'b-',lw=2,label=r'Integer ($\alpha=1$)')
axes[1].plot(t_f2,yf2_9,'r--',lw=2,label=r'Fractional ($\alpha=0.9$)')
axes[1].plot(t_f2,yf2_8,'g-.',lw=2,label=r'Fractional ($\alpha=0.8$)')
axes[1].axhline(0,color='gray',ls=':',lw=1)
axes[1].set_xlabel('Time $t$',fontsize=12); axes[1].set_ylabel('Predator $y(t)$',fontsize=12)
axes[1].set_title('Scenario 2: Predator extinction',fontsize=13)
axes[1].legend(fontsize=9); axes[1].grid(True,alpha=0.3)
plt.tight_layout()
plt.savefig('fig_scenario2.png',dpi=150,bbox_inches='tight')
plt.savefig('fig_scenario2.pdf',dpi=150,bbox_inches='tight')
plt.close(); print("  -> fig_scenario2 saved")

# SECTION 5: FIGURE 4 - SCENARIO 3

print("Generating Figure 4: Scenario 3...")
r,K,a,b,m,hp,hd = 1.2,100,0.01,0.1,0.05,0.03,0.01
x0,y0 = 20,5; T3 = 250
f3 = lambda x,y: r*x*(1-x/K) - a*x*y - hp*x
g3 = lambda x,y: b*a*x*y - m*y - hd*y

t_i3 = np.linspace(0,T3,int(T3/dt_i)+1)
t_f3 = np.linspace(0,T3,int(T3/dt_f)+1)

alphas = [1.0,0.95,0.9,0.85,0.8,0.75,0.7]
colors = ['b','navy','r','orange','g','purple','m']
styles = ['-','--','-.',':','-','--','-.']

fig,axes = plt.subplots(1,2,figsize=(12,5))
for alpha,col,sty in zip(alphas,colors,styles):
    if abs(alpha-1.0)<0.001:
        xs,ys = solve_integer(r,K,a,b,m,hp,hd,x0,y0,T3,dt_i)
        t_plot = t_i3
    else:
        xs,ys = solve_fractional_pc(f3,g3,x0,y0,alpha,T3,dt_f)
        t_plot = t_f3
    lbl = r'Integer ($\alpha=1$)' if alpha==1.0 else r'Frac ($\alpha='+str(alpha)+r'$)'
    axes[0].plot(t_plot,xs,color=col,ls=sty,lw=1.8,label=lbl)
    axes[1].plot(t_plot,ys,color=col,ls=sty,lw=1.8,label=lbl)

axes[0].axhline(60,color='gray',ls=':',lw=1,label=r'$x^*=60$')
axes[0].set_xlabel('Time $t$',fontsize=12); axes[0].set_ylabel('Prey $x(t)$',fontsize=12)
axes[0].set_title(r'Scenario 3: Prey for varying $\alpha$',fontsize=13)
axes[0].legend(fontsize=7); axes[0].grid(True,alpha=0.3)

axes[1].axhline(45,color='gray',ls=':',lw=1,label=r'$y^*=45$')
axes[1].set_xlabel('Time $t$',fontsize=12); axes[1].set_ylabel('Predator $y(t)$',fontsize=12)
axes[1].set_title(r'Scenario 3: Predator for varying $\alpha$',fontsize=13)
axes[1].legend(fontsize=7); axes[1].grid(True,alpha=0.3)
plt.tight_layout()
plt.savefig('fig_scenario3.png',dpi=150,bbox_inches='tight')
plt.savefig('fig_scenario3.pdf',dpi=150,bbox_inches='tight')
plt.close(); print("  -> fig_scenario3 saved")

# SECTION 6: FIGURE 5 - PHASE PORTRAIT

print("Generating Figure 5: Phase portrait...")
r,K,a,b,m,hp,hd = 1.2,100,0.01,0.1,0.05,0.02,0
x0,y0 = 40,9; T = 250
f = lambda x,y: r*x*(1-x/K)-a*x*y-hp*x
g = lambda x,y: b*a*x*y-m*y-hd*y

xi_p,yi_p   = solve_integer(r,K,a,b,m,hp,hd,x0,y0,T,dt_i)
xf9_p,yf9_p = solve_fractional_pc(f,g,x0,y0,0.9,T,dt_f)
xf8_p,yf8_p = solve_fractional_pc(f,g,x0,y0,0.8,T,dt_f)

fig,ax = plt.subplots(figsize=(7,6))
ax.plot(xi_p,yi_p,'b-',lw=2,label=r'Integer ($\alpha=1$)')
ax.plot(xf9_p,yf9_p,'r--',lw=2,label=r'Fractional ($\alpha=0.9$)')
ax.plot(xf8_p,yf8_p,'g-.',lw=2,label=r'Fractional ($\alpha=0.8$)')
ax.plot(50,58,'k*',ms=15,label=r'$E^*=(50,58)$')
ax.plot(x0,y0,'ko',ms=8,label='Initial condition')
ax.set_xlabel('Prey $x(t)$',fontsize=12)
ax.set_ylabel('Predator $y(t)$',fontsize=12)
ax.set_title(r'Phase portrait: convergence to $E^*$',fontsize=13)
ax.legend(fontsize=10); ax.grid(True,alpha=0.3)
plt.tight_layout()
plt.savefig('fig_phase.png',dpi=150,bbox_inches='tight')
plt.savefig('fig_phase.pdf',dpi=150,bbox_inches='tight')
plt.close(); print("  -> fig_phase saved")

# SECTION 7: FIGURES 6 & 7 - SENSITIVITY

print("Generating Figure 6: hp sensitivity...")
r,K,a,b,m,hd = 1.2,100,0.01,0.1,0.05,0
x0,y0 = 40,9; T = 250
t_i = np.linspace(0,T,int(T/dt_i)+1)
t_f = np.linspace(0,T,int(T/dt_f)+1)
hp_vals = [0,0.02,0.05,0.10]
colors4 = ['b','g','orange','r']

fig,axes = plt.subplots(1,2,figsize=(12,5))
for hp,col in zip(hp_vals,colors4):
    f = lambda x,y,hp=hp: r*x*(1-x/K)-a*x*y-hp*x
    g_fn = lambda x,y: b*a*x*y-m*y-hd*y
    xi,yi   = solve_integer(r,K,a,b,m,hp,hd,x0,y0,T,dt_i)
    xf,yf   = solve_fractional_pc(f,g_fn,x0,y0,0.9,T,dt_f)
    axes[0].plot(t_i,xi,color=col,ls='-',lw=1.8,label=f'Integer $h_p={hp}$')
    axes[0].plot(t_f,xf,color=col,ls='--',lw=1.8,label=f'Fractional $h_p={hp}$')
    axes[1].plot(t_i,yi,color=col,ls='-',lw=1.8,label=f'Integer $h_p={hp}$')
    axes[1].plot(t_f,yf,color=col,ls='--',lw=1.8,label=f'Fractional $h_p={hp}$')

axes[0].set_xlabel('Time $t$',fontsize=12); axes[0].set_ylabel('Prey $x(t)$',fontsize=12)
axes[0].set_title('Effect of $h_p$ on prey (solid=integer, dashed=fractional)',fontsize=11)
axes[0].legend(fontsize=7,ncol=2); axes[0].grid(True,alpha=0.3)
axes[1].set_xlabel('Time $t$',fontsize=12); axes[1].set_ylabel('Predator $y(t)$',fontsize=12)
axes[1].set_title('Effect of $h_p$ on predator',fontsize=11)
axes[1].legend(fontsize=7,ncol=2); axes[1].grid(True,alpha=0.3)
plt.tight_layout()
plt.savefig('fig_hp_effect.png',dpi=150,bbox_inches='tight')
plt.savefig('fig_hp_effect.pdf',dpi=150,bbox_inches='tight')
plt.close(); print("  -> fig_hp_effect saved")

print("Generating Figure 7: hd sensitivity...")
r,K,a,b,m,hp = 1.2,100,0.01,0.1,0.05,0.02
hd_vals = [0,0.02,0.05,0.10]

fig,axes = plt.subplots(1,2,figsize=(12,5))
for hd,col in zip(hd_vals,colors4):
    f_fn = lambda x,y: r*x*(1-x/K)-a*x*y-hp*x
    g_fn = lambda x,y,hd=hd: b*a*x*y-m*y-hd*y
    xi,yi = solve_integer(r,K,a,b,m,hp,hd,x0,y0,T,dt_i)
    xf,yf = solve_fractional_pc(f_fn,g_fn,x0,y0,0.9,T,dt_f)
    axes[0].plot(t_i,xi,color=col,ls='-',lw=1.8,label=f'Integer $h_d={hd}$')
    axes[0].plot(t_f,xf,color=col,ls='--',lw=1.8,label=f'Fractional $h_d={hd}$')
    axes[1].plot(t_i,yi,color=col,ls='-',lw=1.8,label=f'Integer $h_d={hd}$')
    axes[1].plot(t_f,yf,color=col,ls='--',lw=1.8,label=f'Fractional $h_d={hd}$')

axes[0].set_xlabel('Time $t$',fontsize=12); axes[0].set_ylabel('Prey $x(t)$',fontsize=12)
axes[0].set_title('Effect of $h_d$ on prey (solid=integer, dashed=fractional)',fontsize=11)
axes[0].legend(fontsize=7,ncol=2); axes[0].grid(True,alpha=0.3)
axes[1].set_xlabel('Time $t$',fontsize=12); axes[1].set_ylabel('Predator $y(t)$',fontsize=12)
axes[1].set_title('Effect of $h_d$ on predator',fontsize=11)
axes[1].legend(fontsize=7,ncol=2); axes[1].grid(True,alpha=0.3)
plt.tight_layout()
plt.savefig('fig_hd_effect.png',dpi=150,bbox_inches='tight')
plt.savefig('fig_hd_effect.pdf',dpi=150,bbox_inches='tight')
plt.close(); print("  -> fig_hd_effect saved")

# SECTION 8: FIGURE 8 - EQUILIBRIUM vs hp

print("Generating Figure 8: Equilibrium vs hp...")
r,K,a,b,m,hd = 1.2,100,0.01,0.1,0.05,0
hp_range = np.linspace(0,0.08,200)
x_star = np.ones_like(hp_range) * (m+hd)/(b*a)   # = 50 always
y_star = (r/a)*(1-x_star/K) - hp_range/a

fig,axes = plt.subplots(1,2,figsize=(12,5))
axes[0].plot(hp_range,x_star,'b-',lw=2,label=r'$x^*=(m+h_d)/(ba)=50$')
axes[0].set_xlabel('Human prey mortality $h_p$',fontsize=12)
axes[0].set_ylabel('Prey equilibrium $x^*$',fontsize=12)
axes[0].set_title('Prey equilibrium vs $h_p$',fontsize=13)
axes[0].legend(fontsize=10); axes[0].grid(True,alpha=0.3)

axes[1].plot(hp_range,y_star,'r-',lw=2,label=r'$y^*$')
axes[1].axhline(0,color='k',ls=':',lw=1)
axes[1].fill_between(hp_range,y_star,0,where=(y_star>0),
                     alpha=0.15,color='green',label='Coexistence')
axes[1].fill_between(hp_range,y_star,0,where=(y_star<=0),
                     alpha=0.15,color='red',label='Collapse')
axes[1].set_xlabel('Human prey mortality $h_p$',fontsize=12)
axes[1].set_ylabel('Predator equilibrium $y^*$',fontsize=12)
axes[1].set_title('Predator equilibrium: coexistence vs collapse',fontsize=11)
axes[1].legend(fontsize=9); axes[1].grid(True,alpha=0.3)
plt.tight_layout()
plt.savefig('fig_equilibrium_hp.png',dpi=150,bbox_inches='tight')
plt.savefig('fig_equilibrium_hp.pdf',dpi=150,bbox_inches='tight')
plt.close(); print("  -> fig_equilibrium_hp saved")

# SECTION 9: FIGURE 9 - SETTLING TIME

print("Generating Figure 9: Settling time vs alpha...")
r,K,a,b,m,hp,hd = 1.2,100,0.01,0.1,0.05,0.02,0
x0,y0 = 40,9; T = 250
f = lambda x,y: r*x*(1-x/K)-a*x*y-hp*x
g = lambda x,y: b*a*x*y-m*y-hd*y
x_eq,y_eq = 50,58; tol = 2.0

alphas_range = np.linspace(0.70,1.0,15)
settle_x = []; settle_y = []

for alpha in alphas_range:
    if abs(alpha-1.0)<0.001:
        xs,ys = solve_integer(r,K,a,b,m,hp,hd,x0,y0,T,dt_i)
        t_arr = np.linspace(0,T,int(T/dt_i)+1)
    else:
        xs,ys = solve_fractional_pc(f,g,x0,y0,alpha,T,dt_f)
        t_arr = np.linspace(0,T,int(T/dt_f)+1)
    tx = 0
    for i in range(len(xs)-1,-1,-1):
        if abs(xs[i]-x_eq)>tol: tx=t_arr[i]; break
    ty = 0
    for i in range(len(ys)-1,-1,-1):
        if abs(ys[i]-y_eq)>tol: ty=t_arr[i]; break
    settle_x.append(tx); settle_y.append(ty)

fig,ax = plt.subplots(figsize=(8,5))
ax.plot(alphas_range,settle_x,'b-o',ms=6,lw=2,label='Prey settling time')
ax.plot(alphas_range,settle_y,'r-s',ms=6,lw=2,label='Predator settling time')
ax.set_xlabel(r'Fractional order $\alpha$',fontsize=12)
ax.set_ylabel('Settling time (time units)',fontsize=12)
ax.set_title(r'Settling time to $E^*=(50,58)$ vs $\alpha$',fontsize=13)
ax.legend(fontsize=11); ax.grid(True,alpha=0.3)
plt.tight_layout()
plt.savefig('fig_settling_time.png',dpi=150,bbox_inches='tight')
plt.savefig('fig_settling_time.pdf',dpi=150,bbox_inches='tight')
plt.close(); print("  -> fig_settling_time saved")

print("\nAll 9 figures generated successfully!")
