import numpy as np
from scipy.integrate import odeint
import matplotlib.pyplot as plt
import scipy.special as sp
"""
Here is the full code for the temperature of the radiation in the early universe. You can manually change the constant gamma0 
to get the different results showed in the paper.
"""

"""Inflaton without Dark Matter"""

#constants (GeV)
m_phi = 10**(9)
g_star = 100
g_mphi = 10**(-5)
g=m_phi*10**(-5)
mP=1.220890*10**(19)
mP_reduced=(1/np.sqrt(8*np.pi))*mP
gamma0=10**(-8)
T_rh=20
m_pl=1.220890*10**(19)
#gamma0=(T_rh**2)/(m_pl)*((8*np.pi**3*g_star)/90)**(1/2)
m_X=1
g_P=1
m_P=1000
Gamma_P=10**2*1.97327e-16 #c*tau=10**-2

#differential equations + temperature, gamma and H implementation
def inflaton_cst(X,t):
    phi,R=X
    Gamma=gamma0
    H=(8*np.pi/3)**(1/2)*(((m_phi)**2)/mP)*((R/t**4)+phi/t**3)**(1/2)
    T=(m_phi/t)*(30*R/((np.pi**2)*g_star))**(1/4)
    dphidt=-(Gamma/(H*t))*phi
    dRdt=(Gamma/H)*phi
    return dphidt,dRdt

def inflaton_dst(X,t):
    phi,R=X
    H=(8*np.pi/3)**(1/2)*(((m_phi)**2)/mP)*((R/t**4)+phi/t**3)**(1/2)
    T=(m_phi/t)*(30*R/((np.pi**2)*g_star))**(1/4)
    Gamma=gamma0*(1+2*(1/(np.exp(m_phi/(2*T))-1)))
    dphidt=-(Gamma/(H*t))*phi
    dRdt=(Gamma/H)*phi
    return dphidt,dRdt


#initial conditions and parameters
T0=0 #if unstable (too steep) with T0=0, you can try 0.0001*m_phi
x0=1
xmax=10**18
ndim=100000
phi0=10**20
R0=((np.pi**2)*g_star/30)*((T0*x0)/m_phi)**4 #R0 calculated from the temperature equation
X0=phi0,R0
x=np.logspace(np.log10(x0),np.log10(xmax), ndim)

#computing and extracting solutions for the constant model
sol_cst=odeint(inflaton_cst,X0,x)
sol_cst_T= sol_cst.T
Solphi_cst=sol_cst_T[0]
SolR_cst=sol_cst_T[1]

#solutions for the model with the Bose-Einstein distribution
sol_dst=odeint(inflaton_dst,X0,x)
sol_dst_T=sol_dst.T
Solphi_dst=sol_dst_T[0]
SolR_dst=sol_dst_T[1]

#Temperature construction
Temp_cst=np.zeros(ndim)
for i in range(np.shape(Temp_cst)[0]):
    if i ==0:
        Temp_cst[0]=T0
    else:
        Temp_cst[i]=((m_phi/x[i])*((30/g_star/np.pi**2)*SolR_cst[i])**(1/4))

Temp_dst=np.zeros(ndim)
for i in range(np.shape(Temp_dst)[0]):
    if i==0:
        Temp_dst[0]=T0
    else:
        Temp_dst[i]=((m_phi/x[i])*((30/g_star/np.pi**2)*SolR_dst[i])**(1/4))

#Check for maximal values
print(np.max(Temp_cst)/m_phi,np.max(Temp_dst)/m_phi)

#Definiton of Tmax and Tr
Tmax= 0.6*(gamma0*mP/g_star)**(1/4)*(3/4*phi0*m_phi**4)**(1/8)
Tr=np.sqrt(gamma0*mP)*(90/(8*np.pi**3*g_star))**(1/4)

#Temperature plot
plt.figure()
plt.xlim(0.95,10**13) #0.95 is an experimental value, it's a bit easier to see the behavior of the temperature at the beginning with this value but it can be changed to 1
plt.ylim(10**(-5),10)
plt.title('T/m_phi in function of x')
plt.xscale('log')
plt.yscale('log')
plt.xlabel('x')
plt.ylabel('T/m_phi')
plt.axhline(Tmax/m_phi,linestyle='dashed',color='black',label='T_max')
plt.axhline(Tr/m_phi,linestyle='dashed',color='darkgreen',label='T_R')
plt.plot(x,Temp_cst/m_phi, color ='red',label='Gamma constant')
plt.plot(x,Temp_dst/m_phi,'--',color='blue',label='Gamma w/ B-E distribution')
plt.legend()
plt.show()

"""Dark Matter"""
#ode
def DM(S,x):
    phi,R,X=S
    T=(m_phi/x)*(30*R/((np.pi**2)*g_star))**(1/4)
    E_X=(m_X**2+9*T**2)**(1/2)
    H=(8*np.pi/3)**(1/2)*m_phi**2/m_pl*((R/x**4)+(phi/x**3))**(1/2)
    Gamma=gamma0*(1+2*(1/(np.exp(m_phi/(2*T))-1)))
    if T/m_phi>0:
        CDM=(g_P/(2*np.pi**2))*Gamma_P*(m_P**2)*T*sp.kn(2,m_P/T)
    else:
        CDM = 0
    dphi=-(Gamma/(H*x))*phi
    dR=(Gamma/H)*phi-(x**3/(H*m_phi**4))*2*E_X*CDM
    dX=x**2/(H*m_phi**3)*CDM
    return dphi,dR,dX

#numerical integration
t0=1
tmax=10**18
tdim=100000
t=np.logspace(np.log10(t0),np.log10(tmax),tdim)
T0=0
phi0=10**20
R0=((np.pi**2)*g_star/30)*((T0*t0)/m_phi)**4
X0=0 
S0=phi0,R0,X0

SolDM=odeint(DM,S0,t)
SolR=SolDM[:,1]
SolX=SolDM[:,2]

def temperature():
    Temp=np.zeros(tdim)
    for i in range(np.shape(Temp)[0]):
        Temp[i]=((m_phi/t[i])*((30/g_star/np.pi**2)*SolR[i])**(1/4))
    return Temp

Temp=temperature()

def darkmatter():
    DMab=np.zeros(tdim)
    for j in range(np.shape(DMab)[0]):
        DMab[j]=SolX[j]*m_phi**3/t[j]**3/(2*np.pi**2/45*g_star*Temp[j]**3)
    return DMab
DMab=darkmatter()

#Temp
plt.figure()
plt.xlim(0.95,tmax)
plt.ylim(10**(-9),10)
plt.title('T/m_phi in function of x')
plt.xscale('log')
plt.yscale('log')
plt.xlabel('x')
plt.ylabel('T/m_phi')
plt.plot(t,Temp/m_phi,color='blue',label='Temperature with DM')
plt.legend()
plt.show()
#DM production
plt.figure()
plt.xscale('log')
plt.yscale('log')
plt.xlabel('x')
plt.ylabel('Dark Matter abundance')
plt.title('Dark Matter production')
plt.plot(t,DMab)
plt.show()

#full plot
plt.figure()
plt.xlim(0.95,10**16)
plt.ylim(10**-7,10)
plt.title('T/m_phi in function of x')
plt.xscale('log')
plt.yscale('log')
plt.xlabel('x')
plt.ylabel('T/m_phi')
plt.plot(t,Temp/m_phi,'o',color='green',label='Gamma B-E distribution with DM')
plt.plot(x,Temp_cst/m_phi, color ='red',label='Gamma constant without DM')
plt.plot(x,Temp_dst/m_phi,'--',color='blue',label='Gamma B-E distribution without DM')
plt.legend()
plt.show()

#DM abundance with different gamma0's
#eqs

def DM_8(S,x):
    gamma0=10**(-8)
    phi,R,X=S
    T=(m_phi/x)*(30*R/((np.pi**2)*g_star))**(1/4)
    E_X=(m_X**2+9*T**2)**(1/2)
    H=(8*np.pi/3)**(1/2)*m_phi**2/m_pl*((R/x**4)+(phi/x**3))**(1/2)
    Gamma=gamma0*(1+2*(1/(np.exp(m_phi/(2*T))-1)))
    if T/m_phi>0:
        CDM=(g_P/(2*np.pi**2))*Gamma_P*(m_P**2)*T*sp.kn(2,m_P/T)
    else:
        CDM = 0
    dphi=-(Gamma/(H*x))*phi
    dR=(Gamma/H)*phi-(x**3/(H*m_phi**4))*2*E_X*CDM
    dX=x**2/(H*m_phi**3)*CDM
    return dphi,dR,dX

def DM_6(S,x):
    gamma0=10**(-6)
    phi,R,X=S
    T=(m_phi/x)*(30*R/((np.pi**2)*g_star))**(1/4)
    E_X=(m_X**2+9*T**2)**(1/2)
    H=(8*np.pi/3)**(1/2)*m_phi**2/m_pl*((R/x**4)+(phi/x**3))**(1/2)
    Gamma=gamma0*(1+2*(1/(np.exp(m_phi/(2*T))-1)))
    if T/m_phi>0:
        CDM=(g_P/(2*np.pi**2))*Gamma_P*(m_P**2)*T*sp.kn(2,m_P/T)
    else:
        CDM = 0
    dphi=-(Gamma/(H*x))*phi
    dR=(Gamma/H)*phi-(x**3/(H*m_phi**4))*2*E_X*CDM
    dX=x**2/(H*m_phi**3)*CDM
    return dphi,dR,dX

def DM_4(S,x):
    gamma0=10**(-4)
    phi,R,X=S
    T=(m_phi/x)*(30*R/((np.pi**2)*g_star))**(1/4)
    E_X=(m_X**2+9*T**2)**(1/2)
    H=(8*np.pi/3)**(1/2)*m_phi**2/m_pl*((R/x**4)+(phi/x**3))**(1/2)
    Gamma=gamma0*(1+2*(1/(np.exp(m_phi/(2*T))-1)))
    if T/m_phi>0:
        CDM=(g_P/(2*np.pi**2))*Gamma_P*(m_P**2)*T*sp.kn(2,m_P/T)
    else:
        CDM = 0
    dphi=-(Gamma/(H*x))*phi
    dR=(Gamma/H)*phi-(x**3/(H*m_phi**4))*2*E_X*CDM
    dX=x**2/(H*m_phi**3)*CDM
    return dphi,dR,dX

#numerical integration
t0=1
tmax=10**18
tdim=100000
t=np.logspace(np.log10(t0),np.log10(tmax),tdim)
T0=0
phi0=10**20
R0=((np.pi**2)*g_star/30)*((T0*t0)/m_phi)**4
X0=0 
S0=phi0,R0,X0

SolDM_8=odeint(DM_8,S0,t)
SolR_8=SolDM_8[:,1]
SolX_8=SolDM_8[:,2]

SolDM_6=odeint(DM_6,S0,t)
SolR_6=SolDM_6[:,1]
SolX_6=SolDM_6[:,2]

SolDM_4=odeint(DM_4,S0,t)
SolR_4=SolDM_4[:,1]
SolX_4=SolDM_4[:,2]

def temperature(SoluR):
    Temp=np.zeros(tdim)
    for i in range(np.shape(Temp)[0]):
        Temp[i]=((m_phi/t[i])*((30/g_star/np.pi**2)*SoluR[i])**(1/4))
    return Temp


def darkmatter(SoluX,Temperat):
    DMab=np.zeros(tdim)
    for j in range(np.shape(DMab)[0]):
        DMab[j]=SoluX[j]*m_phi**3/t[j]**3/(2*np.pi**2/45*g_star*Temperat[j]**3)
    return DMab

Temp8=temperature(SolR_8)
Temp6=temperature(SolR_6)
Temp4=temperature(SolR_4)

DMab8=darkmatter(SolX_8, Temp8)
DMab6=darkmatter(SolX_6,Temp6)
DMab4=darkmatter(SolX_4,Temp4)

plt.figure()
plt.xscale('log')
plt.yscale('log')
plt.xlabel('x')
plt.ylabel('Dark Matter abundance')
plt.xlim(1,tmax)
plt.ylim(10**(-28),10**(-3))
plt.title('Dark Matter production')
plt.plot(t,DMab8,color='blue',label='Gamma_0=10^-8')
plt.plot(t,DMab6,color='red',label='Gamma_0=10^-6')
plt.plot(t,DMab4,color='darkgreen',label='Gamma_0=10^-4')
plt.legend()
plt.show()
