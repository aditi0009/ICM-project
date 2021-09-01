"""This script uses the values of the underlying beta model paramter values 
   computed in a previous step of the pipeline to
   -> Calculate the window function for different values of physical distances 
   theta (Khatri figure 7)
   -> Does a partial fourier tranform (wrt z) for each theta 
   -> Computes N's as a function of theta integrating over k_z and finally plots it

"""

import numpy as np
import matplotlib.pyplot as plt
import scipy.fftpack as fftpack
import scipy.integrate as si

pixsize = 1.7177432059   #size of 1 pixel in arcmin  
arcmin2kpc =  27.052  # conversion factor from arcmin to kpc

m,c,sigma_T = 9.1093e-31, 299792458, 6.652e-29  #electron mass, speed of light and Thompson cross-section

#beta model parameters for 2D y 
y0 = 5.713907103243261e-05
theta_c = 613.0520060958096
beta1 = 1.4086450050856862
#beta model parameters after deprojection
p0 = 1.4840887724753505e-06
r_c = 731.2599585728855
beta2 = 2.410274413306557


# JUST A TEST TO CHECK THE SENSITIVITY OF N FOR DIFFERENT PARAMETERS
# Try the values below to reproduce N's similar to Khatri

# y0=6.5e-5
# theta_c = 419
# beta1 = 1.05

# p0 = 6.4840887724753505e-06
# r_c = 530
# beta2 = 2.05

"""
params:
    -> theta: Physical distance from the centre of the cluster
    -> z: Depth in 3D 
"""
def windowFunc(theta,z):
    w = (p0/y0)*(1+theta**2/theta_c**2)**(beta1)/(1+(theta**2+z**2)/r_c**2)**(beta2)
    return w/arcmin2kpc

zs = np.arange(-2000,2000,1)
ws1 = windowFunc(0,zs)
ws2 = windowFunc(500,zs)

plt.figure()
plt.plot(zs,ws1,label="theta=0 kpc")
plt.plot(zs,ws2,label="theta=500 kpc")
plt.xlabel("z (kpc)")
plt.ylabel("W(z,theta)")
plt.legend()
plt.savefig("../images/Figure 7.png",dpi=400)
plt.show()

ws = np.zeros((2000,4000))
ft_ws = np.zeros((2000,4000))
freqs_ws = np.zeros((2000,4000))

i=0
for theta in range (0,2000,1):
    ws[i,:] = windowFunc(theta,zs)
    ft_ws[i,:] = np.abs(fftpack.fft(ws[i,:]))**2 #square of partial FT (wrt) of window function
    freqs_ws[i,:] = fftpack.fftfreq(len(ws[i,:]), d=(zs[1]-zs[0]))
    i+=1
    
plt.figure()
plt.imshow(ws)
plt.xlabel("z (-2000 to 2000 kpc)")
plt.ylabel("theta (0-2000 kpc)")
plt.title(r"$W(z,\theta)$")
plt.savefig("../images/Window Function.png",dpi=400)
plt.colorbar()
plt.show()

plt.figure()
plt.plot(freqs_ws[0,:],ft_ws[0,:],label="theta=0 kpc")
plt.plot(freqs_ws[500,:],ft_ws[500,:],label="theta=500 kpc")
plt.loglog()
# USE THESE RANGES TO OBTAIN A PLOT SIMILAR TO FIGURE 8 (Khatri)
# plt.xlim([1e-4,2e-3])
# plt.ylim([1e-4,1])
plt.ylabel(r"$\bar{W} (k_z,\theta)$")
plt.xlabel(r"$k_z$")
plt.legend()
plt.savefig("../images/Figure 8.png",dpi=400)
plt.show()

Ns = np.zeros(2000)

i=0
for theta in range (0,2000,1):
    #INCASE YOU WANNA CHECK THE CONTRIBUTION TO THE INTEGRAL FROM PARTCILAR K_Z VALUES
    #idx = np.where(ft_ws[i,:]>1e-6)
    Ns[i] = si.simps(ft_ws[i,:],freqs_ws[i,:])/(2*np.pi)
    i+=1

np.savetxt("../data/Ns.txt",Ns)

plt.figure()
plt.plot(Ns)
plt.xlabel("thetas (kpc)")
plt.ylabel("N")
plt.savefig("../images/Ns vs thetas.png",dpi=400)
plt.show()