"""This script computes both the y and pressure Power spectrum for the normalised
   y fluctuation map obtained in the previous step of the pipeline. 
   
   -> A smooth Gaussian mask is created as described in Khatri et al. , with an 
   apodization scale of 15 arcmin.
   -> Bins are created between 500kpc and 2000kpc. The optimal number of bins 
   have to be chosen such that the final error bars are not too huge. Although,
   we have used the default value of 6, the user can try playing around with this
   number.
   -> The effective l's are computed for each bin followed by the definition of 
   the beam (Refer to beam_test.py for correct usage of beam).
   -> The final masked fluctuation map is plotted 
   -> The angular power spectrum for 2D y is computed using the coupling matrix method 
   and converted to Amplitude of the NON-ANGULAR power spectrum using the flat-
   sky approximation from Khatri et al. (i.e. k**2 P(k) = l**2 Cl (l)), where 
   P(k) is the non-angular spectrum and Cl(l) is the angular power spectrum 
   computed by NaMaster. The formula for Power spectrum aplitude in case of 2D
   is given by: A(k) = ( k**2 P(k) / (2*pi) ) ** (1/2)  
   -> The covairance matrix is calculated to compute the error bars
   -> The user can choose to fit a power law to the obtained power spectrum
   -> A short Parseval theorem check has been carried out (NOT YET WORKING!).
   Check test_parseval.py for details.
   -> The 2D y power spectrum is converted to 3D pressure power spectrum using 
   N's from a previous step of the pipeline and plotted for different values 
   of theta.
   NOTE: The user can reproduce Figure 11 of Khatri et al. by using N = 7e-4
   
   WARNING: 'normalised_y_fluc.txt' and 'Ns.txt' must be present in 'data'

"""

import numpy as np
import pymaster as nmt
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit

pixsize = 1.7177432059  #size of 1 pixel in arcmin
arcmin2kpc =  27.052 # conversion factor from arcmin to kpc

norm_y_fluc = np.loadtxt('../data/normalised_y_fluc.txt')

Lx = 4. * np.pi/180
Ly = 4. * np.pi/180
Nx, Ny = len(norm_y_fluc),  len(norm_y_fluc)

"------------------------------- CREATING MASKS -------------------------------"

mask = np.zeros((Nx,Ny))
#the centre will always be the middle elements becuase of the way the fluctuation maps have been computed
cen_x, cen_y = Nx/2., Ny/2.
cr = 60 #radius of mask in arcmin
I,J=np.meshgrid(np.arange(mask.shape[0]),np.arange(mask.shape[1]))
dist=np.sqrt((I-cen_x)**2+(J-cen_y)**2)
dist = dist * pixsize
idx = np.where(dist<=cr)
theta_ap = 15 #apodization scale in arcmin
mask[idx]=1-np.exp(-9*(dist[idx]-cr)**2/(2*theta_ap**2)) #described in Khatri et al.

"---------------------------- PLOTTING APODIZED MASK --------------------------"

plt.figure()
x_ticks = ['-2', '-1','0','1','2']
y_ticks = ['-2', '-1','0','1','2']
t11 = [0,35,70,105,138]
plt.xticks(ticks=t11, labels=x_ticks, size='small')
plt.yticks(ticks=t11, labels=y_ticks, size='small')
plt.imshow(mask, interpolation='nearest', origin='lower')
plt.xlabel("degrees")
plt.ylabel("degrees")
plt.colorbar()
plt.savefig("../images/Apodized mask.png",dpi = 400)
plt.show()

"------------------------------ CREATING BINS ---------------------------------"

bin_number = 6
#l's have to be converted from kpc using l = pi/angular sep
# We want to use bin sizes between 500 and 2000 kpc in terms of l's
l_min =  (180*60*arcmin2kpc/2000)
l_max = (180*60*arcmin2kpc/500)

bin_size = (l_max-l_min)/bin_number

l0_bins=[]
lf_bins=[]

for i in range (bin_number):
    l0_bins.append(l_min+bin_size*i)
    lf_bins.append(l_min+bin_size*(i+1))

print("\n************************  effective l's  *****************************\n")

b = nmt.NmtBinFlat(l0_bins, lf_bins)
ells_uncoupled = b.get_effective_ells()
print(ells_uncoupled)

lambdas_inv = ells_uncoupled/(arcmin2kpc*60*180)
k = 2*np.pi*lambdas_inv

"----------------------------- DEFINING BEAM ------------------------------------"
#refer to beam_test.py
"""params:
    -> l: array-like; ell values 
   returns:
    -> array-like; value of beam at those ells
"""

def beam(l):
    Planck_res=10./60 
    Planck_sig = Planck_res/2.3548
    return np.exp(-0.5*l*(l+1)*(Planck_sig*np.pi/180)**2)
    
"----------------------------- PLOTTING FIELD WITH MASK -------------------------"

f0 = nmt.NmtFieldFlat(Lx, Ly, mask, [norm_y_fluc] ,beam=[ells_uncoupled, beam(ells_uncoupled)])

plt.figure()
plt.imshow(f0.get_maps()[0] * mask, interpolation='nearest', origin='lower')
plt.colorbar()
plt.savefig('../images/map with mask.png', dpi = 400)
plt.show()

print("\n--------------------------- ANGULAR POWER SPECTRUM ------------------------------------\n")

w00 = nmt.NmtWorkspaceFlat()
w00.compute_coupling_matrix(f0, f0, b)
#Coupling matrix used to estimate angular spectrum
cl00_coupled = nmt.compute_coupled_cell_flat(f0, f0, b)
cl00_uncoupled = w00.decouple_cell(cl00_coupled)[0]
print(cl00_uncoupled)

amp = abs((ells_uncoupled**2)*cl00_uncoupled/(2*np.pi))**(1/2)

print("\n*************************  Covariance matrix  *************************************\n")

cw = nmt.NmtCovarianceWorkspaceFlat()
cw.compute_coupling_coefficients(f0, f0, b)
covar = nmt.gaussian_covariance_flat(cw, 0, 0, 0, 0, ells_uncoupled,
                                     [cl00_uncoupled], [cl00_uncoupled],
                                     [cl00_uncoupled], [cl00_uncoupled], w00)

print(covar)
std_power = (np.diag(covar))
std_amp = np.sqrt(abs((ells_uncoupled**2)*std_power/(2*np.pi))**(1/2))

"--------------------------------- Fitting a power law -------------------------------------"

# def power_law(x,a,p):
#     return a*np.power(x,p)

# a_fit, p_fit = curve_fit(power_law, lambdas_inv, amp, p0 = [1e-2,5/3])[0]

# lambdas_inv_curve = np.linspace(min(lambdas_inv),max(lambdas_inv),100)
# curve = power_law(lambdas_inv_curve, a_fit,p_fit)

"------------------------- Plotting amplitude of Power Spectrum vs 1/lambda ---------------------------"

plt.figure()
plt.plot(lambdas_inv, amp, 'r.', label='Amplitude of power spectrum')
plt.errorbar(lambdas_inv,amp, yerr=std_amp, fmt='r.',ecolor='black',elinewidth=1,
            capsize = 4)
# plt.plot(lambdas_inv_curve, curve, 'b', 
#          label='Best fit: Power law (power = %1.2f)'%p_fit)
plt.xlabel("$1/\lambda$ ($kpc^{-1}$)")
plt.ylabel("Amplitude of power spectrum")
plt.legend()
plt.title("Power Spectrum of normalised map")
plt.savefig("../images/power_spectrum.png", dpi = 400)
plt.show()


print("\n---------------------------- PARSEVAL CHECK ---------------------------------\n")

# NOT YET WORKING! Refer to test_parseval.py
len_norm_y_fluc = np.shape(norm_y_fluc)
variance = np.sum((norm_y_fluc-norm_y_fluc.mean())**2)/(len_norm_y_fluc[0]*len_norm_y_fluc[1])
print("Variance of map =",variance)

meanSq = np.sum(amp**2)/len(amp)
print("Average of amplitude^2 of power = ",meanSq)

print("\n----------------- PRESSURE POWER SPECTRUM ---------------------------------\n")

Ns = np.loadtxt("../data/Ns.txt")
#Ns = 7e-4*np.ones_like(Ns)
amp_pressure = np.zeros((1500,6))

for i in range(500,2000):
    amp_pressure[i-500,:] = abs((ells_uncoupled**2)*cl00_uncoupled*k/(2*np.pi**2*Ns[i]))**(1/2)


plt.figure()
plt.errorbar(lambdas_inv,amp_pressure[0], yerr=std_amp*(k/Ns[0]/np.pi)**(1/2), fmt='r.',ecolor='black',elinewidth=1,
            capsize = 4,label="theta = 500 kpc")
# plt.errorbar(lambdas_inv+1e-5,amp_pressure[500], yerr=std_amp*(k/Ns[500]/np.pi)**(1/2), fmt='b.',ecolor='black',elinewidth=1,
#             capsize = 4,label="theta = 1000 kpc")
# plt.errorbar(lambdas_inv+2e-5,amp_pressure[1000], yerr=std_amp*(k/Ns[1000]/np.pi)**(1/2), fmt='g.',ecolor='black',elinewidth=1,
#             capsize = 4, label='theta = 1500 kpc')
plt.errorbar(lambdas_inv-1e-5,amp_pressure[1499], yerr=std_amp*(k/Ns[1499]/np.pi)**(1/2), fmt='.',ecolor='black',elinewidth=1,
            capsize = 4, label='theta = 2000 kpc')
plt.xlabel("$1/\lambda$ ($kpc^{-1}$)")
plt.ylabel("Amplitude of pressure power spectrum")
plt.legend()
plt.title("Pressure Power Spectrum of Coma")
plt.savefig("../images/PS_pressure.png", dpi = 400)
plt.loglog()
plt.show()