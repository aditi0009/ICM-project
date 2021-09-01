import numpy as np
import pymaster as nmt
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit
import scipy.integrate as si

pixsize = 1.7177432059 #size of 1 pixel in arcmin
arcmin2kpc =  27.052 # conversion factor from arcmin to kpc

norm_y_fluc = np.loadtxt('../data/normalised_y_fluc.txt')
norm_y_fluc_first = np.loadtxt('../data/normalised_y_fluc_first.txt')
norm_y_fluc_last = np.loadtxt('../data/normalised_y_fluc_last.txt')

Lx = 4. * np.pi/180
Ly = 4. * np.pi/180
#first and last maps are both of the same dimensions

Nx, Ny = len(norm_y_fluc_first),  len(norm_y_fluc_first)
Nx1, Ny1 = len(norm_y_fluc),  len(norm_y_fluc)

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
mask[idx]=1-np.exp(-9*(dist[idx]-cr)**2/(2*theta_ap**2))

mask1 = np.zeros((Nx1,Ny1))
#the centre will always be the middle elements becuase of the way the fluctuation maps have been computed
cen_x1, cen_y1 = Nx1/2., Ny1/2.
cr = 60 #radius of mask in arcmin
I1,J1=np.meshgrid(np.arange(mask1.shape[0]),np.arange(mask1.shape[1]))
dist1=np.sqrt((I1-cen_x1)**2+(J1-cen_y1)**2)
dist1 = dist1 * pixsize
idx1 = np.where(dist1<=cr)
theta_ap1 = 15 #apodization scale in arcmin
mask1[idx1]=1-np.exp(-9*(dist1[idx1]-cr)**2/(2*theta_ap1**2)) #described in Khatri et al.


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

#Fiekds for both maps are calculated 
f01 = nmt.NmtFieldFlat(Lx, Ly, mask, [norm_y_fluc] ,beam=[ells_uncoupled, beam(ells_uncoupled)])
f0 = nmt.NmtFieldFlat(Lx, Ly, mask, [norm_y_fluc_first] ,beam=[ells_uncoupled, beam(ells_uncoupled)])
f1 = nmt.NmtFieldFlat(Lx, Ly, mask, [norm_y_fluc_last] ,beam=[ells_uncoupled, beam(ells_uncoupled)])
plt.figure()

print("\n--------------------------- ANGULAR POWER SPECTRUM ------------------------------------\n")

w00 = nmt.NmtWorkspaceFlat()
w00.compute_coupling_matrix(f0, f1, b)
#Coupling matrix used to estimate angular spectrum
cl00_coupled = nmt.compute_coupled_cell_flat(f0, f1, b)
cl00_uncoupled = w00.decouple_cell(cl00_coupled)[0]
print(cl00_uncoupled)

amp = abs((ells_uncoupled**2)*cl00_uncoupled/(2*np.pi))**(1/2)



w001 = nmt.NmtWorkspaceFlat()
w001.compute_coupling_matrix(f01, f01, b)
#Coupling matrix used to estimate angular spectrum
cl00_coupled1 = nmt.compute_coupled_cell_flat(f01, f01, b)
cl00_uncoupled1 = w00.decouple_cell(cl00_coupled1)[0]
print(cl00_uncoupled1)

amp1 = abs((ells_uncoupled**2)*cl00_uncoupled1/(2*np.pi))**(1/2)

print("\n*************************  Covariance matrix  *************************************\n")

cw = nmt.NmtCovarianceWorkspaceFlat()
cw.compute_coupling_coefficients(f0, f1, b)
covar = nmt.gaussian_covariance_flat(cw, 0, 0, 0, 0, ells_uncoupled,
                                     [cl00_uncoupled], [cl00_uncoupled],
                                     [cl00_uncoupled], [cl00_uncoupled], w00)
print(covar)
std_power = (np.diag(covar))
std_amp = np.sqrt(abs((ells_uncoupled**2)*std_power/(2*np.pi))**(1/2))



cw1 = nmt.NmtCovarianceWorkspaceFlat()
cw1.compute_coupling_coefficients(f01, f01, b)
covar1 = nmt.gaussian_covariance_flat(cw1, 0, 0, 0, 0, ells_uncoupled,
                                     [cl00_uncoupled1], [cl00_uncoupled1],
                                     [cl00_uncoupled1], [cl00_uncoupled1], w001)

print(covar1)
std_power1 = (np.diag(covar1))
std_amp1 = np.sqrt(abs((ells_uncoupled**2)*std_power1/(2*np.pi))**(1/2))

"--------------------------------- Fitting a power law -------------------------------------"

#def power_law(x,a,p):
#    return a*np.power(x,p)

#a_fit, p_fit = curve_fit(power_law, lambdas_inv, amp, p0 = [1e-2,5/3])[0]

#lambdas_inv_curve = np.linspace(min(lambdas_inv),max(lambdas_inv),100)
#curve = power_law(lambdas_inv_curve, a_fit,p_fit)

"------------------------- Plotting amplitude of Power Spectrum vs 1/lambda ---------------------------"

plt.figure()
plt.plot(lambdas_inv, amp, 'r.', label='Amplitude of power spectrum')
plt.errorbar(lambdas_inv,amp, yerr=std_amp, fmt='r.',ecolor='black',elinewidth=1,
            capsize = 4)
plt.plot(lambdas_inv, amp1, 'b.', label='Amplitude of power spectrum')
plt.errorbar(lambdas_inv,amp1, yerr=std_amp1, fmt='b',ecolor='cyan',elinewidth=1,
            capsize = 4)
plt.xlabel("$1/\lambda$ ($kpc^{-1}$)")
plt.ylabel("Amplitude of power spectrum")
plt.legend()
plt.title("Cross Power Spectrum of normalised map")
plt.savefig("../images/test.png", dpi = 400)
plt.show()
