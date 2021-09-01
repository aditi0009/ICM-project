"""This script uses an example Power spectrum from NaMaster, 'cls_cmb.txt'
   to test correct usage of PSF (beam) while retrieving a power spectrum from 
   a map. The map in this case has been computed using the synfast_flat 
   functionality (check NaMaster documentation for details). Several different 
   options were tested while trying to simulate a PSF convolved map from the 
   power spectrum. The most accurate retrieval was seen when the power spectrum
   was multiplied by 'beam**2' as opposed to just 'beam' (since we are taking the 
   auto-correlation)
   
   The correct way to use the beam input in NmtFieldFlat is by taking the Fourier 
   transform of the instrumental beam and working in units of radians for the
   instrument's resolution/sigma 
   
   WARNING: The Power Spectrum dataset 'cls_cmb.txt' must be present in 'data'

"""

import numpy as np
import pymaster as nmt
import matplotlib.pyplot as plt
from scipy.signal import convolve2d

#number of pixels along x and y axis for the example chosen
Nx = 602
Ny = 410

#size of 1 pixel
pix=1.7/60 # deg
#physical length of x and y direction
Lx = Nx*pix * np.pi/180
Ly = Ny*pix * np.pi/180

# Gaussian simulations:
# pymaster allows you to generate random realizations of both spherical and
# flat fields given a power spectrum. These are returned as 2D arrays with
# shape (Ny,Nx)
l, cl_tt, cl_ee, cl_bb, cl_te = np.loadtxt('../data/cls_cmb.txt', unpack=True) # this one goes plenty high enough: max(l) = 9134 corresponding to 1.2 arcmin, around the pixel scale
ell_max=np.pi/(pix*np.pi/180) # max expected l based on pixel size

# Generate random realisation of the power spectrum, just in temperature (spin-0).  
# Set random seed for repeatability - doesn't work?!
rseed=10
np.random.seed(rseed)
mpt, = nmt.synfast_flat(Nx, Ny, Lx, Ly, np.expand_dims(cl_tt, 0), [0], seed=rseed)

# Convolve with instrumental beam
x=np.arange(Nx)-Nx/2-1
y=np.arange(Ny)-Ny/2-1
X, Y=np.meshgrid(x, y)
R=np.sqrt(X**2+Y**2)*pix # deg
Planck_res=10./60 # deg
Planck_sig=Planck_res/2./np.sqrt(2*np.log(2.)) # deg
PSF=np.exp(-R**2/2/Planck_sig**2)
mpt_conv=convolve2d(mpt, PSF, mode='same')/np.sum(PSF)

# Check this method looks the same - can't check because random seed doesn't work!!
beam=np.exp(-0.5*l*(l+1)*(Planck_sig*np.pi/180)**2) # as defined in sandbox_validation/check_flag_pure.py.  Makes sense because FT of a Gaussian is a Gaussian, with sigma --> 1/sigma
cl_tt_conv = cl_tt*beam # beam in example 7 of Pymaster API 
cl_tt_conv3 = cl_tt*beam**2 # beam**2 in sandbox_validation/check_flat_pure.py - makes sense because we are taking the autocorrelation
np.random.seed(rseed)
mpt_conv2, = nmt.synfast_flat(Nx, Ny, Lx, Ly, np.expand_dims(cl_tt_conv, 0), [0], seed=rseed)
np.random.seed(rseed)
mpt_conv3, = nmt.synfast_flat(Nx, Ny, Lx, Ly, np.expand_dims(cl_tt_conv3, 0), [0], seed=rseed)

# Write out for reuse later
np.save('../data/mpt.npy', mpt)
np.save('../data/mpt_conv.npy', mpt_conv)
np.save('../data/mpt_conv2.npy', mpt_conv2)
np.save('../data/mpt_conv3.npy', mpt_conv3)

mask=np.ones_like(mpt)
f0 = nmt.NmtFieldFlat(Lx, Ly, mask, [mpt])
f0_conv = nmt.NmtFieldFlat(Lx, Ly, mask, [mpt_conv], beam=[l, beam])
f0_conv2 = nmt.NmtFieldFlat(Lx, Ly, mask, [mpt_conv2], beam=[l, beam])
f0_conv3 = nmt.NmtFieldFlat(Lx, Ly, mask, [mpt_conv3], beam=[l, beam])

l0_bins = np.arange(Nx/8) * 8 * np.pi/Lx
lf_bins = (np.arange(Nx/8)+1) * 8 * np.pi/Lx
b = nmt.NmtBinFlat(l0_bins, lf_bins)
ells_uncoupled = b.get_effective_ells()

w00 = nmt.NmtWorkspaceFlat()
w00.compute_coupling_matrix(f0, f0, b)
w00.write_to("../data/w00_flat.fits")

w00_conv = nmt.NmtWorkspaceFlat()
w00_conv.compute_coupling_matrix(f0_conv, f0_conv, b)
w00_conv.write_to("../data/w00_flat_conv.fits")

w00_conv2 = nmt.NmtWorkspaceFlat()
w00_conv2.compute_coupling_matrix(f0_conv2, f0_conv2, b)
w00_conv2.write_to("../data/w00_flat_conv2.fits")

w00_conv3 = nmt.NmtWorkspaceFlat()
w00_conv3.compute_coupling_matrix(f0_conv3, f0_conv3, b)
w00_conv3.write_to("../data/w00_flat_conv3.fits")

# Computing power spectra:
# As in the full-sky case, you compute the pseudo-CL estimator by
# computing the coupled power spectra and then decoupling them by
# inverting the mode-coupling matrix. This is done in two steps below,
# but pymaster provides convenience routines to do this
# through a single function call
cl00_coupled = nmt.compute_coupled_cell_flat(f0, f0, b)
cl00_uncoupled = w00.decouple_cell(cl00_coupled)

cl00_coupled_conv = nmt.compute_coupled_cell_flat(f0_conv, f0_conv, b)
cl00_uncoupled_conv = w00_conv.decouple_cell(cl00_coupled_conv)

cl00_coupled_conv2 = nmt.compute_coupled_cell_flat(f0_conv2, f0_conv2, b)
cl00_uncoupled_conv2 = w00_conv2.decouple_cell(cl00_coupled_conv2)

cl00_coupled_conv3 = nmt.compute_coupled_cell_flat(f0_conv3, f0_conv3, b)
cl00_uncoupled_conv3 = w00_conv3.decouple_cell(cl00_coupled_conv3)

# Let's look at the results!
plt.figure()
plt.plot(l, cl_tt, 'r-', label='Input TT')
plt.plot(ells_uncoupled, cl00_uncoupled[0], 'r.', label='Uncoupled, no beam')
plt.plot(ells_uncoupled, cl00_uncoupled_conv[0], 'b.', label='Uncoupled, map beam')
plt.plot(ells_uncoupled, cl00_uncoupled_conv2[0], 'k.', label='Uncoupled, cl beam')
plt.plot(ells_uncoupled, cl00_uncoupled_conv3[0], 'kx', label='Uncoupled, cl beam**2')
plt.axvline(x=ell_max, color='r', label='Pixel size')
plt.axvline(x=np.pi/(Planck_res*np.pi/180), color='b', label='Planck FWHM')
plt.legend(loc='best')
plt.loglog()
plt.savefig("../images/beam_test.png",dpi=400)
plt.show()
