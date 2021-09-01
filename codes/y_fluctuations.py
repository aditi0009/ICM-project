"""
This script:
    -> Computes y values in bins calling the 'get_2Dys_in_annuli' function
    -> Interpolates between the bin-averaged values to estimate y at any distance r
       from the centre
    -> Uses the interpolated values to find the fluctuation from the average
       at every pixel
    -> Plots the fluctuation and normalised fluctuation map along with the 
       rectangular patch for which the elliptical model optimzation was carried
       out (refer to minimize_dipole.py and gradient_descent.py) as well as the 
       group of galaxies within the coma cluster (NGC4839)

NOTE:
    -> For the interpolation between r=0 and r=(centre of first bin), we made the 
       choice of y(r=0) = y(r=(centre of first bin)). However, things like using
       the y0 value of the underlying beta model fit can also be used for y(r=0)
    -> The centre of NGC4839 has been estimated using software DS9 which can 
       provide the corresponding pixel value for a given WCS coordinate.
    -> The maximum radius of the fluctuation map (variable: r_ellipse_max) 
       must not exceed the final r (variable: r_final) from Computing_ys_in_annuli.py 
       
WARNING: The y map FITS file being called must be present in 'data'
"""

import numpy as np
import matplotlib.pyplot as plt
from astropy.io import fits
import matplotlib.patches as patches

pixsize = 1.7177432059

from Computing_ys_in_annuli import get_2Dys_in_annuli

"""
For the original map, the optimal centre coordinates were:
[x,y,f,theta] = [352.42445296,349.85768166,1,0] 
The WCS of these coordinates were estimated using DS9

For the first and last maps used to calculate Cross power spectrum,
the centre coordinates corresponding to the above estimated WCS were:
[x,y,f,theta] =  [349.3,351.7,1,0]

For the original map, the optimal elliptical model parameters were:
"""

rs,ys,step_size,maxval = get_2Dys_in_annuli([352.42445296,349.85768166,1,0])
#rs,ys,step_size,maxval = get_2Dys_in_annuli([349.3,351.7,1,0])
#rs,ys,step_size,maxval = get_2Dys_in_annuli([352.42445296,349.85768166,0.89232604,-0.63664394])
#rs,ys,step_size,maxval = get_2Dys_in_annuli([352.4023856 , 350.23374859, 0.52490389, -0.51376102])

#To account for the region between r=0 and r=(centre of first bin)
ys_new = np.zeros(len(ys)+1)
rs_new = np.zeros(len(rs)+1)
rs_new[1:] = rs
ys_new[1:] = ys
ys_new[0]= ys[0]

f = fits.open('../data/map2048_MILCA_Coma_20deg_G.fits')
data = f[1].data

x=np.arange(700) # where npix is the number of pixels in the x direction, ie NAXIS1 from the header
y=np.arange(700) # as above, NAXIS2=NAXIS1

image_length=120*2 #arcmin
print("\nlength of image in arcmin = ",image_length)

x_cen = maxval[0]
y_cen = maxval[1]
f = maxval[2]
theta = maxval[3]

#Note that centre of the cluster will always be the centre of the image 
x_ind = np.nonzero(((x-x_cen)*pixsize>=-(image_length/2)) & (((x-x_cen)*pixsize<=(image_length/2))))
y_ind = np.nonzero(((y-y_cen)*pixsize>=-(image_length/2)) & (((y-y_cen)*pixsize<=(image_length/2))))

y_fluc = np.zeros((len(x_ind[0]),len(y_ind[0])))
normalised_y_fluc = np.zeros_like(y_fluc)

r_ellipse_max = 0.
for t1,rx in enumerate(x_ind[0]):
    for t2,ry in enumerate(y_ind[0]):
        r_ellipse = np.sqrt((f*(np.cos(theta))**2 + 1/f*(np.sin(theta))**2)*(rx - x_cen)**2  \
                        + (f*(np.sin(theta))**2 + 1/f*(np.cos(theta))**2)*(ry - y_cen)**2  \
                        + 2*(np.cos(theta))*(np.sin(theta))*(f-1/f)*(rx - x_cen)*(ry - y_cen))*pixsize
        if r_ellipse>r_ellipse_max:
            r_ellipse_max = r_ellipse
            
        y_radius = np.interp(r_ellipse, rs_new, ys_new)
        y_fluc[t1][t2] = data[rx][ry] - y_radius
        normalised_y_fluc[t1][t2] = y_fluc[t1][t2]/abs(y_radius)

# The maximum value should be less than the r_final from Compute_ys_in_annuli.py
print("\nMaximum radius in arcmin =",r_ellipse_max)

np.savetxt("../data/normalised_y_fluc.txt", normalised_y_fluc)
np.savetxt("../data/y_fluc.txt", y_fluc)

mid = int(len(y_fluc)/2.)
dist = int((len(y_fluc)/2.)/4)
NGC4839 = [345, 381]

print("\nLength of y fluctuation matrix =",len(y_fluc))

from matplotlib.colors import TwoSlopeNorm     #Needed in plotting diverging colormaps

x_ticks = ['-2', '-1','0','1','2']
y_ticks = ['-2', '-1','0','1','2']
t11 = [0,35,70,105,138]

plt.figure()
plt.xticks(ticks=t11, labels=x_ticks, size='small')
plt.yticks(ticks=t11, labels=y_ticks, size='small')

norm = TwoSlopeNorm(vmin=y_fluc.min(), vcenter=0, vmax=y_fluc.max())
pc = plt.pcolormesh(y_fluc, norm=norm, cmap="seismic")     
plt.imshow(y_fluc, cmap = 'seismic')
ax = plt.gca()
rect = patches.Rectangle((mid-dist,mid-dist), 2*dist, 2*dist, linewidth=2,
                         edgecolor='black', facecolor="none")
ax.add_patch(rect)
circ = patches.Circle((NGC4839[0]-x_ind[0][0],NGC4839[1]-y_ind[0][0]), 15/pixsize , linewidth=2,
                         edgecolor='green', facecolor="none")
ax.add_patch(circ)
plt.colorbar(pc)
plt.xlabel("degrees")
plt.ylabel("degrees")
#plt.title("y fluctuations in MILCA (f=%2.2f)"%maxval[2])
plt.savefig("../images/figure 3 (skewed).png", dpi = 400)
plt.show()


plt.figure()
plt.xticks(ticks=t11, labels=x_ticks, size='small')
plt.yticks(ticks=t11, labels=y_ticks, size='small')
 
norm = TwoSlopeNorm(vmin=-1.5, vcenter=0, vmax=1.5) 
pc = plt.pcolormesh( normalised_y_fluc, norm=norm, cmap="seismic") 
plt.imshow(normalised_y_fluc, cmap = 'seismic')
ax = plt.gca()
circ = patches.Circle((NGC4839[0]-x_ind[0][0],NGC4839[1]-y_ind[0][0]), 15/pixsize , linewidth=2,
                         edgecolor='green', facecolor="none")
ax.add_patch(circ)
plt.colorbar(pc)
plt.xlabel("degrees")
plt.ylabel("degrees")
#plt.title("normalised y fluctuations in MILCA (f=%2.2f)"%maxval[2])
#plt.savefig("../images/figure 4.png", dpi = 400)
plt.show()
