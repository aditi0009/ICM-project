"""This script:
    -> Tries to find the optimum parameters of an elliptical model, namely 
       x_centre, y_centre, f and theta
    -> It does so by running through all the values of centre within a reasonable
       range with a reasonable step size (upto to the user depeding on time). 
       This is to get a rough estimate of centre coordinates as well as the 
       elliptical model paramters before running a finer gradient descent in the
       next step starting from the vicinity of the obtained minima.
    -> The optimization criteria is such that the sum of norm of fluctuations
       around the centre within a particular distance is minimized
    -> The 'get_dipoleMetric' function follows the same steps as 'y_flucuations.py'
       except instead of known elliptical paramater values, it starts with arbitrary
       parameters. For more details, refer to 'y_fluctuations.py'
       
NOTE: 
    -> This script might take a while to run depending on the step sizes and the 
         range of the parameters chosen. 
    -> This script finds out the optimal centre coordinates and optimal elliptical
       model coordinates separately but as mentioned, this is just to get a rough 
       estimation subject to refinement using gradient descent
"""

import numpy as np
import matplotlib.pyplot as plt
from astropy.io import fits
from Computing_ys_in_annuli import get_2Dys_in_annuli

pixsize = 1.7177432059 #size of 1 pixel in arcmin
    
f = fits.open('../data/map2048_MILCA_Coma_20deg_G.fits')
data = f[1].data

"""Params:
      params: array-like; corresponding to the value of an elliptical model 
              (Refer to 'Computing_ys_in_annuli.py')
   Returns:
      The value of the metric to measure dipole structure   
"""
def get_dipoleMetric(params):    
    rs,ys,step_size,elliptical_model_vals = get_2Dys_in_annuli(params)
    x_cen = elliptical_model_vals[0]
    y_cen = elliptical_model_vals[1]
    f = elliptical_model_vals[2]
    theta = elliptical_model_vals[3]
    
    ys_new = np.zeros(len(ys)+1)
    rs_new = np.zeros(len(rs)+1)
    rs_new[1:] = rs
    ys_new[1:] = ys

    x=np.arange(700) # where npix is the number of pixels in the x direction, ie NAXIS1 from the header
    y=np.arange(700) # as above, NAXIS2=NAXIS1
    ys_new[0]= ys[0]
    
    image_length=120*2 #arcmin = 4 degrees    
    x_ind = np.nonzero(((x-x_cen)*pixsize>=-(image_length/2)) & 
                       (((x-x_cen)*pixsize<=(image_length/2))))
    y_ind = np.nonzero(((y-y_cen)*pixsize>=-(image_length/2)) & 
                       (((y-y_cen)*pixsize<=(image_length/2))))
    
    y_fluc = np.zeros((len(x_ind[0]),len(y_ind[0])))
    normalised_y_fluc = np.zeros((len(x_ind[0]),len(y_ind[0])))
    
    for t1,rx in enumerate(x_ind[0]):
        for t2,ry in enumerate(y_ind[0]):
            r_ellipse = np.sqrt((f*(np.cos(theta))**2 + 1/f*(np.sin(theta))**2)*(rx - x_cen)**2  \
                            + (f*(np.sin(theta))**2 + 1/f*(np.cos(theta))**2)*(ry - y_cen)**2  \
                            + 2*(np.cos(theta))*(np.sin(theta))*(f-1/f)*(rx - x_cen)*(ry - y_cen))*pixsize
                
            y_radius = np.interp(r_ellipse, rs_new, ys_new)
            y_fluc[t1][t2] = data[rx][ry] - y_radius
            normalised_y_fluc[t1][t2] = y_fluc[t1][t2]/abs(y_radius)
    
    mid = int(len(y_fluc)/2.)  
    dist = int((len(y_fluc)/2.)/4)  #distance from the centre for the optimization criteria of metric
    y_opt = y_fluc[mid-int(dist):mid+int(dist),mid-dist:mid+int(dist)] # ys for which optimization will be carried out
    y_opt = y_opt.reshape(-1)
    idx_neg = np.where(y_opt<0)[0]
    idx_pos = np.where(y_opt>0)[0]
    # you can try assigning different weights to the positive and negative fluctuations 
    dipole_metric = sum(abs(y_opt[i]) for i in idx_neg)+ \
                    sum(abs(y_opt[i]) for i in idx_pos)
    return dipole_metric


#THIS SECTION IS USED TO FIND THE OPTIMAL CENTRE COORDINATES ONLY
"""
xi,xf = 345,355.01
yi,yf = 345,355.01
steps_x = 1
steps_y = 1
size_x = int(((xf-xi)/steps_x)+1)
size_y = int((yf-yi)/steps_y+1)
metrics = np.ones((size_x,size_y))
xc,yc = xi,yi
print(metrics)
itr1, itr2 = 0,0
while xc<=xf:
    while yc<=yf:
        metrics[itr1,itr2] = get_dipoleMetric([xc,yc,1.,0])
        itr2 += 1
        yc+=steps_y
    yc = yi
    itr2 = 0
    itr1+=1
    xc+=steps_x
    print(".",end="")
    
print(metrics)
"""
#THIS SECTION IS USED TO FIND THE OPTIMAL ELLIPTICAL MODEL PARAMETERS ONLY

#initial, final values of f, theta
fi,ff= 0.4,3.01
theta_i,theta_f = 0, np.pi
steps_f = 0.2 #steps for f
steps_theta = 0.5 #steps for theta
#sizes resp.
size_f = int(((ff-fi)/steps_f)+1) 
size_theta = int((theta_f-theta_i)/steps_theta+1)

metrics = np.ones((size_f,size_theta))

fs,thetas = fi, theta_i
itr1, itr2 = 0,0
while fs<ff:
    while thetas<theta_f:
        metrics[itr1,itr2] = get_dipoleMetric([352.8,349.6,fs,thetas])
        itr2 += 1
        thetas+=steps_theta
    thetas = theta_i
    itr2 = 0
    itr1+=1
    fs+=steps_f
    print(".",end="")

minval = np.unravel_index(np.argmin(metrics), metrics.shape) 
f_opt = minval[0]*steps_f+fi
theta_opt = minval[1]*steps_theta+theta_i

print("\n\n")
print(metrics[minval[0],minval[1]],"\t Optimal (f, theta) = " , f_opt,theta_opt)

#USE THIS SECTION WHILE COMPUTING THE OPTIMAL CENTRE COORDINATES. COMMENT OUT THE OTHER PART.
"""
y_opt = minval[0]*steps_x+xi 
x_opt = minval[1]*steps_y+yi
print(metrics[minval[0],minval[1]],"\t Optimum (y,x) = " , y_opt,x_opt)
"""  
plt.imshow(metrics, cmap = 'Greys')
plt.annotate('minima',(minval[1],minval[0]),fontsize=14, color = 'red', alpha = 1)
plt.colorbar()
plt.xlabel("Theta")
plt.ylabel("f")
plt.title("Dipole Metric")
#plt.savefig("../images/Dipole Metric.png", dpi = 400)
plt.show()
