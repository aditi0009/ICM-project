"""This script:
      -> Refines the parameters of elliptical model obtained from 'minimize_dipole.py'
         further using gradient descent to minimize the dipole metric
      -> Check 'minimize_dipole.py' and 'y_fluctuations.py' for refernce 

NOTE: 
      -> The learning rate, threshold and epsilon for numerical gradient computation
         has been manually adjusted for optimal performance after looking at the
         gradient and metric values 

WARNING: This script may take a while to run depeding on the initial chosen 
         parameters, learning rates and threshold of gradient descent
"""
import numpy as np
from astropy.io import fits
from Computing_ys_in_annuli import get_2Dys_in_annuli

pixsize = 1.7177432059   #size of 1 pixel in arcmin
arcmin2kpc =  27.052     # conversion factor from arcmin to kpc
f = fits.open('map2048_MILCA_Coma_20deg_G.fits')
data = f[1].data

"""
params:
    max_iterations: The maximum number of iterations before gradient descent stops
    threshold: The threshold of the difference between successive iteration;
               Stops if difference is less than threshold
    w_init: arraylike; the initial values of the elliptical model parameters 
    obj_func: The function for which gradient descent is to be performed
    learning_rate: This parameter decides how fast the descent will be. Make sure
                   the parameter isn't too large, or else gradient descent will 
                   diverge
 
Returns: 
    The history of parameters and the value of the objective function at those parameters
            
"""
def gradient_descent(max_iterations,threshold,w_init,
                     obj_func,learning_rate):
    
    w = w_init
    print(grad(w))
    w_history = w
    f_history = obj_func(w)
    delta_w = [0,0,0,0]
    i = 0
    diff = 1.0e10
    while  i<max_iterations and diff>threshold:
        delta_w[0] = -learning_rate*grad(w)[0] 
        delta_w[1] = -learning_rate*grad(w)[1] 
        delta_w[2] = -0.01*learning_rate*grad(w)[2] 
        delta_w[3] = -0.01*learning_rate*grad(w)[3] 
        w = [w[i]+delta_w[i] for i in range(4)]

        print(delta_w)
        print(w)
        
        # store the history of w and f
        w_history = np.vstack((w_history,w))
        f_history = np.vstack((f_history,obj_func(w)))
        
        # update iteration number and diff between successive values
        # of objective function
        i+=1
        diff = np.absolute(f_history[-1]-f_history[-2])
    
    return w_history,f_history


"""
params:
    w: The paramters for which to calculate the numerical gradient 

returns:
    arraylike; the partial differential values with respect to each paramter
"""
def grad(w):
    eps = 1e-3 #epsilon to numerically estimate gradient
    return [(fun([w[0]+eps,w[1],w[2],w[3]])-fun([w[0]-eps,w[1],w[2],w[3]]))/(2*eps), \
            (fun([w[0],w[1]+eps,w[2],w[3]])-fun([w[0],w[1]-eps,w[2],w[3]]))/(2*eps), \
            (fun([w[0],w[1],w[2]+eps,w[3]])-fun([w[0],w[1],w[2]-eps,w[3]]))/(2*eps), \
            (fun([w[0],w[1],w[2],w[3]+eps])-fun([w[0],w[1],w[2],w[3]-eps]))/(2*eps)] 

"""
params:
    w: The elliptical model parameters 
returns:
    value of the dipole metric for given parameters
"""
def fun(w):    
    rs,ys,step_size,maxval = get_2Dys_in_annuli(w)
    x_cen = maxval[0]
    y_cen = maxval[1]
    f = maxval[2]
    theta = maxval[3]
    
    ys_new = np.zeros(len(ys)+1)
    rs_new = np.zeros(len(rs)+1)
    
    rs_new[1:] = rs
    ys_new[1:] = ys
    ys_new[0]= ys[0]
    
    x=np.arange(700) # where npix is the number of pixels in the x direction, ie NAXIS1 from the header
    y=np.arange(700) # as above, NAXIS2=NAXIS1
     
    image_length=120*2 #arcmin 
    x_ind = np.nonzero(((x-x_cen)*pixsize>=-(image_length/2)) & (((x-x_cen)*pixsize<=(image_length/2))))
    y_ind = np.nonzero(((y-x_cen)*pixsize>=-(image_length/2)) & (((y-x_cen)*pixsize<=(image_length/2))))
    
    y_fluc = np.zeros((len(x_ind[0]),len(y_ind[0])))
    normalised_y_fluc = np.zeros_like(y_fluc)
        
    for t1,rx in enumerate(x_ind[0]):
        for t2,ry in enumerate(y_ind[0]):
            r_ellipse = np.sqrt((f*(np.cos(theta))**2 + 1/f*(np.sin(theta))**2)*(rx - x_cen)**2  \
                            + (f*(np.sin(theta))**2 + 1/f*(np.cos(theta))**2)*(ry - y_cen)**2  \
                            + 2*(np.cos(theta))*(np.sin(theta))*(f-1/f)*(rx - x_cen)*(ry - y_cen))*pixsize
                
            y_radius = np.interp(r_ellipse, rs_new, ys_new)
            y_fluc[t1][t2] = data[rx][ry] - y_radius
            normalised_y_fluc[t1][t2] = y_fluc[t1][t2]/abs(y_radius)
    
    mid = int(len(y_fluc)/2.)
    dist = int((len(y_fluc)/2.)/4)
    y_centre = y_fluc[mid-int(dist):mid+int(dist),mid-dist:mid+int(dist)]
    y_centre = y_centre.reshape(-1)
    idx_neg = np.where(y_centre<0)[0]
    idx_pos = np.where(y_centre>0)[0]
    dipole_metric = sum(abs(y_centre[i]) for i in idx_neg) +\
                    sum(abs(y_centre[i]) for i in idx_pos)
    return dipole_metric

w_init = [352.32799422, 350.37014835, 0.89435012, -0.64212693]
w_history,f_history = gradient_descent(50,1e-7,w_init,fun,1e2)

print(w_history)
print(f_history)