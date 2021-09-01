"""
This script consists of a single function which extracts data of y-map from a FITS file
and caclculates average 2D ys in bins by assuming an underlying elliptical model.

WARNING: The y map FITS file being called must be present in 'data' 
"""

import numpy as np
from astropy.io import fits

"""
params:
    -> elliptical_model_vals: Array-like with 4 elements 
    -> elliptical_model_vals[0]: x coordinate of the centre of the Cluster
    -> elliptical_model_vals[1]: y coordinate of the centre of the Cluster
    -> elliptical_model_vals[2]: Range:[0,1]; Skewness of an Elliptical model
    -> elliptical_model_vals[3]: Range:[0,pi]; Rotation of the major axis from the x-axis
returns:
    -> rs: arraylike; The centre values for each bin
    -> ys: arraylike; The average ys in bins
    -> step_size: The size of each bin in arcmins
    -> elliptical_model_vals: Described above
"""
def get_2Dys_in_annuli(elliptical_model_vals):
    
    pixsize = 1.7177432059   #size of 1 pixel in arcmin 
    f = fits.open('../data/map2048_MILCA_Coma_20deg_G.fits')   
    data = f[1].data      
    x=np.arange(700) # where npix is the number of pixels in the x direction, ie NAXIS1 from the header
    y=np.arange(700) # as above, NAXIS2=NAXIS1    
    X, Y = np.meshgrid(x, y) # takes your vectors and converts them to a grid
    
    cen_x = elliptical_model_vals[0]
    cen_y = elliptical_model_vals[1]
    f = elliptical_model_vals[2]
    theta = elliptical_model_vals[3]
    
    # calculate distance in pixel coordinates from the centre – note this should not be rounded
    R_ellipse = np.sqrt((f*(np.cos(theta))**2 + 1/f*(np.sin(theta))**2)*(X - cen_x)**2+\
                         (f*(np.sin(theta))**2 + 1/f*(np.cos(theta))**2)*\
                         (Y - cen_y)**2 + 2*(np.cos(theta))*(np.sin(theta))*(f-1/f)*\
                         (X - cen_x)*(Y - cen_y))
    #Refer to y_fluctuations.py script to see how final r should be chosen                    
    r_initial = 0   
    r_final = 180 #arcmin
    step_size = 3 #arcmin
    
    rs_boundary=np.arange(r_initial,r_final,step_size) #computees the values of rs at the bounadries of bins
    
    ys = []
    rs = []
    for r in (rs_boundary):
        in_ann=np.nonzero((R_ellipse*pixsize>r) & (R_ellipse*pixsize<(r+step_size)))
        av=np.mean(data[in_ann])
        ys.append(av)
        rs.append(r+step_size/2)
    
    return np.array(rs), np.array(ys), step_size, elliptical_model_vals
     