"""
This script:
    -> Computes y values in bins calling the 'get_2Dys_in_annuli' function
    -> Fits a beta model to those values and plots it
    
NOTE: The optimized centre values has been computed using a gradient descent 
      whose goal was the minimize the dipole in the y fluctuations map. For more details,
      refer to 'gradient_descent.py', 'minimize_dipole.py' and 'y_fluctuations.py'
"""
import numpy as np
import matplotlib.pyplot as plt
import scipy.optimize as opt

from Computing_ys_in_annuli import get_2Dys_in_annuli
#the centre values have been optimized using gradient_descent.py 
rs,ys,step_size,elliptical_model_vals = get_2Dys_in_annuli([352.42445296,349.85768166,1,0] )

pixsize = 1.7177432059     #size of 1 pixel in arcmin
arcmin2kpc =  27.052    # conversion factor from arcmin to kpc

"""params:
    -> x: The radius at which the value is to be evaluated
    -> a: The value of the model at the centre
    -> b: The scale of the model; 2D: Theta_c, 3D: r_c
    -> c: The beta value of the model
  returns:
    -> The value predicted by the model
"""
def beta_model(x,a,b,c):
    return a/(1+(x/b)**2)**c
#Fitting done only for y's less than 2000 kpc (Khatri+ uses 1200 kpc but the parameters are not too sensitive to that)
idx = np.where (rs<(2000/arcmin2kpc))
rs_fit = rs[idx]
ys_fit = ys[idx]
constants = opt.curve_fit(beta_model, rs_fit, ys_fit,p0=[6e-5,400/arcmin2kpc,1.05])[0] #some inital parameter values given
a_fit = constants[0]
b_fit = constants[1]
c_fit = constants[2]

beta_fit = np.zeros_like(ys)
for i,ri in enumerate (rs):
    beta_fit[i] = beta_model(ri, a_fit, b_fit, c_fit)
    
print("\n")    
print("y0 = "+str(a_fit)+"  theta_c = " +str(arcmin2kpc*b_fit) + "  beta = " +str(c_fit)+ "\n")

idx_plot = np.where (rs<(3000/arcmin2kpc)) #for plotting upto 3000 kpc

plt.plot(rs[idx_plot]*arcmin2kpc, ys[idx_plot],'o',markersize='4',label = 'Milca y-profile')
plt.plot(rs[idx_plot]*arcmin2kpc, beta_fit[idx_plot], label = 'Beta-model fit with beta = %1.3f'%c_fit)
plt.xlabel("Distance from centre of cluster (kpc)" ,fontsize=11)
plt.ylabel("Average y profile" ,fontsize=11)
#plt.title("Avg y profile in bins of 3 arcmin", fontsize=13)
plt.legend()
plt.loglog()
plt.savefig('../images/Figure 2.png', dpi = 600)
plt.show()