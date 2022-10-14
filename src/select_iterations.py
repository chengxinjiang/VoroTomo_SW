import os 
import glob 
import numpy as np 
import pandas as pd 
import subfunction as subf
import matplotlib.pyplot as plt 

'''
approximate the evolution of MSE (mean squared error) with the number 
of iterations for selection of optimal # of iterations

by Chengxin @ANU (Mar2020)
'''

###################################
######## PARAMETER SECTION ########
###################################

# define absolute path
rootpath = '/Users/chengxin/Documents/ANU/St_Helens/Voro_Tomo/real_data/phase/iter2'

# control parameter
plot_each  = False                                                          # plot tomo at each iteration 
real_data  = True                                                           # whether this is for real data or not
write_modle= True                                                           # write the final model into file

# several parameters
nset  = 300                                                                 # number of inversions
per   = 4                                                                   # target period range
nnmod = [2,4,6,8,10,14,18,25,40,60,80,120,160,200,250,300]                      # number of models for sub-sampling
niter = len(nnmod)                                                          # number of final models to output

# check whether dir exists or not
if not os.path.isdir(os.path.join(rootpath,'select_iterations')):
    os.mkdir(os.path.join(rootpath,'select_iterations'))

# define geographic information
latmin,dlat,nlat = 45.0,0.02,150                                            # latitude range of the target region
lonmin,dlon,nlon = -124.3,0.02,250                                          # longitude range of the target region
geogrid = {'latmin':latmin,'lonmin':lonmin,                                 # assemble geographic info into a dict
            'dlat':dlat,'dlon':dlon,
            'nlat':nlat,'nlon':nlon}

# find the observational data
dfile = os.path.join(rootpath,'dispersion_selected/Rayleigh/selection2_'+str(per)+'s.dat')
if not os.path.isfile(dfile):
    raise ValueError('double check! cannot find %s'%dfile)

# get averaged velocity
data  = pd.read_csv(dfile)
if real_data:
    ave = np.mean(data['vel'])
else:
    ave = 4.0

###################################
######## DO SUB-SAMPLING ##########
###################################

# load the velocity model
allfiles = glob.glob(os.path.join(rootpath,'Rayleigh/iterations/'+str(per)+'s_*.npz'))
if not len(allfiles):
    raise ValueError('no model generated! double check')

'''
# loop through each subsampling
for iii in range(niter):
    nmod   = nnmod[iii]
    velall = np.zeros(((nlon+1)*(nlat+1),nmod))                       # vector of the final model averaged over (nsets) times of realizations
    indx = np.random.randint(0,nset-1,size=nmod)

    # load all models into a big matrix
    ii = 0
    for tindx in indx:
        ifile = allfiles[tindx]
        print(ifile,tindx)
        try:
            velall[:,ii] = np.load(ifile)['vel']
            ii+=1
        except Exception as err:
            print('cannot load %s because %s'%(ifile,err))

    # remove empty traces
    velall = velall[:,:ii]

    # averaged model and get stad
    tvel = -velall*ave*ave+ave
    vel_abs = np.mean(tvel,axis=1)
    vel_std = np.std(tvel,axis=1)
    vel_per = (vel_abs-ave)/ave*100

    # output into txt file
    latgrid = np.linspace(latmin+nlat*dlat,latmin,nlat+1) 
    longrid = np.linspace(lonmin,lonmin+nlon*dlon,nlon+1)
    lon,lat = np.meshgrid(longrid,latgrid)

    # calculate residuals 
    sphgrid = subf.construct_grid(geogrid,ave)  
    sphgrid['vel1'] = np.reshape(vel_abs,(1,sphgrid['ntheta'],sphgrid['nphi']))
    fname = os.path.join(rootpath,'select_iterations/residual_'+str(per)+'s_'+str(nmod)+'iter.dat')                     
    res_initial,res_final = subf.calculate_residuals(data,geogrid,sphgrid,fname)

    # format the data column
    if write_modle:
        fout = open(rootpath+'/select_iterations/tomo_'+str(per)+'s'+str(nmod)+'iter.dat','w')
        for ii in range(lon.size):
            fout.write('%6d %8.3f %8.3f %6.3f %6.3f %8.4f\n'%(ii,lon.flatten()[ii],lat.flatten()[ii],vel_per[ii],\
                vel_abs[ii],vel_std[ii]))
        fout.close()
'''

# summarize the MSE and plot the results
mse = np.zeros(len(nnmod),dtype=np.float32)
for ii,nmod in enumerate(nnmod):
    tfile = os.path.join(rootpath,'select_iterations/residual_'+str(per)+'s_'+str(nmod)+'iter.dat')
    tdata = pd.read_csv(tfile)
    res   = np.array(tdata['res_final'])
    mse[ii] = np.sum(res*res)/len(res)

# plot the results
plt.figure(figsize=(6,4))
plt.plot(nnmod,mse,'o-')
plt.savefig(rootpath+'/select_iterations/evolution.pdf', format='pdf', dpi=400)
plt.close()