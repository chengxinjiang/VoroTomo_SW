import os 
import glob 
import numpy as np 
import pandas as pd 
import subfunction as subf 
import matplotlib.pyplot as plt 
from obspy.signal.regression import linear_regression

'''
use outputs from VoroTomo_2D to generate the final model,
calculate the traveltime residuals and make plots if needed

by Chengxin Jiang @ANU (Mar/20)
'''
RR = 6371.

###################################
######## PARAMETER SECTION ########
###################################

# define absolute path
rootpath = '/Users/chengxinjiang/Documents/ANU/Voro_Tomo/synthetic'

# control parameter
cal_resid  = True                                                           # calculate the residual of initial/final model
plot_final = False                                                          # plot the average model
plot_each  = False                                                          # plot tomo at each iteration 
real_data  = False                                                          # whether this is for real data or not

# several useful parameters
pper    = [2,3,4,5,6,8,10,12,14,16,18,20,24,28]                                   # all period range
nset    = 300                                                               # number of inversions

# check whether dir exists or not
if not os.path.isdir(os.path.join(rootpath,'models')):
    os.mkdir(os.path.join(rootpath,'models'))
if not os.path.isdir(os.path.join(rootpath,'residuals')):
    os.mkdir(os.path.join(rootpath,'residuals'))
if plot_final and not os.path.isdir(os.path.join(rootpath,'figures')):
    os.mkdir(os.path.join(rootpath,'figures'))

# define geographic information
latmin,dlat,nlat = 35.5,0.02,100                                            # latitude range of the target region
lonmin,dlon,nlon = -120.0,0.02,100                                          # longitude range of the target region
geogrid = {'latmin':latmin,'lonmin':lonmin,                                 # assemble geographic info into a dict
            'dlat':dlat,'dlon':dlon,
            'nlat':nlat,'nlon':nlon}


#######################################
######## LOOPS THROUGH PERIODS ########
#######################################

# loop through all periods
for iper in range(len(pper)):
    per = pper[iper]
    velall = np.zeros(((nlon+1)*(nlat+1),nset))                             # vector of the final model averaged over (nsets) times of realizations

    # find the observational data
    dfile = os.path.join(rootpath,'dispersion/data_snr8_'+str(per)+'s.dat')
    if not os.path.isfile(dfile):
        raise ValueError('double check! cannot find %s'%dfile)

    # get averaged velocity
    data  = pd.read_csv(dfile)
    if real_data:
        ave = np.mean(data['vel'])
    else:
        ave = 3.0

    # load the velocity model
    allfiles = glob.glob(os.path.join(rootpath,'iterations/'+str(per)+'s_*.npz'))
    if not len(allfiles):
        raise ValueError('no model generated! double check')

    # load all models into a big matrix
    ii = 0
    for ifile in allfiles:
        try:
            tvel = np.load(ifile)['vel']
            if plot_each:
                subf.plot_tomo(geogrid,-tvel*ave*ave+ave,rootpath,ii)
            velall[:,ii]  = tvel
            ii+=1
        except Exception as err:
            print('cannot load %s because %s'%(ifile,err))

    # remove empty traces
    velall = velall[:,:ii]

    # averaged model and get stad
    tvel = np.mean(velall,axis=1)
    vel_std = np.std(velall,axis=1)

    # calculate the absolute vel and vel pertb
    vel_abs = -tvel*ave*ave+ave
    vel_per = (vel_abs-ave)/ave*100
    #print('old and new averages are %6.3f %6.3f'%(ave,nave))

    # output into txt file
    latgrid = np.linspace(latmin+nlat*dlat,latmin,nlat+1) 
    longrid = np.linspace(lonmin,lonmin+nlon*dlon,nlon+1)
    lon,lat = np.meshgrid(longrid,latgrid)

    # format the data column
    fout = open(rootpath+'/models/tomo_'+str(per)+'s.txt','w')
    for ii in range(lon.size):
        fout.write('%6d %8.3f %8.3f %6.3f %6.3f %8.4f\n'%(ii,lon.flatten()[ii],lat.flatten()[ii],vel_per[ii],\
            vel_abs[ii],vel_std[ii]))
    fout.close()


    #######################################
    ######## CALCULATE RESIDUALS ##########
    #######################################

    if cal_resid:
        sphgrid = subf.construct_grid(geogrid,ave)  
        sphgrid['vel1'] = np.reshape(vel_abs,(1,sphgrid['ntheta'],sphgrid['nphi']))
        fname = os.path.join(rootpath,'residuals/tomo_'+str(per)+'s.txt')                     
        res_initial,res_final = subf.calculate_residuals(data,geogrid,sphgrid,fname)
        
        # plot the histogram
        nbin = 30
        plt.close('all')
        plt.hist(res_initial,nbin,density=True,facecolor='g',alpha=0.6,label='initial')
        plt.hist(res_final,nbin,density=True,facecolor='b',alpha=0.6,label='final')
        plt.legend(loc='upper right')
        plt.savefig(rootpath+'/residuals/hist'+str(per)+'s.pdf',format='pdf',mpi=300)

    # plot the model
    if plot_final:
        subf.plot_tomo(geogrid,vel_abs,rootpath,-1)