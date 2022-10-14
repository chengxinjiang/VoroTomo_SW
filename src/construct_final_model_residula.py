import os 
import glob 
import sys
import scipy
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
#rootpath = '/Users/chengxin/Documents/ANU/St_Helens/Voro_Tomo/synthetic/Oct4_WithNoise_0.5s_spacing2/Love'
rootpath = "/Users/chengxin/Documents/ANU/St_Helens/Voro_Tomo/real_data_inversion_Cascadia/Love_phase/roundII"

# control parameter
cal_raypath = True                                                           # calculate the ray path of the averaged model
cal_resid  = False                                                           # calculate the residual of initial/final model
plot_final = False                                                           # plot the average model
plot_each  = False                                                           # plot tomo at each iteration 
write_each = False                                                           # write each model into txt file
real_data  = False                                                           # whether this is for real data or not

# several useful parameters
pper    = [3,4,5,6,8,10,12,14,16,18,20,24,28,32,36,40]                                    # all period range
nset    = 70                                                                 # number of inversions

# check whether dir exists or not
if not os.path.isdir(os.path.join(rootpath,'models')):
    os.mkdir(os.path.join(rootpath,'models'))
if not os.path.isdir(os.path.join(rootpath,'residuals')):
    os.mkdir(os.path.join(rootpath,'residuals'))
if plot_final and not os.path.isdir(os.path.join(rootpath,'figures')):
    os.mkdir(os.path.join(rootpath,'figures'))

# define geographic information
latmin,dlat,nlat = 24.5,0.2,123                                            # latitude range of the target region
lonmin,dlon,nlon = -124.8,0.2,289                                          # longitude range of the target region

latmin,dlat,nlat = 44.5,0.02,190                                            # latitude range of the target region
lonmin,dlon,nlon = -124.8,0.02,310                                          # longitude range of the target region
geogrid = {'latmin':latmin,'lonmin':lonmin,                                 # assemble geographic info into a dict
            'dlat':dlat,'dlon':dlon,
            'nlat':nlat,'nlon':nlon}

# grid for output files
latgrid = np.linspace(latmin+nlat*dlat,latmin,nlat+1) 
longrid = np.linspace(lonmin,lonmin+nlon*dlon,nlon+1)
lon,lat = np.meshgrid(longrid,latgrid)

#######################################
######## LOOPS THROUGH PERIODS ########
#######################################

# loop through all periods
for iper in range(len(pper)):
    tdlat,tnlat = 0.1,38
    tdlon,tnlon = 0.1,62
    tlatgrid = np.linspace(latmin+tnlat*tdlat,latmin,tnlat+1)
    tlongrid = np.linspace(lonmin,lonmin+tnlon*tdlon,tnlon+1)
    tlon,tlat = np.meshgrid(tlongrid,tlatgrid)
    tcount = np.zeros(tlon.size,dtype=np.int16)
    per = pper[iper]
    velall = np.zeros(((nlon+1)*(nlat+1),nset))                             # vector of the final model averaged over (nsets) times of realizations

    # find the observational data
    #dfile = os.path.join(rootpath,'dispersion/Love_phase/new_L{0:02d}_USANT15.txt'.format(per))
    dfile = os.path.join(rootpath,'../dispersion/selection_'+str(per)+'s.dat')
    if not os.path.isfile(dfile):
        raise ValueError('double check! cannot find %s'%dfile)

    # get averaged velocity
    data  = pd.read_csv(dfile)
    if real_data:
        ave = np.mean(data['vel'])
    else:
        ave = 3.0

    # load the velocity model
    allfiles = sorted(glob.glob(os.path.join(rootpath,'iterations/{0:d}s_*.npz'.format(per))))
    if not len(allfiles):
        raise ValueError('no model generated! double check')

    # load all models into a big matrix
    ii = 0
    for ifile in allfiles[:nset]:
        try:
            tvel = np.load(ifile)['vel']
            if plot_each:
                subf.plot_tomo(geogrid,-tvel*ave*ave+ave,rootpath,ii)
            if write_each:
                tvel1 = -tvel*ave*ave+ave
                # format the data column
                fout0 = open(rootpath+'/models/new_tomo_'+str(per)+'s_'+str(ii)+'m.txt','w')
                for jj in range(lon.size):
                    fout0.write('%6d %8.3f %8.3f %6.3f %6.3f 0.0\n'%(jj,lon.flatten()[jj],lat.flatten()[jj],tvel[jj],tvel1[jj]))
                fout0.close()
            velall[:,ii]  = tvel
            ii+=1
        except Exception as err:
            print('cannot load %s because %s'%(ifile,err))

    print('total # of models %d'%ii)
    # remove empty traces
    velall = velall[:,:ii]

    # averaged model and get stad
    tvel = -velall*ave*ave+ave
    vel_abs = np.mean(tvel,axis=1)
    vel_std = np.std(tvel,axis=1)
    vel_per = (vel_abs-ave)/ave*100

    #######################################
    ######## CALCULATE RAYPATPH ###########
    #######################################

    if cal_raypath:
        sphgrid = subf.construct_grid(geogrid,ave)                                  # spherical grid data and initial velocity model
        xygrid = np.zeros(shape=(tlon.size,2),dtype=np.float32)
        ttheta,tphi = subf.geo2sph(tlat.flatten(),tlon.flatten())
        xygrid[:,0],xygrid[:,1] = subf.sph2xyz(ttheta,tphi)
        sphgrid['vel1'] = np.reshape(vel_abs,(1,sphgrid['ntheta'],sphgrid['nphi']))

        #######################################
        ######## CALCULATE RESIDUALS ##########
        #######################################

        if cal_resid:
            fname = os.path.join(rootpath,'residuals/tomo_'+str(per)+'s.txt')                     
            res_initial,res_final = subf.calculate_residuals(data,geogrid,sphgrid,fname)
            
            # plot the histogram
            nbin = 30
            plt.close('all')
            plt.hist(res_initial,nbin,density=True,facecolor='g',alpha=0.6,label='initial')
            plt.hist(res_final,nbin,density=True,facecolor='b',alpha=0.6,label='final')
            plt.legend(loc='upper right')
            plt.savefig(rootpath+'/residuals/hist'+str(per)+'s.pdf',format='pdf',mpi=300)
        
        srcs    = data.evt_id.unique()
        # loop through each virtual source
        for isc in srcs:
            print(len(srcs),isc)

            # source location to initialize the traveltime field
            arr4rc = data[data.evt_id.values==isc]
            lat_s,lon_s   = arr4rc['evt_lat'].iloc[0],arr4rc['evt_lon'].iloc[0]

            # generate parameters
            solver = subf.solve_source_region(geogrid,sphgrid,lat_s,lon_s,1)

            # loop through each receiver 
            for irc in range(len(arr4rc)):
                try:

                    # receiver location
                    lat_r,lon_r = arr4rc['sta_lat'].iloc[irc],arr4rc['sta_lon'].iloc[irc]
                    theta_r,phi_r = subf.geo2sph(lat_r,lon_r)
                    src_pos = (RR,theta_r,phi_r)

                    # do ray tracing here
                    ray = solver.trace_ray(np.array(src_pos))
                    ray = pd.DataFrame(ray,columns=('rho','theta','phi'))
                    ray['x'] = ray['rho']*np.sin(ray['theta'])*np.cos(ray['phi'])
                    ray['y'] = ray['rho']*np.sin(ray['theta'])*np.sin(ray['phi'])
                    dist = scipy.spatial.distance.cdist(ray[['x','y']].values,xygrid)
                    argmin = np.argmin(dist,axis=1)
                    uargmin = np.unique(argmin)
                    tcount[uargmin] += 1

                except Exception as err:
                    print('ERROR for',irc,isc,err)               

    fout1 = open('{0:s}/ray_{1:02d}s.txt'.format(rootpath,per),'w')
    ttlon,ttlat = tlon.flatten(),tlat.flatten()
    for ii in range(tlon.size):
       fout1.write('%8.3f %8.3f %d\n'%(ttlon[ii],ttlat[ii],tcount[ii]))
    fout1.close()
    # format the data column
    fout = open('{0:s}/models/L_tomo_{1:02d}s.txt'.format(rootpath,per),'w')
    for ii in range(lon.size):
        fout.write('%6d %8.3f %8.3f %6.3f %6.3f %8.4f\n'%(ii,lon.flatten()[ii],lat.flatten()[ii],vel_per[ii],\
            vel_abs[ii],vel_std[ii]))
    fout.close()

    # plot the model
    if plot_final:
        subf.plot_tomo(geogrid,vel_abs,rootpath,-1)
