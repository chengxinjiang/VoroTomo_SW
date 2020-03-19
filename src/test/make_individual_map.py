import os
import glob 
import numpy as np     
import pandas as pd
    
'''
this script converts the np format of the model from each 
run into a txt file for plotting purpose

by Chengxin @ANU (Mar2020)
'''

# define absolute  path
rootpath = '/Users/chengxinjiang/Documents/ANU/Voro_Tomo/synthetic'
pper = [10]

# define geographic information
latmin,dlat,nlat = 45.0,0.02,150                                            # latitude range of the target region
lonmin,dlon,nlon = -124.3,0.02,250                                          # longitude range of the target region

# loop through each period
for per in pper:
    allfiles = glob.glob(os.path.join(rootpath,'iterations/'+str(per)+'s_*.npz'))
    dfile    = os.path.join(rootpath,'dispersion_selected/Rayleigh/selection2_'+str(per)+'s.dat')

    # load the real data to get model average used in the inversion
    if not os.path.isfile(dfile):
        raise ValueError('double check! cannot find %s'%dfile)

    # get averaged velocity
    ave  = np.mean(pd.read_csv(dfile)['vel'])

    # loop through each model
    for imod,ifile in enumerate(allfiles):
        data = np.load(ifile)
        tvel = data['vel']

        vel_abs = -tvel*ave*ave+ave
        vel_per = (vel_abs-ave)/ave*100

        # output into txt file
        latgrid = np.linspace(latmin+nlat*dlat,latmin,nlat+1) 
        longrid = np.linspace(lonmin,lonmin+nlon*dlon,nlon+1)
        lon,lat = np.meshgrid(longrid,latgrid)

        # format the data column
        fout = open(rootpath+'/videos_tomo/tomo_'+str(per)+'s_'+str(imod)+'m.txt','w')
        for ii in range(lon.size):
            fout.write('%6d %8.3f %8.3f %6.3f %6.3f\n'%(ii,lon.flatten()[ii],lat.flatten()[ii],vel_per[ii],vel_abs[ii]))
        fout.close()