import os 
import glob 
import pykonal 
import numpy as np 
import pandas as pd 
import subfunction as subf
import matplotlib.pyplot as plt

'''
a script to create synthetic I. checkerboard models and
II. spherical anomaly for VoroTomo_2D

by Chengxin Jiang@ANU (Mar/2020)
'''

# radius of Earth
RR = 6371.
syn_option = 'spherical'

# absoluate path for traveltime data and for output
rootpath = '/Users/chengxinjiang/Documents/ANU/Voro_Tomo/synthetic'
dfiles   = glob.glob(os.path.join(rootpath,'dispersion/Rayleigh/data_snr8_10s.dat'))

# full 2D research area
latmin,dlat,nlat = 35.5,0.02,100                                            # latitude range of the target region
lonmin,dlon,nlon = -120.0,0.02,100                                          # longitude range of the target region
geogrid = {'latmin':latmin,'lonmin':lonmin,                                 # assemble geographic info into a dict 
            'dlat':dlat,'dlon':dlon,
            'nlat':nlat,'nlon':nlon}

if syn_option=='checkerboard':
    # target anomaly information
    anomaly = {'avs':3,'amp':0.3,                                           # amplitude in percentage
                'size':15,'spacing': True,                                  # size of anomaly and whether of spacing
                'noise':0.3}

    # make sphgrid for traveltime prediction
    sphgrid = subf.construct_grid(geogrid,anomaly['avs'])                       # spherical grid data and initial velocity model
    amp0 = subf.make_checkerboard(geogrid,anomaly)
    amp0 = np.array(amp0)
    vel0 = np.reshape(amp0*anomaly['amp']+anomaly['avs'],(1,sphgrid['ntheta'],sphgrid['nphi']))
    sphgrid['vel0'] = vel0

elif syn_option == 'spherical':
    # target anomaly information
    anomaly = {'avs':3,'amp':-0.3,                                               # amplitude in percentage
                'latmin':46.1,'latmax':46.4,                                     # size of anomaly and whether of spacing
                'lonmin':-122.4,'lonmax':-122.1,
                'noise':0.3}

    # make sphgrid for traveltime prediction
    sphgrid = subf.construct_grid(geogrid,anomaly['avs'])                       # spherical grid data and initial velocity model
    vel0 = subf.make_syn(geogrid,anomaly)
    sphgrid['vel0'] = np.reshape(vel0,(1,sphgrid['ntheta'],sphgrid['nphi']))  

# output the starting model    
latgrid = np.linspace(latmin+nlat*dlat,latmin,nlat+1) 
longrid = np.linspace(lonmin,lonmin+nlon*dlon,nlon+1)
lon,lat = np.meshgrid(longrid,latgrid)
fout = open(rootpath+'/input_model.txt','w')
for ii in range(lon.size):
    fout.write('%6d %8.3f %8.3f %6.3f\n'%(ii,lon.flatten()[ii],lat.flatten()[ii],vel0.flatten()[ii]))
fout.close()

#################################
###### LOOP THROUGH PERIODS #####
#################################
if not len(dfiles):
    raise ValueError('no files of %s found'%dfiles)

for iper in range(len(dfiles)):
    # read traveltime info
    data   = pd.read_csv(dfiles[iper])                                              
    srcs    = data.evt_id.unique()

    # output file
    fout  = open(os.path.join(rootpath,dfiles[iper].split('/')[-1]),'w')
    fout.write('index,evt_id,evt_lon,evt_lat,sta_id,sta_lon,sta_lat,vel,dist\n')

    # loop through each virtual source
    for isc in srcs:
        print(len(srcs),isc)

        # source location to initialize the traveltime field
        arr4rc = data[data.evt_id.values==isc]
        lat_s,lon_s   = arr4rc['evt_lat'].iloc[0],arr4rc['evt_lon'].iloc[0]

        # initialize the wavefield near source location on a fine grid (initial/final model)
        solver = subf.solve_source_region(geogrid,sphgrid,lat_s,lon_s)  

        # loop through each receiver 
        for irc in range(len(arr4rc)):
            try:

                # receiver location
                lat_r,lon_r   = arr4rc['sta_lat'].iloc[irc],arr4rc['sta_lon'].iloc[irc]
                theta_r,phi_r = subf.geo2sph(lat_r,lon_r)
                src_pos  = (RR,theta_r,phi_r)
                syn_data = solver.traveltime.value(np.array(src_pos))
                if anomaly['noise']:
                    syn_data += np.random.normal(loc=anomaly['noise'],scale=0.05)
                fout.write('%6d,%6d,%8.3f,%8.3f,%6d,%8.3f,%8.3f,%8.3f,%8.3f\n'%(arr4rc['index'].iloc[irc],\
                    isc,lon_s,lat_s,arr4rc['sta_id'].iloc[irc],lon_r,lat_r,arr4rc['dist'].iloc[irc]/syn_data,\
                    arr4rc['dist'].iloc[irc]))

            except Exception as err:
                print('ERROR for',irc,isc,err)

    fout.close()
    print('synthetic done for %s'%dfiles[iper])