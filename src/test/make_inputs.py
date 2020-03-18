import os 
import glob
import numpy as np 
import pandas as pd 

'''
this script prepares traveltime observations for VoroTomo_SW

by Chengxin Jiang @ANU (Mar2020)
'''

# absolute path for the data
rootpath = '/Users/chengxinjiang/Documents/Github/VoroTomo_SW/example/real_data'
dfiles   = glob.glob(os.path.join(rootpath,'dispersion/LVC_snr8_*s.dat'))

# loop through the file
for ifile in dfiles:
    data = pd.read_csv(ifile)

    # find the unique stations
    slon = np.array(data['lons'])
    slat = np.array(data['lats'])
    rlon = np.array(data['lonr'])
    rlat = np.array(data['latr'])
    vel  = np.array(data['vel'])
    dist = np.array(data['dist'])

    # deal with the source list first
    _,indx1 = np.unique([slon,slat],axis=1,return_index=True)
    _,indx2 = np.unique([rlon,rlat],axis=1,return_index=True)

    # constructing the 2 arrays
    sta_list = []
    indx1 = np.sort(indx1)
    for ii in indx1:
        sta_list.append([slon[ii],slat[ii]])

    # add stations that are missing
    indx2 = np.sort(indx2)
    for ii in indx2:
        try:
            sta_list.index([rlon[ii],rlat[ii]])
        except Exception:
            print('find one missing pair %f %f'%(rlon[ii],rlat[ii]))
            sta_list.append([rlon[ii],rlat[ii]])

    # output to a file
    tfile = rootpath+'/dispersion/new_'+ifile.split('/')[-1]
    fout = open(tfile,'w')
    fout.write('index,evt_id,evt_lon,evt_lat,sta_id,sta_lon,sta_lat,vel,dist\n')

    # loop through each station as a source
    for jj in range(len(slon)):
        indxs = sta_list.index([slon[jj],slat[jj]])
        indxr = sta_list.index([rlon[jj],rlat[jj]])
        fout.write('%6d,%6d,%8.3f,%8.3f,%6d,%8.3f,%8.3f,%8.3f,%8.3f\n'%(jj,indxs,slon[jj],slat[jj],\
            indxr,rlon[jj],rlat[jj],vel[jj],dist[jj]))

    fout.close()
    print('done transformation for %s'%ifile)