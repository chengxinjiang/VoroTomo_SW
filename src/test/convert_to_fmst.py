import os 
import glob 
import numpy as np 
import pandas as pd 

'''
this script reads the traveltime for VoroTomo_2D and convert it
for fmst code
'''

# absolute path for the data
rootpath = '/Users/chengxin/Documents/ANU/St_Helens/Voro_Tomo/synthetic/fmst_tomo/ttime/Rayleigh'
dfiles   = glob.glob(os.path.join(rootpath,'selection*.dat'))

# total number of stations
nsta = 190
std  = 0.2

# loop through each file
for ii in range(len(dfiles)):
    dfile = dfiles[ii]
    per   = dfile.split('/')[-1].split('.')[0].split('_')[-1]

    # load data
    data  = pd.read_csv(dfile)
    indxs = np.array(data['evt_id'])
    indxr = np.array(data['sta_id'])
    vel   = np.array(data['vel'])
    dist  = np.array(data['dist'])

    # ready for conversion
    fout = open(rootpath+'/otimes_'+str(per)+'.dat','w')
    for iss in range(nsta):
        tindx1 = np.where(indxs==iss)[0]
        for irr in range(nsta):
            tindx2 = np.where(indxr==irr)[0]
            tindx  = np.intersect1d(tindx1,tindx2)
            print('found %d for %d and %d pairs'%(len(tindx),iss,irr))

            if len(tindx)==1:
                fout.write('1 %10.6f %3.1f\n'%(dist[tindx]/vel[tindx],std))
            else:
                fout.write('0 0 10\n')
    
    fout.close()