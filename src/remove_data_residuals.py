import os 
import glob
import pykonal 
import numpy as np 
import pandas as pd 
import matplotlib.pyplot as plt 
from obspy.signal.regression import linear_regression

'''
this version removes the traveltime obserations that have residuals
beyond xx times of standard deviations based on the final model

it also plots the residuals as a function of distance and traveltime

by Chengxin Jiang @ANU (Mar/20)
'''

# define rootpath and find all files
rootpath = '/Users/chengxin/Documents/ANU/St_Helens/Voro_Tomo/real_data_inversion_Cascadia/Love_group'
pper = [2,3,4,5,6,8,10,12,14,16,18,20,24,28,32,36,40]

# loop through each period
for ii in range(len(pper)):
    per = pper[ii]

    count = 0
    # find the file
    dfile = rootpath+'/residuals/tomo_'+str(per)+'s.txt'
    if not os.path.isfile(dfile):
        print('continue! cannot find file %s'%dfile)

    # load the data
    data = pd.read_csv(dfile)
    dist = np.array(data['dist'])
    vel  = np.array(data['vel'])
    tt   = np.array(dist/vel)
    res1 = np.array(data['res_initial']).astype(np.float32)
    res2 = np.array(data['res_final']).astype(np.float32)

    # get statistics
    ave1 = np.mean(res1)
    std1 = np.std(res1)
    ave2 = np.mean(res2)
    std2 = np.std(res2)

    # make selections
    indx = np.where((res2>=ave2-2*std2)&(res2<=ave2+2*std2))[0]
    fout = open(rootpath+'/dispersion/selection_'+str(per)+'s.dat','w')
    fout.write('index,evt_id,evt_lon,evt_lat,sta_id,sta_lon,sta_lat,vel,dist\n')
    for jj in range(len(indx)):
        tindx = indx[jj]
        fout.write('%6d,%6d,%8.3f,%8.3f,%6d,%8.3f,%8.3f,%8.3f,%8.3f\n'%(data['index'].iloc[tindx],\
            data['evt_id'].iloc[tindx],data['evt_lon'].iloc[tindx],data['evt_lat'].iloc[tindx],\
            data['sta_id'].iloc[tindx],data['sta_lon'].iloc[tindx],data['sta_lat'].iloc[tindx],\
            vel[tindx],dist[tindx]))

    fout.close()

    # plot the observations 
    plt.figure(figsize=(10,5))
    # initial model
    plt.subplot(121)
    plt.scatter(dist,tt,c=res1,cmap='jet',s=8)
    plt.xlabel('distance [km]')
    plt.ylabel('travel time [s]')
    text = 'ave:'+str(float("{:.3f}".format(ave1)))+'km/s'
    plt.text(15+np.min(dist),0.9*np.max(tt),text)
    text = 'std:'+str(float("{:.3f}".format(std1)))+'km/s'
    plt.text(15+np.min(dist),0.85*np.max(tt),text)
    plt.title(str(per)+'s initial model')
    plt.colorbar()

    plt.subplot(122)
    # final model
    plt.scatter(dist,tt,c=res2,cmap='jet',s=8)
    plt.xlabel('distance [km]')
    plt.ylabel('travel time [s]')
    text = 'ave:'+str(float("{:.3f}".format(ave2)))+'km/s'
    plt.text(15+np.min(dist),0.9*np.max(tt),text)
    text = 'std:'+str(float("{:.3f}".format(std2)))+'km/s'
    plt.text(15+np.min(dist),0.85*np.max(tt),text)
    plt.title(str(per)+'s final model')
    plt.colorbar()
    outname = rootpath+'/dispersion/obs_residual_'+str(per)+'s.pdf'
    plt.savefig(outname,format='pdf',dpi=300)
    #plt.show()
