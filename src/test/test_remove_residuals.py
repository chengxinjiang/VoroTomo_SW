import os 
import glob
import numpy as np 
import pandas as pd 
import matplotlib.pyplot as plt 
from obspy.signal.regression import linear_regression

# define rootpath and find all files
rootpath = '/Users/chengxin/Documents/Research/Harvard/tomo/VoroTomo/real_data'
pper = [3,4,5,6,8,10,12,14,16,18,20,24,28,32]

# loop through each period
for ii in range(len(pper)):
    per = pper[ii]

    count = 0
    # find the file
    #dfile = rootpath+'/dispersion/Rayleigh/data_snr8_'+str(per)+'s.dat'
    dfile = rootpath+'/residuals/tomo_'+str(per)+'s.txt'
    if not os.path.isfile(dfile):
        print('continue! cannot find file %s'%dfile)

    # load the data
    data = pd.read_csv(dfile)
    vel  = np.array(data['vel'])
    dist = np.array(data['dist'])
    tt   = dist/vel
    res  = np.array(data['res_initial'])
    tindx= np.where(np.abs(res)>4)[0]

    # linear regression to get regional velocity average
    ave,unc = linear_regression(tt,dist,np.ones(len(dist)),intercept_origin=True)
    pos1 = np.zeros(shape=(2,2),dtype=np.float32)
    pos2 = np.zeros(shape=(2,2),dtype=np.float32)

    # two ways of removing outliers
    # I: cycle skipping
    pi2 = per 
    pos1[0] = [0,pi2]
    pos1[1] = [np.max(dist),np.max(dist)/ave+pi2]
    pos2[0] = [0,-pi2]
    pos2[1] = [np.max(dist),np.max(dist)/ave-pi2]

    ntt = np.copy(tt)
    # make corrections for each measurements
    for iii in range(len(dist)):
        if dist[iii]/ave - ntt[iii] <pi2 and dist[iii]/ave - ntt[iii] >-pi2:
            count+=1

        while True:
            dres = dist[iii]/ave - ntt[iii]

            if dres>pi2:
                ntt[iii] = ntt[iii]+per
            elif dres<-pi2:
                ntt[iii] = ntt[iii]-per
            else:
                break 

    # II: maximum velocity anomaly of 20%
    xx  = np.linspace(0,np.floor(np.max(dist)),100)
    yy1 = xx/(1.2*ave)
    yy2 = xx/(0.8*ave)

    # plot the observations before and after correction
    plt.figure(figsize=(10,5))
    plt.subplot(121)
    # before correction
    plt.scatter(dist,tt,color='r',s=8)
    plt.scatter(dist[tindx],tt[tindx],color='b',s=8)
    plt.plot(pos1[:,0],pos1[:,1],'--',linewidth=1,color='g')
    plt.plot(pos2[:,0],pos2[:,1],'--',linewidth=1,color='g')
    #plt.plot(xx,yy1,'--',linewidth=1,color='b')
    #plt.plot(xx,yy2,'--',linewidth=1,color='b')
    plt.xlabel('distance [km]')
    plt.ylabel('travel time [s]')
    text = 'ave='+str(float("{:.4f}".format(ave)))+'km/s'
    plt.text(0.7*np.max(dist),0.9*np.max(tt),text)
    plt.title(str(per)+'s before correction; '+str(len(tindx))+' bad')

    plt.subplot(122)
    # after correction
    plt.scatter(dist,ntt,color='b',s=8)
    plt.plot(pos1[:,0],pos1[:,1],'--',linewidth=1,color='g')
    plt.plot(pos2[:,0],pos2[:,1],'--',linewidth=1,color='g')
    plt.xlabel('distance [km]')
    plt.ylabel('travel time [s]')
    plt.title(str(per)+'s after correction; '+str(count)+'out of '+str(len(dist))+' good')
    outname = rootpath+'/dispersion/Rayleigh/obs_'+str(per)+'s.pdf'
    plt.savefig(outname,format='pdf',dpi=300)
    plt.close()

    # plot the geographic information of these arrays
    '''
    rlon = dfile['rlon'];rlat = dfile['rlat'] 
    slon = dfile['slon'];slat = dfile['slat']
    indx = np.where(np.abs(res)>4)[0]

    plt.figure()
    # loop through each segments
    for ii in range(len(indx)):
        tindx = indx[ii]
        plt.plot([slon[tindx],rlon[tindx]],[slat[tindx],rlat[tindx]],'r-')
    '''
        