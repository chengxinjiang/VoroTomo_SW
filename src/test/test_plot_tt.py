import glob
import numpy as np 
import matplotlib.pyplot as plt 

RR = 6371.
def sph2xyz(theta,phi):
    '''
    Map spherical coordinates to Cartesian coordinates.
    
    '''
    xx = RR*np.sin(theta)*np.cos(phi)
    yy = RR*np.sin(theta)*np.sin(phi)
    return xx, yy

# find all npz files
allfiles = glob.glob('st*.npz')

# loop through each result
for ii in range(len(allfiles)):
    tfile = allfiles[ii]

    tmod = np.load(tfile)

    tt = tmod['tt']
    nodes = tmod['nodes']
    xx,yy = sph2xyz(nodes[...,1],nodes[...,2])
    xs,ys = sph2xyz(tmod['source'][0],tmod['source'][1])

    plt.figure()
    plt.scatter(
        xx.flatten(),
        yy.flatten(),
        c=tt.flatten(),
        vmin=tt.min(),
        vmax=tt.max(),
        cmap='jet',
        s=4
    )
    plt.plot(xs,ys,'*')
    plt.colorbar()
    plt.show()