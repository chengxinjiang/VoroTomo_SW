import sys
import pykonal 
import numpy as np 
import matplotlib.pyplot as plt 

def geo2sph(lat, lon):
    '''
    transform from geographic coordinates to spherical on surface

    input parameters:
    -----------------
    lat: latitudes of the grids
    lon: longitude of the grids

    '''
    theta = np.pi/2 - np.radians(lat)
    phi   = np.radians(lon)
    return theta, phi

def sph2xyz(theta,phi):
    '''
    Map spherical coordinates to Cartesian coordinates.
    
    '''
    xx = RR*np.sin(theta)*np.cos(phi)
    yy = RR*np.sin(theta)*np.sin(phi)
    return xx, yy

RR = 6371.

# define the region
latmin,dlat,nlat = 15.0,0.1,500                              # latitude range of the target region
lonmin,dlon,nlon = -120.0,0.1,500                            # longitude range of the target region
latmax = latmin+nlat*dlat
lonmax = lonmin+nlon*dlon
latgrid = np.linspace(latmax,latmin,nlat+1)
longrid = np.linspace(lonmin,lonmax,nlon+1)
lon,lat = np.meshgrid(longrid,latgrid)
vs  = 4*np.ones((nlat+1)*(nlon+1),)   

# detailed parameters for the spherical grid
theta, phi   = geo2sph(lat,lon)
ntheta, nphi = len(np.unique(theta)), len(np.unique(phi))
dtheta       = np.mean(np.diff(np.unique(theta)))
dphi         = np.mean(np.diff(np.unique(phi)))
theta0       = np.min(theta)
phi0         = np.min(phi)
vs0          = vs.reshape((1,ntheta,nphi))

# initalize the velocity grid for the 3D spherical volume
solver = pykonal.solver.PointSourceSolver(coord_sys='spherical')
solver.velocity.min_coords     = RR,theta0,phi0
solver.velocity.node_intervals = 1,dtheta,dphi
solver.velocity.npts           = 1,ntheta,nphi
solver.velocity.values         = vs0

# initialize the wavefield at source
theta_s, phi_s = geo2sph(np.random.rand(1)*(latmax-latmin)+latmin,np.random.rand(1)*(lonmax-lonmin)+lonmin)
solver.src_loc = RR,theta_s,phi_s 
solver.solve()

print('eikonal solver done')

# plot the traveltime field
nodes = solver.tt.nodes
xx,yy = sph2xyz(nodes[...,1],nodes[...,2])
xs,ys = sph2xyz(theta_s,phi_s)

plt.figure()
plt.scatter(
    xx.flatten(),
    yy.flatten(),
    c=solver.tt.values.flatten(),
    vmin=solver.tt.values.min(),
    vmax=solver.tt.values.max(),
    cmap='jet',
    s=4
)
plt.plot(xs,ys,'*')
plt.colorbar()
plt.show()