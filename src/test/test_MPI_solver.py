import sys
import pykonal 
import numpy as np 
from mpi4py import MPI 

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

RR = 6371.

# define the region
latmin,dlat,nlat = 45.0,0.1,30                              # latitude range of the target region
lonmin,dlon,nlon = -124.3,0.1,50                            # longitude range of the target region
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

#--------MPI---------
comm = MPI.COMM_WORLD
rank = comm.Get_rank()
size = comm.Get_size()

if rank==0:
    splits = 3
else:
    splits = [None for _ in range(1)]

# broadcast the variables
splits = comm.bcast(splits,root=0)
extra = splits % size

# loop through each core
for ick in range(rank,splits,size):

    print('core %d'%ick)

    # initalize the velocity grid for the 3D spherical volume
    solver = pykonal.solver.PointSourceSolver(coord_sys='spherical')
    solver.velocity.min_coords     = RR,theta0,phi0
    solver.velocity.node_intervals = 1,dtheta,dphi
    solver.velocity.npts           = 1,ntheta,nphi
    solver.velocity.values         = vs0

    # initialize the wavefield at source
    theta_s, phi_s = geo2sph(np.random.rand()*(latmax-latmin)+latmin,np.random.rand()*(lonmax-lonmin)+lonmin)
    solver.src_loc = RR,theta_s,phi_s 
    solver.solve()

    print('eikonal solver done at %d'%ick)

    # save files to check whether each core works
    np.savez_compressed(
    'st_test_'+str(ick)+'.npz',
    tt=solver.traveltime.values,
    nodes=solver.traveltime.nodes,
    source=np.array([theta_s,phi_s]),
    npts=np.array([ntheta, nphi])
    )

comm.barrier()
if rank == 0:
    sys.exit()