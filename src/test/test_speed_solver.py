import time
import pykonal
import numpy as np 
import subfunction as subf
import matplotlib.pyplot as plt 

'''
this script compares the accuracy of the resulted traveltime 
wavefield from the module of PointSourceSolver and the EikonalSolver
for a 2D sphereical shell
'''

RR = 6371.0

###############################################
######## METHOD I: PointSourceSolver ##########
###############################################

# basic parameters for the grid
latmin,dlat,nlat = 45.0,0.05,120                                             # latitude range of the target region
lonmin,dlon,nlon = -124.3,0.05,200                                          # longitude range of the target region
velall           = np.zeros((nlon+1)*(nlat+1),)                             # vector of the final model averaged over (nsets) times of realizations
geogrid = {'latmin':latmin,'lonmin':lonmin,                                 # assemble geographic info into a dict 
            'dlat':dlat,'dlon':dlon,
            'nlat':nlat,'nlon':nlon}

# construct spherical grid for pykonal solver
ave = 4.0
sphgrid = subf.construct_grid(geogrid,ave)

# get source information
lat_s,lon_s   = 46.15,-123.67
theta_s,phi_s = subf.geo2sph(lat_s,lon_s)

t0 = time.time()
# initalize the velocity grid for the 3D spherical volume
solver = pykonal.solver.PointSourceSolver(coord_sys='spherical')
solver.velocity.min_coords     = RR,sphgrid['theta0'],sphgrid['phi0']
solver.velocity.node_intervals = 1,sphgrid['dtheta'],sphgrid['dphi']
solver.velocity.npts           = 1,sphgrid['ntheta'],sphgrid['nphi']
solver.velocity.values         = sphgrid['vel0']

# initialize the wavefield at source
solver.src_loc = RR,theta_s,phi_s 
solver.solve()
t1 = time.time()

##################################################
############ METHOD II: EikonalSolver ############
##################################################

# define the finer grid around the source
slatmin,sdlat,snlat = lat_s-0.1,0.01,20  
slonmin,sdlon,snlon = lon_s-0.1,0.01,20
sgeogrid = {'latmin':slatmin,'lonmin':slonmin,                                 # assemble geographic info into a dict 
            'dlat':sdlat,'dlon':sdlon,
            'nlat':snlat,'nlon':snlon}

# construct spherical grid for pykonal solver
ssphgrid = subf.construct_grid(sgeogrid,ave)

t2 = time.time()
# coarse grid for the far field
solver_coarse = pykonal.EikonalSolver(coord_sys="spherical")
solver_coarse.vv.min_coords = RR,sphgrid['theta0'],sphgrid['phi0']
solver_coarse.vv.node_intervals = 1,sphgrid['dtheta'],sphgrid['dphi']
solver_coarse.vv.npts = 1,sphgrid['ntheta'],sphgrid['nphi']
solver_coarse.vv.values = sphgrid['vel0']

# find grid for the near field
solver_fine = pykonal.EikonalSolver(coord_sys="spherical")
solver_fine.vv.min_coords = RR,ssphgrid['theta0'],ssphgrid['phi0']
solver_fine.vv.node_intervals = 1, ssphgrid['dtheta'],ssphgrid['dphi']
solver_fine.vv.npts = 1, ssphgrid['ntheta'],ssphgrid['nphi']

# Interpolate the velocity field from the coarse grid onto the fine grid
solver_fine.vv.values = solver_coarse.vv.resample(solver_fine.vv.nodes.reshape(-1, 3)).reshape(solver_fine.vv.npts)

# Initialize the source node. make sure that one node lies directly on the source
src_idx = 0, 10, 10
solver_fine.tt.values[src_idx] = 0
solver_fine.unknown[src_idx] = False
solver_fine.trial.push(*src_idx)
solver_fine.solve()

# Resample the traveltime field from the fine grid onto the coarse grid.
tt = solver_fine.tt.resample(solver_coarse.tt.nodes.reshape(-1, 3)).reshape(solver_coarse.tt.npts)
for idx in np.argwhere(~np.isnan(tt)):
    idx = tuple(idx)
    solver_coarse.tt.values[idx] = tt[idx]
    solver_coarse.unknown[idx] = False
    solver_coarse.trial.push(*idx)
    
solver_coarse.solve()
t3 = time.time()
print('PointSource Solver and EikonalSolver take %6.2f and %6.2f s respectively'%(t1-t0,t3-t2))

#################################
######### PLOT RESULTS ##########
#################################
plt.close("all")
fig = plt.figure(figsize=(12, 6))
ax1 = fig.add_subplot(1, 3, 1, projection="3d")
ax2 = fig.add_subplot(1, 3, 2, projection="3d")
ax3 = fig.add_subplot(1, 3, 3, projection="3d")
ax1.set_title("Traveltime")
ax2.set_title("Traveltime")
ax3.set_title("Traveltime")

nodes = solver_coarse.vv.nodes
xx = nodes[...,0] * np.sin(nodes[...,1]) * np.cos(nodes[..., 2])
yy = nodes[...,0] * np.sin(nodes[...,1]) * np.sin(nodes[..., 2])
zz = nodes[...,0] * np.cos(nodes[...,1])
ax1.scatter(
    xx.flatten(),
    yy.flatten(),
    zz.flatten(),
    c=solver.tt.values.flatten(),
    cmap=plt.get_cmap('jet_r')
)
ax2.scatter(
    xx.flatten(),
    yy.flatten(),
    zz.flatten(),
    c=solver_coarse.tt.values.flatten(),
    cmap=plt.get_cmap("jet_r")
)
p3= ax3.scatter(
    xx.flatten(),
    yy.flatten(),
    zz.flatten(),
    c=solver_coarse.tt.values.flatten()-solver.tt.values.flatten(),
    cmap=plt.get_cmap("jet_r")
)
fig.colorbar(p3)