#-.-.-.-.-.-.-.-.-.-.-.-.-.-.-.-.-.-.-.-.-.-.-.-.-.-.-.-.-.
# Random projections based seismic tomography using PyKonal
# Created By : Hongjian Fang: hfang@mit.edu at 22-10-2019
# Reference: Fang et al., (2020) doi:10.1785/0220190141
#
# Modified By Chengxin Jiang (chengxin.jiang1@anu.edu.au)
# at Mar-2020 for 2D phase velocity map inversions
#
# Also see https://github.com/malcolmw/pykonal for details on
# PyKonal package by Malcolm White (malcolcw@usc.edu)
#_._._._._._._._._._._._._._._._._._._._._._._._._._._._._.

import os
import sys
import glob
import pykonal
import numpy as np
import pandas as pd
import scipy.spatial
from mpi4py import MPI
import subfunction as subf
from scipy.sparse import coo_matrix
from scipy.sparse.linalg import lsmr
from obspy.signal.regression import linear_regression

'''
Notes:
    - updated with MPI (Feb26/2020)
    - use fine-grid at source locaiton for accurate traveltime estimation (Mar02/2020)
    - loops through all periods for each core (which represents each set of inversion)
    - fix one bug (hopefully the final one) in the script of calculating residuals
'''

# radius of Earth
RR = 6371.

##########################################
######## PARAMETER & SETUP SECTION #######
##########################################

# define the absolute path of data and project
rootpath = '/Users/chengxin/Documents/ANU/St_Helens/Voro_Tomo/synthetic/dispersion/Love'
data = glob.glob(os.path.join(rootpath,'syn_selection_*.dat'))

# useful parameters for location and inversion
ncell = 350                                                                 # number of Voronoi cells for the target region
nsets = 50                                                                 # number of realizations
latmin,dlat,nlat = 44.5,0.05,76                                             # latitude range of the target region
lonmin,dlon,nlon = -124.8,0.05,124                                          # longitude range of the target region
velall           = np.zeros(((nlon+1)*(nlat+1),nsets))                      # vector of the final model averaged over (nsets) times of realizations
geogrid = {'latmin':latmin,'lonmin':lonmin,                                 # assemble geographic info into a dict
            'dlat':dlat,'dlon':dlon,
            'nlat':nlat,'nlon':nlon}

#---------MPI---------
comm = MPI.COMM_WORLD
rank = comm.Get_rank()
size = comm.Get_size()
#---------------------

if rank==0:
    # make sub-directory to save intermediate files
    if not os.path.isdir(os.path.join(rootpath,'iterations')):
        os.mkdir(os.path.join(rootpath,'iterations'))

    # set up parameters to share between cores
    splits = int(nsets)
else:
    splits = None

splits     = comm.bcast(splits,root=0)

###############################################
########### LOOPS OF PV-BASED INVERSION #######
###############################################

for iset in range(rank,splits,size):
    print('iset #'+str(iset))

    # loop through each period
    for iper in range(len(data)):
        pper  = data[iper].split('/')[-1].split('.')[0].split('_')[-1]
        dfile = pd.read_csv(data[iper])

        # define the size of subsampling
        obs_all = len(dfile)
        obs_sub = int(np.floor(obs_all*0.8))

        # read traveltime info
        #ave     = float("{:.3f}".format(np.mean(dfile['vel'])))                     # averaged velocity from the observation
        ave     = 3
        sphgrid = subf.construct_grid(geogrid,ave)                                  # spherical grid data and initial velocity model

        # generate parameters
        cellpos, nearcells, Gp = subf.outputproj(geogrid,ncell)
        dfile_sub = dfile.sample(n=obs_sub)
        srcs    = dfile_sub.evt_id.unique()
        syndata = []
        dobs    = []
        colidp = []
        rowidp = []
        nonzerop = []
        ridx = 0

        # loop through each virtual source
        for isc in srcs:
            print(ridx,len(srcs),isc,end='\r')

            # source location to initialize the traveltime field
            arr4rc = dfile_sub[dfile_sub.evt_id.values==isc]
            lat_s,lon_s   = arr4rc['evt_lat'].iloc[0],arr4rc['evt_lon'].iloc[0]

            # initialize the wavefield near source location on a fine grid
            solver = subf.solve_source_region(geogrid,sphgrid,lat_s,lon_s)

            # loop through each receiver 
            for irc in range(len(arr4rc)):
                try:

                    # receiver location
                    lat_r,lon_r = arr4rc['sta_lat'].iloc[irc],arr4rc['sta_lon'].iloc[irc]
                    theta_r,phi_r = subf.geo2sph(lat_r,lon_r)
                    src_pos = (RR,theta_r,phi_r)

                    # find the travetime at receiver location
                    syndata = solver.traveltime.value(np.array(src_pos))

                    # do ray tracing here
                    ray = solver.trace_ray(np.array(src_pos))
                    ray = pd.DataFrame(ray,columns=('rho','theta','phi'))
                    ray = subf.findrayidx(ray,cellpos)
                    ray = ray.groupby('cellidx').count()
                    dres = (arr4rc['dist'].iloc[irc]/arr4rc['vel'].iloc[irc])-syndata

                    # remove really bad ones if necessary
                    if np.abs(dres)>4:continue
                    dobs.append(dres)

                    # construct the sensitivity kernels
                    for iseg in range(len(ray)):
                        colidp.append(ray.index.values[iseg])
                        rowidp.append(ridx)
                        nonzerop.append(solver.step_size*ray['phi'].iloc[iseg])

                    ridx += 1
                except Exception as err:
                    print('ERROR for',irc,isc,err)
                    syndata = np.nan

        G = coo_matrix((nonzerop,(rowidp,colidp)),shape=(ridx,ncell))

        # control parameters for LSQR
        atol = 1e-3
        btol = 1e-4
        maxiter = 100
        conlim = 50
        damp = 1.0

        # LSQR algorithm for inversion
        x = lsmr(G,dobs,damp,atol,btol,conlim,maxiter,show=False)
        x = x[0]
        vel = Gp*x

        # output velocity model for each iteration
        tname = rootpath+'/iterations/'+str(pper)+'_inversion'+str(iset)+'.npz'
        np.savez_compressed(tname,vel=vel)

        # plot each inversion results
        #subf.plot_tomo(geogrid,vel,rootpath,iset)

############################################################
######### RUN construct_final_model_residuals NEXT #########
############################################################

# syn time to exit
comm.barrier()
if rank == 0:
    sys.exit()
