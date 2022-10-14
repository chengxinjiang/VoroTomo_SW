
import os
import pykonal 
import itertools
import numpy as np
import scipy.spatial
from scipy.spatial import Delaunay
from scipy.sparse import coo_matrix
from collections import defaultdict
from matplotlib import pyplot as plt
from scipy.ndimage import gaussian_filter

'''
subfunctions used in VoroTomo_2D

see each subfunction for details
'''

# radius of the Earth
RR = 6371.0

def construct_grid(geogrid,avs=4):
    '''
    construct the intial spherical grid needed for the eikonal solver

    input parameters:
    -----------------
    geogrid: a dict containing required geographic info about the study region
    avs: np.float32, an uniform velocity as an initial model

    returns:
    -------
    sphgrid: a dict of spherical grid parameters and intial velocity models for eikonal solver
    '''
    # load parameters from the geogrid dict
    latmin,dlat,nlat = geogrid['latmin'],geogrid['dlat'],geogrid['nlat']
    lonmin,dlon,nlon = geogrid['lonmin'],geogrid['dlon'],geogrid['nlon']
    # averaged velocity from the observation        
    vs  = avs*np.ones((nlat+1)*(nlon+1),)                       

    # setup the 2D/3D geographic grid
    latmax = latmin+nlat*dlat
    lonmax = lonmin+nlon*dlon
    latgrid = np.linspace(latmax,latmin,nlat+1)
    longrid = np.linspace(lonmin,lonmax,nlon+1)
    lon,lat = np.meshgrid(longrid,latgrid)

    # detailed parameters for the spherical grid
    theta, phi   = geo2sph(lat,lon)
    ntheta, nphi = len(np.unique(theta)), len(np.unique(phi))
    dtheta       = np.mean(np.diff(np.unique(theta)))
    dphi         = np.mean(np.diff(np.unique(phi)))
    theta0       = np.min(theta)
    phi0         = np.min(phi)
    vel0 = vs.reshape((1,ntheta, nphi))
    vel1 = vs.reshape((1,ntheta, nphi))

    # plug everything into a dict
    sphgrid = {'theta':theta,'phi':phi,
                'ntheta':ntheta,'nphi':nphi,
                'dtheta':dtheta,'dphi':dphi,
                'theta0':theta0,'phi0':phi0,
                'vel0':vel0,'vel1':vel1}
    return sphgrid


def voronoicells(latmin,latmax,lonmin,lonmax,ncell = 300):
    '''
    generate the voronoi cells by using the Delaunay module from scipy.spatial

    input parameters:
    -----------------
    latmin: np.float32, min lat for the study region
    latmax: np.float32, max lat for the study region
    lonmin: np.float32, min lon for the study region
    lonmax: np.float32, max lon for the study region
    ncell : np.int8, number of voronoi cell to regularize the study region

    returns:
    --------
    '''

    # initalize the voronoicell position (vertices) in spherical coordinates
    pos = np.zeros((ncell,2))
    lat = np.random.rand(ncell,)*(latmax-latmin)+latmin
    lon = np.random.rand(ncell,)*(lonmax-lonmin)+lonmin
    theta, phi = geo2sph(lat,lon)

    # transform to cartisian coordinates
    pos[:,0],pos[:,1] = sph2xyz(theta,phi)

    # link voronoi cell with delaunay triangulation
    tri = Delaunay(pos)
    neiList=defaultdict(list)
    
    # not sure what is this: vertices index
    for p in tri.vertices:
        for i,j in itertools.combinations(p,2):
            neiList[i+1].append(j+1)
            neiList[j+1].append(i+1)
    for p in range(1,1+len(pos)):
        neiList[0].append(p)
    neiList[0] = np.unique(neiList[0])
    for p in range(1,len(pos)+1):
        neiList[p].append(p)
        neiList[p] = np.unique(neiList[p])
    return pos,neiList,

def outputproj(geogrid,ncell=300):
    '''
    vectorized version of generating projecting matrix

    input parameters:
    -----------------
    geogrid: a dict containing required geographic info about the study region
    ncell : np.int8, number of voronoi cell to regularize the study region

    returns:
    --------
    cellpos: center location of each voronoi cell
    nearcells: not used (to be figured out)
    Gp: projection matrix from lower dimension to origianl high dimension
    '''
    # load basic geographic info
    latmin,dlat,nlat = geogrid['latmin'],geogrid['dlat'],geogrid['nlat']
    lonmin,dlon,nlon = geogrid['lonmin'],geogrid['dlon'],geogrid['nlon']
    latmax = latmin+nlat*dlat
    lonmax = lonmin+nlon*dlon

    # generate voronoi cells
    cellpos, nearcells = voronoicells(latmin=latmin,latmax=latmax,lonmin=lonmin,\
            lonmax=lonmax, ncell=ncell)
    mdim = (nlat+1)*(nlon+1)
    latgrid = np.linspace(latmax,latmin,nlat+1)
    longrid = np.linspace(lonmin,lonmax,nlon+1)
    latgrid,longrid = geo2sph(latgrid,longrid)
    
    # generate mesh grid
    longrids,latgrids = np.meshgrid(longrid,latgrid)
    
    # get coordinates in cartisian system
    xpts = RR*np.sin(latgrids.flatten())*np.cos(longrids.flatten())
    ypts = RR*np.sin(latgrids.flatten())*np.sin(longrids.flatten())
    
    # record useful parameters for making projection matrix 
    dist = scipy.spatial.distance.cdist(np.vstack([xpts,ypts]).T, cellpos)
    colid = np.argmin(dist, axis=1)
    rowid = np.arange(mdim)
    
    # construct projection matrix
    Gp = coo_matrix((np.ones(mdim,),(rowid,colid)),shape=(mdim,ncell))
    return cellpos,nearcells,Gp

def solve_source_region(geogrid,sphgrid,lat_s,lon_s,flag=0):
    '''
    solve the traveltime at near-source region in a fine grid

    input parameters:
    -----------------
    geogrid: a dict containing required geographic info about the study region
    sphgrid: a dict of spherical info about the grid
    lat_s: np.float32, latitude of the source point
    lon_s: np.float32, longitude of the source point
    flag: boolen, True to use final model and False to use initial model

    returns:
    --------
    '''
    sdlat = geogrid['dlat']/5
    sdlon = geogrid['dlon']/5
    # define the near-source region (20by20 points)
    slatmin,snlat = lat_s-sdlat*20,40  
    slonmin,snlon = lon_s-sdlon*20,40
    sgeogrid = {'latmin':slatmin,'lonmin':slonmin,                                 # assemble geographic info into a dict 
                'dlat':sdlat,'dlon':sdlon,
                'nlat':snlat,'nlon':snlon}

    # construct spherical grid for pykonal solver
    ssphgrid = construct_grid(sgeogrid)

    # initalize the velocity grid for the far-field coarse grid
    solver = pykonal.EikonalSolver(coord_sys='spherical')
    solver.vv.min_coords     = RR,sphgrid['theta0'],sphgrid['phi0']
    solver.vv.node_intervals = 1,sphgrid['dtheta'],sphgrid['dphi']
    solver.vv.npts           = 1,sphgrid['ntheta'],sphgrid['nphi']
    if flag:
        solver.vv.values     = sphgrid['vel1']
    else:
        solver.vv.values     = sphgrid['vel0']

    # initalize the velocity grid for the near-field fine grid
    solver_fine = pykonal.EikonalSolver(coord_sys="spherical")
    solver_fine.vv.min_coords     = RR,ssphgrid['theta0'],ssphgrid['phi0']
    solver_fine.vv.node_intervals = 1, ssphgrid['dtheta'],ssphgrid['dphi']
    solver_fine.vv.npts           = 1, ssphgrid['ntheta'],ssphgrid['nphi']

    # Interpolate the velocity field from the coarse grid onto the fine grid
    solver_fine.vv.values = solver.vv.resample(solver_fine.vv.nodes.reshape(-1, 3)).reshape(solver_fine.vv.npts)

    # Initialize the source node on the near-field
    src_idx = 0, 20, 20
    solver_fine.tt.values[src_idx] = 0
    solver_fine.unknown[src_idx] = False
    solver_fine.trial.push(*src_idx)
    solver_fine.solve() 

    # interpolate the traveltime from fine grid (near-filed) to the coarse grid (far-field)
    tt = solver_fine.tt.resample(solver.tt.nodes.reshape(-1, 3)).reshape(solver.tt.npts)
    for idx in np.argwhere(~np.isnan(tt)):
        idx = tuple(idx)
        solver.tt.values[idx] = tt[idx]
        solver.unknown[idx] = False
        solver.trial.push(*idx)
    solver.solve() 

    return solver   

def findrayidx(ray,cellpos):
    '''
    vectorized version of finding ray index

    input parameters:
    -----------------

    returns:
    --------
    '''
    # transfer each ray into a cartersian space
    ray['x'] = ray['rho'] * np.sin(ray['theta']) * np.cos(ray['phi'])
    ray['y'] = ray['rho'] * np.sin(ray['theta']) * np.sin(ray['phi'])

    # find the block for each ray segment based on dist
    dist = scipy.spatial.distance.cdist(ray[['x', 'y']].values, cellpos)
    argmin = np.argmin(dist, axis=1)
    ray['cellidx'] = argmin
    return (ray)


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

def sph2geo(theta,phi):
    '''
    transform spherical coordinates to geographic coordinates.
    
    input parameters:
    -----------------
    theta: polo angle of the grids
    phi: azimuthal angle of the grids

    '''
    lat = np.degrees(np.pi/2-theta)
    lon = np.degrees(phi)
    return lat, lon

def sph2xyz(theta,phi):
    '''
    Map spherical coordinates to Cartesian coordinates.
    
    '''
    xx = RR*np.sin(theta)*np.cos(phi)
    yy = RR*np.sin(theta)*np.sin(phi)
    return xx, yy


def make_checkerboard(geogrid,anomaly_info):
    '''
    make synthetic traveltime data for test purpose

    input parameters:
    -----------------
    geogrid: a dict containing required geographic info about the study region
    anomaly_info: a dict holding all info about the target anomaly

    returns:
    --------
    sphgrid: a dict of spherical grid parameters used for synthetic tests    
    '''

    # check the basic info of the targeted checkerboard
    checker_size = anomaly_info['size']
    checker_spa  = anomaly_info['spacing']

    # load parameters from the geogrid dict
    latmin,dlat,nlat = geogrid['latmin'],geogrid['dlat'],geogrid['nlat']+1
    lonmin,dlon,nlon = geogrid['lonmin'],geogrid['dlon'],geogrid['nlon']+1

    # construct one basic anomaly
    if checker_spa:
        # spike input
        anomaly1 = np.ones(2*(checker_spa+1)*checker_size,dtype=np.float32)
        anomaly1[:checker_size] = -1
        anomaly1[checker_size:(checker_spa+1)*checker_size] = 0
        anomaly1[(checker_spa+2)*checker_size:] = 0
        anomaly2 = anomaly1*-1
        #anomaly3 = np.concatenate((anomaly1[checker_size:],anomaly1[:checker_size]))
    else:
        # checkerboard input
        anomaly1 = np.ones(2*checker_size,dtype=np.float32)
        anomaly1[:checker_size] = -1
        anomaly2 = anomaly1*-1

    # construct the overall checkerboard
    amp  = []
    nchecker_lon = int(np.round(nlon/checker_size)) 
    nchecker_lat = int(np.round(nlat/checker_size)) 
    column_anomaly = np.tile(anomaly1,nchecker_lat)[:nlat]
    for ii in range(nlat):
        if column_anomaly[ii] == -1:
            amp.append(np.tile(anomaly1,nchecker_lon)[:nlon])
        elif column_anomaly[ii] == 1:
            amp.append(np.tile(anomaly2,nchecker_lon)[:nlon])
        else:
            amp.append(np.zeros(nlon))

    # smooth using a gaussian filter
    amp= gaussian_filter(amp, sigma=1.5)

    return amp

def make_syn(geogrid,anomaly):
    '''
    make synthetic traveltime data for test purpose

    input parameters:
    -----------------
    geogrid: a dict containing required geographic info about the study region
    anomaly: a dict holding all info about the target anomaly

    returns:
    --------
    sphgrid: a dict of spherical grid parameters used for synthetic tests
    '''
    # load parameters from the geogrid dict
    latmin,dlat,nlat = geogrid['latmin'],geogrid['dlat'],geogrid['nlat']
    lonmin,dlon,nlon = geogrid['lonmin'],geogrid['dlon'],geogrid['nlon']
    # averaged velocity from the observation
    avs = anomaly['avs']     
    vs  = avs*np.ones((nlat+1)*(nlon+1),)                       

    # setup the 2D/3D geographic grid
    latmax = latmin+nlat*dlat
    lonmax = lonmin+nlon*dlon
    latgrid = np.linspace(latmax,latmin,nlat+1)
    longrid = np.linspace(lonmin,lonmax,nlon+1)
    lon,lat = np.meshgrid(longrid,latgrid)

    # find the index for the target region
    indx1 = np.where((lon.flatten()<=anomaly['lonmax']) & (lon.flatten()>=anomaly['lonmin']))[0]
    indx2 = np.where((lat.flatten()<=anomaly['latmax']) & (lat.flatten()>=anomaly['latmin']))[0]
    indx  = np.intersect1d(indx1,indx2)

    # change the velocity at grid
    vs[indx] = vs[indx]+anomaly['amp']

    return vs

def calculate_residuals(data,geogrid,sphgrid,fname):
    '''

    input parameters:
    -----------------

    return:
    -------
    '''
    # initial and final residual
    res_initial = [] 
    res_final   = []
    fout = open(fname,'w')
    fout.write('index,evt_id,evt_lon,evt_lat,sta_id,sta_lon,sta_lat,vel,dist,res_initial,res_final\n')

    srcs    = data.evt_id.unique()
    # loop through each virtual source
    for isc in srcs:
        print(len(srcs),isc)

        # source location to initialize the traveltime field
        arr4rc = data[data.evt_id.values==isc]
        lat_s,lon_s   = arr4rc['evt_lat'].iloc[0],arr4rc['evt_lon'].iloc[0]

        # initialize the wavefield near source location on a fine grid (initial/final model)
        solver_initial = solve_source_region(geogrid,sphgrid,lat_s,lon_s)  
        solver_final   = solve_source_region(geogrid,sphgrid,lat_s,lon_s,1)

        # loop through each receiver 
        for irc in range(len(arr4rc)):
            try:

                # receiver location
                lat_r,lon_r   = arr4rc['sta_lat'].iloc[irc],arr4rc['sta_lon'].iloc[irc]
                theta_r,phi_r = geo2sph(lat_r,lon_r)
                src_pos  = (RR,theta_r,phi_r)
                syn_initial = solver_initial.traveltime.value(np.array(src_pos))
                syn_final   = solver_final.traveltime.value(np.array(src_pos))
                dd1 = (arr4rc['dist'].iloc[irc]/arr4rc['vel'].iloc[irc])-syn_initial
                dd2 = (arr4rc['dist'].iloc[irc]/arr4rc['vel'].iloc[irc])-syn_final
                res_initial.append(dd1)
                res_final.append(dd2)
                fout.write('%6d,%6d,%8.3f,%8.3f,%6d,%8.3f,%8.3f,%8.3f,%8.3f,%8.4f,%8.4f\n'%(arr4rc['index'].iloc[irc],\
                    isc,lon_s,lat_s,arr4rc['sta_id'].iloc[irc],lon_r,lat_r,arr4rc['vel'].iloc[irc],\
                    arr4rc['dist'].iloc[irc],dd1,dd2))

            except Exception as err:
                print('ERROR for',irc,isc,err)

    fout.close()
    return res_initial,res_final

def plot_tomo(geogrid,vel,rootpath,iset=-1):
    '''
    plot the tomography model and output as a PDF file

    input parameters:
    -----------------
    geogrid: a dict containing required geographic info about the study region
    vel: 2d matrix of resulted velocity anomaly
    rootpath: absolute path for saving the figures
    iset: index of the current model

    '''
    # load geographic information
    latmin,dlat,nlat = geogrid['latmin'],geogrid['dlat'],geogrid['nlat']
    lonmin,dlon,nlon = geogrid['lonmin'],geogrid['dlon'],geogrid['nlon']

    # setup the 2D/3D geographic grid
    latmax = latmin+nlat*dlat
    lonmax = lonmin+nlon*dlon    

    # plot figures
    plt.figure(figsize=(10,10))
    plt.imshow(vel.reshape(nlat+1,nlon+1),cmap='jet_r',extent=(lonmin,lonmax,latmin,latmax))
    plt.gca().invert_yaxis()
    plt.xlim([lonmin,lonmax])
    plt.ylim([latmin,latmax])
    plt.colorbar()
    #plt.clim(3.6,4)
    if iset==-1:
        outfname = rootpath+'/figures/tomo_final.pdf'
    elif iset==-2:
        outfname = rootpath+'/figures/tomo_syn.pdf'
    else:
        outfname = rootpath+'/figures/tomo_'+str(iset)+'.pdf'
    plt.savefig(outfname, format='pdf', dpi=400)
    plt.close()
