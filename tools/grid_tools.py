import numpy as np
import netCDF4

"""
This function takes care of the boundary issue when finding 
the mid point of longitude. It outputs range 0 to 360
"""
def calMidLon(lon1, lon2):

    lon1 = lon1 % 360
    lon2 = lon2 % 360

    if lon1 - lon2 > 180:   # 180 is just arbitrarily large so that we know it crosses the boundary
        lon2 += 360

    elif lon1 - lon2 < 180:
        lon1 += 360
        

    mid_lon = ((lon1 + lon2) / 2) % 360
    
    return mid_lon
    

def genSCRIPFileWithCornerMissing_regular(output_filename, lat, lon, imask, rev_lat=False, rev_lon=False):
  
    lat = np.array(lat) 
    lon = np.array(lon) % 360


    imask = np.array(imask)
 
    Nx = len(lon)
    Ny = len(lat)
    
    grid_corners = 4
    grid_size = Ny * Nx

    grid_center_lat = np.zeros((Ny, Nx))
    grid_center_lon = np.zeros((Ny, Nx))

    grid_corner_lat = np.zeros((Ny, Nx, grid_corners))
    grid_corner_lon = np.zeros((Ny, Nx, grid_corners))

    # Need to assign center first 
    for j in range(Ny):
        for i in range(Nx):
            grid_center_lat[j, i] = lat[j]
            grid_center_lon[j, i] = lon[i]

   
    for j in range(Ny):
        for i in range(Nx):
            
            j_N = min(j+1, Ny-1)
            j_S = max(j-1,    0)
            
            i_W = (i - 1) % Nx
            i_E = (i + 1) % Nx
            
            lat_N = (grid_center_lat[j, i] + grid_center_lat[j_N, i]) / 2
            lat_S = (grid_center_lat[j, i] + grid_center_lat[j_S, i]) / 2

            lon_W = calMidLon(grid_center_lon[j, i], grid_center_lon[j, i_W])
            lon_E = calMidLon(grid_center_lon[j, i], grid_center_lon[j, i_E])

            grid_corner_lat[j, i, 0] = lat_S
            grid_corner_lat[j, i, 1] = lat_S
            grid_corner_lat[j, i, 2] = lat_N
            grid_corner_lat[j, i, 3] = lat_N
 
            grid_corner_lon[j, i, 0] = lon_W
            grid_corner_lon[j, i, 1] = lon_E
            grid_corner_lon[j, i, 2] = lon_E
            grid_corner_lon[j, i, 3] = lon_W


    if rev_lon:
        print("rev_lon is True. Reverse the points...")
        grid_corner_lon[:, :, 0], grid_corner_lon[:, :, 1] = grid_corner_lon[:, :, 1], grid_corner_lat[:, :, 0]
        grid_corner_lon[:, :, 2], grid_corner_lon[:, :, 3] = grid_corner_lon[:, :, 3], grid_corner_lat[:, :, 2]


    if rev_lat:
        print("rev_lat is True. Reverse the points...")
        grid_corner_lat[:, :, 0], grid_corner_lat[:, :, 3] = grid_corner_lat[:, :, 3], grid_corner_lat[:, :, 0]
        grid_corner_lat[:, :, 1], grid_corner_lat[:, :, 2] = grid_corner_lat[:, :, 2], grid_corner_lat[:, :, 1]

    if np.any(grid_corner_lat[:, :, 0] > grid_corner_lat[:, :, 3]) or np.any(grid_corner_lat[:, :, 1] > grid_corner_lat[:, :, 2]):
        raise Exception("Something is wrong") 

    writeSCRIPFile(output_filename, grid_center_lat, grid_center_lon, grid_corner_lat, grid_corner_lon, imask)



def genSCRIPFileWithCornerMissing_curvlinear(output_filename, lat, lon, imask):
   
    lat = np.array(lat)  
    lon = np.array(lon) % 360
    imask = np.array(imask)
 
    Ny, Nx = lat.shape
    
    grid_corners = 4
    grid_size = Ny * Nx

    grid_center_lat = lat
    grid_center_lon = lon

    grid_corner_lat = np.zeros((Ny, Nx, grid_corners))
    grid_corner_lon = np.zeros((Ny, Nx, grid_corners))
    

    for j in range(Ny):
        for i in range(Nx):
            
            j_N = min(j+1, Ny-1)
            j_S = max(j-1,    0)
            
            i_W = (i - 1) % Nx
            i_E = (i + 1) % Nx
            
            lat_N = (grid_center_lat[j, i] + grid_center_lat[j_N, i]) / 2
            lat_S = (grid_center_lat[j, i] + grid_center_lat[j_S, i]) / 2

            lon_W = calMidLon(grid_center_lon[j, i], grid_center_lon[j, i_W])
            lon_E = calMidLon(grid_center_lon[j, i], grid_center_lon[j, i_E])

            grid_corner_lat[j, i, 0] = lat_S
            grid_corner_lat[j, i, 1] = lat_S
            grid_corner_lat[j, i, 2] = lat_N
            grid_corner_lat[j, i, 3] = lat_N
 
            grid_corner_lon[j, i, 0] = lon_W
            grid_corner_lon[j, i, 1] = lon_E
            grid_corner_lon[j, i, 2] = lon_E
            grid_corner_lon[j, i, 3] = lon_W
    

    writeSCRIPFile(output_filename, grid_center_lat, grid_center_lon, grid_corner_lat, grid_corner_lon, imask)


def writeSCRIPFile(filename, grid_center_lat, grid_center_lon, grid_corner_lat, grid_corner_lon, imask):

    Ny, Nx, grid_corners = np.array(grid_corner_lat).shape

    grid_size =  Ny * Nx
    grid_dims = [Nx, Ny] 
    grid_rank = 2

    with netCDF4.Dataset(filename, mode='w', format='NETCDF4_CLASSIC') as ds: 

        grid_size_dim    = ds.createDimension('grid_size', grid_size)
        grid_corners_dim = ds.createDimension('grid_corners', grid_corners)
        grid_rank_dim    = ds.createDimension('grid_rank', grid_rank)


        var_grid_dims       = ds.createVariable('grid_dims',       np.float32, ('grid_rank',))
        
        var_grid_center_lat = ds.createVariable('grid_center_lat', np.float32, ('grid_size',)) ; var_grid_center_lat.units='degrees'
        var_grid_center_lon = ds.createVariable('grid_center_lon', np.float32, ('grid_size',)) ; var_grid_center_lon.units='degrees'

        var_grid_corner_lat = ds.createVariable('grid_corner_lat', np.float32, ('grid_size', 'grid_corners',)) ; var_grid_corner_lat.units='degrees'
        var_grid_corner_lon = ds.createVariable('grid_corner_lon', np.float32, ('grid_size', 'grid_corners',)) ; var_grid_corner_lon.units='degrees'
        
        var_imask = ds.createVariable('grid_imask', np.int32, ('grid_size',))
         

        var_grid_dims[:] = grid_dims

        var_grid_center_lat[:] = np.reshape(grid_center_lat, (grid_size,))
        var_grid_center_lon[:] = np.reshape(grid_center_lon, (grid_size,))

        var_grid_corner_lat[:, :] = np.reshape(grid_corner_lat, (grid_size, grid_corners))
        var_grid_corner_lon[:, :] = np.reshape(grid_corner_lon, (grid_size, grid_corners))

        var_imask[:] = np.reshape(imask, (grid_size,))
