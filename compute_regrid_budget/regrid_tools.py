import numpy as np
import operator
import xarray as xr
from functools import reduce
import scipy.sparse as sparse

def loadRegridInfo(regrid_file):

    ds_regrid = xr.load_dataset(regrid_file)

    lat_idx = ds_regrid["lat_idx"].to_numpy()
    lon_idx = ds_regrid["lon_idx"].to_numpy()

    lat_regrid_bnds = ds_regrid["lat_regrid_bnd"].to_numpy()   
    lon_regrid_bnds = ds_regrid["lon_regrid_bnd"].to_numpy()   

    lat_regrid = (lat_regrid_bnds[1:] + lat_regrid_bnds[:-1])/2
    lon_regrid = (lon_regrid_bnds[1:] + lon_regrid_bnds[:-1])/2
    
    regrid_shape = (len(lat_regrid), len(lon_regrid))
    
    regrid_info = constructAvgMtx(lat_idx, lon_idx, lat_regrid, lon_regrid)

    return regrid_info


def constructAvgMtx(lat_idx, lon_idx, lat_regrid, lon_regrid):
    
    nbox_lat = len(lat_regrid)
    nbox_lon = len(lon_regrid) 
    original_shape = lat_idx.shape
    original_total_grids = reduce(operator.mul, original_shape)
    
    total_boxes = nbox_lat * nbox_lon
    
    print("original_shape = ", original_shape)
   
    # This is the numbering of each regridded boxes 
    regrid_row_idx = np.arange(total_boxes).reshape((nbox_lat, nbox_lon)) 

    # This is the numbering of the original grids
    original_grid_idx = np.arange(original_total_grids).reshape(original_shape)
   
    print("shape of regrid_row_idx: ", regrid_row_idx.shape) 
    
    row_idxes = []
    col_idxes = []
    for index in np.ndindex(original_shape):
        _lon_idx = lon_idx[index]
        _lat_idx = lat_idx[index]
      
        if _lon_idx >= 0 and _lat_idx >= 0:

            _row_idx = regrid_row_idx[ _lat_idx, _lon_idx]
            _col_idx = original_grid_idx[index]

            row_idxes.append(_row_idx)
            col_idxes.append(_col_idx)
             
    vals = np.ones((len(row_idxes), ))
    
    avg_mtx = sparse.coo_array((vals, (row_idxes, col_idxes)), shape=(total_boxes, original_total_grids), dtype=np.float32)
    
    wgt = avg_mtx.sum(axis=1)
    
    wgt_mtx = sparse.dia_array( ([wgt**(-1),], [0,]), shape=(total_boxes, total_boxes))
    avg_mtx = wgt_mtx @ avg_mtx 

    regrid_info = dict(
        avg_mtx = avg_mtx,
        shape_original = original_shape,
        shape_regrid = (nbox_lat, nbox_lon),
        lat = lat_regrid,
        lon = lon_regrid,
    )

    return regrid_info

def regrid(regrid_info, arr):
    
    flattened_arr = np.array(arr).flatten()

    if len(flattened_arr) != reduce(operator.mul, regrid_info["shape_original"]):
        raise Exception("Dimension of input array does not match avg_info.")
    
    result = regrid_info["avg_mtx"] @ np.array(arr).flatten()
    result = np.reshape(result, regrid_info["shape_regrid"])

    return result 



