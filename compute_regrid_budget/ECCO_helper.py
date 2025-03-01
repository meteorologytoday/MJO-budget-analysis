from datetime import (datetime, timedelta)
import os.path
import xarray as xr
import xgcm
import ecco_v4_py as ecco
import numpy as np

rhoConst = 1029.0
c_p = 3994.0
R = 0.62
zeta1 = 0.06
zeta2 = 20.0    
Omega = (2*np.pi)/86164

ECCO_root_dir = "data/ECCO_LLC"
ECCO_grid_dir = "ECCO_L4_GEOMETRY_LLC0090GRID_V4R4"
ECCO_grid_filename = "GRID_GEOMETRY_ECCO_V4r4_native_llc0090.nc"

ECCO_mapping = {

    "TEMP_SALINITY" : {
        "fileprefix": "OCEAN_TEMPERATURE_SALINITY",
        "varnames": ["THETA", "SALT"],
    },

    "OCEAN_3D_TEMPERATURE_FLUX" : {
        "fileprefix": "OCEAN_3D_TEMPERATURE_FLUX",
        "varnames": ["ADVx_TH", "ADVy_TH", "ADVr_TH", "DFxE_TH", "DFyE_TH", "DFrE_TH", "DFrI_TH"],
    },

    "MIXED_LAYER_DEPTH" : {
        "fileprefix": "OCEAN_MIXED_LAYER_DEPTH",
        "varnames": ["MXLDEPTH",],
    },

    
    # oceQnet = EXFls + EXFlh - (EXFlwnet + EXFswnet)
    # oceQsw = - EXFswnet

    # EXFqnet = - oceQnet = EXFlwnet + EXFswnet - EXFlh - EXFhs
    # TFLUX = surForcT + oceQsw + oceFreez + [PmEpR*SST]*Cp
    # oceFWflx = [PmEpR]

    # In our case where sea ice does not involve
    # TFLUX = oceQsw + EXFhl + EXFhs - EXFlwnet + [PmEpR*SST]*Cp
    #                                                  |
    #                                                  +--> the loss/gain of ocean mass * c_p

    "HEAT_FLUX" : {
        "fileprefix" : "OCEAN_AND_ICE_SURFACE_HEAT_FLUX",
        "varnames" : ["oceQsw", "TFLUX", "EXFhs", "EXFhl", "EXFlwnet"],
    },

    "FRESH_FLUX" : {
        "fileprefix" : "OCEAN_AND_ICE_SURFACE_FW_FLUX",
        "varnames" : ["EXFempmr", "EXFpreci", "EXFroff", "EXFevap"],
    },


    "OCEAN_VEL" : {
        "fileprefix" : "OCEAN_VELOCITY",
        "varnames" : ["UVEL", "VVEL", "WVEL"],
    },

    "SSH" : {
        "fileprefix" : "SEA_SURFACE_HEIGHT",
        "varnames" : ["SSH", "ETAN"],
    },

    "DENS_STRAT_PRESS" : {
        "fileprefix" : "OCEAN_DENS_STRAT_PRESS",
        "varnames" : ["RHOAnoma", "PHIHYDcR"],
    },

    "POSTPROC_GS_TERMS" : {
        "fileprefix": "GS_TERMS",
        "varnames": ["Gs_ttl", "Gs_hadv", "Gs_vadv", "Gs_hdiff", "Gs_vdiff",
                     "Gs_frc_sw", "Gs_frc_lw", "Gs_frc_sh", "Gs_frc_lh", "Gs_frc_fwf",
                     "Gs_sum", "Gs_res"],
    },

    "POSTPROC_ADV_TERMS" : {
        "fileprefix": "ADV",
        "varnames": ["HADV_g", "HADV_ag", "VADV", "U_g", "V_g", "U_ag", "V_ag"],
    },



    "POSTPROC_MXLANA" : {
        "fileprefix": "MXLANA",
        "varnames": [

            "MLT", "dMLTdt",
            


            "MLG_ttl", "MLG_adv", "MLG_hadv", "MLG_vadv", "MLG_hdiff", "MLG_vdiff",
            "MLG_frc_sw", "MLG_frc_lw", "MLG_frc_sh", "MLG_frc_lh", "MLG_frc_fwf", "MLG_frc_dilu",
            "MLG_sum", "MLG_res", "MLG_ent", "MLG_ent_wep", "MLG_ent_wen", "MLG_rescale",

            "MLGs_ttl", "MLGs_hadv", "MLGs_vadv", "MLGs_hdiff", "MLGs_vdiff",
            "MLGs_frc_sw", "MLGs_frc_lw", "MLGs_frc_sh", "MLGs_frc_lh", "MLGs_frc_fwf",
            "MLGs_sum", "MLGs_res", "MLGs_ent", "MLGs_ent_wep", "MLGs_ent_wen",

            "MLD", "dMLDdt", "dMLTdx", "dMLTdy", "dMLSdx", "dMLSdy",
            "MLU", "MLV", "MLU_g", "MLV_g", "MLU_ag", "MLV_ag",
            "dTdz_b", "dSdz_b", "w_b", "T_b",
    
            "MLHADVT_g", "MLHADVT_ag", "ENT_ADV",

        ],
    },


}


map_varname_pathinfo = {}

for dirmidfix, info in ECCO_mapping.items():
    for varname in info["varnames"]:
        map_varname_pathinfo[varname] = {
            "dirmidfix" : dirmidfix,
            "fileprefix" : info["fileprefix"],
        }



grid_mapping = {

    "LATLON" : {
        "dir" : "05DEG",
        "file" : "latlon_0p50deg",
    },

    "LLC" : {
        "dir" : "LLC0090GRID",
        "file" : "native_llc0090",
    },

}

time_character_mapping = {

    "DAILY" : {
        "dir" : "DAILY",
        "file" : "day_mean",
    },

    "SNAPSHOT" : {
        "dir" : "SNAPSHOT",
        "file" : "snap",
    },

}


def getECCOGrid():
        
    print("%s/%s" % (ECCO_root_dir, ECCO_grid_dir))
    print(ECCO_grid_filename)

    ecco_grid = ecco.load_ecco_grid_nc("%s/%s" % (ECCO_root_dir, ECCO_grid_dir), ECCO_grid_filename)
    
    return ecco_grid 


def getECCOFilename(varname, time_character, target_datetime, grid="LLC", version=4, release=4, extra_dirsuffix=""):


    if time_character not in ["DAILY", "SNAPSHOT"]:
        raise Exception("Unknown `time_character`: %s" % (str(time_character),))

    if grid not in ["LLC", "LATLON"]:
        raise Exception("Unknown `grid`: %s" % (str(grid),))


    dirrelease = "V%dR%d" % (version, release)
    dirsuffix = "%s_%s_%s" % (grid_mapping[grid]["dir"], time_character_mapping[time_character]["dir"], dirrelease)
    dirmidfix = map_varname_pathinfo[varname]["dirmidfix"]

    dirname = "ECCO_L4_%s_%s%s" % (dirmidfix, dirsuffix, extra_dirsuffix)

   

    if time_character == "DAILY" :
        time_str_suffix = ""

    elif time_character == "SNAPSHOT":
        time_str_suffix = "T000000"
        
    time_str = "%s%s" % ( target_datetime.strftime("%Y-%m-%d"), time_str_suffix)
    filerelase = "V%dr%d" % (version, release)
    fileprefix = map_varname_pathinfo[varname]["fileprefix"]
    filesuffix = "%s_%s_ECCO_%s_%s" % (time_character_mapping[time_character]["file"], time_str, filerelase, grid_mapping[grid]["file"])
    filename = "%s_%s.nc" % (fileprefix, filesuffix)


    return dirname, filename




def loadECCOData_continuous(
    beg_datetime,
    ndays = 1,
    snp_varnames = [],
    ave_varnames = [],
    return_xgcm_grid = False,
):

    full_list = []
    
    # ndays + 1 SNAPSHOTS are needed
    for _t in range(ndays+1):

        _now_datetime = beg_datetime + timedelta(days=1) * _t

        for varname in snp_varnames:

            dirname, filename = getECCOFilename(varname, "SNAPSHOT", _now_datetime)
            fullpath = "data/ECCO_LLC/%s/%s" % (dirname, filename)

            if not os.path.isfile(fullpath):
                raise Exception("File %s does not exist." % (fullpath,))

            new_varname = "%s_snp" % (varname,)
            _tmp = xr.open_dataset(fullpath)[[varname,]].rename({'time':'time_snp', varname : new_varname})

            full_list.append(_tmp) 
 
    for _t in range(ndays):
        
        _now_datetime = beg_datetime + timedelta(days=1) * _t

        for varname in ave_varnames:

            dirname, filename = getECCOFilename(varname, "DAILY", _now_datetime)
            fullpath = "data/ECCO_LLC/%s/%s" % (dirname, filename)
            if not os.path.isfile(fullpath):
                raise Exception("File %s does not exist." % (fullpath,))


            _tmp = xr.open_dataset(fullpath)[[varname,]]
            
            full_list.append(_tmp)


    ds = xr.merge(full_list)

    if "time_snp" in ds:
        ds.time_snp.attrs['c_grid_axis_shift'] = - 0.5

    else:
        if len(snp_varnames) > 0:
            raise Exception("Axis `time_snp` does not exist but snapshot variables' are loaded.")


    return ds





