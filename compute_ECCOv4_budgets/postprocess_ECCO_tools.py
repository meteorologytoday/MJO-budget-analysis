
from pathlib import Path
import os.path
import xarray as xr
import ECCO_helper
import ecco_v4_py as ecco

import numpy as np
import netCDF4
#import earth_constants as ec
import calculus_tools

from datetime import (datetime, timedelta)

# Before 2023/8/30 : using Bryan & Cox 1972 linearized buoyancy
#from buoyancy_linear import TS2rho

# Starting 2023/8/30 : using Millero & Poisson 1981 nonlinear buoyancuy
from buoyancy_nonlinear import TS2rho

default_fill_value = np.nan
default_fill_value_int = -1

# RHO_CONST number is read from
# https://ecco-v4-python-tutorial.readthedocs.io/Thermal_wind.html#Viewing-and-Plotting-Density
RHO_CONST = 1029.0
OMEGA     = (2*np.pi)/86164

def calddz(Q, xgcm_grid, ecco_grid, interp=False):
    dQdz = xgcm_grid.diff(Q, axis='Z', boundary='fill') #/ ecco_grid.drC

    if interp:
        dQdz = xgcm_grid.interp(dQdz, 'Z')
    
    return dQdz

def calGrad(Q, xgcm_grid, ecco_grid):

    dQdx = (xgcm_grid.diff(Q, axis="X", boundary='extend')) / ecco_grid.dxC
    dQdy = (xgcm_grid.diff(Q, axis="Y", boundary='extend')) / ecco_grid.dyC

    #dQdx.data = dQdx.values
    #dQdy.data = dQdy.values

    #_interp = xgcm_grid.interp_2d_vector({"X": dQdx, "Y": dQdy}, boundary='extend')

    dQdx = xgcm_grid.interp(dQdx, "X")#_interp["X"]
    dQdy = xgcm_grid.interp(dQdy, "Y")
    #dQdy = _interp["Y"]

    return dQdx, dQdy


def rotateVector2LatLon(u, v, ecco_grid):

    u_e = u * ecco_grid['CS'] - v * ecco_grid['SN']
    v_n = u * ecco_grid['SN'] + v * ecco_grid['CS']
    
    return u_e, v_n


def detectMLNz(h, z_W, mask=None, fill_value=default_fill_value_int):

    if mask is None:
        mask = np.ones((Ny, Nx), dtype=np.int32)

    Ny, Nx = h.shape
    Nz = len(z_W) - 1
    MLNz = np.zeros((Ny, Nx), dtype=np.int32)

    for j in range(Ny):
        for i in range(Nx):

            if mask[j, i] == 0:
                MLNz[j, i] = fill_value

            else:

                z = - h[j, i]
                for k in range(Nz):

                    if z_W[k] >= z and z >= z_W[k+1]:   # Using 2 equalities so that the last grid-box will include z = z_bottom
                        MLNz[j, i] = k
                        break
 
                    elif k == Nz-1:
                        MLNz[j, i] = k
    
    return MLNz


def evalAtMLD_W(fi, h, z_W, mask=None, fill_value=default_fill_value):
  
    Nzp1, Ny, Nx = fi.shape

    if mask is None:
        mask = np.ones((Ny, Nx), dtype=np.int32)

    fo = np.zeros((Ny, Nx))
    Nz_h = detectMLNz(h, z_W, mask=mask)
    
    dz_T = z_W[:-1] - z_W[1:]

    for j in range(Ny):
        for i in range(Nx):
            
            if mask[j, i] == 0:
                fo[j, i] = fill_value
            
            else:
                
                _Nz = Nz_h[j, i]
                _h = h[j, i]

                fo[j, i] = fi[_Nz+1, j, i] + (fi[_Nz, j, i] - fi[_Nz+1, j, i]) / dz_T[_Nz] * (- _h - z_W[_Nz+1])
   
    return fo


def evalAtMLD_T(fi, h, z_W, mask=None, fill_value=default_fill_value):
  
    Nz, Ny, Nx = fi.shape

    if mask is None:
        mask = np.ones((Ny, Nx), dtype=np.int32)

    fo = np.zeros((Ny, Nx))
    Nz_h = detectMLNz(h, z_W, mask=mask)
    
    for j in range(Ny):
        for i in range(Nx):
            
            if mask[j, i] == 0:
                fo[j, i] = fill_value
            
            else:
                
                _Nz = Nz_h[j, i]
                fo[j, i] = fi[_Nz, j, i]
   
    return fo



def computeMLMean(fi, h, z_W, mask=None, fill_value=default_fill_value):
  
    Nz, Ny, Nx = fi.shape 

    if mask is None:
        mask = np.ones((Ny, Nx), dtype=np.int32)

    dz_T = z_W[:-1] - z_W[1:]
    fo = np.zeros((Ny, Nx))

    Nz_h = detectMLNz(h, z_W, mask=mask)

    for j in range(Ny):
        for i in range(Nx):
           

            _Nz = Nz_h[j, i]
            

            if mask[j, i] == 0:
                fo[j, i] = fill_value

            else:


                _tmp = 0.0
                if _Nz > 0:
                    _tmp += np.sum(dz_T[:_Nz] * fi[:_Nz, j, i])

                _tmp += (z_W[_Nz] + h[j, i]) * fi[_Nz, j, i]
                
                fo[j, i] = _tmp / h[j, i]
    
    return fo


def findMLD_rho(rho, z_T, dev=0.03, mask=None, Nz_bot=None, fill_value=default_fill_value):


    Nz, Ny, Nx = rho.shape

    if mask is None:
        mask = np.ones((Ny, Nx), dtype=np.int32)
 
    if Nz_bot is None:
        Nz_bot = np.zeros((Ny, Nx), dtype=np.int32)
        Nz_bot += Nz
        
        
    MLD = np.zeros((Ny, Nx))
    for j in range(Ny):
        for i in range(Nx):

            if mask[j, i] == 0:
                MLD[j, i] = fill_value
                continue 

            SSrho = rho[0, j, i]
            rho_threshold = SSrho + dev

            _Nz_bot = Nz_bot[j, i]
            
            for k in range(_Nz_bot):
                
                if rho[k, j, i] >= rho_threshold:
                
                    MLD[j, i] = - z_T[k-1] + (z_T[k-1] - z_T[k]) * (rho_threshold - rho[k-1, j, i])/(rho[k, j, i] - rho[k-1, j, i]) 
                    break

                if k == _Nz_bot-1:  # cannot find the mixed layer depth
                        MLD[j, i] = - z_T[k]

    # Sanity check
    if np.any(MLD[np.isfinite(MLD)] <= 0):
        throw(ErrorException("Some MLD is negative."))

    return MLD


def processECCO(
    target_datetime,
    output_filename,
    MLD_dev = 0.03,
    fixed_MLD = -1.0, # if this option is positive, then MLD will be assignd with this value
    debug = False,
):

    print("[processECCO] Target datetime: ", target_datetime)
    print("[processECCO] Output filename: ", output_filename)


    if fixed_MLD <= 0:
        
        print("Notice: the parameter `fixed_MLD` is negative. Later on the mixed-layer depth will be determined dynamically.")

    else:
        
        print("Notice: the parameter `fixed_MLD` = %f is positive. Later on the mixed-layer depth will be assigned to this value." % (fixed_MLD, ))

    beg_datetime = target_datetime # - timedelta(days=1)

    snp_varnames = ["THETA", "SALT", "ETAN"]
    ave_varnames = ["MXLDEPTH", "Gs_hadv", "Gs_vadv", "Gs_hdiff", "Gs_vdiff", "Gs_frc_sw", "Gs_frc_lw", "Gs_frc_sh", "Gs_frc_lh", "THETA", "SALT", "UVEL", "VVEL", "WVEL", "RHOAnoma", "PHIHYDcR", "EXFempmr", "TFLUX", "oceQsw", "EXFhl", "EXFhs", "EXFlwnet"]

    ds = ECCO_helper.loadECCOData_continuous(
        beg_datetime = beg_datetime,
        ndays = 1,
        snp_varnames = snp_varnames,
        ave_varnames = ave_varnames,
    )

    xgcm_grid = ecco.get_llc_grid(ds)
    ecco_grid = ECCO_helper.getECCOGrid()
    
    s_snp = 1 + ds.ETAN_snp / ecco_grid.Depth
    
    s_0 = s_snp[0, :, : ,:].to_numpy()
    s_1 = s_snp[1, :, : ,:].to_numpy()

    THETA_snp = ds.THETA_snp
    SALT_snp  = ds.SALT_snp

    sTHETA_snp = ds.THETA_snp * s_snp
    sSALT_snp  = ds.SALT_snp  * s_snp

    sTHETA_snp = sTHETA_snp.rename("sTHETA_snp")
    
    Nt, Nz, Nl, Nx, Ny = (ds.dims['time'], ds.dims['k'], ds.dims['tile'], ds.dims['j'], ds.dims['i'])
  
    if Nt != 1:
        raise Exception("Too many records. I only need one.")
 
    Zl = ecco_grid.Zl.load()
    Zu = ecco_grid.Zu.load()

    z_W = np.zeros((len(Zl)+1,),)


    z_W[:-1] = Zl
    z_W[-1] = Zu[-1]
    
    z_T = (z_W[:-1] + z_W[1:]) / 2.0

 
    sample2D_snp = ds.THETA_snp[:, 0, :, :, :]
    sample2D_ave = ds.MXLDEPTH[:, :, :, :]

    ML_snp = {}
    ML_ave = {}

    ML_snp_varnames = ["MLDs_snp", "MLD_snp", "MLT_snp", "MLS_snp"]
    ML_ave_varnames = [
        "MLT", "dMLTdt", "dMLSdt", "detadt", "w_eta",
        "MLG_hadv", "MLG_vadv", "MLG_adv",
        "MLG_vdiff", "MLG_hdiff",
        "MLG_frc_sw", "MLG_frc_lw", "MLG_frc_sh", "MLG_frc_lh",
        "MLG_rescale", "MLG_frc_dilu",
    ]

    for varname in ML_snp_varnames:
        ML_snp[varname] = xr.zeros_like(sample2D_snp).rename(varname)

    for varname in ML_ave_varnames:
        ML_ave[varname] = xr.zeros_like(sample2D_ave).rename(varname)

 
    rho_snp = TS2rho(THETA_snp, SALT_snp).rename('rho_snp')
    
    mask   = [ THETA_snp[0, :, l, :, :].notnull().rename('mask').astype('i4').to_numpy() for l in range(Nl) ]
    mask2D = [ mask[l][0, :, :] for l in range(Nl) ]

    Nz_bot = [ np.sum(mask[l], axis=0) for l in range(Nl) ]

    print("Compute MLD, MLT and MLS")
    # Compute variable at the time_bnds
    for s in range(2):
        for l in range(Nl):

            if fixed_MLD <= 0:
                ML_snp["MLDs_snp"][s, l, :, :] = findMLD_rho(
                    rho_snp[s, :, l, :, :].to_numpy(),
                    z_T,
                    mask=mask2D[l],
                    Nz_bot=Nz_bot[l],
                    dev=MLD_dev,
                )
                
            else:
                ML_snp["MLDs_snp"][s, l, :, :] = fixed_MLD
               
 
            ML_snp["MLD_snp"][s, l, :, :] = ML_snp["MLDs_snp"][s, l, :, :] / s_snp[s, l, :, :]

            # Computing mean values in z* space
            ML_snp["MLT_snp"][s, l, :, :] = computeMLMean(
                THETA_snp[s, :, l, :, :].to_numpy(),
                ML_snp["MLDs_snp"][s, l, :, :].to_numpy(),
                z_W,
                mask=mask2D[l]
            )

            # Computing mean values in z* space
            ML_snp["MLS_snp"][s, l, :, :] = computeMLMean(
                SALT_snp[s, :, l, :, :].to_numpy(),
                ML_snp["MLDs_snp"][s, l, :, :].to_numpy(),
                z_W,
                mask=mask2D[l]
            )

    dt = 86400.0
 
    ML_ave["dMLTdt"][0, :, :, :] = (ML_snp["MLT_snp"][1, :, :, :] - ML_snp["MLT_snp"][0, :, :, :]) / dt
    ML_ave["dMLSdt"][0, :, :, :] = (ML_snp["MLS_snp"][1, :, :, :] - ML_snp["MLS_snp"][0, :, :, :]) / dt
    
    ML_ave["detadt"][0, :, :, :] = (ds.ETAN_snp[1, :, :, :] - ds.ETAN_snp[0, :, :, :]) / dt
 
    ML_ave["MLT"][0, :, :, :] = (ML_snp["MLT_snp"][0, :, :, :] + ML_snp["MLT_snp"][1, :, :, :]) / 2.0
 
    # Compute dMLDdt
    MLDs_snp = [ ML_snp["MLDs_snp"][s, :, :, :].to_numpy() for s in range(2) ]
    MLD_snp = [ ML_snp["MLD_snp"][s, :, :, :].to_numpy() for s in range(2) ]
    
    ML_ave["dMLDdt"] = xr.zeros_like(sample2D_ave).rename("dMLDdt") 
    for l in range(Nl):
        ML_ave["dMLDdt"][0, l, :, :] = ( MLD_snp[1][l, :, :] - MLD_snp[0][l, :, :] ) / dt
   
    print("Compute MLG terms")
    # Process the source terms
    for phy_proc in ["hadv", "vadv", "vdiff", "hdiff", "frc_sw", "frc_lw", "frc_sh", "frc_lh"]:
        varname = "Gs_%s" % phy_proc
        ML_varname = "MLG_%s" % phy_proc
        for l in range(Nl):
                
            # Computing mean values in z* space
            ML_ave[ML_varname][0, l, :, :] = computeMLMean(
                ds[varname][0, :, l, :, :].to_numpy(),
                MLDs_snp[1][l, :, :],
                z_W,
                mask=mask2D[l]
            ) / s_0[l, :, :]

    # Compute entrainment term explicitly
    ML_ave["MLG_ent"]  = xr.zeros_like(sample2D_ave).rename("MLG_ent")
    s_ratio = s_1 / s_0 
    
    print("Compute MLG_ent")
    for l in range(Nl):
        
        _THETA = THETA_snp[0, :, l, :, :].to_numpy()
            
        # Computing mean values in z* space, need to convert the coordinate
        # through s*
        ML_ave["MLG_ent"][0, l, :, :] = ( computeMLMean(
            _THETA,
            MLDs_snp[1][l, :, :] * s_ratio[l, :, :],
            z_W,
            mask=mask2D[l]
        ) -  computeMLMean(
            _THETA,
            MLDs_snp[0][l, :, :],
            z_W,
            mask=mask2D[l]
        )) / dt
       

    print("Separate MLG_ent into two parts based on the sign of dMLDdt")
    ML_ave["MLG_ent_wep"] = xr.where( ML_ave["dMLDdt"] >= 0, ML_ave["MLG_ent"], 0.0, keep_attrs=True).rename("MLG_ent_wep")
    ML_ave["MLG_ent_wen"] = xr.where( ML_ave["dMLDdt"]  < 0, ML_ave["MLG_ent"], 0.0, keep_attrs=True).rename("MLG_ent_wen")


   
    ML_ave["MLG_rescale"][0, :, :, :] = - ML_snp["MLT_snp"][1, :, :, :] / s_0 * (s_1 - s_0) / dt
    ML_ave["MLG_rescale"] = ML_ave["MLG_rescale"].rename('MLG_rescale')

    # Failed method: use w_eta, PmEmR, and detadt to guess.
    # This method results in a lot of singularity points because
    # detadt can be quite small

    """
    print("Indirectly compute (PmE - w_\eta) * ( \Theta_\eta - \vavg{\Theta} ) ...")
    # Indirectly infer (PmE - w_\eta) * ( \Theta_\eta - \vavg{\Theta} ) 
    
    partial_adv = ML_ave["MLG_hadv"] + ML_ave["MLG_vadv"] + ML_ave["MLG_rescale"]
    ML_ave["MLG_top_ent"] = ML_ave["dMLTdt"] - (
              ML_ave["MLG_ent_wep"] 
            + ML_ave["MLG_ent_wen"] 
            + partial_adv
            + ML_ave["MLG_hdiff"]
            + ML_ave["MLG_vdiff"]
            + ML_ave["MLG_frc_sw"]
            + ML_ave["MLG_frc_lw"]
            + ML_ave["MLG_frc_sh"]
            + ML_ave["MLG_frc_lh"]
    )
 
    print("Estimating dilution effect...")
  
    ML_ave["PmEmR"] = - ds["EXFempmr"] 
    ML_ave["w_eta"] = ML_ave["detadt"] - ML_ave["PmEmR"]
    for l in range(Nl):
        _tmp = ML_ave["MLG_top_ent"][0, l, :, :].to_numpy()
        ML_ave["MLG_frc_dilu"][0, l, :, :] = _tmp * ML_ave["PmEmR"][0, l, :, :].to_numpy() / ML_ave["detadt"][0, l, :, :].to_numpy()

    ignore_threshold = 1e-8
    ML_ave["MLG_frc_dilu"] = xr.where( np.abs(ML_ave["w_eta"]) < ignore_threshold, 0.0, ML_ave["MLG_frc_dilu"], keep_attrs=True)

    
    ML_ave["MLG_adv_partial"] = partial_adv
    ML_ave["MLG_adv"] = partial_adv + (ML_ave["MLG_top_ent"] - ML_ave["MLG_frc_dilu"])
    ML_ave["MLG_adv"] = ML_ave["MLG_adv"].rename("MLG_adv")


    for l in range(Nl):
        ML_ave["MLG_frc_dilu"][0, l, :, :] = _tmp * ML_ave["PmEmR"][0, l, :, :].to_numpy() / ML_ave["detadt"][0, l, :, :].to_numpy()

    ignore_threshold = 1e-8
    ML_ave["MLG_frc_dilu"] = xr.where( np.abs(ML_ave["w_eta"]) < ignore_threshold, 0.0, ML_ave["MLG_frc_dilu"], keep_attrs=True)

    
    ML_ave["MLG_adv_partial"] = partial_adv
    ML_ave["MLG_adv"] = partial_adv + (ML_ave["MLG_top_ent"] - ML_ave["MLG_frc_dilu"])
    ML_ave["MLG_adv"] = ML_ave["MLG_adv"].rename("MLG_adv")
    """


    # Current method (2023/10/03): use fwf to have <PmEmR * \Theta_{PmEmR}>
    # and we also have <PmEmR> and <\vavg{\Theta}>.
    # This means we can get <PmEmR * \Theta_{PmEmR}> - <PmEmR> and <\vavg{\Theta}>
    # or <PmEmR> (<\Theta_{PmEmR}> - <\vavg{\Theta}>) + <(PmEmR)' * (\Theta_{PmEmR})'>
    # This method needs to assume the last correlation term is negligible
    ML_ave["PmEmR"] = - ds["EXFempmr"]
    
    EXFfwf = ds.TFLUX - ds.oceQsw - ds.EXFhl - ds.EXFhs + ds.EXFlwnet
    PmEmR_THETA_rain = EXFfwf / (ECCO_helper.rhoConst * ECCO_helper.c_p)  # = <PmEmR * \Theta_{PmEmR}>
 
    for l in range(Nl):
        ML_ave["MLG_frc_dilu"][0, l, :, :] = ( PmEmR_THETA_rain[0, l, :, :] - ML_ave["MLT"][0, l, :, :] * ML_ave["PmEmR"][0, l, :, :] ) / ML_snp["MLD_snp"][1, l, :, :]

    ML_ave["MLG_adv"] = ML_ave["dMLTdt"] - (
              ML_ave["MLG_ent_wep"]
            + ML_ave["MLG_ent_wen"]
            + ML_ave["MLG_hdiff"]
            + ML_ave["MLG_vdiff"]
            + ML_ave["MLG_frc_sw"]
            + ML_ave["MLG_frc_lw"]
            + ML_ave["MLG_frc_sh"]
            + ML_ave["MLG_frc_lh"]
            + ML_ave["MLG_frc_dilu"]
    )
    

    # Estimate the residue
    ML_ave["dMLTdt_res"] = ML_ave["dMLTdt"] - (
              ML_ave["MLG_ent_wep"] 
            + ML_ave["MLG_ent_wen"] 
            + ML_ave["MLG_adv"]
            + ML_ave["MLG_hdiff"]
            + ML_ave["MLG_vdiff"]
            + ML_ave["MLG_frc_sw"]
            + ML_ave["MLG_frc_lw"]
            + ML_ave["MLG_frc_sh"]
            + ML_ave["MLG_frc_lh"]
            + ML_ave["MLG_frc_dilu"]
    )
 
    ML_ave["dMLTdt_res"] = ML_ave["dMLTdt_res"].rename("dMLTdt_res")
   
    ####################################################
    # compute MLU, MLV, dMLTdx, dMLTdy, dMLSdx, dMLSdy
    #         U_g, V_g
    # with MLD in the end of the time interval
    ####################################################

    print("Compute advection related terms...")
    def calGrad_wrap(Q):
        return calGrad(Q, xgcm_grid, ecco_grid)
    
    ML_ave2 = {}

    for varname in ["MLT", "MLS", "MLU", "MLV", "MLU_g", "MLV_g", "MLU_ag", "MLV_ag", "dTdz_b", "dSdz_b", "T_b", "w_b", "ENT_ADV"]:
         ML_ave2[varname] = xr.zeros_like(ML_ave["MLG_adv"]).rename(varname)

    
    # Compute geostrophic balance. The code is from 
    # https://ecco-v4-python-tutorial.readthedocs.io/Geostrophic_balance.html#Right-hand-side
    dens  = ds.RHOAnoma + RHO_CONST
    pressanom = ds.PHIHYDcR

    dpdx, dpdy = calGrad_wrap(RHO_CONST * pressanom)

    f_co = 2 * OMEGA * np.sin(ecco_grid.YC * np.pi / 180)
    U_g = - dpdy / RHO_CONST / f_co
    V_g =   dpdx / RHO_CONST / f_co


    # Compute velocity at T grid
    vel_interp = xgcm_grid.interp_2d_vector({'X':ds.UVEL,'Y':ds.VVEL},boundary='extend')
    UVEL = vel_interp['X']
    VVEL = vel_interp['Y']

    for MLvarname, var in {
        "MLT" : ds["THETA"],
        "MLS" : ds["SALT"],
        "MLU" : UVEL,
        "MLV" : VVEL,
        "MLU_g" : U_g,
        "MLV_g" : V_g,
    }.items():

        for l in range(Nl):

            MLD = MLDs_snp[1][l, :, :]

            ML_ave2[MLvarname][0, l, :, :] = computeMLMean(
                var[0, :, l, :, :].to_numpy(),
                MLD,
                z_W,
                mask=mask2D[l]
            )

    
    ML_ave2["dMLTdx"], ML_ave2["dMLTdy"] = calGrad_wrap(ML_ave2["MLT"])
    ML_ave2["dMLSdx"], ML_ave2["dMLSdy"] = calGrad_wrap(ML_ave2["MLS"])

    # rotate to latlon
    UVEL, VVEL = rotateVector2LatLon(UVEL, VVEL, ecco_grid)
    U_g, V_g   = rotateVector2LatLon(U_g,  V_g, ecco_grid)
    ML_ave2["dMLTdx"], ML_ave2["dMLTdy"] = rotateVector2LatLon(ML_ave2["dMLTdx"], ML_ave2["dMLTdy"], ecco_grid) 
    ML_ave2["MLU"], ML_ave2["MLV"] = rotateVector2LatLon(ML_ave2["MLU"], ML_ave2["MLV"], ecco_grid) 
    ML_ave2["MLU_g"], ML_ave2["MLV_g"] = rotateVector2LatLon(ML_ave2["MLU_g"], ML_ave2["MLV_g"], ecco_grid) 

    ML_ave2["MLU_ag"] = ML_ave2["MLU"]  - ML_ave2["MLU_g"]
    ML_ave2["MLV_ag"] = ML_ave2["MLV"]  - ML_ave2["MLV_g"]

    # Compute advection
    ML_ave2["MLHADVT"]    = - ( ML_ave2["dMLTdx"] * ML_ave2["MLU"]    + ML_ave2["dMLTdy"] * ML_ave2["MLV"] )
    ML_ave2["MLHADVT_g"]  = - ( ML_ave2["dMLTdx"] * ML_ave2["MLU_g"]  + ML_ave2["dMLTdy"] * ML_ave2["MLV_g"] )
    ML_ave2["MLHADVT_ag"] = - ( ML_ave2["dMLTdx"] * ML_ave2["MLU_ag"] + ML_ave2["dMLTdy"] * ML_ave2["MLV_ag"] )

    for l in range(Nl):
        MLD = MLDs_snp[1][l, :, :]

        THETA = ds["THETA"][0, :, l, :, :].to_numpy()
        SALT  = ds["SALT"][0, :, l, :, :].to_numpy()
        WVEL  = ds["WVEL"][0, :, l, :, :].to_numpy()


        dTdz = calculus_tools.W_ddz_T(THETA, z_W=z_W)
        dSdz = calculus_tools.W_ddz_T(SALT, z_W=z_W)

        ML_ave2["dTdz_b"][0, l, :, :] = evalAtMLD_W(dTdz, MLD, z_W, mask=mask2D[l])
        ML_ave2["dSdz_b"][0, l, :, :] = evalAtMLD_W(dSdz, MLD, z_W, mask=mask2D[l])
        
        ML_ave2["T_b"][0, l, :, :] = evalAtMLD_T(THETA, MLD, z_W, mask=mask2D[l])
        ML_ave2["w_b"][0, l, :, :] = evalAtMLD_W(WVEL, MLD, z_W, mask=mask2D[l])
        ML_ave2["ENT_ADV"][0, l, :, :] = - ( ML_ave2["MLT"][0, l, :, :] - ML_ave2["T_b"][0, l, :, :] ) * ML_ave2["w_b"][0, l, :, :] / MLD

    print("Output data...")
    output_data = []

    for k, var in ML_ave.items():
        var = var.rename(k)
        output_data.append(var)

    for k, var in ML_ave2.items():
        var = var.rename(k)
        output_data.append(var)

    output_data.append(ML_snp["MLDs_snp"][1:2, :, :, :].rename("MLDs"))
    output_data.append(ML_snp["MLD_snp"][1:2, :, :, :].rename("MLD"))

    for var in output_data:
        for attr in ["valid_min", "valid_max"]:
            if attr in var.attrs:
                del(var.attrs[attr])

    ds_out = xr.merge(output_data, compat='override')


    print("Output: ", output_filename)

    dir_name = os.path.dirname(output_filename)
    if not os.path.isdir(dir_name):
        print("Create dir: %s" % (dir_name,))
        Path(dir_name).mkdir(parents=True, exist_ok=True)


    ds_out.to_netcdf(output_filename)

    if debug:

        print("Debug mode on.")
        print("Load Matplotlib")
        import matplotlib.pyplot as plt
        import matplotlib.transforms as transforms
        print("Loading complete.")
        sel_lat = 30.0
        sel_lon = 170.0
        sel_tile = 7

        idx = ( (ds.XC - sel_lon)**2 + (ds.YC - sel_lat)**2).argmin(dim=("tile", "j", "i"))
        print("Selected idx: ", idx)
        ds = ds.isel(time=0).isel(idx)
        ds_out = ds_out.isel(time=0, time_snp=0).isel(idx)

        print("Selected Lon: ", ds.coords["i"])
        print("Selected Lat: ", ds.coords["j"])


        VEL = xr.merge([U_g.rename("U_g"), V_g.rename("V_g"), UVEL.rename("UVEL"), VVEL.rename("VVEL")])
        VEL = VEL.isel(time=0).isel(idx)

        fig, ax = plt.subplots(1, 3, figsize=(12, 6))

        # Temperature
        _ax = ax[0]
        _ax.plot(ds.THETA, - ds.coords["Z"], label="$\\Theta$")

       
        trans = transforms.blended_transform_factory(_ax.transData, _ax.transAxes)
        _ax.plot([ds_out.MLT.to_numpy(), ] * 2, [0, 1], transform=trans, label="MLT")
        
        
        # Velocity U 
        _ax = ax[1]
        _ax.set_title("Velocity - U (zonal)")
        _ax.plot(VEL.UVEL, - ds.coords["Z"], "k-", label="UVEL")
        _ax.plot(VEL.U_g, - ds.coords["Z"], "r-", label="U_g")
        _ax.plot(VEL.UVEL - VEL.U_g, - ds.coords["Z"], "b-", label="U_ag")

        trans = transforms.blended_transform_factory(_ax.transData, _ax.transAxes)
        _ax.plot([ds_out.MLU.to_numpy(), ] * 2, [0, 1], color="gray", linestyle="dashed", transform=trans, label="MLU")
        _ax.plot([ds_out.MLU_g.to_numpy(), ] * 2, [0, 1], color="magenta", linestyle="dashed", transform=trans, label="MLU_g")
        _ax.plot([ds_out.MLU_ag.to_numpy(), ] * 2, [0, 1], color="dodgerblue", linestyle="dashed", transform=trans, label="MLU_ag")
 

        # Velocity V 
        _ax = ax[2]
        _ax.set_title("Velocity - V (meridional)")
        _ax.plot(VEL.VVEL, - ds.coords["Z"], "k-", label="VVEL")
        _ax.plot(VEL.V_g, - ds.coords["Z"], "r-", label="V_g")
        _ax.plot(VEL.VVEL - VEL.V_g, - ds.coords["Z"], "b-", label="V_ag")

        trans = transforms.blended_transform_factory(_ax.transData, _ax.transAxes)
        _ax.plot([ds_out.MLV.to_numpy(), ] * 2, [0, 1], color="gray", linestyle="dashed", transform=trans, label="MLV")
        _ax.plot([ds_out.MLV_g.to_numpy(), ] * 2, [0, 1], color="magenta", linestyle="dashed", transform=trans, label="MLV_g")
        _ax.plot([ds_out.MLV_ag.to_numpy(), ] * 2, [0, 1], color="dodgerblue", linestyle="dashed", transform=trans, label="MLV_ag")
 


        for _ax in ax.flatten():
            trans = transforms.blended_transform_factory(_ax.transAxes, _ax.transData)
            _ax.plot([0, 1], [ds_out.MLD.to_numpy(), ] * 2, transform=trans)
     
            _ax.set_ylim([0, 100])
            
            _ax.invert_yaxis()
            _ax.grid()
            _ax.legend()

        plt.show()
            



if __name__ == "__main__" : 

    print("*** This is for testing ***") 

    target_datetime = datetime(1992, 11,  4)
    output_filename = "data/ECCO_LLC/%s/%s" % ECCO_helper.getECCOFilename("MLT", "DAILY", target_datetime)
    
    output_filename = "./%s" % ECCO_helper.getECCOFilename("MLT", "DAILY", target_datetime)[1]
    
    print("Output file: ", output_filename)
    processECCO(
        target_datetime,
        output_filename,
        debug = True,
    ) 
   
     
