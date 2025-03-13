from ECCO_helper import *
import xarray as xr
import xgcm
import ecco_v4_py as ecco
import numpy as np
import warnings


# Reference: https://ecco-v4-python-tutorial.readthedocs.io/ECCO_v4_Heat_budget_closure.html
def computeTendency(target_datetime, grid=None):
    
    snp_varnames = ["THETA", "ETAN"]
    ave_varnames = [ 
        "TFLUX", "oceQsw", "EXFlwnet", "EXFhl", "EXFhs",
        "ADVx_TH", "ADVy_TH", "ADVr_TH", "DFxE_TH", "DFyE_TH", "DFrE_TH", "DFrI_TH", ] 

    ds = loadECCOData_continuous(
        beg_datetime = target_datetime,
        ndays = 1,
        snp_varnames = snp_varnames,
        ave_varnames = ave_varnames,
    )
    xgcm_grid = ecco.get_llc_grid(ds)

 
    delta_t = xgcm_grid.diff(ds.time_snp, 'T', boundary='fill', fill_value=np.nan).astype('f4') / 1e9 # nanosec to sec 

    ecco_grid = getECCOGrid()
    vol = (ecco_grid.rA*ecco_grid.drF*ecco_grid.hFacC).transpose('tile','k','j','i')

    s_star_snap = 1 + ds.ETAN_snp / ecco_grid.Depth
    sTHETA = ds.THETA_snp * s_star_snap
    G_ttl = xgcm_grid.diff(sTHETA, 'T', boundary='fill', fill_value=0.0)/delta_t

    ADVxy_diff = xgcm_grid.diff_2d_vector({'X' : ds.ADVx_TH, 'Y' : ds.ADVy_TH}, boundary = 'fill')

    adv_hConvH = (-(ADVxy_diff['X'] + ADVxy_diff['Y']))

    ADVr_TH = ds.ADVr_TH.transpose('time','tile','k_l','j','i')
    adv_vConvH = xgcm_grid.diff(ADVr_TH, 'Z', boundary='fill')

    G_hadv = adv_hConvH / vol
    G_vadv = adv_vConvH / vol


    DFxyE_diff = xgcm_grid.diff_2d_vector({'X' : ds.DFxE_TH, 'Y' : ds.DFyE_TH}, boundary = 'fill')

    # Convergence of horizontal diffusion (degC m^3/s)
    dif_hConvH = (-(DFxyE_diff['X'] + DFxyE_diff['Y']))

    # Load monthly averages of vertical diffusive fluxes
    DFrE_TH = ds.DFrE_TH.transpose('time','tile','k_l','j','i')
    DFrI_TH = ds.DFrI_TH.transpose('time','tile','k_l','j','i')

    # Convergence of vertical diffusion (degC m^3/s)
    dif_vConvH = xgcm_grid.diff(DFrE_TH, 'Z', boundary='fill') + xgcm_grid.diff(DFrI_TH, 'Z', boundary='fill')

    G_hdiff = dif_hConvH / vol
    G_vdiff = dif_vConvH / vol

    Z = ecco_grid.Z.load()
    RF = np.concatenate([ecco_grid.Zp1.values[:-1],[np.nan]])

    q1 = R*np.exp(1.0/zeta1*RF[:-1]) + (1.0-R)*np.exp(1.0/zeta2*RF[:-1])
    q2 = R*np.exp(1.0/zeta1*RF[1:]) + (1.0-R)*np.exp(1.0/zeta2*RF[1:])


    zCut = np.where(Z < -200)[0][0]
    q1[zCut:] = 0
    q2[zCut-1:] = 0


    q1 = xr.DataArray(q1,coords=[Z.k],dims=['k'])
    q2 = xr.DataArray(q2,coords=[Z.k],dims=['k'])

    mskC = ecco_grid.hFacC.copy(deep=True).load()

    # Change all fractions (ocean) to 1. land = 0
    mskC.values[mskC.values>0] = 1
    forcH_subsurf_sw = ((q1*(mskC==1)-q2*(mskC.shift(k=-1)==1))*ds.oceQsw).transpose('time','tile','k','j','i')


    forcH_surf_sw = ( (q1[0]-q2[0]) * ds.oceQsw
              *mskC[0]).transpose('time','tile','j','i').assign_coords(k=0).expand_dims('k')

    forcH_sw = xr.concat([forcH_surf_sw,forcH_subsurf_sw[:,:,1:]], dim='k').transpose('time','tile','k','j','i')

    forcH_surf_nonsw_shape = (( ds.TFLUX * 0 + 1 )
              *mskC[0]).transpose('time','tile','j','i').assign_coords(k=0).expand_dims('k')
    forcH_subsurf_nonsw_shape = forcH_subsurf_sw * 0

    forcH_nonsw_shape = xr.concat([forcH_surf_nonsw_shape,forcH_subsurf_nonsw_shape[:,:,1:]], dim='k').transpose('time','tile','k','j','i')

    
    EXFfwf = ds.TFLUX - ds.oceQsw - ds.EXFhl - ds.EXFhs + ds.EXFlwnet

    G_frc_sw  = forcH_sw                            / (rhoConst*c_p) / (ecco_grid.hFacC*ecco_grid.drF)
    G_frc_lw  = forcH_nonsw_shape * (- ds.EXFlwnet) / (rhoConst*c_p) / (ecco_grid.hFacC*ecco_grid.drF)
    G_frc_sh  = forcH_nonsw_shape * ds.EXFhs        / (rhoConst*c_p) / (ecco_grid.hFacC*ecco_grid.drF)
    G_frc_lh  = forcH_nonsw_shape * ds.EXFhl        / (rhoConst*c_p) / (ecco_grid.hFacC*ecco_grid.drF)
    G_frc_fwf = forcH_nonsw_shape * EXFfwf          / (rhoConst*c_p) / (ecco_grid.hFacC*ecco_grid.drF)


    G_sum = G_hadv + G_vadv + G_hdiff + G_vdiff + G_frc_sw + G_frc_lw + G_frc_sh + G_frc_lh + G_frc_fwf
    G_res = G_sum - G_ttl

    result = {
        "Gs_ttl"     : G_ttl,
        "Gs_hadv"    : G_hadv,
        "Gs_vadv"    : G_vadv,
        "Gs_frc_sw"  : G_frc_sw,
        "Gs_frc_lw"  : G_frc_lw,
        "Gs_frc_sh"  : G_frc_sh,
        "Gs_frc_lh"  : G_frc_lh,
        "Gs_frc_fwf" : G_frc_fwf,
        "Gs_hdiff"   : G_hdiff,
        "Gs_vdiff"   : G_vdiff,
        "Gs_sum"     : G_sum,
        "Gs_res"     : G_res,
    }

    for k, v in result.items():
        result[k] = v.transpose('time', 'k', 'tile', 'j', 'i')



    """
    # I realize that data is already there. No need to do this
    result2D = {
        # My convention for vertical flux: positive upward
        "EXF_sw"     : - ds.oceQsw,
        "EXF_lw"     :   ds.EXFlwnet,
        "EXF_sh"     : - ds.EXFhs,
        "EXF_lh"     : - ds.EXFhl,
        "EXF_fwf"    : - EXFfwf,
    }
    """



    return result




def computeTendencyAdv(target_datetime):
 
    ecco_grid = getECCOGrid()
 
    # compute f from latitude of grid cell centers
    lat = ecco_grid.YC

    f = 2.0 * Omega * np.sin(np.deg2rad(lat))

    snp_varnames = []
    ave_varnames = [ 
        "THETA", "RHOAnoma", "UVEL", "VVEL", "WVEL", "PHIHYDcR",
    ]

    ds = loadECCOData_continuous(
        beg_datetime = target_datetime,
        ndays = 1,
        snp_varnames = snp_varnames,
        ave_varnames = ave_varnames,
    )

    xgcm_grid = ecco.get_llc_grid(ds)

    rho = rhoConst + ds.RHOAnoma
    rho = rho.rename("rho")

    pressanom = ds.PHIHYDcR
    

    # compute derivatives of pressure in x and y
    d_press_dx = (xgcm_grid.diff(rhoConst * pressanom, axis="X", boundary='extend')) / ecco_grid.dxC
    d_press_dy = (xgcm_grid.diff(rhoConst * pressanom, axis="Y", boundary='extend')) / ecco_grid.dyC
    
    d_THETA_dx = (xgcm_grid.diff(ds.THETA, axis="X", boundary='extend')) / ecco_grid.dxC
    d_THETA_dy = (xgcm_grid.diff(ds.THETA, axis="Y", boundary='extend')) / ecco_grid.dyC
   
    drC = ecco_grid.drC.rename(dict(k_p1='k_l'))[0:-1]  # the axis name is inconsistent so I have to do it manually
    d_THETA_dz = - xgcm_grid.diff(ds.THETA, axis="Z", boundary='extend') / drC
    #d_THETA_dz = xgcm_grid.interp(d_THETA_dz, axis="Z", boundary='fill', fill_value=0.0) 
    d_THETA_dz = d_THETA_dz.rename("d_THETA_dz")

    VADV = xgcm_grid.interp(- d_THETA_dz * ds.WVEL, axis="Z", boundary="fill", fill_value=0.0)
    VADV = VADV.rename("VADV")
        

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")    # use this to ignore future warnings caused by interp_2d_vector function
        press_grads_interp = xgcm_grid.interp_2d_vector(
            {
                "X" : d_press_dx,
                "Y" : d_press_dy,
            },
            boundary='extend',
        )
 
        THETA_grads_interp = xgcm_grid.interp_2d_vector(
            {
                "X" : d_THETA_dx,
                "Y" : d_THETA_dy,
            },
            boundary='extend',
        )


        
    
    dp_dx = press_grads_interp['X'].rename("dp_dx")
    dp_dy = press_grads_interp['Y'].rename("dp_dy")
    
    d_THETA_dx = THETA_grads_interp['X'].rename("dTHETAdx")
    d_THETA_dy = THETA_grads_interp['Y'].rename("dTHETAdy")

    V_g =   dp_dx / (f * rho)
    U_g = - dp_dy / (f * rho)

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        vel_interp = xgcm_grid.interp_2d_vector({'X':ds.UVEL,'Y':ds.VVEL}, boundary = 'extend')
    
    U = vel_interp['X']
    V = vel_interp['Y']

   
    U = U.where(ecco_grid.maskC).rename("U") 
    V = V.where(ecco_grid.maskC).rename("V")
 
    U_g = U_g.where(ecco_grid.maskC).rename("U_g")
    V_g = V_g.where(ecco_grid.maskC).rename("V_g")

    U_ag = U - U_g
    V_ag = V - V_g
    

    U_ag = U_ag.rename("U_ag")
    V_ag = V_ag.rename("V_ag")
    
    HADV_g  = - (U_g * d_THETA_dx + V_g * d_THETA_dy).rename("HADV_g")
    HADV_ag = - (U_ag * d_THETA_dx + V_ag * d_THETA_dy).rename("HADV_ag")


    HADV_g  = HADV_g.where(ecco_grid.maskC)
    HADV_ag = HADV_ag.where(ecco_grid.maskC)
    VADV    = VADV.where(ecco_grid.maskC)

    Ue = U * ecco_grid['CS'] - V * ecco_grid['SN']
    Vn = U * ecco_grid['SN'] + V * ecco_grid['CS']

    Ue_g = U_g * ecco_grid['CS'] - V_g * ecco_grid['SN']
    Vn_g = U_g * ecco_grid['SN'] + V_g * ecco_grid['CS']

    Ue_ag = U_ag * ecco_grid['CS'] - V_ag * ecco_grid['SN']
    Vn_ag = U_ag * ecco_grid['SN'] + V_ag * ecco_grid['CS']


    Ue = Ue.rename("Ue")
    Vn = Vn.rename("Vn")

    Ue_g = Ue_g.rename("Ue_g")
    Vn_g = Vn_g.rename("Vn_g")

    Ue_ag = Ue_ag.rename("Ue_ag")
    Vn_ag = Vn_ag.rename("Vn_ag")


    new_ds = xr.merge([
         Ue, Vn,
        Ue_g, Vn_g, Ue_ag, Vn_ag,
         U_g,  V_g,  U_ag,  V_ag, 
        HADV_g, HADV_ag, 
        VADV,
    ])

    

    return new_ds
   
    


if __name__ == "__main__":
    
    output_filename = "output.nc"
  

    ds = computeTendencyAdv(datetime(2005, 12, 12))
 
    print("Output file: ", output_filename) 
    ds.to_netcdf(output_filename)
    
     
    
    




 
