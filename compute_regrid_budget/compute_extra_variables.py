import numpy as np
import xarray as xr
import argparse
from pathlib import Path

def loadData(root, varname, stat):
    filename  = Path(root) / f"{stat:s}_{varname:s}.nc"
    ds = xr.open_dataset(filename)[varname]
    return ds


if __name__ == "__main__":
    
    parser = argparse.ArgumentParser(
                        prog = 'plot_skill',
                        description = 'Plot prediction skill of GFS on AR.',
    )

    parser.add_argument('--input-dir', type=str, help='Output directory', required=True)
    args = parser.parse_args()
    print(args)

    
    load_varnames = [
        "MLG_frc_sw",
        "MLG_frc_lw",
        "MLG_frc_sh",
        "MLG_frc_lh",
        "MLG_frc_dilu",
        "MLG_adv",
        "MLG_hdiff",
        "MLG_vdiff",
        "MLG_ent_wep",
        "MLG_ent_wen",
    ]

    ds = dict()

    for varname in load_varnames:
        print("Loading varname: ", varname)
        ds[varname] = loadData(args.input_dir, varname, "anom")

    print("Computing extra variables")
    extra_da = dict(
        MLG_frc = ( ds['MLG_frc_sw'] + ds['MLG_frc_lw'] + ds['MLG_frc_sh']  + ds['MLG_frc_lh'] + ds['MLG_frc_dilu'] ),
        MLG_vmix = (ds['MLG_vdiff'] + ds['MLG_ent_wep']),
        MLG_nonfrc = (ds['MLG_adv'] + ds['MLG_hdiff'] + ds['MLG_vdiff'] + ds['MLG_ent_wep'] + ds['MLG_ent_wen']),
    )

    for varname, da in extra_da.items():
        output_file = Path(args.input_dir) / f"anom_{varname:s}.nc"
        
        da = da.rename(varname)
        print("Writing output: ", str(output_file))
        da.to_netcdf(output_file)


