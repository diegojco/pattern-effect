""" Module containing convenience functions CMIP experiments for analysing the
    upwelling-ozone project
"""
""" External Modules
"""

import calctool as clima
import disttool as dst
import model_cmip as mcmip
import matplotlib.pyplot as plt
import matplotlib as mpl
from matplotlib.ticker import LogFormatter

""" Matplotlib customised tick formatters
"""

class HectoPascalLogFormatter(LogFormatter):
    def _num_to_string(self, x, vmin, vmax):
        return '{:g}'.format(x / 100)
class normalFormatter(LogFormatter):
    def _num_to_string(self, x, vmin, vmax):
        return '{:g}'.format(x)

""" Constants
"""

cst = {
    "time_avgs": {
        "mean": "_meanstate",
        "yearly": "_yr",
    },
    "space_avgs": {
        "full": "",
        "zonal": "_zon",
        "tropical 05": "_trop_05",
        "tropical 10": "_trop_10",
        "tropical 15": "_trop_15",
        "tropical 20": "_trop_20",
        "tropical 25": "_trop_25",
        "tropical 30": "_trop_30",
        "tropical 35": "_trop_35",
        "tropical 40": "_trop_40",
        "global": "_global",
    },
    "base": "/work/bm1205/m300556/CMIP/Extra_data",
    "models": {
        "ALL": mcmip.models,
        "all": (
            "ACCESS1-0",
            "ACCESS1-3",
            "BNU-ESM",
            "CCSM4",
            "CNRM-CM5-2",
            "CNRM-CM5",
            "CanESM2",
            "FGOALS-g2",
            "GFDL-ESM2G",
            "GFDL-ESM2M",
            "GISS-E2-H_a",
            "GISS-E2-H_b",
            "GISS-E2-H_c",
            "IPSL-CM5A-MR",
            "IPSL-CM5B-LR",
            "MIROC-ESM",
            "MIROC5",
            "MPI-ESM-LR",
            "MPI-ESM-MR",
            "MPI-ESM-P",
            "MRI-CGCM3",
            "NorESM1-M",
            "bcc-csm1-1-m",
            "bcc-csm1-1",
            "inmcm4",
            "ACCESS-CM2",
            "ACCESS-ESM1-5",
            "BCC-CSM2-MR",
            "CAMS-CSM1-0",
            "CESM2-FV2",
            "CESM2-WACCM-FV2",
            "CESM2-WACCM",
            "CESM2",
            "CMCC-CM2-SR5",
            "CNRM-CM6-1-HR",
            "CNRM-ESM2-1",
            "CanESM5_a",
            "CanESM5_b",
            "FGOALS-g3",
            "GFDL-CM3",
            "GFDL-CM4",
            "GFDL-ESM4",
            "GISS-E2-1-G",
            "GISS-E2-1-H",
            "INM-CM5-0",
            "IPSL-CM6A-LR",
            "MIROC-ES2L",
            "MPI-ESM1-2-HR",
            "MPI-ESM1-2-LR",
            "NorCPM1",
            "NorESM2-MM",
            "SAM0-UNICON",
            "TaiESM1",
        ),
        "excluded": ("CNRM-CM6-1", "GISS-E2-R", "HadGEM2-ES"),
        "CMIP5": (
            "ACCESS1-0",
            "ACCESS1-3",
            "BNU-ESM",
            "CCSM4",
            "CNRM-CM5-2",
            "CNRM-CM5",
            "CanESM2",
            "FGOALS-g2",
            "GFDL-ESM2G",
            "GFDL-ESM2M",
            "GISS-E2-H_a",
            "GISS-E2-H_b",
            "GISS-E2-H_c",
            "IPSL-CM5A-MR",
            "IPSL-CM5B-LR",
            "MIROC-ESM",
            "MIROC5",
            "MPI-ESM-LR",
            "MPI-ESM-MR",
            "MPI-ESM-P",
            "MRI-CGCM3",
            "NorESM1-M",
            "bcc-csm1-1-m",
            "bcc-csm1-1",
            "inmcm4",
        ),
        "CMIP6": (
            "ACCESS-CM2",
            "ACCESS-ESM1-5",
            "BCC-CSM2-MR",
            "CAMS-CSM1-0",
            "CESM2-FV2",
            "CESM2-WACCM-FV2",
            "CESM2-WACCM",
            "CESM2",
            "CMCC-CM2-SR5",
            "CNRM-CM6-1-HR",
            "CNRM-ESM2-1",
            "CanESM5_a",
            "CanESM5_b",
            "FGOALS-g3",
            "GFDL-CM3",
            "GFDL-CM4",
            "GFDL-ESM4",
            "GISS-E2-1-G",
            "GISS-E2-1-H",
            "INM-CM5-0",
            "IPSL-CM6A-LR",
            "MIROC-ES2L",
            "MPI-ESM1-2-HR",
            "MPI-ESM1-2-LR",
            "NorCPM1",
            "NorESM2-MM",
            "SAM0-UNICON",
            "TaiESM1",
        ),
        "ozone": (
            "CNRM-CM5",
            "CNRM-CM5-2",
            "GFDL-CM3",
            "GISS-E2-H_a",
            "GISS-E2-H_b",
            "GISS-E2-H_c",
            "CESM2-WACCM",
            "CESM2-WACCM-FV2",
            "CNRM-ESM2-1",
            "GFDL-ESM4"
        )
    },
    "coords": {
        "t": {
            "name": "time",
        },
        "lat": {
            "name": "lat",
        },
        "lon": {
            "name": "lon",
        },
        "p": {
            "name": "plev",
        },
    },
    "vars": {
        "T_s": {
            "name": "ts",
            "type": "2d",
        },
        "φ_t": {
            "name": "rnt",
            "type": "2d",
        },
        "φ_tsw↑": {
            "name": "rsut",
            "type": "2d",
        },
        "φ_tsw↓": {
            "name": "rsdt",
            "type": "2d",
        },
        "φ_tlw↑": {
            "name": "rlut",
            "type": "2d",
        },
        "φ_s": {
            "name": "rns",
            "type": "2d",
        },
        "φ_ssw↑": {
            "name": "rsus",
            "type": "2d",
        },
        "φ_ssw↓": {
            "name": "rsds",
            "type": "2d",
        },
        "φ_slw↑": {
            "name": "rlus",
            "type": "2d",
        },
        "φ_slw↓": {
            "name": "rlds",
            "type": "2d",
        },
        "φl_s": {
            "name": "hfls",
            "type": "2d",
        },
        "φs_s": {
            "name": "hfss",
            "type": "2d",
        },
        "T": {
            "name": "ta",
            "type": "3d",
        },
        "u": {
            "name": "ua",
            "type": "3d",
        },
        "v": {
            "name": "va",
            "type": "3d",
        },
        "ω": {
            "name": "wap",
            "type": "3d",
        },
        "q": {
            "name": "hus",
            "type": "3d",
        },
        "zg": {
            "name": "zg",
            "type": "3d",
        },
    },
}

""" Create data structure
"""

def load_coordinate(
    coordinate,
    model_id="CESM2-WACCM",
    exp_type="piControl",
    time_avg="yearly",
    space_avg="global",
    base=cst["base"],
):
    pth = path(
        model_id=model_id, exp_type=exp_type, time_avg=time_avg, space_avg=space_avg,
        base=base
    )
    if isinstance(pth, str):
        data = load_coord(coordinate, pth, time_avg, space_avg)
    else:
        data = [load_coord(coordinate, p, time_avg, space_avg) for p in pth]    
    return data

def load_coord(
    coord,
    pth,
    time_avg,
    space_avg,
):
    with clima.ncload(pth, mode="r") as data:
        temp = data.variables[cst["coords"][coord]["name"]][:]
    return temp.data

def load_variable(
    variable,
    model_id="CESM2-WACCM",
    exp_type="piControl",
    time_avg="yearly",
    space_avg="global",
    base=cst["base"],
):
    pth = path(
        model_id=model_id, exp_type=exp_type, time_avg=time_avg, space_avg=space_avg,
        base=base
    )
    if isinstance(pth, str):
        data = load_var(variable, pth, time_avg, space_avg)
    else:
        data = [load_var(variable, p, time_avg, space_avg) for p in pth]    
    return data

def load_var(
    var,
    pth,
    time_avg,
    space_avg,
):
    with clima.ncload(pth, mode="r") as data:
        temp1 = data.variables[cst["vars"][var]["name"]]
        temp2 = cst["vars"][var]["type"]
        if time_avg != "mean":
            if space_avg == "full":
                if temp2 == "2d":
                    temp1 = temp1[:, :, :]
                elif temp2 == "3d":
                    temp1 = clima.np.transpose(
                        temp1[:, :, :, :], axes=[0, 2, 3, 1]
                    )
            elif space_avg == "zonal":
                if temp2 == "2d":
                    temp1 = temp1[:, :, 0]
                elif temp2 == "3d":
                    temp1 = temp1[:, :, :, 0]
                    temp1 = clima.np.transpose(
                        temp1, axes=[0, 2, 1]
                    )
            else:
                if temp2 == "2d":
                    temp1 = temp1[:, 0, 0]
                elif temp2 == "3d":
                    temp1 = temp1[:, :, 0, 0]
        else:
            if space_avg == "full":
                if temp2 == "2d":
                    temp1 = temp1[0, :, :]
                elif temp2 == "3d":
                    temp1 = temp1[0, :, :, :]
                    temp1 = clima.np.transpose(
                        temp1, axes=[1, 2, 0]
                    )
            elif space_avg == "zonal":
                if temp2 == "2d":
                    temp1 = temp1[0, :, 0]
                elif temp2 == "3d":
                    temp1 = temp1[0, :, :, 0]
                    temp1 = clima.np.transpose(
                        temp1, axes=[1, 0]
                    )
            else:
                if temp2 == "2d":
                    temp1 = temp1[0, 0, 0]
                elif temp2 == "3d":
                    temp1 = temp1[0, :, 0, 0]
    return temp1.data

def path(
    model_id="CESM2-WACCM",
    exp_type="piControl",
    time_avg="yearly",
    space_avg="global",
    base=cst["base"],
):
    if isinstance(cst["models"]["ALL"][model_id][exp_type]["r"], int):
        temp = (
            cst["models"]["ALL"][model_id]["model"],
            cst["models"]["ALL"][model_id][exp_type]["r"],
            cst["models"]["ALL"][model_id][exp_type]["i"],
            cst["models"]["ALL"][model_id][exp_type]["p"],
            cst["models"]["ALL"][model_id][exp_type]["f"],
            exp_type,
            time_avg,
            space_avg,
            base,
        )
        pth = path_cmip(*temp)
    else:
        pth = []
        for r in cst["models"]["ALL"][model_id][exp_type]["r"]:
            temp = (
                cst["models"]["ALL"][model_id]["model"],
                r,
                cst["models"]["ALL"][model_id][exp_type]["i"],
                cst["models"]["ALL"][model_id][exp_type]["p"],
                cst["models"]["ALL"][model_id][exp_type]["f"],
                exp_type,
                time_avg,
                space_avg,
                base,
            )
            pth.append(path_cmip(*temp))
    return pth

def path_cmip(
    model,
    r,
    i,
    p,
    f,
    exp_type,
    time_avg,
    space_avg,
    base,
):
    tavg = cst["time_avgs"][time_avg]
    savg = cst["space_avgs"][space_avg]
    if f != 0:
        return path_cmip6(
            model, r, i, p, f, exp_type, tavg, savg, base,
        )
    else:
        return path_cmip5(
            model, r, i, p, exp_type, tavg, savg, base,
        )

def path_cmip5(
    model,
    r,
    i,
    p,
    exp_type,
    time_avg,
    space_avg,
    base,
):
    temp = exp_type
    if exp_type == "abrupt-4xCO2":
        temp = "abrupt4xCO2"
    path = f"{base}/{model}_r{r}i{i}p{p}_{temp}{time_avg}{space_avg}.nc"
    return path

def path_cmip6(
    model,
    r,
    i,
    p,
    f,
    exp_type,
    time_avg,
    space_avg,
    base,
):
    path = f"{base}/{model}_r{r}i{i}p{p}f{f}_{exp_type}{time_avg}{space_avg}.nc"
    return path