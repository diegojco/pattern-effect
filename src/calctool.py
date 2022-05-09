""" Module containing functions for calculating derived quantities from common
    atmospheric variables
"""
import copy
import numpy as np
import scipy.constants as spc
import scipy.stats as st
import scipy.integrate as integ
import scipy.interpolate as interpol
from netCDF4 import Dataset as ncload
import regrtool as regr
import disttool as dst

""" Physical Constants
"""

s_per_a = 365 * 24 * 60 * 60
s_per_d = 24 * 60 * 60
s_per_h = 60 * 60
d_per_a = 365

g = spc.g
lv = 2501000.0
R = spc.gas_constant
Cp = 1003.5

Mid = {
    "N2": 28.014e-3,
    "O2": 31.998e-3,
    "Ar": 39.950e-3,
    "CO2": 44.009e-3,
}
Nid = {
    "N2": 78.09e-2,
    "O2": 20.95e-2,
    "Ar": 0.93e-2,
    "CO2": 0.03e-2,
}
Md = np.array([Mid[comp] * Nid[comp] for comp in Mid]).sum()
molar_mass_dry_air = Md #28.9645e-3
Mv = 18.01528e-3
molar_mass_water = Mv
MO3 = 47.998e-3
molar_mass_ozone = MO3

fid = {
    comp: (Mid[comp] * Nid[comp]) / Md
    for comp in Mid
}
Rid = {
    comp: R / Mid[comp]
    for comp in Mid
}
fidRid = {
    comp: (fid[comp] * Rid[comp])
    for comp in Mid
}
Rd = np.array([fidRid[comp] for comp in Mid]).sum()
Rv = R / Mv
χd = Rd / Cp

triple_point_water = 273.16
Γ_d = g / Cp
Rd_by_g = Rd / g
r_avg_Earth = 6371008.7714

""" Radiation
"""

def φ_sw(φ_swu, φ_swd):
    """ Function that calculates the net shortwave radiative flux
    """
    return φ_swu + φ_swd

def φ_lw(φ_lwu, φ_lwd):
    """ Function that calculates the net longwave radiative flux
    """
    return φ_lwu + φ_lwd

def φ_u(φ_swu, φ_lwu):
    """ Function that calculates the net upwelling radiative flux
    """
    return φ_swu + φ_lwu

def φ_d(φ_swd, φ_lwd):
    """ Function that calculates the net downwelling radiative flux
    """
    return φ_swd + φ_lwd

def φ(φ_swu, φ_lwu, φ_swd, φ_lwd):
    """ Function that calculates the net radiative flux
    """
    return φ_swu + φ_lwu + φ_swd + φ_lwd

def φ_s(φ_swu, φ_lwu, φ_swd, φ_lwd):
    """ Function that calculates the net surface radiative flux
    """
    return (φ_swu + φ_lwu + φ_swd + φ_lwd)[..., 0]


""" Equilibrium climate sensitivity
"""

def ECS(Ts, N, nxCO2=2, size=0, p=0.05, konrad=False):
    """ Function that calculates the equilibrium climate sensitivity
    """
    if konrad:
        return ECS_konrad(Ts, N, nxCO2=nxCO2, size=size, p=p)
    else:
        return ECS_CMIP(Ts, N, nxCO2=nxCO2, size=size, p=p)

def ECS_konrad(Ts, N, nxCO2=2, size=0, p=0.05):
    """ Function that calculates the equilibrium climate sensitivity for `konrad`
    """
    if size == 0:
        b0 = regr.OLS_lin1d_xi(
            Ts, N,
            x_intercept=Ts[-1]
        ).beta
        lambd = b0[0]
        if nxCO2 >= 1:
            ecs = Ts[-1] / (nxCO2 / 2.0)
        else:
            ecs = - Ts[-1] / (nxCO2 / 2.0)
        Feff = - lambd * ecs
        
        return ecs, Feff, lambd
        
    else:
        b0, dist_param = regr.dist_OLS_lin1d_xi(
            Ts, N,
            x_intercept=Ts[-1], size=size, multi=False
        )
        
        lambd = dist_param[0](size=int(size / 10))
        if nxCO2 >= 1:
            ecs = Ts[-1] / (nxCO2 / 2.0)
        else:
            ecs = - Ts[-1] / (nxCO2 / 2.0)
        Feff = - lambd * ecs
        lambd = dst.percentile_range(lambd, p=p)
        Feff = dst.percentile_range(Feff, p=p)
        
        return ecs, Feff, lambd, b0, dist_param

def NT_slice_konrad(Ts, N, nxCO2=2):
    """ Function that find the appropriate indices to subset the NT-diagram for `konrad`
    """
    Ts_max_idx = np.where(Ts == Ts.max())[0][0]
    
    if nxCO2 < 1:
        # It finds the first index where N attains its minimum
        min_idx = np.where(N == N[1:].min())[0][0]
        # It finds the first index where N is non-negative
        max_idx = np.where(N < 0)[0][-1] + 1
        
    # Second, the anomalous cases for factors greater than one
    #     If the first N value is non-positive; the last Ts value is negative or the N
    #     value at the index of the maximum Ts is lesser than -0.1
    elif (
        N[0] <= 0 and
        (
            Ts[-1] <  0 or
            N[Ts_max_idx] < -0.1
        ) and
        nxCO2 >= 1
    ):
        # The minimum index is where Ts has its maximum
        min_idx = Ts_max_idx
        # The maximum index is the last index where N is greater than or equal to the
        #     value at the maximum Ts but is non-positive
        max_idx = np.where(np.logical_and(N >= N[min_idx], N <= 0))[0][-1]
        
    # In any other case
    else:
        # The minimum index is where N attains its maximum
        min_idx = np.where(N == N.max())[0][0]
        # The maximum index is the first index where DN is non-positive
        max_idx = np.where(N > 0)[0][-1] + 1
    
    return min_idx + 20, max_idx

def ECS_CMIP(Ts, N, nxCO2=2, size=0, p=0.05):
    """ Function that calculates the equilibrium climate sensitivity in CMIP models
    """
    if size == 0:
        b0 = regr.OLS_lin2d(Ts, N).beta
        Feff = b0[0]
        lambd = b0[1]
        ecs = - (Feff  * (2 / nxCO2)) / lambd
        
        return ecs, Feff, lambd
        
    else:
        b0, dist_param = regr.dist_OLS_lin2d(Ts, N, size=size, multi=True)
        joint_dist = dist_param[1]
        
        Flambda = dist_param[1](size / 10)
        Feff = dst.percentile_range(Flambda[:,0], p=p)
        lambd = dst.percentile_range(Flambda[:,1], p=p)
        ecs = dst.percentile_range(- (Flambda[:,0] * (2 / nxCO2)) / Flambda[:,1], p=p)
        
        return ecs, Feff, lambd, b0, dist_param

""" Tropopause and TTL
"""

def variable_at_plev_calc(variable, plev_index):
    """ Subsets a field to the values at a given pressure
    """
    urshape = variable.shape
    if len(urshape) == 4 or len(urshape) == 2:
        variable_plev = variable_at_plev_time(variable, plev_index)
    elif len(urshape) == 3 or len(urshape) == 1:
        variable_plev = variable_at_plev_calcs(variable, plev_index)
    
    return variable_plev

def variable_at_plev_time(variable, plev_index):
    """ Subsets a field to the values at a given pressure (several times)
    """
    t = variable.shape[0]
    variable_plev = np.zeros_like(variable[...,0])
    def f(i):
        return variable_at_plev_calcs(variable[i], plev_index)
    
    for i in range(t):
        variable_plev[i] = f(i)
    
    return variable_plev

def variable_at_plev_calcs(variable, plev_index):
    """ Subsets a field to the values at a given pressure (one time)
    """
    if len(variable.shape) == 3:
        variable_plev = variable_at_plev_columns(variable, plev_index)
    elif len(variable.shape) == 1:
        variable_plev = variable_at_plev_column(variable, plev_index)
    
    return variable_plev

def variable_at_plev_columns(variable, plev_index):
    """ Subsets a field to the values at a given pressure in columns
    """
    nxmxk = variable.shape
    nxm = variable[...,0].shape
    npm = variable[...,0].size
    npmxk = (npm, nxmxk[-1])
    variable_plev = np.zeros_like(variable[...,0]).reshape(npm)
    def f(i):
        return variable_at_plev_column(
            variable.reshape(npmxk)[i], plev_index
        )
    
    for i in range(npm):
        variable_plev[i] = f(i)
    
    variable_plev = variable_plev.reshape(nxm)
    
    return variable_plev

def variable_at_plev_column(variable, plev_index):
    """ Subsets a field to the values at a given pressure in Pa
    """
    variable_plev = variable[plev_index]
    
    return variable_plev

def pressure_level_index(p, plev):
    """ Gets the index corresponding to a given pressure in Pa
    """
    plev_index = np.where(p == plev)[0]
    if plev_index.size == 0:
        idxh = np.where(p < plev)[0][0]
        idxl = np.where(p > plev)[0][-1]
        diffh = plev - p[idxh]
        diffl = p[idxl] - plev
        if diffh > diffl:
            plev_index = idxl
        elif diffh <= diffl:
            plev_index = idxh
    else:
        plev_index = plev_index[0]
        
    return plev_index

def variable_at_cold_point_calc(variable, cp_index):
    """ Subsets a field to the values at the cold-point tropopause
    """
    urshape = variable.shape
    if len(urshape) == 4 or len(urshape) == 2:
        variable_cp = variable_at_cold_point_time(variable, cp_index)
    elif len(urshape) == 3 or len(urshape) == 1:
        variable_cp = variable_at_cold_point_calcs(variable, cp_index)
    
    return variable_cp

def variable_at_cold_point_time(variable, cp_index):
    """ Subsets a field to the values at the cold-point tropopause (several times)
    """
    t = variable.shape[0]
    variable_cp = np.zeros_like(variable[...,0])
    def f(i):
        return variable_at_cold_point_calcs(variable[i], cp_index[i])
    
    for i in range(t):
        variable_cp[i] = f(i)
    
    return variable_cp

def variable_at_cold_point_calcs(variable, cp_index):
    """ Subsets a field to the values at the cold-point tropopause (one time)
    """
    if len(variable.shape) == 3:
        variable_cp = variable_at_cold_point_columns(variable, cp_index)
    elif len(variable.shape) == 1:
        variable_cp = variable_at_cold_point_column(variable, cp_index)
    
    return variable_cp

def variable_at_cold_point_columns(variable, cp_index):
    """ Subsets a field to the values at the cold-point tropopause in columns
    """
    nxmxk = variable.shape
    nxm = variable[...,0].shape
    npm = variable[...,0].size
    npmxk = (npm, nxmxk[-1])
    variable_cp = np.zeros_like(variable[...,0]).reshape(npm)
    def f(i):
        return variable_at_cold_point_column(
            variable.reshape(npmxk)[i], cp_index.reshape(npm)[i]
        )
    
    for i in range(npm):
        variable_cp[i] = f(i)
    
    variable_cp = variable_cp.reshape(nxm)
    
    return variable_cp

def variable_at_cold_point_column(variable, cp_index):
    """ Subsets a field to the values at the cold-point tropopause in a column
    """
    variable_cp = variable[cp_index]
    
    return variable_cp

def cold_point_tropopause_index_calc(p, T, z, geopotential=False):
    """ Function that calculates the cold-point tropopause index
    """
    urshape = T.shape
    if len(urshape) == 4 or len(urshape) == 2:
        index = cold_point_tropopause_index_time(p, T, z, geopotential=geopotential)
    elif len(urshape) == 3 or len(urshape) == 1:
        index = cold_point_tropopause_index_calcs(p, T, z, geopotential=geopotential)
    
    return index

def cold_point_tropopause_index_time(p, T, z, geopotential=False):
    """ Function that calculates the cold-point tropopause index (several times)
    """
    t = T.shape[0]
    index = np.zeros_like(T[...,0]).astype(int)
    def f(i):
        return cold_point_tropopause_index_calcs(
            p, T[i], z[i], geopotential=geopotential
        )
    
    for i in range(t):
        index[i] = f(i)
    
    return index

def cold_point_tropopause_index_calcs(p, T, z, geopotential=False):
    """ Function that calculates the cold-point tropopause index (one time)
    """
    if len(T.shape) == 3:
        index = cold_point_tropopause_index_columns(p, T, z, geopotential=geopotential)
    elif len(T.shape) == 1:
        index = cold_point_tropopause_index_column(p, T, z, geopotential=geopotential)
    
    return index

def cold_point_tropopause_index_columns(p, T, z, geopotential=False):
    """ Function that calculates the cold-point tropopause index in columns
    """
    nxmxk = T.shape
    nxm = T[...,0].shape
    npm = T[...,0].size
    npmxk = (npm, nxmxk[-1])
    index = np.zeros_like(T[...,0]).reshape(npm)
    def f(i):
        return cold_point_tropopause_index_column(
            p, T.reshape(npmxk)[i], z.reshape(npmxk)[i], geopotential=geopotential
        )
    
    for i in range(npm):
        index[i] = f(i)
    
    index = index.reshape(nxm).astype(int)
    
    return index

def cold_point_tropopause_index_column(p, T, z, geopotential=False):
    """ Function that calculates the cold-point tropopause index in a column
    """
    Γ = Γ_calc(T, z, geopotential=geopotential)
    
    p_mask1 = p >= 1000
    p_mask2 = p <= 20000
    p_mask = np.logical_and(p_mask1, p_mask2)
    
    Γ_region = Γ[p_mask]
    T_region = T[p_mask]
    
    index = np.where(T == T_region.min())[0][0]
    test = (Γ[index + 1] >= 0) and (Γ[index - 1] <= 0)
    while (not test) and (index + 1 < (Γ.size - 1)):
        index += 1
        test = (Γ[index + 1] >= 0) and (Γ[index - 1] <= 0)
    
    index = int(index)
    
    return index

""" Height lapse rate with pressure
"""

def dzdp_calc(z, p, geopotential=False):
    """ Function that calculates the height lapse rate
    """
    z_work = copy.deepcopy(z)
    if geopotential:
        z_work = zz_g_calc(z)
    
    urshape = z.shape
    if len(urshape) == 4 or len(urshape) == 2:
        dzdp = dzdp_time(z_work, p)
    elif len(urshape) == 3 or len(urshape) == 1:
        dzdp = dzdp_calcs(z_work, p)
    
    return dzdp

def dzdp_time(z, p):
    """ Function that calculates the height lapse rate (several times)
    """
    t = z.shape[0]
    dzdp = np.zeros_like(z)
    def f(i):
        return dzdp_calcs(z[i], p)
    
    for i in range(t):
        dzdp[i] = f(i)
    
    return dzdp

def dzdp_calcs(z, p):
    """ Function that calculates the height lapse rate (one time)
    """
    if len(z.shape) == 3:
        dzdp = dzdp_columns(z, p)
    elif len(z.shape) == 1:
        dzdp = dzdp_column(z, p)
    
    return dzdp

def dzdp_columns(z, p):
    """ Function that calculates the height lapse rate in columns
    """
    nxmxk = z.shape
    npm = z[...,0].size
    npmxk = (npm, nxmxk[-1])
    dzdp = np.zeros_like(z).reshape(npmxk)
    def f(i):
        return dzdp_column(z.reshape(npmxk)[i], p)
    
    for i in range(npm):
        dzdp[i] = f(i)
    
    dzdp = dzdp.reshape(nxmxk)
    
    return dzdp

def dzdp_column(z, p):
    """ Function that calculates the height lapse rate in a column
    """
    return np.gradient(z, p)

""" Temperature lapse rate
"""

def ΔΓ_calc(T, z, geopotential=False):
    """ Function that calculates the difference between adiabatic and environmental
        lapse rate
    """
    return Γ_calc(T, z, geopotential=False) + Γ_d

def Γ_calc(T, z, geopotential=False):
    """ Function that calculates the temperature lapse rate
    """
    z_work = copy.deepcopy(z)
    if geopotential:
        z_work = zz_g_calc(z)
    
    urshape = T.shape
    if len(urshape) == 4 or len(urshape) == 2:
        Γ = Γ_time(T, z_work)
    elif len(urshape) == 3 or len(urshape) == 1:
        Γ = Γ_calcs(T, z_work)
    
    return Γ

def Γ_time(T, z):
    """ Function that calculates the temperature lapse rate (several times)
    """
    t = T.shape[0]
    Γ = np.zeros_like(T)
    def f(i):
        return Γ_calcs(T[i], z[i])
    
    for i in range(t):
        Γ[i] = f(i)
    
    return Γ

def Γ_calcs(T, z):
    """ Function that calculates the temperature lapse rate (one time)
    """
    if len(T.shape) == 3:
        Γ = Γ_columns(T, z)
    elif len(T.shape) == 1:
        Γ = Γ_column(T, z)
    
    return Γ

def Γ_columns(T, z):
    """ Function that calculates the temperature lapse rate in columns
    """
    nxmxk = T.shape
    npm = T[...,0].size
    npmxk = (npm, nxmxk[-1])
    Γ = np.zeros_like(T).reshape(npmxk)
    def f(i):
        return Γ_column(T.reshape(npmxk)[i], z.reshape(npmxk)[i])
    
    for i in range(npm):
        Γ[i] = f(i)
    
    Γ = Γ.reshape(nxmxk)
    
    return Γ

def Γ_column(T, z):
    """ Function that calculates the temperature lapse rate in a column
    """
    return np.gradient(T, z)

""" Height from pressure considering hydrostatic balance
"""

def zp_calc(p, T, q_wv, z0=0, p0=100000):
    """ Transforms pressure to height considering hydrostatic balance
    """
    urshape = T.shape
    if len(urshape) == 4 or len(urshape) == 2:
        zp = zp_time(p, T, q_wv, z0, p0)
    elif len(urshape) == 3 or len(urshape) == 1:
        zp = zp_calcs(p, T, q_wv, z0=z0, p0=p0)
    
    return zp

def zp_time(p, T, q_wv, z0, p0):
    """ Transforms pressure to height (several times) considering hydrostatic balance
    """
    t = T.shape[0]
    zp = np.zeros_like(T)
    if (
        ((type(p0) == np.ndarray) and (type(z0) == np.ndarray)) or
        ((type(p0) == np.ndarray) and (type(z0) != np.ndarray))
    ):
        def f(i):
            return zp_calcs(p, T[i], q_wv[i], z0=z0, p0=p0[i])
    elif (type(p0) != np.ndarray) and (type(z0) != np.ndarray):
        def f(i):
            return zp_calcs(p, T[i], q_wv[i], z0=z0, p0=p0)
    
    for i in range(t):
        zp[i] = f(i)
    
    return zp

def zp_calcs(p, T, q_wv, z0=0, p0=100000):
    """ Transforms pressure to height (one time) considering hydrostatic balance
    """
    if len(T.shape) == 3:
        zp = zp_columns(p, T, q_wv, z0, p0)
    elif len(T.shape) == 1:
        zp = zp_column(p, T, q_wv, z0=z0, p0=p0)
    
    return zp

def zp_columns(p, T, q_wv, z0, p0):
    """ Transforms pressure to height in columns considering hydrostatic balance
    """
    nxmxk = T.shape
    npm = T[...,0].size
    npmxk = (npm, nxmxk[-1])
    zp = np.zeros_like(T).reshape(npmxk)
    if (type(p0) == np.ndarray) and (type(z0) == np.ndarray):
        def f(i):
            return zp_column(
                p,
                T.reshape(npmxk)[i],
                q_wv.reshape(npmxk)[i],
                z0=z0.reshape(npm)[i],
                p0=p0.reshape(npm)[i],
            )
    elif (type(p0) == np.ndarray) and (type(z0) != np.ndarray):
        def f(i):
            return zp_column(
                p,
                T.reshape(npmxk)[i],
                q_wv.reshape(npmxk)[i],
                z0=z0,
                p0=p0.reshape(npm)[i],
            )
    elif (type(p0) != np.ndarray) and (type(z0) == np.ndarray):
        def f(i):
            return zp_column(
                p,
                T.reshape(npmxk)[i],
                q_wv.reshape(npmxk)[i],
                z0=z0.reshape(npm)[i],
                p0=p0,
            )
    else:
        def f(i):
            return zp_column(
                p,
                T.reshape(npmxk)[i],
                q_wv.reshape(npmxk)[i],
                z0=z0,
                p0=p0,
            )
    
    for i in range(npm):
        zp[i] = f(i)
    
    zp = zp.reshape(nxmxk)
    
    return zp

def zp_column(p, T, q_wv, z0=0, p0=100000):
    """ Transforms pressure to height in a column considering hydrostatic balance
    """
    zp = -Rd_by_g * T_v_calc(T, q_wv) / p
    dp = np.diff(p, prepend=p0)
    zp *= dp
    zp = np.cumsum(zp)
    zp += z0
    
    return zp

""" Virtual temperature
"""

def T_v_calc(T, q_wv):
    """ Calculate the virtual temperature
    """
    return (1 + (((Rv / Rd) - 1) * q_wv)) * T

""" Height from geopotential height
"""

def zz_g_calc(zg):
    """ Calculate the geometrical height from the geopotential height
    """
    return (zg * r_avg_Earth) / (r_avg_Earth - zg)

""" Potential temperature calculations
"""

def dθdt(p, T, z, w, dt=1, xi=χd, R=Rd, geopotential=False, components=False):
    """ Calculates the total derivative of the potential temperature
    """
    Γ = -Γ_calc(T, z, geopotential=geopotential)
    Γ_ad = Γ_d
    κ_hydro = (-1 / (ρ(p, T, R=Rd) * g * dzdp_calc(z, p, geopotential=geopotential)))
    dTdt = tder_calc(T, dt=dt)
    θdT = θ(p, T, xi=xi) / T
    Tdθ = T / θ(p, T, xi=xi)
    
    dθdt = θdT * (dTdt - (w * (Γ - (Γ_ad * κ_hydro))))
    
    if components:
        return dθdt, θdT, dTdt, Γ, κ_hydro, Tdθ
    else:
        return dθdt

def dθdt_direct(p, T, z, w, dt=1, xi=χd, geopotential=False):
    """ Calculates the total derivative of the potential temperature directly
    """
    return (
        tder_calc(θ(p, T, xi=xi), dt=dt) +
        (w * Γ_calc(θ(p, T, xi=xi), z, geopotential=geopotential))
    )

def θ(p, T, xi=χd):
    """ Calculates the potential temperature from the temperature and the pressure
        The reference value of pressure is 1000 hPa
    """
    return ((100000 / p) ** xi) * T

""" Ideal gas calculations
"""

def ρ(p, T, R=Rd):
    """ Calculates the density from the temperature and the pressure considering an
        ideal gas
    """
    return p / (Rd * T)

""" Time derivative
"""

def tder_calc(variable, dt=1):
    """ Calculates the time derivative of a variable with dt spacing
    """
    urshape = variable.shape
    if len(urshape) == 4 or len(urshape) == 3:
        tder = tder_columns(variable, dt=dt)
    elif len(urshape) == 2:
        tder = tder_column(variable, dt=dt)
    elif len(urshape) == 1:
        tder = tder_location(variable, dt=dt)
    
    return tder

def tder_columns(variable, dt=1):
    """ Calculates the time derivative of a variable in several columns with dt spacing
    """
    txnxmxk = variable.shape
    npm = variable[0,...,0].size
    txnpmxk = (txnxmxk[0], npm, txnxmxk[-1])
    tder = np.zeros_like(variable).reshape(txnpmxk)
    def f(i):
        return tder_column(variable.reshape(txnpmxk)[:, i, :], dt=dt)
    
    for i in range(npm):
        tder[:, i, :] = f(i)
    
    tder = tder.reshape(txnxmxk)
    
    return tder

def tder_column(variable, dt=1):
    """ Calculates the time derivative of a variable in a column with dt spacing
    """
    txk = variable.shape
    tder = np.zeros_like(variable)
    def f(i):
        return tder_location(variable[:, i], dt=dt)
    
    for i in range(txk[-1]):
        tder[:, i] = f(i)
    
    return tder

def tder_location(variable, dt=1):
    """ Calculates the time derivative of a variable in a location with dt spacing
    """
    return np.gradient(variable, dt)

""" Vertical integrals
"""

def fvint_z(variable, z, zlevref=0, geopotential=False):
    """ Calculates the vertical primitive of a variable in height coordinates
    """
    primitive = np.zeros_like(variable)
    k = variable.shape[-1]
    def f(i):
        if i == 0:
            return 0
        else:
            return vint_calc_z(
                variable, z, zlevi=0, zlevf=i, geopotential=geopotential
            )
    
    if zlevref != 0:
        integral_ref = vint_calc_z(
            variable, z, zlevi=0, zlevf=zlevref, geopotential=geopotential
        )
        def f(i):
            if i == 0:
                return -integral_ref
            else:
                return (
                    vint_calc_z(
                        variable, z, zlevi=0, zlevf=i, geopotential=geopotential
                    ) - integral_ref
                )
    
    for i in range(k):
        primitive[...,i] = f(i)
    
    return primitive

def vint_calc_z(variable, z, zlevi=0, zlevf=-1, geopotential=False):
    """ Calculates the integral of a variable in height coordinates
    """
    z_work = copy.deepcopy(z)
    if geopotential:
        z_work = zz_g_calc(z)
    
    urshape = variable.shape
    if len(urshape) == 4 or len(urshape) == 2:
        integral = vint_time_z(variable, z_work, zlevi=zlevi, zlevf=zlevf)
    elif len(urshape) == 3 or len(urshape) == 1:
        integral = vint_calcs_z(variable, z_work, zlevi=zlevi, zlevf=zlevf)
    
    return integral

def vint_time_z(variable, z, zlevi=0, zlevf=-1):
    """ Calculates the integral of a variable in height coordinates (several times)
    """
    t = variable.shape[0]
    integral = np.zeros_like(variable[...,0])
    def f(i):
        return vint_calcs_z(variable[i], z[i], zlevi=zlevi, zlevf=zlevf)
    
    for i in range(t):
        integral[i] = f(i)
    
    return integral

def vint_calcs_z(variable, z, zlevi=0, zlevf=-1):
    """ Calculates the integral of a variable in height coordinates (one time)
    """
    if len(variable.shape) == 3:
        integral = vint_columns_z(variable, z, zlevi=zlevi, zlevf=zlevf)
    elif len(variable.shape) == 1:
        integral = vint_column_z(variable, z, zlevi=zlevi, zlevf=zlevf)
    
    return integral

def vint_columns_z(variable, z, zlevi=0, zlevf=-1):
    """ Calculates the integral of a variable in height coordinates in columns
    """
    nxmxk = variable.shape
    npm = variable[...,0].size
    npmxk = (npm, nxmxk[-1])
    integral = np.zeros_like(variable[...,0]).reshape(npm)
    def f(i):
        return vint_column_z(
            variable.reshape(npmxk)[i], z.reshape(npmxk)[i], zlevi=zlevi, zlevf=zlevf
        )
    
    for i in range(npm):
        integral[i] = f(i)
    
    integral = integral.reshape(nxmxk[:-1])
    
    return integral

def vint_column_z(variable, z, zlevi=0, zlevf=-1):
    """ Calculates the integral of a variable in height coordinates
    """
    integral = integ.simpson(variable, z)
    if zlevi != 0:
        integral -= integ.simpson(variable[:zlevi + 1], z[:zlevi + 1])
    if (zlevf != -1) or (zlevf != z.shape[-1]):
        integral -= integ.simpson(variable[zlevf:], z[zlevf:])
    return integral

def fvint_p(variable, p, plevref=0):
    """ Calculates the vertical primitive of a variable in pressure coordinates
    """
    primitive = np.zeros_like(variable)
    k = variable.shape[-1]
    def f(i):
        if i == 0:
            return 0
        else:
            return vint_calc_p(variable, p, plevi=0, plevf=i)
    
    if plevref != 0:
        integral_ref = vint_calc_p(variable, p, plevi=0, plevf=plevref)
        def f(i):
            if i == 0:
                return -integral_ref
            else:
                return (
                    vint_calc_p(variable, p, plevi=0, plevf=i) -
                    integral_ref
                )
    
    for i in range(k):
        primitive[...,i] = f(i)
    
    return primitive

def vint_calc_p(variable, p, plevi=0, plevf=-1):
    """ Calculates the integral of a variable in pressure coordinates
    """
    urshape = variable.shape
    if len(urshape) == 4 or len(urshape) == 2:
        integral = vint_time_p(variable, p, plevi=plevi, plevf=plevf)
    elif len(urshape) == 3 or len(urshape) == 1:
        integral = vint_calcs_p(variable, p, plevi=plevi, plevf=plevf)
    
    return integral

def vint_time_p(variable, p, plevi=0, plevf=-1):
    """ Calculates the integral of a variable in pressure coordinates (several times)
    """
    t = variable.shape[0]
    integral = np.zeros_like(variable[...,0])
    def f(i):
        return vint_calcs_p(variable[i], p, plevi=plevi, plevf=plevf)
    
    for i in range(t):
        integral[i] = f(i)
    
    return integral

def vint_calcs_p(variable, p, plevi=0, plevf=-1):
    """ Calculates the integral of a variable in pressure coordinates (one time)
    """
    if len(variable.shape) == 3:
        integral = vint_columns_p(variable, p, plevi=plevi, plevf=plevf)
    elif len(variable.shape) == 1:
        integral = vint_column_p(variable, p, plevi=plevi, plevf=plevf)
    
    return integral

def vint_columns_p(variable, p, plevi=0, plevf=-1):
    """ Calculates the integral of a variable in pressure coordinates in columns
    """
    nxmxk = variable.shape
    npm = variable[...,0].size
    npmxk = (npm, nxmxk[-1])
    integral = np.zeros_like(variable[...,0]).reshape(npm)
    def f(i):
        return vint_column_p(variable.reshape(npmxk)[i], p, plevi=plevi, plevf=plevf)
    
    for i in range(npm):
        integral[i] = f(i)
    
    integral = integral.reshape(nxmxk[:-1])
    
    return integral

def vint_column_p(variable, p, plevi=0, plevf=-1):
    """ Calculates the integral of a variable in pressure coordinates
    """
    integral = integ.simpson(variable, p)
    if plevi != 0:
        integral -= integ.simpson(variable[:plevi + 1], p[:plevi + 1])
    if (plevf != -1) or (plevf != p.size):
        integral -= integ.simpson(variable[plevf:], p[plevf:])
    return integral

""" O3
"""

def O3_rv_to_r(rv):
    """ Calculates the O3 mass mixing ratio from the volume one
    """
    return (rv / (1 - rv)) * (molar_mass_ozone / molar_mass_dry_air)

def O3_r_to_q(r):
    """ Calculates the specific O3 content from the O3 mass mixing ratio
    """
    return r / (r + 1)

def O3_rv_to_q(rv):
    """ Calculates the specific O3 content from the volume mixing ratio
    """
    return O3_r_to_q(O3_rv_to_r(rv))

""" Humidity
"""

def WV_rv_to_r(rv):
    """ Calculates the water-vapour mass mixing ratio from the volume one
    """
    return (rv / (1 - rv)) * (molar_mass_water / molar_mass_dry_air)

def WV_r_to_q(r):
    """ Calculates the specific humidity from the mass mixing ratio
    """
    return r / (r + 1)

def WV_rv_to_q(rv):
    """ Calculates the specific humidity from the volume mixing ratio
    """
    return WV_r_to_q(WV_rv_to_r(rv))

def WV_rv_to_q_gpkg(rv):
    """ Calculates the specific humidity from the volume mixing ratio in g kg^{-1}
    """
    return WV_rv_to_q(rv) * 1000

def WV_rv_to_r_gpkg(rv):
    """ Calculates the water-vapour mass mixing ratio from the volume one in g kg^{-1}
    """
    return WV_rv_to_r(rv) * 1000

def WV_e_sat_liq(T):
    """ Calculates the water-vapour saturation partial pressure on liquid water
    """
    e_sat_liq = (
        54.842763
        - (6763.22 / T)
        - (4.21 * np.log(T))
        + (0.000367 * T)
        + (
            np.tanh(0.0415 * (T - 218.8))
            * (
                53.878
                - (1331.22 / T)
                - (9.44523 * np.log(T))
                + (0.014025 * T)
            )
          )
    )
    return np.exp(e_sat_liq)

def WV_e_sat_ice(T):
    """ Calculates the water-vapour saturation partial pressure on ice
    """
    e_sat_ice = (
        9.550426
        - (5723.265 / T)
        + (3.53068 * np.log(T))
        - (0.00728332 * T)
    )
    return np.exp(e_sat_ice)

def WV_e_sat(T):
    """ Calculates the water-vapour saturation partial pressure
    """
    e_sat_liq = WV_e_sat_liq(T)
    e_sat_ice = WV_e_sat_ice(T)
    e_sat = (
        e_sat_ice
        + (
            (e_sat_liq - e_sat_ice)
            * (((T - triple_point_water + 23) / 23) ** 2)
        )
    )
    if isinstance(T, np.ndarray):
        is_liq = np.where(T > triple_point_water)
        is_ice = np.where(T < triple_point_water - 23.0)
        for i in range(is_liq[0].size):
            e_sat[is_liq[0][i],is_liq[1][i]] = e_sat_liq[is_liq[0][i],is_liq[1][i]]
        for i in range(is_ice[0].size):
            e_sat[is_ice[0][i],is_ice[1][i]] = e_sat_ice[is_ice[0][i],is_ice[1][i]]
        return e_sat
    else:
        is_liq = T > triple_point_water
        is_ice = T < triple_point_water - 23.0
        if is_liq:
            return e_sat_liq
        elif is_ice:
            return e_sat_ice
        else:
            return e_sat

def WV_rv_sat(T, p):
    """ Calculates the water-vapour saturation volume mixing ratio
    """
    return WV_e_sat(T) / p

def WV_r_sat(T, p):
    """ Calculates the water-vapour saturation mass mixing ratio
    """
    return WV_rv_to_r(WV_rv_sat(T, p))

def WV_q_sat(T, p):
    """ Calculates the water-vapour saturation specific humidity
    """
    return WV_r_to_q(WV_r_sat(T, p))

def WV_r_sat_gpkg(T, p):
    """ Calculates the water-vapour saturation mass mixing ratio
    """
    return WV_r_sat(T, p) * 1000

def WV_q_sat_gpkg(T, p):
    """ Calculates the water-vapour saturation specific humidity
    """
    return WV_q_sat(T, p) * 1000

def WV_RH(rv, T, p):
    """ Calculates the relative humidity
    """
    return  (rv * p) / WV_e_sat(T)

def Γ_m_calc(T, p):
    """ Calculates the approximated moist_adiabatic_lapse_rate
    """
    Γ_m = Γ_d * (
        (1 + (lv * (WV_r_sat(T, p) / (Rd * T))))
        / (1 + ((lv ** 2) * (WV_r_sat(T, p) / (Cp * Rd * (T ** 2)))))
    )
    return Γ_m

# def height_to_pressure(height, temperature, wv_q, z0, p0):
#     """ Transforms height to pressure considering hydrostatic balance
#     """
#     pz = pressure / virtual_temperature(temperature, wv_q)
#     if type(z)
#     dz = np.diff(height, prepend=z0)
#     pz *= dz
#     pz = -np.cumsum(pz, axis=-1) / Rd_by_g
#     pz += p0
#     return pz

# def pressure_to_height(pressure, temperature, wv_q, z0=0, p0=100000):
#     """ Transforms pressure to height considering hydrostatic balance
#     """
#     zp = virtual_temperature(temperature, wv_q) / pressure
#     dp = np.diff(pressure, prepend=p0)
#     zp *= dp
#     zp = -np.cumsum(zp) * Rd_by_g
#     zp += z0
#     return zp

# def geopotential_height(geopotential):
#     """ Calculates the geopotential height
#     """
#     return geopotential / g

# def lnp_to_pressure(lnpressure):
#     """ Calculates the pressure from the logarithm of pressure
#     """
#     return np.exp(lnpressure)

# """ Thermodynamics
# """

# def virtual_temperature(temperature, wv_q):
#     """ Calculate the virtual temperature
#     """
#     return (1 + (((Rv / Rd) - 1) * wv_q)) * temperature

# def lapse_rate_column(temperature, height):
#     """ Calculates the lapse rate of a profile
#     """
#     return np.gradient(temperature, height)

# def lapse_rate(temperature, height):
#     """ Calculate the lapse rate
#     """
#     if len(temperature.shape) > 1:
#         lapse_rate = np.zeros_like(temperature)
#         for i in range(temperature.shape[0]):
#             lapse_rate[i, :] = lapse_rate_column(temperature[i, :], height[i, :])
#     else:
#         lapse_rate = lapse_rate_column(temperature, height)
#     return lapse_rate

# def tropopause_WMO_column(temperature, height):
#     """ Calculate the tropopause level for a profile, according to the WMO
#     """
#     lr = lapse_rate(temperature, height) * 1000
#     mask = np.where(lr > -2)[0]
#     tropopause_index = False
#     mask_index = 0
#     while not tropopause_index:
#         index = mask[mask_index]
#         index_2km = np.where(height - height[index] >= 2000)[0][0]
#         region = lr[index:index_2km + 1]
#         if np.all(region >= -2):
#             tropopause_index = index
#         mask_index += 1
#     return tropopause_index

# def tropopause_WMO(temperature, height):
#     """ Calculate the tropopause level according to the WMO
#     """
#     if len(temperature.shape) > 1:
#         tropopause_index = np.zeros_like(temperature[:,0])
#         for i in range(temperature.shape[0]):
#             tropopause_index[i] = tropopause_WMO_column(temperature[i, :], height[i, :])
#     else:
#         tropopause_index = lapse_rate_column(temperature, height)
#     return tropopause_index

def TLM_params_Geoffroy_b(ΔTs, N, ϵ=1e-10, it_max=10):
    """ Calculates the modified TLM parameters according to Geoffroy et al. (2013b)
    """
    
    parameters_0 = TLM_params_Geoffroy_b_init(ΔTs[0:], N[0:])
    parameters_1 = TLM_params_Geoffroy_b_iter(ΔTs[0:], N[0:], parameters_0)
    test = np.absolute(parameters_1 - parameters_0)
    it=1
    
    while (test.max() >= ϵ or it < it_max):
        parameters_0 = parameters_1[0:]
        parameters_1 = TLM_params_Geoffroy_b_iter(ΔTs[0:], N[0:], parameters_0)
        test = np.absolute(parameters_1 - parameters_0)
        it += 1
        
    F = parameters_1[0]
    Cu = parameters_1[1]
    Cd = parameters_1[2]
    λ = parameters_1[3]
    γ = parameters_1[4]
    ε = parameters_1[5]
    ΔTs_eq = - F / λ
    τ_fast, τ_slow = TLM_timescales_alt(Cu, Cd, λ, γ)
    
    return F, Cu, Cd, λ, γ, ε, ΔTs_eq, τ_fast, τ_slow

def TLM_params_Geoffroy_b_init(ΔTs, N):
    """ Calculates an initial guess of the modified TLM parameters according to
        Geoffroy et al. (2013a)
    """
    
    F0, Cu0, Cd0, λ0, γ0, ΔTs_eq0, τ_fast0, τ_slow0 = TLM_params_Geoffroy_a(ΔTs[0:], N[0:])
    
    return np.array([F0, Cu0, Cd0, λ0, γ0, 1.0])

def TLM_params_Geoffroy_b_iter(ΔTs, N, parameters):
    """ Calculates an iteration to get the modified TLM parameters according to
        Geoffroy et al. (2013b)
    """
    
    n = ΔTs.size
    x = np.zeros((2, n))
    t = np.array(range(n))
    
    sol, Nc = Two_Layer_Model(t, *parameters)
    
    x[0, :] = ΔTs[0:]
    x[1, :] = Nc[:, 1]
    beta = regr.OLS_lin3d(x, N).beta
    F = beta[0]
    λ = beta[1]
    ε = 1 - beta[2]
    ΔTs_eq = - F / λ
    
    Cu, Cd, γ, τ_fast, τ_slow = TLM_circulation(ΔTs[0:], λ, ΔTs_eq)
    
    Cd = Cd / ε
    γ = γ / ε
    
    return np.array([F, Cu, Cd, λ, γ, ε])

def TLM_params_Geoffroy_a(ΔTs, N):
    """ Calculates TLM parameters according to Geoffroy et al. (2013a)
    """
    
    F, λ, ΔTs_eq = TLM_radiative(ΔTs[0:], N[0:])
    Cu, Cd, γ, τ_fast, τ_slow = TLM_circulation(ΔTs[0:], λ, ΔTs_eq)
    
    return F, Cu, Cd, λ, γ, ΔTs_eq, τ_fast, τ_slow

def TLM_radiative(ΔTs, N):
    """ Calculates TLM radiative parameters from NT-diagram regression
    """
    
    regr0 = regr.OLS_lin2d(ΔTs[0:], N[0:]).beta
    F = regr0[0]
    λ = regr0[1]
    ΔTs_eq = - F / λ
    
    return F, λ, ΔTs_eq

def TLM_circulation(ΔTs, λ, ΔTs_eq):
    """ Calculates TLM circulation parameters from timescales
    """
    
    τ_fast, τ_slow, a_fast, a_slow = TLM_timescales(ΔTs[0:], ΔTs_eq)
    Cu = (-λ / ((a_fast / τ_fast) + (a_slow / τ_slow)))
    Cd = (-λ * ((τ_fast * a_fast) + (τ_slow * a_slow))) - Cu
    γ = Cd / ((τ_fast * a_slow) + (τ_slow * a_fast))
    
    return Cu, Cd, γ, τ_fast, τ_slow

def TLM_timescales(ΔTs, ΔTs_eq):
    """ Calculates TLM circulation timescales from initial and final periods 
    """
    
    n = ΔTs.size
    t = np.array(range(n))
    y = - (ΔTs[0:] / ΔTs_eq)
    y = np.log1p(y)
    
    idxi = 30
    idxf = 151
    regr1 = regr.OLS_lin2d(t[idxi:idxf], y[idxi:idxf]).beta
    a_slow = np.exp(regr1[0])
    τ_slow = - 1 / regr1[1]
    
    idxi = 2
    idxf = 12
    a_fast = 1 - a_slow
    τ_fast = - (ΔTs[idxi:idxf] / ΔTs_eq) - (a_slow * np.exp(-(t[idxi:idxf] / τ_slow)))
    τ_fast = np.nanmean(t[idxi:idxf] / (np.log(a_fast) - np.log1p(τ_fast)))
    
    return τ_fast, τ_slow, a_fast, a_slow

def TLM_timescales_alt(Cu, Cd, λ, γ):
    """ Calculates TLM circulation timescales from parameters 
    """
    
    b = ((λ + γ) / Cu) + (γ / Cd)
    bstar = ((λ + γ) / Cu) - (γ / Cd)
    δ = (b ** 2) - (4 * (λ * γ) / (Cu * Cd))
    τ_fast = ((Cu * Cd) / (2 * λ * γ)) * (b - np.sqrt(δ))
    τ_slow = ((Cu * Cd) / (2 * λ * γ)) * (b + np.sqrt(δ))
    
    return τ_fast, τ_slow

def Two_Layer_Model_interpolator(
    F, Cu, Cd, λ, γ, ε=1,
    t_min=0, t_max=300,
):
    """ Uses the solutions of the TLM to get interpolators for the solutions
    """
    n = round(t_max - t_min + 1)
    t = np.array(range(n))
    sol, Nc = Two_Layer_Model(t, F, Cu, Cd, λ, γ, ε)
    Tu = interpol.interp1d(t, sol[:, 0])
    Td = interpol.interp1d(t, sol[:, 1])
    Nu = interpol.interp1d(t, Nc[:, 0])
    Nd = interpol.interp1d(t, Nc[:, 1])
    N = interpol.interp1d(t, Nc[:, 2])
    
    return Tu, Td, Nu, Nd, N

def Two_Layer_Model(
    t, F, Cu, Cd, λ, γ, ε=1,
    ΔT_0=np.array([0.0, 0.0])
):
    """ Uses an ODE integrator to solve the TLM with constant forcing
    """

    sol = integ.odeint(
        Two_Layer_Model_rhs, ΔT_0, t, args=(F, Cu, Cd, λ, γ, ε),
        tfirst=True
    )
    
    Nc = np.zeros((sol[:,0].size, 3))
    for i in range(Nc.shape[0]):
        Nc[i, :2] = Two_Layer_Model_rhs_alt(t[i], sol[i], F, Cu, Cd, λ, γ, ε)
    Nc[:, 2] = Nc[:, 0] + Nc[:, 1]
    
    return sol, Nc

def Two_Layer_Model_rhs(
    t, ΔT,
    F, Cu, Cd, λ, γ, ε
):
    """ Defines the a function to evaluate the right-hand side of TLM.
    """
    ΔTu, ΔTd = ΔT
    dΔTdt = [(F + (λ * ΔTu) - (ε * γ * (ΔTu - ΔTd))) / Cu, γ * (ΔTu - ΔTd) / Cd]
    
    return dΔTdt

def Two_Layer_Model_rhs_alt(
    t, ΔT,
    F, Cu, Cd, λ, γ, ε
):
    """ Defines the a function to evaluate the right-hand side of TLM.
    """
    ΔTu, ΔTd = ΔT
    dΔTdt = [F + (λ * ΔTu) - (ε * γ * (ΔTu - ΔTd)), γ * (ΔTu - ΔTd)]
    
    return dΔTdt

def TLM_t_rev(Cu, Cd, λ, γ, ε):
    """ Calculates the time of reversal from the modified TLM parameters.
    """
    
    κ, Z = TLM_calculate_extra(Cu, Cd, λ, γ, ε)
    
    return (2 / κ) * np.arctanh(np.absolute(Z))

def TLM_net_feedback_mean(to, tf, Cu, Cd, λ, γ, ε):
    """ Mean net radiative feedback between two times (in years)
    """
    
    κ, Z = TLM_calculate_extra(Cu, Cd, λ, γ, ε)
    
    F_res = (ε + 1) / (2 * ε)
    F_pat_stat = Cu * (γ / np.absolute(λ)) * ((ε / Cu) + (1 / Cd))
    F_pat_dyn = Cu * (κ / np.absolute(λ)) * (1 / (tf - to)) * (
        ((1 / np.cosh(((κ / 2) * tf) + np.arctanh(Z))) ** 2) - 
        ((1 / np.cosh(((κ / 2) * to) + np.arctanh(Z))) ** 2)
    )
    F_pat = ((ε - 1) / (2 * ε)) * (F_pat_stat - F_pat_dyn)
    
    return (F_res + F_pat) * λ

def TLM_net_feedback(Cu, Cd, λ, γ, ε):
    """ Net radiative feedback at a given time (in years): evaluable function
    """
    
    return lambda t: TLM_net_feedback_def(t, Cu, Cd, λ, γ, ε)

def TLM_net_feedback_def(t, Cu, Cd, λ, γ, ε):
    """ Net radiative feedback at a given time (in years)
    """
    
    κ, Z = TLM_calculate_extra(Cu, Cd, λ, γ, ε)
    F_res = (ε + 1) / (2 * ε)
    F_pat_stat = Cu * (γ / np.absolute(λ)) * ((ε / Cu) + (1 / Cd))
    F_pat_dyn = Cu * (κ / np.absolute(λ)) * np.tanh(((κ / 2) * t) + np.arctanh(Z))
    F_pat = ((ε - 1) / (2 * ε)) * (F_pat_stat - F_pat_dyn)
    
    return (F_res + F_pat) * λ

def TLM_calculate_extra(Cu, Cd, λ, γ, ε):
    """ Calculate κ and Z
    """
    
    λp = λ / Cu
    γp = γ / Cu
    γpd = γ / Cd
    λh = λp - (ε * γp) - γpd 
    κ = np.sqrt((λh ** 2) + (4 * λp * γpd))
    Z = (λh + (2 * γpd)) / κ
    
    return κ, Z