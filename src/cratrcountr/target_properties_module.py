from .prior_module import *
from .error_plotting_module import *


def π2(g, mi, vi, ρi):
    return g / vi**2 * (mi / ρi)**(1 / 3)


def Y(Y0, D_est, n=4):
    return Y0 * (0.1 / D_est)**(1 / n)


def π3(Y, ρt, vi):
    return Y / (ρt * vi**2)


def π4(ρt, ρi):
    return (ρt / ρi)


def πV(K1, π2, π3, π4, μ, ν):
    e1 = (6 * ν - 2 - μ) / (3 * μ)
    e2 = (6 * ν - 2) / (3 * μ)
    e3 = (2 + μ) / 2
    e4 = (-3 * μ) / (2 + μ)
    return K1 * (π2 * π4**e1 + (π3 * π4**e2)**e3)**e4


def V(πV, mi, ρt):
    return πV * mi / ρt


def D_final(V, Kr=1.1):
    return 2 * Kr * V**(1 / 3)


def D_guess(mi, K1, μ, ν, Y0, ρt, g, vi, ρi, D_est=50):
    return D_final(
        V(
            πV(
                K1,
                π2(g, mi, vi, ρi),
                π3(Y(Y0, D_est), ρt, vi),
                π4(ρt, ρi),
                μ,
                ν
            ),
            mi,
            ρt
        )
    )


def D(mi, K1, μ, ν, Y0, ρt, g, vi, ρi, D_est=50, n_it=5):
    D = D_est
    for i in range(n_it):
        D = D_guess(mi, K1, μ, ν, Y0, ρt, g, vi, ρi, D_est=D)
    return D
    

_g = 1.62
_mi = 1000
_vi = 19000 * np.cos(math.pi / 4)
_ρi = 2500
_Y0_ej = 5E5
_Y0_sr = 1.44E7
_D_est = 50
_ρt_ej = 2000
_ρt_sr = 3000
_K1_ej = 0.55
_K1_sr = 1.05
_μ_ej = 0.42
_μ_sr = 0.38
_ν = 1 / 3
_Kr = 1.1
moon_scaling_params = tuple([_g, _vi, _ρi])
ejecta_scaling_params = tuple([_K1_ej, _μ_ej, _ν, _Y0_ej, _ρt_ej])
solid_rock_scaling_params = tuple([_K1_sr, _μ_sr, _ν, _Y0_sr, _ρt_sr])


def adjust_npf(
    ej=ejecta_scaling_params, sr=solid_rock_scaling_params, 
    moon=moon_scaling_params, mi_array=10**np.linspace(0, 20, 10000)
):
    ej_D = D(mi_array, *ej, *moon)
    sr_D = D(mi_array, *sr, *moon)
    d = sr_D / 1000
    sr_d = d
    sr2ej = np.interp(d, sr_D / 1000, ej_D / sr_D)
    f = 1 - np.log10(d[(d >= 0.25) & (d < 2)] / 0.25) / np.log10(2 / 0.25)
    sr_shift = np.piecewise(
        d, 
        [d < 0.25, (d >= 0.25) & (d < 2), d >= 2], 
        [sr2ej[d < 0.25],
         (f * sr2ej[(d >= 0.25) & (d < 2)] + (1 - f)),
         1]
    )

    d = ej_D / 1000
    ej_d = d
    ej2sr = np.interp(d, ej_D / 1000, sr_D / ej_D)
    f = np.log10(d[(d >= 0.25) & (d < 2)] / 0.25) / np.log10(2 / 0.25)
    ej_shift = np.piecewise(
        d, 
        [d < 0.25, (d >= 0.25) & (d < 2), d >= 2], 
        [1,
         (f * ej2sr[(d >= 0.25) & (d < 2)] + (1 - f)),
         ej2sr[d >= 2]]
    )

    return sr_d, sr_shift, ej_d, ej_shift


_sr_d, _sr_shift, _ej_d, _ej_shift = adjust_npf()
def npf_solid_rock(d):
    return np.interp(d, _sr_d / _sr_shift, npf_new(_sr_d))


def npf_ejecta(d):
    return np.interp(d, _ej_d / _ej_shift, npf_new(_ej_d))


_X = np.log10(_sr_d / _sr_shift)
_Y = np.log10(npf_new(_sr_d))
solid_rock_npf_Fit = get_fit(polynomial_degree_11, _X, _Y)


_X = np.log10(_ej_d / _ej_shift)
_Y = np.log10(npf_new(_ej_d))
ejecta_npf_Fit = get_fit(polynomial_degree_11, _X, _Y)




