"""Recompute every number quoted in the manuscript from both result sets and compare."""
import numpy as np


def cross(x, y, level=0.5):
    """First x at which y crosses `level` (linear interpolation); nan if never."""
    y = np.asarray(y); x = np.asarray(x)
    idx = np.where((y[:-1] < level) & (y[1:] >= level))[0]
    if len(idx) == 0:
        return np.nan if y[0] < level else x[0]
    i = idx[0]
    return x[i] + (level - y[i]) * (x[i + 1] - x[i]) / (y[i + 1] - y[i])


def numbers(suf):
    R = np.load(f'results{suf}.npz'); R2 = np.load(f'results2{suf}.npz')
    R3 = np.load(f'results3{suf}.npz'); S = np.load(f'resultsS1{suf}.npz')
    mus = R['mus']; mi = {round(float(m), 1): i for i, m in enumerate(mus)}
    out = {}
    b = R['E1_lam_basin'][:, 0]
    for m in [0.0, 0.4, 0.5, 0.7, 0.8, 0.9]:
        out[f'basin λ=1, μ={m}'] = b[mi[m]]
    for m in [0.7, 0.8, 0.9]:
        out[f'λ* (prosp=0.5), μ={m}'] = cross(R['E1_lam_vals'], R['E1_lam_prosp'][mi[m]])
    for m in [0.5, 0.6, 0.7, 0.8]:
        out[f'κ* (basin=0.5), μ={m}'] = cross(R['E1_kappa_vals'], R['E1_kappa_basin'][mi[m]])
    out['max prosp over κ≤1, μ=0.9'] = R['E1_kappa_prosp'][mi[0.9]].max()
    g = R['E1_gamma_vals']; gb = R['E1_gamma_basin']
    out['basin γ=0.99, μ=0'] = gb[mi[0.0], -1]
    out['max basin over γ, μ=0.8'] = gb[mi[0.8]].max()
    out['basin τ=0.7, μ=0'] = R['E1_tau_basin'][mi[0.0], list(np.round(R['E1_tau_vals'], 2)).index(0.7)]
    v = R['E2_lam_vals']; pb = R['E2_lam_basin']
    out['λ where basin first >0.05 (μ=0.8)'] = v[np.argmax(pb > 0.05)]
    out['λ where basin first >0.95 (μ=0.8)'] = v[np.argmax(pb > 0.95)]
    t = R['E2_lam_tconv']
    out['tconv λ=1'] = t[0]; out['tconv peak'] = t.max(); out['λ at tconv peak'] = v[t.argmax()]
    out['max tconv λ≥2.6'] = t[v >= 2.6].max()
    kv = R['E2_kappa_vals']
    out['κ impactors all invest (>0.99)'] = kv[np.argmax(R['E2_kappa_imp'] > 0.99)]
    out['κ dependents all invest (>0.99)'] = kv[np.argmax(R['E2_kappa_dep'] > 0.99)]
    tv = R['E2_tau_vals']; m5 = tv >= 0.5 - 1e-9
    out['min impactor invest τ≥0.5'] = R['E2_tau_imp'][m5].min()
    out['max dependent invest τ≥0.5'] = R['E2_tau_dep'][m5].max()
    out['prosp τ≥0.5 (mean)'] = R['E2_tau_prosp'][m5].mean()
    v1 = list(np.round(R2['E6_kt_v1'], 2)); v2 = list(np.round(R2['E6_kt_v2'], 2))
    out['prosp κ=0.8, τ=0'] = R2['E6_kt_prosp'][v1.index(0.8), v2.index(0.0)]
    out['prosp κ=0.8, τ=0.5'] = R2['E6_kt_prosp'][v1.index(0.8), v2.index(0.5)]
    l1 = list(np.round(R2['E6_lk_v1'], 2)); l2 = list(np.round(R2['E6_lk_v2'], 2))
    P = R2['E6_lk_prosp']
    out['prosp λ=1.2 alone'] = P[l1.index(1.2), l2.index(0.0)]
    out['prosp κ=0.4 alone'] = P[l1.index(1.0), l2.index(0.4)]
    out['prosp λ=1.2 + κ=0.4'] = P[l1.index(1.2), l2.index(0.4)]
    out['scaffold κ=0.9→0 final prosp'] = R2['E4b_k09_0_prosp'][-1]
    out['scaffold κ=0.9→0 min after withdrawal'] = R2['E4b_k09_0_prosp'][100:].min()
    out['scaffold κ=0.5 at withdrawal'] = R2['E4b_k05_0_prosp'][99]
    out['scaffold κ=0.5 final'] = R2['E4b_k05_0_prosp'][-1]
    out['no intervention final'] = R2['E4b_k0_0_prosp'][-1]
    lv = list(np.round(R3['levels'], 2))
    for x in [0.05, 0.1, 0.2, 0.3, 0.4]:
        out[f'entrenched mix from {x}'] = R3['mix'][lv.index(x)]
    out['entrenched κ=0.9 from 0.05'] = R3['kappa09'][lv.index(0.05)]
    out['entrenched λ=3 from 0.05'] = R3['lam30'][lv.index(0.05)]
    n = len(R2['E3b_kappa_path']) // 2
    out['slow ramp κ: max prosp while raising'] = R2['E3b_kappa_prosp'][:n].max()
    for a in ['private', 'monoculture', 'pooled']:
        bb = R[f'E5_{a}'][:, 0]
        out[f'info {a}: mean basin'] = bb.mean(); out[f'info {a}: sd'] = bb.std()
        out[f'info {a}: P(basin<0.1)'] = (bb < 0.1).mean()
    out['info error-free basin'] = R['E5_truth'][0]
    gS = np.round(S['g'], 4)
    out['S1 coop γ=0.70'] = S['coop'][list(gS).index(0.7)]
    out['S1 coop γ=0.85'] = S['coop'][list(gS).index(0.85)]
    out['S1 tconv peak γ'] = S['g'][np.argmax(S['t'])]; out['S1 tconv peak'] = np.max(S['t'])
    return out


if __name__ == '__main__':
    a, b = numbers(''), numbers('_pycrld')
    worst = 0
    print(f"{'quantity':45s} {'NumPy':>9s} {'pyCRLD':>9s}  |diff|")
    for k in a:
        d = abs(a[k] - b[k]) if np.isfinite(a[k]) and np.isfinite(b[k]) else (0 if np.isnan(a[k]) == np.isnan(b[k]) else np.inf)
        flag = '' if d < 0.005 or 'tconv' in k else '  <-- differs'
        if 'tconv' not in k: worst = max(worst, d)
        print(f"{k:45s} {a[k]:9.3f} {b[k]:9.3f}  {d:.3g}{flag}")
    print('worst non-timing difference:', worst)
    import json
    json.dump([[k, float(a[k]), float(b[k])] for k in a], open('compare.json', 'w'), ensure_ascii=False)
