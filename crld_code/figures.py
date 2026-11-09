import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

R = np.load('results.npz'); R2 = np.load('results2.npz')
INK, INK2, GRID = '#0b0b0b', '#52514e', '#e4e3df'
S1, S2, S3, S4 = '#2a78d6', '#eb6834', '#1baf7a', '#eda100'
plt.rcParams.update({'font.family': 'DejaVu Sans', 'font.size': 8.5, 'axes.edgecolor': INK2,
                     'axes.labelcolor': INK, 'xtick.color': INK2, 'ytick.color': INK2,
                     'axes.spines.top': False, 'axes.spines.right': False,
                     'axes.titlesize': 9.5, 'axes.titleweight': 'bold', 'axes.titlelocation': 'left',
                     'legend.frameon': False, 'savefig.dpi': 300})
mus = R['mus']


def heat(ax, Z, xv, yv, xlab, ylab, fmtx='{:g}', fmty='{:.1f}', nx=4, ny=4):
    im = ax.imshow(Z, origin='lower', aspect='auto', cmap='Blues', vmin=0, vmax=1,
                   extent=[0, Z.shape[1], 0, Z.shape[0]])
    ax.contour(np.arange(Z.shape[1]) + .5, np.arange(Z.shape[0]) + .5, Z, levels=[0.5],
               colors=S2, linewidths=1.6)
    ix = np.unique(np.linspace(0, len(xv) - 1, nx).astype(int))
    iy = np.unique(np.linspace(0, len(yv) - 1, ny).astype(int))
    ax.set_xticks(ix + .5); ax.set_xticklabels([fmtx.format(xv[i]) for i in ix])
    ax.set_yticks(iy + .5); ax.set_yticklabels([fmty.format(yv[i]) for i in iy])
    ax.set_xlabel(xlab, fontsize=7.6); ax.set_ylabel(ylab, fontsize=7.6)
    return im


# ---------------- Fig 2: lever frontiers (ecosystem outcome) -------------------------
labels = {'lam': 'Risk-pricing intensity λ\n(1 = accurate pricing)',
          'kappa': 'Materiality bridging κ',
          'tau': 'PES / water-fund transfer τ',
          'gamma': 'Decision horizon γ'}
fig, axs = plt.subplots(1, 4, figsize=(7.4, 2.4), constrained_layout=True)
for ax, (k, lab), tag in zip(axs, labels.items(), 'ABCD'):
    im = heat(ax, R[f'E1_{k}_prosp'], R[f'E1_{k}_vals'], mus, lab,
              'Materiality mismatch μ' if tag == 'A' else '')
    if tag != 'A': ax.set_yticklabels([])
    ax.set_title(tag)
cb = fig.colorbar(im, ax=axs, shrink=0.85, pad=0.01)
cb.set_label('Long-run share of time\necosystem is prosperous', fontsize=7.5)
fig.savefig('fig2_frontiers.png'); plt.close(fig)

# ---------------- Fig 3: critical transitions at mu = 0.8 ---------------------------
fig, axs = plt.subplots(2, 3, figsize=(7.4, 3.9), constrained_layout=True, sharex='col')
for j, (k, lab) in enumerate([('lam', 'Risk-pricing intensity λ'),
                              ('kappa', 'Materiality bridging κ'),
                              ('tau', 'PES transfer τ')]):
    v = R[f'E2_{k}_vals']; ax = axs[0, j]
    ax.plot(v, R[f'E2_{k}_imp'], color=S2, lw=2, label='Impactors invest')
    ax.plot(v, R[f'E2_{k}_dep'], color=S1, lw=2, label='Dependents invest')
    ax.plot(v, R[f'E2_{k}_prosp'], color=INK, lw=1.2, ls='--', label='Ecosystem prosperous')
    ax.set_ylim(-0.03, 1.03); ax.grid(color=GRID, lw=.6); ax.set_title('ABC'[j])
    if j == 0: ax.set_ylabel('Share / probability'); ax.legend(fontsize=6.5, loc='center right')
    ax = axs[1, j]
    ax.plot(v, R[f'E2_{k}_tconv'], color=INK2, lw=2)
    ax.grid(color=GRID, lw=.6); ax.set_xlabel(lab); ax.set_title('DEF'[j])
    if j == 0: ax.set_ylabel('Median learning steps\nto convergence')
fig.savefig('fig3_transitions.png'); plt.close(fig)

# ---------------- Fig 4: temporary scaffold + entrenchment -------------------------
R3 = np.load('results3.npz')
fig, axs = plt.subplots(1, 2, figsize=(7.4, 2.6), constrained_layout=True)
ax = axs[0]
t = np.arange(len(R2['E4b_k09_0_prosp'])) * 10
for key, col, lab, lw, ls in [('k09_0', S2, 'κ=0.9, then removed (κ=0)', 2.6, '-'),
                              ('k09_03', S1, 'κ=0.9, then reduced to 0.3', 1.4, '--'),
                              ('k05_0', S3, 'κ=0.5, then removed', 2, '-'),
                              ('k0_0', INK2, 'No intervention', 1.2, '-')]:
    ax.plot(t, R2[f'E4b_{key}_prosp'], color=col, lw=lw, ls=ls, label=lab)
ax.axvline(1000, color=INK2, lw=.8, ls=':'); ax.text(1040, 0.55, 'policy\nwithdrawn', fontsize=6.5, color=INK2)
ax.set_xlabel('Learning step'); ax.set_ylabel('Ecosystem prosperous')
ax.set_title('A  Temporary scaffold'); ax.grid(color=GRID, lw=.6); ax.legend(fontsize=6.3, loc='center right'); ax.set_ylim(-.03, 1.03)
ax = axs[1]
lv = R3['levels']
for key, col, lab in [('kappa09', S2, 'Strong bridging (κ=0.9)'), ('lam30', S1, 'Strong risk loading (λ=3)'),
                      ('mix', S3, 'Moderate mix (λ=1.2, κ=0.4)')]:
    ax.plot(lv, R3[key], color=col, lw=3 if key=='kappa09' else 1.6, ls='--' if key=='lam30' else '-', marker='o', ms=4, label=lab)
ax.set_xlabel('Mean initial investment probability\n(lower = more entrenched non-investment)')
ax.set_ylabel('Ecosystem prosperous'); ax.set_ylim(-.03, 1.03)
ax.set_title('B  Entrenchment'); ax.grid(color=GRID, lw=.6); ax.legend(fontsize=6.3, loc='center right')
fig.savefig('fig4_hysteresis.png'); plt.close(fig)

# ---------------- Fig 5: policy mix ---------------------------------------------------
fig, axs = plt.subplots(1, 3, figsize=(7.4, 2.5), constrained_layout=True)
v1, v2 = R2['E6_kt_v1'], R2['E6_kt_v2']
heat(axs[0], R2['E6_kt_prosp'], v2, v1, 'PES transfer τ', 'Materiality bridging κ', fmty='{:.1f}')
axs[0].set_title('A  Ecosystem prosperous')
heat(axs[1], R2['E6_kt_dep'], v2, v1, 'PES transfer τ', '', fmty='{:.1f}')
axs[1].set_title('B  Dependents invest'); axs[1].set_yticklabels([])
im = heat(axs[2], R2['E6_lk_prosp'], R2['E6_lk_v2'], R2['E6_lk_v1'], 'Materiality bridging κ',
          'Risk-pricing intensity λ', fmty='{:.1f}')
axs[2].set_title('C  Ecosystem prosperous')
cb = fig.colorbar(im, ax=axs, shrink=0.85, pad=0.01); cb.set_label('Share', fontsize=7.5)
fig.savefig('fig5_policymix.png'); plt.close(fig)

# ---------------- Fig 6: information architecture -----------------------------------
fig, axs = plt.subplots(1, 2, figsize=(7.4, 2.5), constrained_layout=True)
names = [('private', 'Private, independent assessments', S1),
         ('monoculture', 'Single shared provider (monoculture)', S2),
         ('pooled', 'Pooled independent assessments', S3)]
bins = np.linspace(0, 1, 21)
for nm, lab, col in names:
    axs[0].hist(R[f'E5_{nm}'][:, 0], bins=bins, histtype='step', lw=2, color=col, label=lab, density=True)
axs[0].axvline(R['E5_truth'][0], color=INK, lw=1, ls=':')
axs[0].set_xlabel('Basin of collective investment (per error draw)')
axs[0].set_ylabel('Density'); axs[0].set_title('A'); axs[0].legend(fontsize=6.3, loc='upper center', bbox_to_anchor=(0.5, 1.0))
axs[0].set_ylim(0, 8)
ax = axs[1]; x = np.arange(3); w = .38
mean = [R[f'E5_{n}'][:, 0].mean() for n, _, _ in names]
fail = [(R[f'E5_{n}'][:, 0] < 0.1).mean() for n, _, _ in names]
ax.bar(x - w / 2, mean, w, color=S1, label='Mean basin')
ax.bar(x + w / 2, fail, w, color=S2, label='P(collective failure: basin < 0.1)')
for xi, (m, f) in enumerate(zip(mean, fail)):
    ax.text(xi - w / 2, m + .02, f'{m:.2f}', ha='center', fontsize=6.5, color=INK)
    ax.text(xi + w / 2, f + .02, f'{f:.2f}', ha='center', fontsize=6.5, color=INK)
ax.set_xticks(x); ax.set_xticklabels(['Private', 'Monoculture', 'Pooled'])
ax.set_ylim(0, .8); ax.set_title('B'); ax.legend(fontsize=6.3, loc='upper left'); ax.grid(axis='y', color=GRID, lw=.6)
fig.savefig('fig6_information.png'); plt.close(fig)
print('figures written')
