import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch

INK, INK2 = '#0b0b0b', '#52514e'
S1, S2, S3, S4 = '#2a78d6', '#eb6834', '#1baf7a', '#eda100'
plt.rcParams.update({'font.family': 'DejaVu Sans'})

fig, ax = plt.subplots(figsize=(10.6, 5.0))
ax.set_xlim(0, 100); ax.set_ylim(0, 52); ax.axis('off')


def box(x, y, w, h, title, body, col):
    ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle='round,pad=0.4,rounding_size=1.2',
                                fc=col + '1f', ec=col, lw=1.4))
    ax.text(x + 1, y + h - 1.6, title, fontsize=7.6, weight='bold', color=INK, va='top')
    ax.text(x + 1, y + h - 5.4, body, fontsize=6.3, color=INK2, va='top', linespacing=1.35)


def arrow(x1, y1, x2, y2):
    ax.add_patch(FancyArrowPatch((x1, y1), (x2, y2), arrowstyle='-|>', mutation_scale=10,
                                 color=INK2, lw=1.1))


# row 1: pipeline
box(1, 29, 22, 21, '1  Corporate data layer',
    'Firm-site impact & dependency\n(ENCORE, asset-location data,\nwatershed/basin overlays)\nDisclosures: CDP Water/Forests,\nTNFD/CSRD-ESRS E4, SBTN\nCapex, contracts, lending\n(WRDS/Compustat, LSEG,\nBloomberg, Orbis)', S1)
box(26, 29, 22, 21, '2  Calibrated game',
    'N firms per basin;\nimpact shares h, dependency d\n→ materiality mismatch μ\ncosts c, synergy f from capex\ncollapse/recovery qc, qr from\nhydro-ecological tipping models\nhorizon γ from investor base', S4)
box(51, 29, 22, 21, '3  Collective RL dynamics',
    'Deterministic CRLD\n(Barfuss et al. 2019; 2025)\nbasins of attraction,\ncritical transitions,\nhysteresis, slowing down;\nstochastic MARL & LLM-agent\nrobustness checks', S2)
box(76, 29, 23, 21, '4  Policy-lever atlas',
    'Lever frontiers λ*(μ), κ*(μ),\nτ*(μ), γ*(μ)\nScaffold vs permanent policy\nInformation architecture\n(private / monoculture / pooled)\nEarly-warning indicators', S3)
for x in (23.8, 48.8, 73.8):
    arrow(x, 39.5, x + 2, 39.5)
arrow(87, 28.6, 87, 25.3)
ax.annotate('', xy=(12, 28.6), xytext=(12, 25.3),
            arrowprops=dict(arrowstyle='-|>', color=INK2, lw=1.1, mutation_scale=10))

# row 2: validation loop
box(1, 2, 98, 21.5, '5  Empirical test and feedback (out-of-sample)', '', INK2)
ax.text(2.5, 17.5,
        'Predictions tested against observed corporate behaviour:  (i) basin-level co-investment events (water funds, landscape programmes, '
        'collective action\nplatforms) vs. modelled basin size;  (ii) staggered adoption of disclosure mandates (CSRD, TNFD uptake) as risk-pricing '
        'shocks λ;\n(iii) supply-chain / covenant linkages as κ;  (iv) rating-provider concentration in nature metrics as the monoculture '
        'parameter ρ;\n(v) persistence of collective action after programme funding ends (hysteresis).  '
        'Mismatched predictions update the calibrated game (step 2).',
        fontsize=6.4, color=INK2, va='top', linespacing=1.45)
fig.savefig('fig1_pipeline.png', dpi=300, bbox_inches='tight')
print('ok')
