const fs = require('fs');
const {
  Document, Packer, Paragraph, TextRun, HeadingLevel, AlignmentType, ImageRun, Table, TableRow,
  TableCell, WidthType, ShadingType, BorderStyle, Footer, SimpleField, LevelFormat, PageBreak
} = require('docx');

const NAVY = '1F3A5F', ACCENT = 'C8552B', GREY = '52514E', LIGHT = 'EEF2F7';
const FONT = 'Arial';
const W = 9026; // A4 text width at 1" margins (DXA)

// ---------- helpers ----------
function runs(text, base = {}) {
  // supports **bold** and *italic*
  const out = [];
  const re = /(\*\*[^*]+\*\*|\*[^*]+\*)/g;
  let last = 0, m;
  while ((m = re.exec(text)) !== null) {
    if (m.index > last) out.push(new TextRun({ text: text.slice(last, m.index), ...base }));
    const t = m[0];
    if (t.startsWith('**')) out.push(new TextRun({ text: t.slice(2, -2), bold: true, ...base }));
    else out.push(new TextRun({ text: t.slice(1, -1), italics: true, ...base }));
    last = m.index + t.length;
  }
  if (last < text.length) out.push(new TextRun({ text: text.slice(last), ...base }));
  return out;
}
const P = (text, opts = {}) => new Paragraph({ children: runs(text, opts.run || {}), spacing: { after: 120, line: 276 }, alignment: opts.align || AlignmentType.JUSTIFIED, ...(opts.p || {}) });
const H1 = t => new Paragraph({ heading: HeadingLevel.HEADING_1, children: [new TextRun(t)] });
const H2 = t => new Paragraph({ heading: HeadingLevel.HEADING_2, children: [new TextRun(t)] });
const H3 = t => new Paragraph({ heading: HeadingLevel.HEADING_3, children: [new TextRun(t)] });
const B = (text, level = 0) => new Paragraph({ numbering: { reference: 'bullets', level }, children: runs(text), spacing: { after: 60, line: 264 } });
const N = (text, ref = 'qs') => new Paragraph({ numbering: { reference: ref, level: 0 }, children: runs(text), spacing: { after: 60, line: 264 } });

function pull(text) {
  return new Paragraph({
    children: runs(text, { color: NAVY, size: 22, italics: true }),
    spacing: { before: 160, after: 200, line: 288 },
    indent: { left: 567, right: 567 },
    border: { left: { style: BorderStyle.SINGLE, size: 24, color: ACCENT, space: 12 } },
  });
}

function figure(file, w, h, caption) {
  const img = fs.readFileSync(file);
  const scale = 600 / w; // ~6.25 in wide
  return [
    new Paragraph({ alignment: AlignmentType.CENTER, spacing: { before: 120, after: 60 }, keepNext: true,
      children: [new ImageRun({ type: 'png', data: img, transformation: { width: Math.round(w * scale), height: Math.round(h * scale) } })] }),
    new Paragraph({ children: runs(caption, { size: 17, color: GREY }), spacing: { after: 220, line: 252 }, alignment: AlignmentType.JUSTIFIED }),
  ];
}

const border = { style: BorderStyle.SINGLE, size: 4, color: 'BFC7D1' };
const borders = { top: border, bottom: border, left: border, right: border };
function table(headers, rows, widths) {
  const total = widths.reduce((a, b) => a + b, 0);
  const cell = (t, i, head) => new TableCell({
    borders, width: { size: widths[i], type: WidthType.DXA },
    shading: head ? { fill: NAVY, type: ShadingType.CLEAR, color: 'auto' } : undefined,
    margins: { top: 60, bottom: 60, left: 100, right: 100 },
    children: [new Paragraph({ children: runs(t, head ? { bold: true, color: 'FFFFFF', size: 17 } : { size: 17 }), spacing: { after: 0, line: 240 } })],
  });
  return new Table({
    width: { size: total, type: WidthType.DXA }, columnWidths: widths,
    rows: [new TableRow({ tableHeader: true, children: headers.map((h, i) => cell(h, i, true)) }),
      ...rows.map((r, k) => new TableRow({ children: r.map((t, i) => {
        const c = cell(t, i, false);
        return k % 2 ? new TableCell({ borders, width: { size: widths[i], type: WidthType.DXA }, shading: { fill: 'F6F8FB', type: ShadingType.CLEAR, color: 'auto' }, margins: { top: 60, bottom: 60, left: 100, right: 100 }, children: [new Paragraph({ children: runs(t, { size: 17 }), spacing: { after: 0, line: 240 } })] }) : c;
      }) }))],
  });
}
function boxed(title, paras) {
  return new Table({
    width: { size: W, type: WidthType.DXA }, columnWidths: [W],
    rows: [new TableRow({ children: [new TableCell({
      borders: { top: { style: BorderStyle.SINGLE, size: 12, color: NAVY }, bottom: border, left: border, right: border },
      width: { size: W, type: WidthType.DXA }, shading: { fill: LIGHT, type: ShadingType.CLEAR, color: 'auto' },
      margins: { top: 140, bottom: 140, left: 200, right: 200 },
      children: [new Paragraph({ children: [new TextRun({ text: title, bold: true, color: NAVY, size: 22 })], spacing: { after: 120 } }), ...paras],
    })] })],
  });
}
const spacer = () => new Paragraph({ children: [], spacing: { after: 120 } });

// ---------- content ----------
const kids = [];

kids.push(new Paragraph({ children: [new TextRun({ text: 'RESEARCH ARTICLE  ·  DRAFT FOR SCIENCE  ·  5 NOVEMBER 2026', size: 16, color: ACCENT, bold: true })], spacing: { after: 160 } }));
kids.push(new Paragraph({ heading: HeadingLevel.TITLE, children: [new TextRun('Materiality mismatch traps private nature finance; collective-intelligence levers can release it')] }));
kids.push(P('Maganizo Kruger Nyasulu', { run: { bold: true }, align: AlignmentType.LEFT, p: { spacing: { after: 40 } } }));
kids.push(P('Oxford Sustainable Finance Group, School of Geography and the Environment, University of Oxford, Oxford, UK', { run: { size: 18, color: GREY }, align: AlignmentType.LEFT, p: { spacing: { after: 200 } } }));

kids.push(pull('**One-sentence summary.** Pricing nature risk cannot mobilise the firms that degrade an ecosystem when they are not the firms that depend on it; instruments that transmit losses across firms, applied early and temporarily, can.'));

kids.push(H2('Abstract'));
kids.push(P('Policy relies on pricing nature-related financial risk to draw private capital into nature recovery. This fails when the firms that drive ecosystem degradation are not those financially exposed to it. Using multi-agent reinforcement-learning dynamics, we model firms that share a tipping-prone ecosystem and learn whether to invest in nature-positive action. Accurate risk pricing sustains collective investment when impacts and dependencies align, but its basin of attraction vanishes as they diverge. Mismatch removes the basin, not the cooperative equilibrium, which turns the problem into one of equilibrium selection. Instruments that make firms internalise others’ losses restore collective investment; strong, time-limited versions lock it in after withdrawal; payment transfers crowd out dependent firms’ own action; and a single shared risk-assessment provider nearly triples the probability of collective failure.'));

// ---------------- Introduction
kids.push(H1('Introduction'));
kids.push(P('More than half of global economic output is moderately or highly dependent on nature, and the financial system has been asked to treat its degradation as a source of risk (1–3). The dominant policy theory of change is informational: if nature-related risks are measured, disclosed and priced, capital will be reallocated toward firms that protect the ecosystems they rely on, and those firms will invest in nature recovery (3, 4). Alongside offset and credit markets, risk pricing is the principal route through which policy expects private capital to reach nature-positive action.'));
kids.push(P('The theory carries a hidden premise: that the firms whose operations degrade an ecosystem are the firms whose balance sheets bear the cost of its degradation. In shared landscapes the premise is often false. An upstream agribusiness, mine or plantation may account for most land conversion and water abstraction in a catchment while bearing little of the loss when the catchment tips; downstream beverage producers, hydropower operators, municipal utilities and their insurers bear the loss but control little of the pressure. We call this divergence between the distribution of impacts and the distribution of financial dependencies **materiality mismatch**. Under mismatch, accurately pricing each firm’s own financial exposure tells the impactor what it already knows: that degradation is not its problem.'));
kids.push(P('Materiality mismatch makes private nature finance a problem of collective action among heterogeneous, adaptive organisations embedded in a dynamic environment. Two recent syntheses supply the tools to analyse it. Barfuss et al. (5) propose a science of *collective cooperative intelligence* that couples complex-systems science (CSS) with multi-agent reinforcement learning (MARL), using deterministic collective reinforcement-learning dynamics (CRLD) to expose multistability, critical transitions and hysteresis in how learning agents come to cooperate under ecological tipping. Burton et al. (6) identify diversity, individual competence and aggregation as the components of collective intelligence, and warn that shared algorithmic information sources, including large language models, can homogenise judgement and create illusions of consensus. Neither framework has been applied to corporate nature finance, where firms learn from returns rather than reason from first principles and increasingly rely on a small number of shared data and AI providers to assess nature risk.'));
kids.push(P('We ask which policy avenues can mobilise private nature-positive investment when risk pricing fails, and why. We build a corporate nature-commons game calibrated to the structure of a shared catchment, derive its learning dynamics, and map how four families of instrument (risk pricing, materiality bridging, payment transfers and investor horizon) and three information architectures shape whether firms converge on collective investment. We then specify an empirical pipeline that maps each model parameter to observable corporate data, so that the predictions can be tested against firms’ actual decisions.'));

kids.push(boxed('Box 1  Research question and question stream', [
  P('**Core research question.** Under what conditions, and through which policy levers, do private firms that share a tipping-prone ecosystem but differ in their impacts on it and their financial dependence on it learn to co-invest in nature-positive action, when pricing nature risk alone fails to make degradation financially material to the firms that cause it?', { align: AlignmentType.LEFT }),
  P('**Question stream** (each answer conditions the next; *M* = model, *E* = empirical, *P* = policy translation).', { align: AlignmentType.LEFT }),
  P('**A. Diagnosing mismatch**', { align: AlignmentType.LEFT, p: { spacing: { after: 40 } } }),
  N('*E* How large is materiality mismatch μ across the world’s river basins, measured as the distance between firms’ shares of impact and of financial dependency?'),
  N('*E* In which sectors and basin types (headwater versus downstream-heavy, plantation versus industrial) is μ high, and who are the structural impactors and dependents?'),
  N('*E* Does high μ predict weaker corporate engagement in basin-level collective action after controlling for exposure, size and disclosure?'),
  P('**B. Dynamics of corporate learning**', { align: AlignmentType.LEFT, p: { spacing: { after: 40 } } }),
  N('*M* For a calibrated basin, is collective investment a stable equilibrium, and how large is its basin of attraction?'),
  N('*M* Where are the critical thresholds in mismatch, horizon and ecosystem fragility at which collective investment becomes unreachable?'),
  N('*M* Do convergence times rise near those thresholds (critical slowing down), and can that rise be read from firms’ observed decision volatility?'),
  N('*M* Are the deterministic findings robust to stochastic exploration, partial observability of the ecosystem state, more firms and continuous investment levels?'),
  P('**C. Policy levers**', { align: AlignmentType.LEFT, p: { spacing: { after: 40 } } }),
  N('*M* What intensity of each lever (λ, κ, τ, γ) is needed to restore collective investment at a given μ, and which levers are substitutes or complements?'),
  N('*M/E* Which real instruments transmit others’ losses to impactors (supply-contract clauses, sustainability-linked loan KPIs, insurance pools, double-materiality reporting), and how strong is each in practice?'),
  N('*M* How should payment schemes such as water funds be designed so that transfers to impactors do not crowd out dependents’ own action?'),
  N('*M* Is a strong, time-limited intervention more effective than a permanent moderate one or a gradual phase-in, and when can it be withdrawn without unravelling?'),
  P('**D. Information, data and AI**', { align: AlignmentType.LEFT, p: { spacing: { after: 40 } } }),
  N('*M* How does correlation in firms’ nature-risk assessments (a single rating, dataset or AI provider) change the distribution of collective outcomes?'),
  N('*E* How concentrated are the providers and models that firms and lenders actually use to assess nature dependency, and how divergent are their scores?'),
  N('*M/E* Do pooled, independent assessment infrastructures (data utilities, basin observatories) preserve functional diversity while improving accuracy?'),
  N('*M* Can LLM-mediated deliberation among basin stakeholders lower coordination costs without homogenising their assessments?'),
  P('**E. Validation**', { align: AlignmentType.LEFT, p: { spacing: { after: 40 } } }),
  N('*E* Does modelled basin size predict which basins host corporate co-investment platforms, and when they form?'),
  N('*E* Do staggered disclosure mandates (risk-pricing shocks) raise engagement among dependents but not impactors, as the model predicts?'),
  N('*E* Does collective action persist after programme funding ends, as hysteresis predicts, and does persistence depend on how strong the programme was?'),
  P('**F. Policy translation**', { align: AlignmentType.LEFT, p: { spacing: { after: 40 } } }),
  N('*P* Which supervisors, standard-setters and market actors control each lever, and where do mandates leave gaps?'),
  N('*P* How should disclosure scoping (for example, EU sustainability-reporting thresholds) be set so that structural impactors remain in scope?'),
]));
kids.push(spacer());

// ---------------- Model
kids.push(H1('A corporate nature commons with materiality mismatch'));
kids.push(P('We extend the ecological public-goods stochastic game of Barfuss et al. (5, 7) to firms with heterogeneous impact and dependency. N firms share an ecosystem that is either prosperous or degraded. In each period every firm chooses whether to invest in nature-positive action (restoration, abstraction limits, conversion-free sourcing) at cost c. Investments generate a shared co-benefit with synergy factor f, split equally. In the prosperous state, the per-period probability of collapse is q_c·Σ h_i over non-investing firms, where h_i is firm i’s share of ecosystem pressure (Σh_i = 1). A degraded ecosystem recovers with probability q_r and, while degraded, firm i incurs a loss m·d_i, where d_i is its relative financial dependency. With h_i = 1/N and d_i = 1 the game reduces exactly to the environment in (5). In general it is pyCRLD’s EcologicalPublicGood environment with firm-specific collapse leverage N·q_c·h_i and collapse impact m·d_i; payment transfers are added through a subclass.'));
kids.push(P('We consider four firms in two groups: impactors with h = (1+μ)/4 and d = 1−μ, and dependents with h = (1−μ)/4 and d = 1+μ. The parameter μ ∈ [0,1] equals the total-variation distance between the impact distribution and the dependency distribution, ½Σ|h_i − d_i/Σd|, so it is directly measurable from firm-level data. At μ = 0 every firm bears risk in proportion to the pressure it exerts; at μ → 1 impactors exert all pressure and bear no loss.'));
kids.push(P('Firms learn by temporal-difference reinforcement learning. Following (5, 8), we study the deterministic limit of strategy actor-critic learning, in which each firm’s probability of investing in each state is updated multiplicatively by the exponentiated, strategy-averaged temporal-difference error. The dynamics are fast to integrate and expose the full stability landscape. All results are computed with pyCRLD, the library used in (5), and checked against an independent NumPy implementation (Materials and Methods). The unmodified library reproduces the benchmark of (5): for two symmetric firms, full non-investment is globally stable below a discount factor of about 0.70, a bistable regime follows, cooperation dominates above about 0.85, and convergence times peak near 0.75 (fig. S1; compare Fig. 2 of ref. 5).'));
kids.push(P('Four instruments enter as parameters. **Risk pricing** λ scales each firm’s perceived loss; λ = 1 is accurate pricing of the firm’s own exposure and λ > 1 represents loadings beyond expected loss, such as capital surcharges or tail-risk weighting. **Materiality bridging** κ makes each firm internalise a share κ of the mean loss of the other firms, representing supply contracts conditioned on basin outcomes, sustainability-linked loan terms, shared insurance pools or double-materiality obligations; formally it is the other-regarding reward shown to promote cooperation in MARL (5, 9). **Payment transfers** τ pay each investing impactor τ·c, funded by dependents, as in water funds and payment-for-ecosystem-services schemes (10). **Investor horizon** γ is the discount factor, the lever emphasised in (7). The main analyses use f = 1.2, c = 5, m = −5, q_c = 0.8, q_r = 0.01, γ = 0.9 and learning rate 0.05; outcomes are averaged over 64 random initial strategies. The headline outcome is the long-run share of time the ecosystem spends in the prosperous state.'));

kids.push(...figure('figs/fig1_pipeline.png', 2524, 1215,
  '**Fig. 1. Applied research pipeline.** Corporate data (1) calibrate a basin-level game (2) whose learning dynamics (3) yield a policy-lever atlas (4). Predictions are tested against observed corporate behaviour (5), and mismatches update the calibration. Table 2 operationalises each step.'));

// ---------------- Results
kids.push(H1('Results'));
kids.push(H2('Accurate risk pricing fails as mismatch rises'));
kids.push(P('With accurate pricing (λ = 1), the share of initial conditions from which all firms converge on investment falls from 0.75 at μ = 0 to 0.47 at μ = 0.4, 0.27 at μ = 0.5, 0.02 at μ = 0.7 and zero at μ ≥ 0.8 (λ = 1 in Fig. 2A). Collective investment is reached only by loading perceived losses beyond expected values: between λ = 1.5 and 2 at μ = 0.7, λ ≈ 2.2 at μ = 0.8 and λ ≈ 3.7 at μ = 0.9. Because loading multiplies each firm’s own dependency, the required loading diverges as the impactors’ dependency approaches zero. Disclosure that reveals true exposure cannot supply this; it is the regime in which risk pricing, as currently conceived, has nothing to price.'));
kids.push(pull('Mismatch does not destroy the cooperative equilibrium. It erases the path to it.'));
kids.push(H2('Mismatch removes the basin, not the equilibrium'));
kids.push(P('At μ = 0.8 and λ = 1 no random starting point leads to collective investment, yet collective investment remains a stable fixed point: firms that are already investing stay there (see Fig. 5A). For impactors, the expected cost of triggering collapse through their own large share of pressure, discounted over the recovery time, still exceeds the saving from not investing once others invest. Mismatch therefore converts nature finance from a problem of incentives into a problem of equilibrium selection. This distinction matters for policy. An instrument need not make investment individually optimal for every firm in every state; it needs to move firms into the basin of the cooperative equilibrium, after which the equilibrium can hold itself.'));

kids.push(...figure('figs/fig2_frontiers.png', 2220, 720,
  '**Fig. 2. Lever atlas.** Long-run share of time the ecosystem is prosperous as a function of materiality mismatch μ and (A) risk-pricing intensity λ, (B) materiality bridging κ, (C) payment transfer τ and (D) investor horizon γ. Orange line: 0.5 contour. Each cell averages 64 learning trajectories from random initial strategies.'));

kids.push(H2('Materiality bridging restores collective investment; horizon alone does not'));
kids.push(P('Transmitting a share of other firms’ losses to each firm restores the cooperative basin over most of the mismatch range (Fig. 2B). The κ needed to reach a basin of one half rises from about 0.1 at μ = 0.5 to 0.2 at μ = 0.6, 0.3 at μ = 0.7 and 0.6 at μ = 0.8. At μ = 0.9, κ ≤ 1 lifts ecosystem prosperity only to 0.2, because dependents then hold too little leverage to be worth their own investment. Lengthening investor horizons helps when impacts and dependencies align (γ = 0.99 raises the basin from 0.75 to 0.91 at μ = 0) but does almost nothing under mismatch (basin ≤ 0.03 at μ = 0.8; Fig. 2D): a patient impactor still has nothing at stake.'));
kids.push(P('At μ = 0.8 the routes to collective investment differ in character (Fig. 3). Risk loading produces an abrupt transition between λ = 1.8 and 2.4 in which both groups switch together. Materiality bridging produces a graded transition in which impactors move first (all investing by κ ≈ 0.55) and dependents follow (κ ≈ 0.8). Near the risk-loading threshold, median convergence time rises from 357 steps at λ = 1 to about 7,800 at λ = 2.0, against fewer than 120 above λ = 2.6 (Fig. 3D): critical slowing down, the signature of an approaching transition (11). For a supervisor or a basin initiative, rising volatility and slower settling of firms’ engagement decisions would indicate that an instrument is close to its tipping intensity.'));

kids.push(...figure('figs/fig3_transitions.png', 2220, 1170,
  '**Fig. 3. Critical transitions at μ = 0.8.** (A to C) Investment probability of impactors and dependents and share of time the ecosystem is prosperous, against each lever. (D to F) Median learning steps to convergence. Risk loading shows an abrupt transition with sharp critical slowing down; bridging shows a graded transition led by impactors; payment transfers convert impactors but dependents stop investing.'));

kids.push(H2('Payment transfers shift the burden rather than share it'));
kids.push(P('Payment transfers to impactors appear the most direct remedy, and they do convert impactors: at μ = 0.8 and τ ≥ 0.5 impactors invest with probability above 0.98. But dependents stop investing entirely, because the transfer becomes their contribution. Since ecosystems recover far more slowly than they collapse, the residual pressure from non-investing dependents keeps the ecosystem prosperous only 11% of the time (Fig. 3C). Transfers also undercut bridging: at κ = 0.8, prosperity falls from 0.97 without transfers to 0.21 with τ = 0.5 (Fig. 4A, B). Under low mismatch, moderate transfers (τ = 0.2 to 0.3) do help, but large transfers again shrink the basin (to 0.11 at μ = 0, τ = 0.7; Fig. 2C). The finding is specific to transfer designs in which payers buy others’ action without retaining their own obligations. It implies that water funds and similar schemes should tie payments to basin outcomes and pair them with commitments from contributors, not substitute for them.'));
kids.push(P('Risk loading and bridging are strong complements (Fig. 4C). At μ = 0.8, a modest loading (λ = 1.2) alone yields prosperity of 0.01 and moderate bridging (κ = 0.4) alone yields 0.26, but together they yield 0.94. Risk pricing is therefore not redundant; it is insufficient on its own and most valuable as a complement to instruments that link firms’ fortunes.'));

kids.push(...figure('figs/fig5_policymix.png', 2220, 750,
  '**Fig. 4. Policy mixes at μ = 0.8.** (A) Ecosystem prosperity and (B) dependents’ investment under combinations of materiality bridging κ and payment transfer τ: transfers erode the effect of bridging by displacing dependents’ own action. (C) Ecosystem prosperity under risk loading λ and bridging κ: the two are complements. Orange line: 0.5 contour.'));

kids.push(H2('Strong, early, temporary scaffolds lock in collective investment'));
kids.push(P('Because the cooperative equilibrium survives mismatch, an instrument can be withdrawn once firms are inside its basin. Applying strong bridging (κ = 0.9) for 1,000 learning steps and then removing it entirely leaves the ecosystem prosperous for the remainder of the run (Fig. 5A). A weaker scaffold (κ = 0.5) secures a partial gain (0.38 while active, 0.32 after withdrawal), compared with 0.01 without intervention. This is hysteresis in the sense of (5): the system’s state depends on its policy history, which functions as collective memory.'));
kids.push(P('The same property cuts the other way. When non-investment is already entrenched, moderate instruments fail. The moderate mix that succeeds from random starting points (λ = 1.2, κ = 0.4) achieves prosperity of 0.10 or less when firms start with mean investment probability of 0.2 or below, rising to 0.78 at 0.3 and 0.99 at 0.4 (Fig. 5B). Strong instruments (κ = 0.9 or λ = 3) succeed even from 5%. Gradual phase-ins are the worst case: raising κ slowly from 0 to 1 over 30,000 learning steps from an entrenched start never unlocks investment, because firms settle deeper into non-investment during the sub-threshold phase. Timing and intensity matter more than permanence.'));

kids.push(...figure('figs/fig4_hysteresis.png', 2220, 780,
  '**Fig. 5. Scaffolding and entrenchment at μ = 0.8.** (A) Ecosystem prosperity when bridging is applied for 1,000 learning steps and then removed or reduced (dotted line). The κ = 0.9 trajectories with full and partial withdrawal coincide. (B) Ecosystem prosperity against firms’ initial investment probability for strong and moderate instruments; the strong-bridging and strong-loading curves coincide at 1.'));

kids.push(H2('A single shared assessment provider nearly triples the risk of collective failure'));
kids.push(P('Firms do not observe their own dependency directly; they estimate it with ratings, datasets and, increasingly, AI tools. Following the collective-intelligence components identified by Burton et al. (6), we compared three information architectures at μ = 0.5 and γ = 0.95, drawing multiplicative log-normal errors (σ = 0.6) in each firm’s perceived dependency over 300 draws: private, independent assessments; a single shared provider whose error is common to all firms (a monoculture); and pooled independent assessments, in which every firm uses the average of all firms’ estimates. Mean basins were similar (0.37, 0.42 and 0.36), but the monoculture polarised outcomes (Fig. 6). The probability of collective failure (basin below 0.1) was 0.23 under the monoculture against 0.08 with private assessments and 0.09 with pooling; the standard deviation of the basin rose from 0.28 to 0.38. A shared error moves all firms in the same direction, so their collective fate depends on the provider’s error rather than on the ecosystem. This is the corporate counterpart of algorithmic monoculture (12) and of the homogenisation risk that Burton et al. attribute to reliance on a few LLMs (6). Pooling independent assessments retains the benefits of a shared signal without the correlated tail.'));

kids.push(...figure('figs/fig6_information.png', 2220, 750,
  '**Fig. 6. Information architecture at μ = 0.5.** (A) Distribution of the basin of collective investment across 300 draws of assessment error under three architectures; dotted line, error-free basin (0.34). (B) Mean basin and probability of collective failure (basin < 0.1).'));

// ---------------- Discussion
kids.push(H1('Discussion: policy avenues for private entities'));
kids.push(P('The results reframe the case for private nature finance. Risk pricing works where it was designed to work, among firms that bear the consequences of their own pressure, and fails where nature finance is most needed, in landscapes where pressure and exposure sit on different balance sheets. The policy question is therefore not how to price nature more accurately but how to move heterogeneous firms into the basin of a cooperative equilibrium that already exists. Table 1 summarises the avenues the model supports, in order of leverage.'));

kids.push(table(['Policy avenue', 'Model lever and finding', 'Private-sector instruments', 'Who controls it'], [
  ['1. Materiality bridging', 'κ: restores the cooperative basin up to μ ≈ 0.8; impactors move first', 'Supply contracts conditioned on basin outcomes; sustainability-linked loans with landscape KPIs; basin insurance pools; double-materiality reporting kept in scope for structural impactors', 'Buyers, lenders, insurers; reporting standard-setters'],
  ['2. Risk loading as a complement', 'λ: insufficient alone under mismatch; strong complementarity with κ', 'Supervisory add-ons for concentrated basin exposure; tail-risk weighting in credit models', 'Prudential supervisors; credit-risk teams'],
  ['3. Strong, early, time-limited scaffolds', 'Hysteresis: strong scaffolds persist after withdrawal; gradual phase-ins fail from entrenchment', 'Front-loaded blended finance with sunset clauses; time-bound buyer coalitions; transition-plan milestones', 'DFIs, philanthropy, buyer coalitions'],
  ['4. Payments with reciprocal obligations', 'τ: transfers displace payers’ own action; erode bridging', 'Water funds and PES paying for basin outcomes, with contributor commitments retained', 'Water-fund boards, utilities, PES designers'],
  ['5. Diverse, pooled information', 'Monoculture nearly triples collective-failure risk; pooling avoids it', 'Shared nature-data utilities; multiple-provider requirements; disclosure of model provenance for AI-based nature tools', 'Data infrastructures, supervisors, standard-setters'],
  ['6. Patient capital', 'γ: helps only at low mismatch', 'Stewardship codes; long-horizon incentive plans', 'Asset owners, boards'],
  ['7. Early-warning monitoring', 'Critical slowing down near lever thresholds', 'Tracking volatility in firms’ engagement to time interventions', 'Basin initiatives, supervisors'],
], [1700, 2350, 3176, 1800]));
kids.push(new Paragraph({ children: runs('**Table 1. Policy avenues ranked by model leverage.** Instruments are illustrative mappings to be tested empirically (Box 1, questions 9 and 16 to 18).', { size: 17, color: GREY }), spacing: { before: 80, after: 220 } }));

kids.push(P('Three implications follow for policy aimed at private entities. First, instruments should be judged by whether they connect firms’ fortunes, not by whether they measure exposure more precisely. Supply contracts, loan covenants and insurance pools already link firms across basins; the opportunity is to condition them on basin-level ecological outcomes. Disclosure scope matters here, because double-materiality reporting is one of the few channels through which an impactor’s pressure becomes visible to the firms and lenders exposed to it; descoping that removes structural impactors would remove the bridge. Second, collective-action programmes should be designed as scaffolds: strong enough to carry firms past the separatrix, early enough to precede entrenchment, and time-limited, since persistence after withdrawal is the expected outcome rather than a hoped-for one. Third, the information infrastructure for nature finance is itself a lever. As AI tools for nature-risk assessment spread, regulators should treat provider concentration as a source of correlated error and favour pooled, independent assessment.'));
kids.push(P('The analysis has limits that define the next steps (Box 1). Learning is deterministic and noise-free; exploration noise is known to promote cooperation (13) and is likely to soften the entrenchment results. Firms choose a binary action in a two-state ecosystem; continuous investment, more states and spatially coupled sub-basins are needed for calibration to real catchments. Payment designs other than the one modelled may avoid crowding out. Most importantly, the predictions require empirical testing against corporate decisions, for which Table 2 specifies the pipeline.'));

// ---------------- Pipeline table
kids.push(H1('Empirical pipeline'));
kids.push(P('Each model quantity has an observable counterpart. The pipeline proceeds basin by basin, so that every prediction is falsifiable at the level where collective action happens.'));
kids.push(table(['Step', 'Model quantity', 'Data sources', 'Operationalisation and test'], [
  ['1. Basins and firms', 'Set of firms N per basin', 'HydroBASINS levels 6 to 8; asset-location data (Spatial Finance Initiative GeoAsset, Orbis, Bloomberg); CDP site-level water data', 'Firm-by-basin exposure matrix; basins with at least two material firms'],
  ['2. Impact shares', 'h_i', 'ENCORE pressure ratings × production; CDP Water withdrawals; deforestation and land-conversion layers', 'Pressure share of each firm within the basin, normalised to sum to one'],
  ['3. Dependency', 'd_i, m', 'ENCORE dependency ratings; segment and site revenue (Compustat segments via WRDS, Bloomberg); water-intensity data', 'Loss given degradation as share of revenue at risk in the basin'],
  ['4. Mismatch', 'μ', 'Steps 2 and 3', 'μ = ½Σ|h_i − d_i/Σd|; global map and sector decomposition (Box 1, questions 1 to 3)'],
  ['5. Costs and co-benefits', 'c, f', 'Corporate nature capex in sustainability and CDP reports; water-fund cost data', 'Cost of investment relative to basin revenue'],
  ['6. Ecosystem dynamics', 'q_c, q_r', 'Hydro-ecological models (for example LPJmL), regime-shift databases, recovery-time literature', 'Collapse and recovery hazards per basin'],
  ['7. Horizon', 'γ', 'Institutional ownership horizon (13F data on WRDS; investor turnover classifications)', 'Ownership-weighted investor horizon per firm'],
  ['8. Levers', 'λ, κ, τ, ρ', 'Timing of disclosure mandates and TNFD adoption (λ); supply-chain links (FactSet Revere, Bloomberg SPLC) and sustainability-linked loans (LSEG Deals) (κ); water-fund membership (τ); provider concentration in nature metrics (ρ)', 'Lever intensity per firm-basin-year'],
  ['9. Dynamics', 'Basin, thresholds, convergence time', 'Calibrated CRLD; stochastic MARL and LLM-agent checks', 'Predicted probability of collective investment per basin'],
  ['10. Test', 'Observed co-investment', 'Water Action Hub projects, water-fund and landscape-programme membership, CDP engagement responses', 'Out-of-sample prediction of participation and timing; staggered difference-in-differences around mandates; persistence after programme end'],
], [1500, 1400, 3226, 2900]));
kids.push(new Paragraph({ children: runs('**Table 2. From corporate data to falsifiable predictions.**', { size: 17, color: GREY }), spacing: { before: 80, after: 220 } }));

kids.push(table(['Collective-intelligence concept (source)', 'Corporate interpretation', 'Model element'], [
  ['Basin of attraction as a measure of collective cooperative intelligence (5)', 'Likelihood that a basin’s firms converge on co-investment', 'Outcome metric'],
  ['Other-regarding rewards promote cooperation (5, 9)', 'Contracts and finance that transmit others’ losses', 'κ'],
  ['Caring for the future (7)', 'Investor and managerial horizon', 'γ'],
  ['Critical transitions and slowing down (5, 11)', 'Instrument intensity thresholds; early warning', 'Fig. 3'],
  ['Hysteresis as collective memory (5)', 'Persistence of collective action after programmes end', 'Fig. 5'],
  ['Functional diversity and independence (6, 14)', 'Independent nature-risk assessments', 'Private architecture'],
  ['Aggregation mechanisms (6)', 'Pooled assessment infrastructure', 'Pooled architecture'],
  ['Homogenisation by shared LLMs; illusion of consensus (6, 12)', 'Single dominant rating, dataset or AI provider', 'Monoculture architecture'],
], [3600, 3426, 2000]));
kids.push(new Paragraph({ children: runs('**Table 3. Mapping collective-intelligence concepts to corporate nature finance.**', { size: 17, color: GREY }), spacing: { before: 80, after: 220 } }));

// ---------------- Methods
kids.push(H1('Materials and Methods'));
kids.push(H3('Learning dynamics'));
kids.push(P('Firm i’s strategy X_i(s,a) is its probability of action a in state s. Each step applies X_i(s,a) ← X_i(s,a)·exp[α·δ_i(s,a)]/Z_i(s), where δ_i(s,a) = R_i(s,a) + γ·Σ_s′ T_i(s,a,s′)V_i(s′) − V_i(s) is the temporal-difference error averaged over other firms’ strategies and environmental transitions, V_i solves the Bellman equation under the joint strategy, and Z_i normalises (5, 8). We integrate it with pyCRLD’s strategy actor-critic agent (stratAC, without prefactor; commit a370d51), vectorised over initial conditions with jax.vmap, in double precision. The library’s default single precision freezes slow strategy drift near basin boundaries: after withdrawal of a weak scaffold (κ = 0.5) it reports prosperity of 0.38 instead of 0.32. Trajectories run to a tolerance of 10⁻⁵ in maximum strategy change or a step cap (8,000 to 15,000). Basin size is the share of 64 uniformly drawn initial strategies converging to joint investment (all firms’ investment probability above 0.5). Ecosystem prosperity is the stationary probability of the prosperous state under the converged joint strategy.'));
kids.push(H3('Correction to pyCRLD for three or more agents'));
kids.push(P('pyCRLD’s summation over the other agents’ actions (_OtherAgentsActionsSummationTensor) admits every ordering of the other agents. For N ≥ 3 it therefore sums strategy-averaged rewards, transitions and next-state values (N − 1)! times: rows of the action-conditioned transition matrix sum to 2 (N = 3), 6 (N = 4) or 24 (N = 5) instead of 1, and permuted terms pair agents with one another’s strategies. We restricted the tensor to the single canonical ordering. The correction leaves two-agent dynamics unchanged (difference exactly zero), so the results of ref. 5, all for N = 2, are unaffected. With it, pyCRLD and the independent NumPy implementation agree to 10⁻¹⁵ per step and 10⁻¹³ after 2,000 steps in double precision, and every reported quantity is the same in both (table S1). The uncorrected library would have reversed the conclusions: for the four-firm game it predicts ecosystem prosperity of 0.89 to 1.00 in all seven headline configurations tested, including accurate pricing at μ = 0.8, where the corrected value is 0.01.'));
kids.push(H3('Experiments'));
kids.push(P('Lever atlases use 10 values of μ and 8 to 15 values per lever. Threshold analyses use 36 to 41 values at μ = 0.8. Policy-mix maps use 11 × 11 grids. Scaffold runs apply the lever for 1,000 steps then switch. Entrenchment runs draw initial investment probabilities uniformly within ±50% of a target mean. The slow-ramp run raises κ in 51 increments of 600 steps from a 5% start. Information-architecture runs draw ε_i ~ N(0,1) independently (private), as one common draw (monoculture), or as the mean of four independent draws applied to all firms (pooled), with perceived dependency d_i·exp(σε_i − σ²/2).'));
kids.push(H3('Validation, code and data availability'));
kids.push(P('The unmodified pyCRLD library reproduces the two-firm discount-factor thresholds and critical slowing down of ref. 5 (fig. S1). Code is provided with this draft: pycrld_bridge.py (pyCRLD environment, corrected agent and batched runner), crld.py (independent NumPy implementation), experiment scripts that run on either engine, check_equivalence.py and compare_numbers.py. The original materials of ref. 5 are at github.com/wbarfuss/collective-cooperative-intelligence (Zenodo 10.5281/zenodo.11611242) and build on pyCRLD (github.com/barfusslab/pyCRLD). Ref. 6 is a Perspective without a code repository.'));

// ---------------- References
kids.push(H1('References'));
const refs = [
  'World Economic Forum, *Nature Risk Rising: Why the Crisis Engulfing Nature Matters for Business and the Economy* (2020).',
  'P. Dasgupta, *The Economics of Biodiversity: The Dasgupta Review* (HM Treasury, 2021).',
  'Network for Greening the Financial System, *Nature-related Financial Risks: A Conceptual Framework to Guide Action by Central Banks and Supervisors* (2023).',
  'Taskforce on Nature-related Financial Disclosures, *Recommendations of the Taskforce on Nature-related Financial Disclosures* (2023).',
  'W. Barfuss et al., Collective cooperative intelligence. *Proc. Natl. Acad. Sci. U.S.A.* **122**, e2319948121 (2025).',
  'J. W. Burton et al., How large language models can reshape collective intelligence. *Nat. Hum. Behav.* **8**, 1643–1655 (2024).',
  'W. Barfuss, J. F. Donges, V. V. Vasconcelos, J. Kurths, S. A. Levin, Caring for the future can turn tragedy into comedy for long-term collective action under risk of collapse. *Proc. Natl. Acad. Sci. U.S.A.* **117**, 12915–12922 (2020).',
  'W. Barfuss, J. F. Donges, J. Kurths, Deterministic limit of temporal difference reinforcement learning for stochastic games. *Phys. Rev. E* **99**, 043305 (2019).',
  'E. Hughes et al., Inequity aversion improves cooperation in intertemporal social dilemmas. *Adv. Neural Inf. Process. Syst.* **31** (2018).',
  'L. L. Bremer et al., One size does not fit all: Natural infrastructure investments within the Latin American Water Funds Partnership. *Ecosyst. Serv.* **17**, 217–236 (2016).',
  'M. Scheffer et al., Early-warning signals for critical transitions. *Nature* **461**, 53–59 (2009).',
  'J. Kleinberg, M. Raghavan, Algorithmic monoculture and social welfare. *Proc. Natl. Acad. Sci. U.S.A.* **118**, e2018340118 (2021).',
  'W. Barfuss, J. M. Meylahn, Intrinsic fluctuations of reinforcement learning promote cooperation. *Sci. Rep.* **13**, 1309 (2023).',
  'L. Hong, S. E. Page, Groups of diverse problem solvers can outperform groups of high-ability problem solvers. *Proc. Natl. Acad. Sci. U.S.A.* **101**, 16385–16389 (2004).',
  'F. Berg, J. F. Kölbel, R. Rigobon, Aggregate confusion: The divergence of ESG ratings. *Rev. Finance* **26**, 1315–1344 (2022).',
  'E. Ostrom, *Governing the Commons: The Evolution of Institutions for Collective Action* (Cambridge Univ. Press, 1990).',
];
refs.forEach((r, i) => kids.push(new Paragraph({ children: [new TextRun({ text: `${i + 1}. `, size: 18 }), ...runs(r, { size: 18 })], spacing: { after: 60, line: 252 }, indent: { left: 360, hanging: 360 } })));

kids.push(H1('Supplementary figure'));
kids.push(...figure('figs/figS1_replication_pycrld.png', 2220, 720,
  '**Fig. S1. Replication of the benchmark in ref. 5 with the unmodified pyCRLD library.** Two symmetric firms (q_c = 0.2, learning rate 0.1, 250 random initial strategies). (A) Share converging to mutual cooperation against the discount factor: 0.07 at γ = 0.70 and 0.91 at γ = 0.85, with the bistable window shaded. (B) Median steps to convergence peak at γ = 0.75 (174 steps), reproducing critical slowing down.'));
const cmp = JSON.parse(fs.readFileSync('compare.json', 'utf8'));
const fmt = (k, v) => (Math.abs(v) >= 10 ? v.toFixed(0) : v.toFixed(2));
kids.push(table(['Quantity reported in the text', 'NumPy implementation', 'pyCRLD (corrected)'],
  cmp.map(([k, a, b]) => [k, fmt(k, a), fmt(k, b)]), [5226, 1900, 1900]));
kids.push(new Paragraph({ children: runs('**Table S1. Every quantity reported in the text, computed with both engines.** Both runs use double precision, the same seeds and the same scripts; convergence times are medians in learning steps. λ*, κ*: lever intensity at which prosperity or basin first reaches 0.5 (linear interpolation).', { size: 17, color: GREY }), spacing: { before: 80, after: 220 } }));
// ---------------- Pre-submission notes
kids.push(new Paragraph({ children: [new PageBreak()] }));
kids.push(boxed('Pre-submission notes (remove before submission)', [
  B('**Empirical component.** Science will expect at least one empirical test. The strongest candidate is Box 1, question 16: whether modelled basin size, computed from steps 1 to 8 of Table 2, predicts which basins host corporate co-investment, out of sample.'),
  B('**pyCRLD correction.** Report the N ≥ 3 summation-tensor issue to the maintainers before submission (draft issue with reproduction and patch in the code bundle, pyCRLD_issue_draft.md), and cite the fixed release or commit once available. Consider contacting W. Barfuss directly, since the fix supports this paper’s Methods.'),
  B('**Robustness.** Extend beyond the deterministic limit: stochastic MARL agents, exploration noise, N = 6 to 12 firms (now possible with the corrected tensor, though Omega has N^N·M^(2N) entries, 1.9 × 10⁸ for N = 6, so it needs a sparse rewrite beyond N ≈ 5) and partial observability. The entrenchment and transfer results are the most likely to be sensitive.'),
  B('**Parameters.** q_c = 0.8 was chosen so that the symmetric four-firm benchmark sustains cooperation; at the paper’s q_c = 0.2 four firms never cooperate for γ ≤ 0.95 (the large-collective problem of ref. 5). Report this as a sensitivity result, not only a calibration choice.'),
  B('**Payment design (now tested; analysis notebook §5).** Crowding-out is specific to the buy-out design: a budget-balanced levy-and-rebate that rewards any investing firm reaches full prosperity from τ = 0.5 at μ = 0.8 (buy-out: 0.11). Add this to the Results and Table 1, avenue 4.'),
  B('**Group size and fragility (notebook §6).** The failure of accurate pricing at μ = 0.9 holds for N = 2, 4, 6 and q_c = 0.4 to 1.0, but thresholds move a lot: two firms tolerate μ up to 0.75, six firms fail even at μ = 0, and q_c = 0.4 removes cooperation entirely. State that four-firm thresholds are illustrative and must be calibrated per basin.'),
  B('**Sampled learners (notebook §7).** With slow strategy updates, sampled actor-critic learners approach the deterministic results (0.87–0.89 vs 1.00 under strong bridging, still rising); with fast updates, dependents often lock into non-investment (0.45). Report the time-scale condition, and note that reducing feedback noise (monitoring, multi-year KPIs) is itself a lever.'),
  B('**Targeting (notebook §3–4).** In an illustrative five-firm basin, bridging one or two firms is not enough; covering all upstream impactors is. Invasion and defection margins give a simulation-free diagnostic that reproduces the simulated thresholds and could appear in the Discussion as a supervisory tool.'),
  B('**Supplementary figure S1** (two-firm replication of ref. 5) is appended after the references; move it to the Supplementary Materials.'),
  B('**References** were compiled from the two source papers and standard literature; check every entry against the published record before submission.'),
  B('**Format.** Science Research Articles are limited to about 4,500 words and six figures; this draft is within the figure limit, and Tables 2 and 3 can move to the Supplementary Materials if needed.'),
]));

// ---------- document ----------
const doc = new Document({
  creator: 'Maganizo Kruger Nyasulu',
  title: 'Materiality mismatch traps private nature finance',
  styles: {
    default: { document: { run: { font: FONT, size: 20 } } },
    paragraphStyles: [
      { id: 'Title', name: 'Title', basedOn: 'Normal', run: { font: FONT, size: 38, bold: true, color: NAVY }, paragraph: { spacing: { before: 0, after: 200 }, keepNext: true } },
      { id: 'Heading1', name: 'Heading 1', basedOn: 'Normal', next: 'Normal', quickFormat: true, run: { font: FONT, size: 28, bold: true, color: NAVY }, paragraph: { spacing: { before: 320, after: 140 }, keepNext: true, outlineLevel: 0 } },
      { id: 'Heading2', name: 'Heading 2', basedOn: 'Normal', next: 'Normal', quickFormat: true, run: { font: FONT, size: 23, bold: true, color: ACCENT }, paragraph: { spacing: { before: 220, after: 100 }, keepNext: true, outlineLevel: 1 } },
      { id: 'Heading3', name: 'Heading 3', basedOn: 'Normal', next: 'Normal', quickFormat: true, run: { font: FONT, size: 21, bold: true, color: GREY }, paragraph: { spacing: { before: 160, after: 80 }, keepNext: true, outlineLevel: 2 } },
    ],
  },
  numbering: { config: [
    { reference: 'bullets', levels: [{ level: 0, format: LevelFormat.BULLET, text: '•', alignment: AlignmentType.LEFT, style: { paragraph: { indent: { left: 540, hanging: 270 } } } }] },
    { reference: 'qs', levels: [{ level: 0, format: LevelFormat.DECIMAL, text: '%1.', alignment: AlignmentType.LEFT, style: { paragraph: { indent: { left: 540, hanging: 360 } } } }] },
  ] },
  sections: [{
    properties: { page: { size: { width: 11906, height: 16838 }, margin: { top: 1440, bottom: 1440, left: 1440, right: 1440 } } },
    footers: { default: new Footer({ children: [new Paragraph({ alignment: AlignmentType.CENTER, children: [
      new TextRun({ text: 'Nyasulu · Materiality mismatch and collective nature finance · draft · ', size: 16, color: GREY }),
      new SimpleField('PAGE')] })] }) },
    children: kids,
  }],
});
Packer.toBuffer(doc).then(b => { fs.writeFileSync('/home/claude/Materiality_Mismatch_Collective_Nature_Finance_Science_draft.docx', b); console.log('written'); });
