"""
LiquidBiopsyEngine: Cell-Free DNA / ctDNA Analysis Pipeline
- Fragment length analysis (nucleosome positioning signal)
- Tumor fraction estimation (copy number-based)
- Low-VAF somatic variant calling
- Methylation-based tissue-of-origin deconvolution
- Longitudinal tumor burden tracking
"""
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from scipy import stats
from scipy.signal import savgol_filter
import warnings
warnings.filterwarnings('ignore')

np.random.seed(42)

print("="*60)
print("LiquidBiopsyEngine v1.0")
print("Cell-Free DNA / ctDNA Analysis Pipeline")
print("="*60)

# ─── 1. SYNTHETIC cfDNA DATA ─────────────────────────────────
N_PATIENTS = 40
N_TIMEPOINTS = 5
N_BINS = 500       # genomic bins (1Mb each)
N_VARIANTS = 2000  # candidate somatic variants

print(f"\n[Data] {N_PATIENTS} patients, {N_TIMEPOINTS} time points")
print(f"  {N_BINS} genomic bins (1Mb), {N_VARIANTS} candidate variants")

# Patient metadata
cancer_types = np.random.choice(['LUAD', 'BRCA', 'CRC', 'PRAD', 'HCC'], N_PATIENTS)
true_tumor_fractions = np.random.beta(2, 8, N_PATIENTS)  # mostly low TF (0.01-0.3)
true_tumor_fractions = true_tumor_fractions.clip(0.005, 0.5)

# ─── 2. FRAGMENT LENGTH ANALYSIS ─────────────────────────────
print("\n[Fragment] Analyzing cfDNA fragment length distribution...")

def simulate_fragment_lengths(n_frags, tumor_fraction, seed_offset=0):
    """
    Simulate cfDNA fragment lengths.
    Normal cfDNA: ~167bp (nucleosome-protected) + ~10bp linker
    ctDNA: shorter fragments (~145bp) due to different nucleosome positioning
    """
    np.random.seed(42 + seed_offset)
    n_tumor = int(n_frags * tumor_fraction)
    n_normal = n_frags - n_tumor

    # Normal cfDNA: bimodal (mono + di-nucleosomal)
    mono = np.random.normal(167, 15, int(n_normal * 0.75)).astype(int)
    di = np.random.normal(334, 20, int(n_normal * 0.25)).astype(int)
    normal_frags = np.concatenate([mono, di])

    # ctDNA: shorter, less nucleosomal periodicity
    tumor_frags = np.random.normal(145, 20, n_tumor).astype(int)

    all_frags = np.concatenate([normal_frags, tumor_frags])
    return np.clip(all_frags, 50, 600)

# Compute fragment length profiles for all patients
frag_profiles = []
for i in range(N_PATIENTS):
    frags = simulate_fragment_lengths(10000, true_tumor_fractions[i], seed_offset=i)
    # Histogram: 50-400bp
    hist, bins = np.histogram(frags, bins=np.arange(50, 401, 5))
    frag_profiles.append(hist / hist.sum())

frag_profiles = np.array(frag_profiles)
bin_centers = np.arange(52.5, 400, 5)

# Fragment length ratio: short/long (tumor enriched in short)
short_mask = (bin_centers >= 100) & (bin_centers <= 150)
long_mask = (bin_centers >= 160) & (bin_centers <= 180)
frag_ratio = frag_profiles[:, short_mask].sum(axis=1) / (frag_profiles[:, long_mask].sum(axis=1) + 1e-6)

r_frag_tf, p_frag_tf = stats.pearsonr(frag_ratio, true_tumor_fractions)
print(f"  Fragment ratio (short/long) vs tumor fraction: r={r_frag_tf:.3f}")
print(f"  Mean fragment ratio: {frag_ratio.mean():.3f} ± {frag_ratio.std():.3f}")

# ─── 3. TUMOR FRACTION ESTIMATION (ichorCNA-style) ───────────
print("\n[TumorFraction] Estimating tumor fraction from copy number...")

def estimate_tumor_fraction_cn(true_tf, n_bins=N_BINS):
    """
    Simulate copy number-based tumor fraction estimation.
    Uses read depth variation across genomic bins.
    """
    # True copy number alterations (CNAs)
    n_cna = np.random.poisson(lam=20)  # ~20 CNA segments
    cna_bins = np.random.choice(n_bins, n_cna, replace=False)
    cna_states = np.random.choice([-1, 1, 2], n_cna, p=[0.3, 0.4, 0.3])  # loss/gain/amp

    # Expected read depth
    expected_depth = np.ones(n_bins) * 100
    for bin_idx, state in zip(cna_bins, cna_states):
        # Observed depth = (1-tf)*2 + tf*(2+state) normalized
        cn_tumor = 2 + state
        cn_normal = 2
        expected_depth[bin_idx] = 100 * ((1 - true_tf) * cn_normal + true_tf * cn_tumor) / 2

    # Add noise
    observed_depth = np.random.poisson(expected_depth)

    # Estimate TF from variance of log2 ratios
    log2_ratio = np.log2(observed_depth / 100 + 0.01)
    # TF estimate: scaled MAD of log2 ratios
    mad = np.median(np.abs(log2_ratio - np.median(log2_ratio)))
    tf_estimate = np.clip(mad * 3.5, 0.001, 0.99)
    return tf_estimate, log2_ratio

estimated_tfs = []
cn_profiles = []
for i in range(N_PATIENTS):
    tf_est, cn_prof = estimate_tumor_fraction_cn(true_tumor_fractions[i])
    estimated_tfs.append(tf_est)
    cn_profiles.append(cn_prof)

estimated_tfs = np.array(estimated_tfs)
cn_profiles = np.array(cn_profiles)

r_tf, p_tf = stats.pearsonr(true_tumor_fractions, estimated_tfs)
rmse_tf = np.sqrt(np.mean((true_tumor_fractions - estimated_tfs)**2))
print(f"  TF estimation: r={r_tf:.3f}, RMSE={rmse_tf:.4f}")
print(f"  True TF range: {true_tumor_fractions.min():.3f} - {true_tumor_fractions.max():.3f}")
print(f"  Estimated TF range: {estimated_tfs.min():.3f} - {estimated_tfs.max():.3f}")

# ─── 4. LOW-VAF SOMATIC VARIANT CALLING ──────────────────────
print("\n[Variants] Calling low-VAF somatic variants...")

# Simulate variant calling
N_TRUE_SOMATIC = 150
true_somatic_idx = np.random.choice(N_VARIANTS, N_TRUE_SOMATIC, replace=False)
true_somatic = np.zeros(N_VARIANTS, dtype=bool)
true_somatic[true_somatic_idx] = True

# VAF distribution: mostly low (ctDNA)
true_vafs = np.zeros(N_VARIANTS)
true_vafs[true_somatic] = np.random.beta(1.5, 10, N_TRUE_SOMATIC) * 0.3  # 0-30% VAF
true_vafs[~true_somatic] = 0

# Simulate read counts
total_depth = np.random.poisson(1000, N_VARIANTS)  # deep sequencing
alt_counts = np.random.binomial(total_depth, np.clip(true_vafs + np.random.normal(0, 0.002, N_VARIANTS), 0, 1))
observed_vafs = alt_counts / (total_depth + 1)

# Variant calling: binomial test against error rate
error_rate = 0.003
pvals_var = np.array([stats.binomtest(alt_counts[i], total_depth[i], error_rate, alternative='greater').pvalue
                      if alt_counts[i] > 0 else 1.0
                      for i in range(N_VARIANTS)])

# BH FDR
n = len(pvals_var)
sorted_idx = np.argsort(pvals_var)
fdr_var = np.zeros(n)
for rank, idx in enumerate(sorted_idx):
    fdr_var[idx] = min(1.0, pvals_var[idx] * n / (rank + 1))
for i in range(len(sorted_idx)-2, -1, -1):
    fdr_var[sorted_idx[i]] = min(fdr_var[sorted_idx[i]], fdr_var[sorted_idx[i+1]])

called_variants = (fdr_var < 0.01) & (observed_vafs >= 0.005)
tp_var = (called_variants & true_somatic).sum()
fp_var = (called_variants & ~true_somatic).sum()
fn_var = (~called_variants & true_somatic).sum()
prec_var = tp_var / max(tp_var + fp_var, 1)
recall_var = tp_var / max(tp_var + fn_var, 1)
f1_var = 2 * prec_var * recall_var / max(prec_var + recall_var, 1e-6)

print(f"  Called variants: {called_variants.sum()} (TP={tp_var}, FP={fp_var})")
print(f"  Precision={prec_var:.3f}, Recall={recall_var:.3f}, F1={f1_var:.3f}")
print(f"  Median VAF of called somatic: {observed_vafs[called_variants & true_somatic].mean():.4f}")

# ─── 5. TISSUE-OF-ORIGIN DECONVOLUTION ───────────────────────
print("\n[Deconvolution] Methylation-based tissue-of-origin...")

# Reference methylation profiles for 5 cancer types
CANCER_REFS = ['LUAD', 'BRCA', 'CRC', 'PRAD', 'HCC']
N_CpG_MARKERS = 200  # tissue-specific CpG markers

# Reference methylation matrix (markers x cancer types)
ref_methylation = np.random.dirichlet(np.ones(5) * 0.5, N_CpG_MARKERS).T  # 5 x 200

# Patient cfDNA methylation (mixture of tumor + normal)
patient_methylation = np.zeros((N_PATIENTS, N_CpG_MARKERS))
for i in range(N_PATIENTS):
    ct_idx = CANCER_REFS.index(cancer_types[i])
    tf = true_tumor_fractions[i]
    # Mixture: tf * tumor_profile + (1-tf) * normal_profile
    normal_profile = np.random.beta(2, 8, N_CpG_MARKERS)
    patient_methylation[i] = tf * ref_methylation[ct_idx] + (1-tf) * normal_profile
    patient_methylation[i] += np.random.normal(0, 0.02, N_CpG_MARKERS)
    patient_methylation[i] = patient_methylation[i].clip(0, 1)

# Deconvolution: NNLS
from scipy.optimize import nnls
predicted_types = []
for i in range(N_PATIENTS):
    # Solve: ref_methylation.T @ w ≈ patient_methylation[i]
    w, _ = nnls(ref_methylation.T, patient_methylation[i])
    w = w / (w.sum() + 1e-6)
    predicted_types.append(CANCER_REFS[np.argmax(w)])

accuracy = np.mean([predicted_types[i] == cancer_types[i] for i in range(N_PATIENTS)])
print(f"  Tissue-of-origin accuracy: {accuracy:.3f} ({int(accuracy*N_PATIENTS)}/{N_PATIENTS})")

# ─── 6. LONGITUDINAL TUMOR BURDEN TRACKING ───────────────────
print("\n[Longitudinal] Tracking tumor burden over time...")

# Simulate 5 patients with longitudinal data
N_LONG = 5
timepoints = np.arange(N_TIMEPOINTS)  # months 0,3,6,9,12

# Treatment response patterns
patterns = ['responder', 'partial', 'progressor', 'stable', 'mixed']
longitudinal_tfs = np.zeros((N_LONG, N_TIMEPOINTS))
for i, pattern in enumerate(patterns):
    base_tf = np.random.uniform(0.05, 0.25)
    if pattern == 'responder':
        longitudinal_tfs[i] = base_tf * np.exp(-0.5 * timepoints)
    elif pattern == 'partial':
        longitudinal_tfs[i] = base_tf * (0.5 + 0.5 * np.exp(-0.3 * timepoints))
    elif pattern == 'progressor':
        longitudinal_tfs[i] = base_tf * np.exp(0.3 * timepoints)
    elif pattern == 'stable':
        longitudinal_tfs[i] = base_tf * np.ones(N_TIMEPOINTS) + np.random.normal(0, 0.01, N_TIMEPOINTS)
    else:  # mixed
        longitudinal_tfs[i] = base_tf * (1 + 0.3*np.sin(timepoints))
    longitudinal_tfs[i] = longitudinal_tfs[i].clip(0.001, 0.8)
    longitudinal_tfs[i] += np.random.normal(0, 0.005, N_TIMEPOINTS)

print(f"  Longitudinal patterns: {patterns}")
for i, pattern in enumerate(patterns):
    print(f"    {pattern}: TF {longitudinal_tfs[i,0]:.3f} → {longitudinal_tfs[i,-1]:.3f}")

# ─── 7. VISUALIZATION ────────────────────────────────────────
print("\n[Viz] Generating dashboard...")

fig = plt.figure(figsize=(18, 14))
fig.patch.set_facecolor('#0a0a0a')
gs_main = gridspec.GridSpec(3, 3, figure=fig, hspace=0.45, wspace=0.4)

# Panel 1: Fragment length profiles
ax1 = fig.add_subplot(gs_main[0, 0])
ax1.set_facecolor('#111111')
high_tf_idx = np.argmax(true_tumor_fractions)
low_tf_idx = np.argmin(true_tumor_fractions)
ax1.plot(bin_centers, frag_profiles[high_tf_idx], color='#FF5722', linewidth=1.5,
         label=f'High TF ({true_tumor_fractions[high_tf_idx]:.2f})')
ax1.plot(bin_centers, frag_profiles[low_tf_idx], color='#2196F3', linewidth=1.5,
         label=f'Low TF ({true_tumor_fractions[low_tf_idx]:.3f})')
ax1.axvspan(100, 150, alpha=0.15, color='#FF5722', label='Short (ctDNA)')
ax1.axvspan(160, 180, alpha=0.15, color='#2196F3', label='Mono-nuc')
ax1.set_xlabel('Fragment Length (bp)', color='white', fontsize=9)
ax1.set_ylabel('Frequency', color='white', fontsize=9)
ax1.set_title('cfDNA Fragment Length', color='white', fontsize=10, fontweight='bold')
ax1.tick_params(colors='white', labelsize=7)
for spine in ax1.spines.values(): spine.set_color('#333333')
ax1.legend(fontsize=6, facecolor='#222222', labelcolor='white')

# Panel 2: TF estimation
ax2 = fig.add_subplot(gs_main[0, 1])
ax2.set_facecolor('#111111')
ax2.scatter(true_tumor_fractions, estimated_tfs, c='#E9ED4C', s=30, alpha=0.8)
lim = max(true_tumor_fractions.max(), estimated_tfs.max()) * 1.1
ax2.plot([0, lim], [0, lim], 'w--', alpha=0.4, linewidth=1)
ax2.set_xlabel('True Tumor Fraction', color='white', fontsize=9)
ax2.set_ylabel('Estimated Tumor Fraction', color='white', fontsize=9)
ax2.set_title(f'TF Estimation\n(r={r_tf:.3f}, RMSE={rmse_tf:.4f})', color='white', fontsize=10, fontweight='bold')
ax2.tick_params(colors='white', labelsize=7)
for spine in ax2.spines.values(): spine.set_color('#333333')

# Panel 3: VAF distribution
ax3 = fig.add_subplot(gs_main[0, 2])
ax3.set_facecolor('#111111')
ax3.hist(observed_vafs[called_variants & true_somatic], bins=20, color='#4CAF50', alpha=0.8, label=f'TP ({tp_var})')
ax3.hist(observed_vafs[called_variants & ~true_somatic], bins=10, color='#FF5722', alpha=0.8, label=f'FP ({fp_var})')
ax3.set_xlabel('Variant Allele Frequency', color='white', fontsize=9)
ax3.set_ylabel('Count', color='white', fontsize=9)
ax3.set_title(f'Low-VAF Variant Calling\nF1={f1_var:.3f}', color='white', fontsize=10, fontweight='bold')
ax3.tick_params(colors='white', labelsize=7)
for spine in ax3.spines.values(): spine.set_color('#333333')
ax3.legend(fontsize=7, facecolor='#222222', labelcolor='white')

# Panel 4: Copy number profile
ax4 = fig.add_subplot(gs_main[1, 0])
ax4.set_facecolor('#111111')
sample_cn = cn_profiles[high_tf_idx]
ax4.plot(range(N_BINS), sample_cn, color='#607D8B', linewidth=0.5, alpha=0.7)
smoothed = savgol_filter(sample_cn, 21, 3)
ax4.plot(range(N_BINS), smoothed, color='#E9ED4C', linewidth=1.5)
ax4.axhline(y=0, color='white', linestyle='--', alpha=0.3, linewidth=0.8)
ax4.set_xlabel('Genomic Bin (1Mb)', color='white', fontsize=9)
ax4.set_ylabel('log2(Depth/Expected)', color='white', fontsize=9)
ax4.set_title(f'Copy Number Profile\n(TF={true_tumor_fractions[high_tf_idx]:.2f})', color='white', fontsize=10, fontweight='bold')
ax4.tick_params(colors='white', labelsize=7)
for spine in ax4.spines.values(): spine.set_color('#333333')

# Panel 5: Tissue-of-origin
ax5 = fig.add_subplot(gs_main[1, 1])
ax5.set_facecolor('#111111')
from collections import Counter
ct_colors_map = {'LUAD': '#2196F3', 'BRCA': '#E91E63', 'CRC': '#4CAF50', 'PRAD': '#FF9800', 'HCC': '#9C27B0'}
correct = [predicted_types[i] == cancer_types[i] for i in range(N_PATIENTS)]
ax5.scatter(range(N_PATIENTS), true_tumor_fractions,
            c=[ct_colors_map[cancer_types[i]] for i in range(N_PATIENTS)],
            marker=np.where(correct, 'o', 'x')[0] if False else 'o',
            s=40, alpha=0.8)
# Mark incorrect predictions
incorrect_idx = [i for i in range(N_PATIENTS) if not correct[i]]
ax5.scatter(incorrect_idx, true_tumor_fractions[incorrect_idx],
            c='red', marker='x', s=80, linewidths=2, zorder=5, label='Misclassified')
from matplotlib.patches import Patch
legend_els = [Patch(facecolor=col, label=ct) for ct, col in ct_colors_map.items()]
ax5.legend(handles=legend_els, fontsize=6, facecolor='#222222', labelcolor='white')
ax5.set_xlabel('Patient', color='white', fontsize=9)
ax5.set_ylabel('Tumor Fraction', color='white', fontsize=9)
ax5.set_title(f'Tissue-of-Origin (acc={accuracy:.2f})', color='white', fontsize=10, fontweight='bold')
ax5.tick_params(colors='white', labelsize=7)
for spine in ax5.spines.values(): spine.set_color('#333333')

# Panel 6: Longitudinal tracking
ax6 = fig.add_subplot(gs_main[1, 2])
ax6.set_facecolor('#111111')
long_colors = ['#4CAF50', '#FF9800', '#FF5722', '#2196F3', '#9C27B0']
month_labels = [0, 3, 6, 9, 12]
for i, (pattern, col) in enumerate(zip(patterns, long_colors)):
    ax6.plot(month_labels, longitudinal_tfs[i], 'o-', color=col, linewidth=2, markersize=5, label=pattern)
ax6.set_xlabel('Time (months)', color='white', fontsize=9)
ax6.set_ylabel('Tumor Fraction (ctDNA)', color='white', fontsize=9)
ax6.set_title('Longitudinal Tumor Burden', color='white', fontsize=10, fontweight='bold')
ax6.tick_params(colors='white', labelsize=7)
for spine in ax6.spines.values(): spine.set_color('#333333')
ax6.legend(fontsize=7, facecolor='#222222', labelcolor='white')

# Panel 7: Fragment ratio vs TF
ax7 = fig.add_subplot(gs_main[2, 0])
ax7.set_facecolor('#111111')
ct_colors_arr = [ct_colors_map[ct] for ct in cancer_types]
ax7.scatter(true_tumor_fractions, frag_ratio, c=ct_colors_arr, s=30, alpha=0.8)
m, b = np.polyfit(true_tumor_fractions, frag_ratio, 1)
x_line = np.linspace(true_tumor_fractions.min(), true_tumor_fractions.max(), 50)
ax7.plot(x_line, m*x_line+b, 'w--', linewidth=1, alpha=0.6)
ax7.set_xlabel('True Tumor Fraction', color='white', fontsize=9)
ax7.set_ylabel('Short/Long Fragment Ratio', color='white', fontsize=9)
ax7.set_title(f'Fragment Ratio vs TF\n(r={r_frag_tf:.3f})', color='white', fontsize=10, fontweight='bold')
ax7.tick_params(colors='white', labelsize=7)
for spine in ax7.spines.values(): spine.set_color('#333333')

# Panel 8: Variant calling ROC
ax8 = fig.add_subplot(gs_main[2, 1])
ax8.set_facecolor('#111111')
thresholds = np.logspace(-4, 0, 50)
tprs, fprs = [], []
for thresh in thresholds:
    called = fdr_var < thresh
    tpr = (called & true_somatic).sum() / max(true_somatic.sum(), 1)
    fpr = (called & ~true_somatic).sum() / max((~true_somatic).sum(), 1)
    tprs.append(tpr); fprs.append(fpr)
ax8.plot(fprs, tprs, color='#E9ED4C', linewidth=2)
ax8.plot([0,1],[0,1],'w--',alpha=0.3,linewidth=0.8)
auc = np.trapz(tprs[::-1], fprs[::-1])
ax8.set_xlabel('False Positive Rate', color='white', fontsize=9)
ax8.set_ylabel('True Positive Rate', color='white', fontsize=9)
ax8.set_title(f'Variant Calling ROC\n(AUC={auc:.3f})', color='white', fontsize=10, fontweight='bold')
ax8.tick_params(colors='white', labelsize=7)
for spine in ax8.spines.values(): spine.set_color('#333333')

# Panel 9: Summary
ax9 = fig.add_subplot(gs_main[2, 2])
ax9.set_facecolor('#111111'); ax9.axis('off')
summary = [
    "LiquidBiopsyEngine v1.0", "",
    f"Patients: {N_PATIENTS}",
    f"Time points: {N_TIMEPOINTS}",
    f"Genomic bins: {N_BINS}", "",
    f"Fragment analysis:",
    f"  Short/long r={r_frag_tf:.3f}", "",
    f"Tumor fraction:",
    f"  r={r_tf:.3f}, RMSE={rmse_tf:.4f}", "",
    f"Variant calling:",
    f"  F1={f1_var:.3f}",
    f"  AUC={auc:.3f}", "",
    f"Tissue-of-origin:",
    f"  Accuracy={accuracy:.3f}",
    f"  ({int(accuracy*N_PATIENTS)}/{N_PATIENTS} correct)",
]
for i, line in enumerate(summary):
    ax9.text(0.05, 0.97-i*0.056, line, transform=ax9.transAxes,
             color='#E9ED4C' if i==0 else 'white', fontsize=8.5, va='top',
             fontweight='bold' if i==0 else 'normal')

fig.suptitle('LiquidBiopsyEngine: ctDNA Analysis Dashboard',
             color='white', fontsize=14, fontweight='bold', y=0.98)
plt.savefig('/workspace/liquid_biopsy_dashboard.png', dpi=150, bbox_inches='tight', facecolor='#0a0a0a')
plt.close()
print("  Dashboard saved.")

print("\n"+"="*60)
print("LiquidBiopsyEngine COMPLETE")
print(f"  Patients: {N_PATIENTS} | Timepoints: {N_TIMEPOINTS}")
print(f"  TF estimation: r={r_tf:.3f}, RMSE={rmse_tf:.4f}")
print(f"  Variant calling: F1={f1_var:.3f}, AUC={auc:.3f}")
print(f"  Tissue-of-origin: accuracy={accuracy:.3f}")
print(f"  Fragment ratio vs TF: r={r_frag_tf:.3f}")
print("="*60)
