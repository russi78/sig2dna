"""
Test suite for signomics module

Case study: classification of synthetic mixtures using multiple distance metrics

Generative Simulation Initiative\olivier.vitrac@gmail.com

"""

import os
from sig2dna_core.signomics import signal_collection, DNAsignal

# %% Output configuration
outputfolder = "./images" if os.path.isdir("./images") else ("../images" if os.path.isdir("../images") else None)

# %% main

# ----------------------------------------------
# Case Study: Classification of Synthetic Mixtures
# ----------------------------------------------

# Generate synthetic signal mixtures
Smix, pSmix, idSmix = signal_collection.generate_mixtures(
    n_mixtures=30,              # Number of distinct mixtures to generate
    max_peaks=16,               # Maximum number of distinct peaks available (used to construct mixtures)
    peaks_per_mixture=(4, 8),   # Random number of peaks per mixture, uniformly sampled in this range (inclusive)
    amplitude_range=(0.5, 2),   # Amplitude multiplier range for each included peak in a mixture
    n_signals=1,                # Number of replicate signals per mixture (e.g., noise variants)
    kinds=("gauss",),           # Allowed peak shapes (Gaussian only in this case)
    width_range=(0.5, 3),       # Range of peak widths when defining the original peak dictionary
    height_range=(1.0, 5.0),    # Range of peak heights (used to scale the original peaks)
    x_range=(0, 500),           # Domain range for the signal (x-axis)
    n_points=2048,              # Number of sampling points in each signal
    normalize=False,            # Do not normalize the resulting signals (preserve original intensities)
    seed=123                    # Random seed for reproducibility
)

# Symbolic transformation
scales = [1, 2, 4, 8, 16, 32]
dnaSmix = Smix._toDNA(scales=scales)

# Detect the pattern YAZB in the first sample at scale 4
# the mask is used by the Jaccard criterion
dnaSmix[0].codesfull[4].extract_motifs("YAZB", minlen=4, plot=True)

# Run distance metrics and dimension diagnostics
results = {}

D = DNAsignal._pairwiseEntropyDistance(dnaSmix, scale=4, engine="bio")
D.name = "Excess Entropy"
Ddhalf, figD = D.dimension_variance_curve()
figD.print("Entropy_dimensions", outputfolder)
results["Excess Entropy"] = Ddhalf

J = DNAsignal._pairwiseJaccardMotifDistance(dnaSmix, scale=4,plot=True)
J.name = "Jaccard"
Jdhalf, figJ = J.dimension_variance_curve()
figJ.print("Jaccard_dimensions", outputfolder)
results["Jaccard"] = Jdhalf

L = DNAsignal._pairwiseLevenshteinDistance(dnaSmix, scale=4)
L.name = "Levenshtein"
Ldhalf, figL = L.dimension_variance_curve()
figL.print("Levenshtein_dimensions", outputfolder)
results["Levenshtein"] = Ldhalf

S = DNAsignal._pairwiseJensenShannonDistance(dnaSmix, scale=4)
S.name = "Jensen-Shannon"
Sdhalf, figS = S.dimension_variance_curve()
figS.print("Jensen_shannon_dimensions", outputfolder)
results["Jensen-Shannon"] = Sdhalf


# Dendrograms and scatter3D for visual clustering
figD_dendro = D.plot_dendrogram()
figD_dendro.print("Entropy_dendrogram", outputfolder)
figD_scatter = D.scatter3d(n_clusters=5)
figD_scatter.print("Entropy_scatter3", outputfolder)

figJ_dendro = J.plot_dendrogram()
figJ_dendro.print("Jaccard_dendrogram", outputfolder)
figJ_scatter = J.scatter3d(n_clusters=5)
figJ_scatter.print("Jaccard_scatter3", outputfolder)

figL_dendro = L.plot_dendrogram()
figL_dendro.print("Levenshtein_dendrogram", outputfolder)
figL_scatter = L.scatter3d(n_clusters=5)
figL_scatter.print("Levenshtein_scatter3", outputfolder)

figS_dendro = S.plot_dendrogram()
figS_dendro.print("Jensen_shannon_dendrogram", outputfolder)
figS_scatter = S.scatter3d(n_clusters=5)
figS_scatter.print("Jensen_shannon_scatter3", outputfolder)


# Summary output
print("\nSummary of dhalf dimensions (50% info content):\n")
print(f"{'Metric':<20} | {'dhalf':>6}")
print("-" * 30)
for k, v in results.items():
    print(f"{k:<20} | {v:6}")
