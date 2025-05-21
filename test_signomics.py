"""
Test suite for signomics module

Basic functionality test (signal creation, noise, and DNA encoding)

Generative Simulation Initiative\olivier.vitrac@gmail.com

"""

import os
from sig2dna_core.signomics import peaks, signal_collection, DNAsignal

# %% Output configuration
outputfolder = "./images" if os.path.isdir("./images") else ("../images" if os.path.isdir("../images") else None)

# %% ----------------------------------------------
# 1. Basic Functionality Test
# ----------------------------------------------

# Define basic peak signal
p = peaks()
p.add(x=10, w=1, h=1)
p.add(x=20, w=2, h=1)
p.add(x=30, w=3, h=1)
s = p.to_signal()
fig, _ = s.plot()
fig.print("test_simple_signal", outputfolder)

# Create variants
s_noisy = s.add_noise(kind="gaussian", scale=0.01, bias=0.5)
s_scaled = s * 1.5
collection = signal_collection(s, s_noisy, s_scaled)
fig = collection.plot()
fig.print("test_signal_collection", outputfolder)

# Encode into DNA signal
s_dna = DNAsignal(s)
s_dna.compute_cwt()
s_dna.encode_dna()
fig4 = s_dna.plot_codes(4)
fig8 = s_dna.plot_codes(8)
fig16 = s_dna.plot_codes(16)
fig4.print("test_simple_signal_scale4", outputfolder)
fig8.print("test_simple_signal_scale8", outputfolder)
fig16.print("test_simple_signal_scale16", outputfolder)

# Encode into full DNA with repetitions
# align scales 4 and 16, plot the alignment
s_dna.encode_dna_full()
s_dna.codesfull[4].align(s_dna.codesfull[16],"bio")
s_dna.codesfull[4].plot_mask() # plot alignment mask
fig4vs16,_ = s_dna.codesfull[4].plot_alignment()
fig4vs16.print("test_simple_alignment_scale16vs4")
