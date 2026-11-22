"""The size of the breathing displacement in the video: the RMS, in pixels, of the torso displacement observation in
the respiratory band (0.08-0.5 Hz, the band of the belt references), per trial, with its median and quartiles over
the trials. It sets the scale against which the motion perturbations of motion_perturbation.sh are chosen. The band
also holds head and posture motion, which the value includes beside the breath.

usage: torso_displacement_rms.py <observation data dir>
"""
import glob
import os
import pickle
import sys

import numpy as np
from scipy.signal import butter, sosfiltfilt

BAND_HZ = (0.08, 0.5)

rms = {}
for path in sorted(glob.glob(os.path.join(sys.argv[1], "*.pkl"))):
    trial = pickle.load(open(path, "rb"))
    displacement = next(np.asarray(e["estimate"]["signal_hat"], float) for e in trial["estimates"] if e["method"] == "torso_shift")
    band = butter(2, BAND_HZ, btype="band", fs=trial["fps"], output="sos")
    rms[os.path.basename(path)[:-4]] = np.sqrt(np.mean(sosfiltfilt(band, displacement - displacement.mean()) ** 2))

for trial, value in rms.items():
    print(f"{trial} {value:.3f}")
q1, median, q3 = np.percentile(list(rms.values()), [25, 50, 75])
print(f"{len(rms)} trials: median {median:.3f} px, IQR {q1:.3f}-{q3:.3f} px")
