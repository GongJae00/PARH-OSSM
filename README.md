<h1 align="center">Video-Based Respiratory Rate Estimation via<br>Reliability-Weighted Multichannel State-Space Fusion</h1>

<p align="center">
Young-Jae Cho<sup>1,†</sup>, Do-Yup Kim<sup>2,†</sup>, Dae-Yeol Kim<sup>1,&#42;</sup>, Youngjun Kim<sup>3,&#42;</sup><br>
<sup>1</sup>University of Ulsan &nbsp; <sup>2</sup>Incheon National University &nbsp; <sup>3</sup>Changwon National University<br>
<sup>†</sup>Equal contribution &nbsp; <sup>&#42;</sup>Corresponding authors<br>
<i>Scientific Reports</i>, in revision
</p>

## What the code computes

Breathing moves the chest and shoulders by about a pixel in an ordinary video, and that movement reaches the camera
mixed with head and posture motion, changes of illumination, and sensor noise. This code estimates the respiratory
rate and a respiratory waveform from such video without a contact sensor. It reads the movement through nine
observations, each a different measurement of the same displacement, and estimates one latent breathing oscillation
from all of them with a state-space filter, PARH-OSSM, which weights each observation by how much of that
oscillation it appears to carry.

<br>
<p align="center"><img src="figures/F1_architecture.png" width="100%"></p>
<p align="center"><sub><b>Figure 1.</b> PARH-OSSM: the nine video-derived observations, the reliability-weighted
observation law, the state-space model, and the separate rate and waveform outputs.</sub></p>
<br>

1. **Observations.** Each observation projects the chest displacement in its own way. Optical flow (OF) measures the
   vertical velocity of the chest region, the difference of frames (DoF) counts the pixels whose brightness
   changes, the one-dimensional profile observations (P1D) measure the vertical shift of a row-intensity profile,
   and torso displacement measures the position of the upper body in a box below the face. Each therefore adds its
   own bias, such as harmonics of the breath, drift, or head motion, and its own noise.
2. **Estimation.** The state of the filter is one oscillation with its second harmonic, a baseline drift, and an
   aperiodic residual, and every observation is a noisy view of that state. A trust computed from the video alone,
   from the filter's innovations, the agreement between observations, and their signal quality, sets the
   measurement variance of each observation in the Kalman filter and the Rauch–Tung–Striebel smoother. A readout
   turns the smoothed state into a rate track and a waveform. No step reads the respiratory belt at inference.
3. **Evaluation.** The rate is scored in 30-s windows under three definitions of the belt's rate (Hilbert, Welch,
   and breath count), against methods that read the same nine observations and against a constant predictor that
   outputs the typical rate of the cohort. The waveform is scored by its concordance with the belt within ±2 s,
   against the belts of other participants and against phase-randomized copies of the trial's own belt. COHFACE
   was used to build the method. MAHNOB-HCI participants were split into 9 development and 18 test participants,
   and the test participants were scored once, after the method was fixed.

## Layout

The folders follow the computation, from video to the reported tables.

```
observations/   video → chest ROI and torso box → the nine observations
estimate/       OSSM-KF, PARH-OSSM, reliability priors, stage and control variants
evaluation/     belt references, window rates, tests, waveform agreement, table rows
figures/        the figure scripts of the article
configs/        extraction settings of each dataset
tests/          tests that pin the filters, the reference rates, and the controls
```

| In the article | Code |
|---|---|
| Video-derived observations | `observations/chest_roi.py`, `chest_motion.py`, `observation_signals.py` |
| State-space model, observation law | `estimate/parh_ossm.py`, `multichannel_fusion.py` |
| Reliability priors | `estimate/reliability_priors.py` |
| OSSM-KF | `estimate/ossm_kf.py` |
| Stages and controls | `estimate/materialize.sh` (one variant per run) |
| Hilbert, Welch, and breath-count references | `evaluation/belt_reference.py`, `rate_tracks.py` |
| Tables 2 and 3 | `evaluation/main_tables.sh`, `main_tables.py` |
| Table 4, waveform agreement | `evaluation/bounded_lag_waveform.py`, `phase_randomized_null.py`, `onset_alignment_waveform.py`, `waveform_table.py` |
| Figure 2 | `figures/case_figure.py` |
| Confirmatory tests | `evaluation/confirmatory_test.py`, `mahnob_test_scores.sh` |
| Agreement across trials | `evaluation/rate_agreement.py`, `stage_slopes.py`, `observation_track_slope.py`, `figures/rate_agreement_figure.py` |
| Error under motion | `evaluation/head_motion.py`, `motion_tertiles.py`, `motion_perturbation.sh` |
| Supplementary tables | `evaluation/supplementary_tables.py`, `parameter_table.py` |

## Running

The code runs on Python 3.10 with the packages in `requirements.txt`. COHFACE
(<https://www.idiap.ch/en/scientific-research/data/cohface>) and MAHNOB-HCI (<https://mahnob-db.eu/>) are obtained
from their providers under their access terms and are not redistributed here. From the repository root, the
COHFACE result of PARH-OSSM is reproduced in four steps:

1. Extract the nine observations from the videos:
   ```
   python observations/run_shards.py --dataset COHFACE --dataset-root <COHFACE folder> \
       --config configs/cohface_first_frame_roi.json --work <work dir> --shards 8
   ```
2. Fit the reliability priors on the observations:
   ```
   python estimate/reliability_priors.py --data-dir <work dir>/data --dataset-label COHFACE \
       --out-edge <priors dir>/edges.csv --out-group <priors dir>/group.csv \
       --out-summary <priors dir>/summary.csv --window-sec 30 --window-stride-sec 10 --jobs 8
   ```
3. Run PARH-OSSM with the reported settings (`frozen`) and the four stages of Table 3:
   ```
   for run in frozen plain_multichannel readout observation_law observability; do
       estimate/materialize.sh $run <work dir>/data <priors dir>/group.csv <runs dir>/$run 8
   done
   ```
4. Compute the window rates and the rows of Tables 2 and 3:
   ```
   evaluation/window_rates.sh cohface <work dir>/data <runs dir> <records dir> 8 - \
       frozen plain_multichannel readout observation_law observability
   evaluation/bank_rates.sh cohface <work dir>/data <records dir> 8
   evaluation/main_tables.sh <records dir> cohface > table_rows.tex
   ```
   The row of PARH-OSSM on the eight chest observations needs one more run, `chest_only`, made as the header of
   `evaluation/main_tables.sh` describes; the controls and leave-one-out runs are listed in
   `estimate/materialize.sh`.

`python -m pytest` runs the tests.

## Credits

The OF, DoF, and P1D observation operators originate in resPyre (<https://github.com/phuselab/resPyre>), described in
Boccignone et al., *Remote Respiration Measurement with RGB Cameras: A Review and Benchmark*, ACM Computing Surveys
58(5), 2025. Like resPyre, this code is released under the GNU General Public License v3.0.
