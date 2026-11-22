<h1 align="center">Video-Based Respiratory Rate Estimation via<br>Reliability-Weighted Multichannel State-Space Fusion</h1>

<p align="center">
Young-Jae Cho<sup>1,†</sup>, Do-Yup Kim<sup>2,†</sup>, Dae-Yeol Kim<sup>1,&#42;</sup>, Youngjun Kim<sup>3,&#42;</sup><br>
<sup>1</sup>University of Ulsan &nbsp; <sup>2</sup>Incheon National University &nbsp; <sup>3</sup>Changwon National University<br>
<sup>†</sup>Equal contribution &nbsp; <sup>&#42;</sup>Corresponding authors<br>
<i>Scientific Reports</i>, in revision
</p>

## What the Code Computes

Breathing moves the upper torso by about a pixel in an ordinary video, and that movement reaches the camera
mixed with head and posture motion, changes of illumination, and sensor noise. This code estimates the respiratory
rate and a respiratory waveform from such video without a contact sensor. It reads the movement through three
observations, each a different measurement of the same displacement, and estimates one latent breathing oscillation
from all of them with a state-space filter, PARH-OSSM, which weights each observation by how much of that
oscillation it appears to carry.

<br>
<p align="center"><img src="figures/F1_architecture.png" width="100%"></p>
<p align="center"><sub><b>Figure 1.</b> PARH-OSSM. (a) The chest strip, placed from the shoulder landmarks of the
first frame, and the torso box below the face. (b) The three observations of one trial and, below them, the share of
each in the Kalman update; where a transient breaks the chest observations (grey line), the update moves to torso
displacement. (c) The state-space model and its smoothed state: the breathing oscillation, its second harmonic, a
slow baseline, and an aperiodic residual. (d) The rate track and the smoothed waveform.</sub></p>
<br>

1. **Data.** COHFACE (160 trials of 40 participants) and MAHNOB-HCI (488 sessions) film seated participants
   who wear a respiratory belt, and both frame the face large and central. The belt signal is the reference: no
   step of the method reads it, and the evaluation reads it to score the estimates. COHFACE was used to build the
   method. MAHNOB-HCI participants were split into 9 development and 18 test participants; the development
   participants were used for the choices of the revision, and the test participants were scored once, after the
   method was fixed. The MAHNOB-HCI belt recording is longer than the video and does not store where the video lies
   in it, so the reference starts at the stimulus onset, the position the development participants supported, and
   every MAHNOB-HCI result is also computed with the reference ending with the recording.
2. **Observations.** Optical flow (OF) measures the vertical velocity of the texture in a chest strip placed from
   the shoulder landmarks of the first frame, and the one-dimensional profile (P1D) measures the vertical shift of
   the strip's row-intensity profile between frames, with the cross-correlation peak refined by quadratic
   interpolation. Torso displacement is the vertical shift of a box below the face against a keyframe renewed every
   second, so it stays in the image when the framing pushes the shoulders out of it; its errors accumulate at each
   renewal. These three form the bank, the observations the method reads. The extraction computes six further
   signals that the bank leaves out (`estimate/bank.py`). Four repeat a measurement already in the bank: the OF
   bridge integrates OF, and the linear, cubic, and consensus P1D variants refine the same correlation peak. Because
   the filter of item 3 treats the errors of its observations as independent, a repeated measurement would count
   its error twice. The other two, the difference of frames (DoF) and the DoF bridge, count and sign the pixels
   whose brightness changes, so they also respond to changes of light. Before filtering, every observation is
   aligned to one anchor observation, which removes the quarter cycle by which a velocity leads a displacement.
   Head and posture motion enter all three observations, each in its own way, so the observation that carries the
   breath can change within a trial.
3. **Estimation.** Each observation is first tracked alone by an oscillator Kalman filter, OSSM-KF, which is also
   the single-observation method of the article. PARH-OSSM then filters the three together as noisy views of one
   state: a breathing oscillation, its second harmonic for the asymmetry of inhalation and exhalation, a baseline
   drift, and an aperiodic residual, so that drift and irregular motion go to their own components rather than to
   the oscillation. The observation law sets how much each observation counts. The measurement variance of each
   observation, estimated from its innovations, is divided by a trust computed at each frame from signal quality,
   judged by those innovations, the agreement between observations in rate and phase, and target observability,
   which scores how clearly the window shows one respiratory rate. The variances of all observations rise together
   when that rate is ambiguous. The Kalman filter and the Rauch–Tung–Striebel smoother run with these variances,
   and the waveform is read from the smoothed state. The rate combines the phase rate of the smoothed oscillation
   with a readout of the OSSM-KF tracks, in which each observation forms its own family, weighted by reliability
   priors of each family and corrected by a posterior over candidate rates. Reliability thus enters twice, as the
   trust of each observation in the filter and as the prior of each family in the readout; both are computed from
   the video observations alone.
4. **Evaluation.** The rate is scored in 30-s windows under three definitions of the belt's rate (Hilbert, Welch,
   and breath count), because a method scores best under the definition that shares its estimator. It is compared
   with methods that read the same three observations, with the method as submitted, which reads eight chest
   observations, and with a constant predictor that outputs the typical rate of the cohort, which shows how much of
   a low error comes from staying near that rate. On the MAHNOB-HCI test participants, six confirmatory comparisons
   set PARH-OSSM against the constant predictor and against the submitted version under each reference definition,
   corrected together for multiple testing. The slope of the trial-mean estimate on the trial-mean reference shows
   whether a method follows the differences in rate between trials. The waveform is scored by its concordance
   correlation coefficient (CCC) with the belt after a shift of up to ±2 s. Two nulls bound that agreement: the
   belts of other participants give the agreement expected by chance, and phase-randomized copies of the trial's
   own belt keep its rate and change only the timing of its breaths. Table 3 of the article adds the parts of item
   3 one at a time: the multichannel filter, the readout, the observation law, target observability and signal
   quality, and the rate posterior, which completes PARH-OSSM. Three controls each remove one use of reliability
   and keep everything else: `equal_filter` replaces the trust of each observation by its mean over the
   observations at the same frame, `equal_readout` gives every family the mean of the family priors in each
   window, and `shuffled_readout` permutes the windows of each family's priors within the trial. A fourth control
   averages the observations' OSSM-KF tracks without weights, in place of the filter and the readout
   (`evaluation/bank_rates.sh`). Each route
   exclusion reruns PARH-OSSM without one observation of the bank, its route, and keeps the priors of the whole
   bank.

## Layout

The folders follow the computation, from video to the reported tables.

```
observations/   video → chest region and torso box → the nine observations
estimate/       the bank, OSSM-KF, PARH-OSSM, reliability priors, stages, controls
evaluation/     belt references, window rates, statistics, waveform agreement, table rows
figures/        the figure scripts of the article
configs/        extraction settings, including the submitted MAHNOB-HCI run
tests/          tests that pin the filters, the reference rates, the bank, the controls
```

| In the article | Code |
|---|---|
| Chest observations | `observations/chest_roi.py`, `chest_motion.py`, `observation_signals.py` |
| Torso displacement | `observations/chest_motion.py` (`torso_shift`), `observations/torso_shift_cache.py` (MAHNOB-HCI) |
| The bank and its route exclusions | `estimate/bank.py`, `drop_observations.py`, `bank_priors.sh` |
| OSSM-KF | `estimate/ossm_kf.py` |
| State-space model, observation law | `estimate/parh_ossm.py` |
| Anchor alignment, readout, observability, rate posterior | `estimate/multichannel_fusion.py` |
| Reliability priors | `estimate/reliability_priors.py` |
| Stages (Table 3), controls | `estimate/materialize.sh`, one run each |
| Belt references | `evaluation/belt_reference.py`, `rate_tracks.py` |
| Onset placement, MAHNOB-HCI | `evaluation/onset_reference.py`, `onset_alignment_waveform.py` |
| Sessions and split, MAHNOB-HCI | `evaluation/mahnob_common488_sessions.txt`, `mahnob_subject_split.csv` |
| Tables 2 and 3, constant predictor | `evaluation/main_tables.sh`, `main_tables.py` |
| Table 4, waveform agreement | `evaluation/bounded_lag_waveform.py`, `waveform_table.py` |
| Waveform nulls | `evaluation/phase_randomized_null.py` |
| Figures 1 and 2 | `figures/architecture_figure.py`, `case_figure.py` |
| Confirmatory tests | `evaluation/confirmatory_test.py`, `mahnob_test_scores.sh` |
| Agreement across trials | `evaluation/rate_agreement.py`, `stage_slopes.py`, `observation_track_slope.py` |
| Agreement figures | `figures/rate_agreement_figure.py` |
| Error under motion | `evaluation/head_motion.py`, `motion_tertiles.py`, `motion_perturbation.sh` |
| Perturbation scale | `evaluation/torso_displacement_rms.py` |
| Supplementary tables | `evaluation/cohface_scores.sh`, `supplementary_tables.py`, `parameter_table.py` |

The extraction stores nine observations in each trial's pickle, under the names `of_farneback`, `of_disp_bridge`,
`DoF`, `dof_disp_bridge`, `profile1D linear`, `profile1D quadratic`, `profile1D cubic`, `profile1d_consensus`, and
`torso_shift`, in the order of item 2. The bank keeps `of_farneback`, `profile1D quadratic`, and `torso_shift`;
`python estimate/bank.py labels` prints them and `python estimate/bank.py drop` prints the six it removes. A row
ending in `__kfstd` holds the OSSM-KF track of one observation, and a row ending in `__parh_ossm` the PARH filter run
on that observation alone, so removing an observation removes these two rows with it. A run is a folder of
pickles written by `estimate/materialize.sh`, and its window rates are written to
`<dataset>_<run>_window_rates_three_references.csv`, with `_onset` before `_window` for the MAHNOB-HCI belt that
starts at the stimulus onset. The runs are these:

| Run | Method | In the article |
|---|---|---|
| `frozen` | PARH-OSSM | Tables 2–4 |
| `plain_multichannel`, `readout`, `observation_law`, `observability` | the stages before the rate posterior | Table 3 |
| `equal_filter`, `equal_readout`, `shuffled_readout` | reliability controls | Supplementary |
| `no_of`, `no_p1d`, `no_torso` | route exclusions | Supplementary |
| `with_dof` | the bank with DoF and the DoF bridge added back | Supplementary |
| `chest_only` | COHFACE: PARH-OSSM on the eight chest observations | Table 2, as submitted |
| `w2` | MAHNOB-HCI: the submitted method with its stored priors | Table 2, as submitted |

`python estimate/bank.py routes` prints the route runs that `mahnob_test_scores.sh` and `cohface_scores.sh` read.

## Running

The code runs on Python 3.10 with the packages in `requirements.txt`. COHFACE
(<https://www.idiap.ch/en/scientific-research/data/cohface>) and MAHNOB-HCI (<https://mahnob-db.eu/>) are obtained
from their providers under their access terms and are not redistributed here. Commands run from the repository
root, and each `<...>` is a folder of your choice. `python -m pytest` runs the tests, which pin the filters, the
reference rates, the bank, and the controls. `--no-caches` makes the extraction compute every observation
from the video even when the dataset folder holds observation files of an earlier run. The COHFACE and MAHNOB-HCI
steps give the rows of Tables 2 and 3; the steps after them give the reliability controls and route exclusions, the
confirmatory tests on the MAHNOB-HCI test participants, Table 4, and the Supplementary tables, and the last
step draws the figures.

### COHFACE

1. Beside each COHFACE `data.avi`, make `data.cfr.avi`, a copy at a constant 20 frames per second. The reported
   copies were made with ffmpeg 6.1 under settings that were not recorded; this command makes a copy of the same
   frame rate and codec, though not a byte-identical one:
   ```
   ffmpeg -i data.avi -vsync cfr -r 20 -c:v mpeg4 -q:v 2 data.cfr.avi
   ```
2. Extract the nine observations from `data.cfr.avi`. The chest box stays where the first frame placed it
   (`configs/cohface_first_frame_roi.json`), and the torso displacement is computed inside the extraction:
   ```
   python observations/run_shards.py --dataset COHFACE --dataset-root <COHFACE folder> \
       --config configs/cohface_first_frame_roi.json --work <cohface dir> --shards 8 --no-caches
   ```
3. Remove the six observations the bank leaves out, and compute the reliability priors of the bank. The same script
   with `chest_only` gives the eight chest observations and their priors, the input of the row "as submitted" of
   Table 2, and with `with_dof` the bank with DoF and the DoF bridge added back:
   ```
   estimate/bank_priors.sh COHFACE <cohface dir>/data <cohface bank> 8
   estimate/bank_priors.sh COHFACE <cohface dir>/data <cohface chest> 8 chest_only
   estimate/bank_priors.sh COHFACE <cohface dir>/data <cohface with_dof> 8 with_dof
   ```
4. Run PARH-OSSM (`frozen`), the four stages of Table 3 before it, PARH-OSSM on the chest bank, and PARH-OSSM on the
   bank with DoF added back:
   ```
   for run in frozen plain_multichannel readout observation_law observability; do
       estimate/materialize.sh $run <cohface bank>/data <cohface bank>/priors_group.csv \
           <cohface runs>/$run 8
   done
   estimate/materialize.sh frozen <cohface chest>/data <cohface chest>/priors_group.csv \
       <cohface runs>/chest_only 8
   estimate/materialize.sh frozen <cohface with_dof>/data \
       <cohface with_dof>/priors_group.csv <cohface runs>/with_dof 8
   ```
5. Compute the window rates, those of the methods without a filter (`bank_rates.sh`), and the table rows; the `-`
   stands for the MAHNOB-HCI folder, which COHFACE does not use:
   ```
   evaluation/window_rates.sh cohface <cohface dir>/data <cohface runs> <records> 8 - \
       frozen plain_multichannel readout observation_law observability chest_only with_dof
   evaluation/bank_rates.sh cohface <cohface bank>/data <records> 8
   evaluation/main_tables.sh <records> cohface > <records>/cohface_main_tables.tex
   ```

### MAHNOB-HCI

MAHNOB-HCI differs in four ways. Its 488 sessions with all eight chest observations are used
(`evaluation/mahnob_common488_sessions.txt`), and the torso displacement is measured once per session and passed to
the extraction as a cache. The row "as submitted" of Table 2 is the submitted method itself, which reads the eight
chest observations of the submitted extraction (`configs/mahnob.json`, a chest box refreshed every 10 frames) with
the reliability priors stored with the submitted analysis (`estimate/submitted_reliability_priors.csv.gz`); they
were computed from observation files that were not kept, so they cannot be recomputed. The belt is read under two
placements, ending with the recording and starting at the stimulus onset, the primary one, and `window_rates.sh`
writes both when it is given the MAHNOB-HCI folder.

1. Measure the torso displacement of the 488 sessions, then extract the nine observations with the chest box
   fixed at the first frame (`configs/mahnob_first_frame_roi.json`):
   ```
   python observations/torso_shift_cache.py <MAHNOB-HCI folder> <torso cache> 8 \
       evaluation/mahnob_common488_sessions.txt
   python observations/run_shards.py --dataset MAHNOB --dataset-root <MAHNOB-HCI folder> \
       --config configs/mahnob_first_frame_roi.json --work <mahnob dir> --shards 8 \
       --trials evaluation/mahnob_common488_sessions.txt --torso-cache <torso cache> --no-caches
   ```
2. Remove the six observations the bank leaves out, compute the reliability priors of the bank and of the bank with
   DoF added back, and extract the eight chest observations of the submitted method. The first argument of
   `bank_priors.sh` only names the dataset in the priors table; the priors read no belt, so it sets no placement:
   ```
   estimate/bank_priors.sh MAHNOB_tailaligned <mahnob dir>/data <mahnob bank> 8
   estimate/bank_priors.sh MAHNOB_tailaligned <mahnob dir>/data <mahnob with_dof> 8 with_dof
   python observations/run_shards.py --dataset MAHNOB --dataset-root <MAHNOB-HCI folder> \
       --config configs/mahnob.json --work <mahnob submitted> --shards 8 \
       --trials evaluation/mahnob_common488_sessions.txt --no-caches
   ```
3. Run PARH-OSSM, the four stages before it, PARH-OSSM on the bank with DoF added back, and the submitted method
   (`w2`):
   ```
   for run in frozen plain_multichannel readout observation_law observability; do
       estimate/materialize.sh $run <mahnob bank>/data <mahnob bank>/priors_group.csv \
           <mahnob runs>/$run 8
   done
   estimate/materialize.sh frozen <mahnob with_dof>/data \
       <mahnob with_dof>/priors_group.csv <mahnob runs>/with_dof 8
   estimate/materialize.sh frozen <mahnob submitted>/data \
       estimate/submitted_reliability_priors.csv.gz <mahnob runs>/w2 8
   ```
4. Compute the window rates under both placements, the rates without a filter, and the table rows of the development
   participants; `mahnob_test_scores.sh` below writes those of the test participants:
   ```
   evaluation/window_rates.sh mahnob <mahnob dir>/data <mahnob runs> <records> 8 \
       <MAHNOB-HCI folder> frozen plain_multichannel readout observation_law \
       observability with_dof
   evaluation/window_rates.sh mahnob <mahnob submitted>/data <mahnob runs> <records> 8 \
       <MAHNOB-HCI folder> w2
   evaluation/bank_rates.sh mahnob <mahnob bank>/data <records> 8
   evaluation/main_tables.sh <records> mahnob development \
       > <records>/mahnob_development_main_tables.tex
   ```

### Controls, tests, and the Supplementary

The reliability controls and the route exclusions run on the bank of each dataset with its priors. Each exclusion
(`no_of`, `no_p1d`, `no_torso`) is run on a copy of the bank without that observation and keeps the priors of the
whole bank. `mahnob_test_scores.sh` then runs the confirmatory tests on the MAHNOB-HCI test participants and writes
the Supplementary tables of both datasets.

1. Run the controls and exclusions, and compute their window rates:
   ```
   variants="equal_readout shuffled_readout equal_filter no_of no_p1d no_torso"
   for run in $variants; do
       estimate/materialize.sh $run <cohface bank>/data <cohface bank>/priors_group.csv \
           <cohface runs>/$run 8
       estimate/materialize.sh $run <mahnob bank>/data <mahnob bank>/priors_group.csv \
           <mahnob runs>/$run 8
   done
   evaluation/window_rates.sh cohface <cohface dir>/data <cohface runs> <records> 8 - \
       $variants
   evaluation/window_rates.sh mahnob <mahnob dir>/data <mahnob runs> <records> 8 \
       <MAHNOB-HCI folder> $variants
   ```
2. Measure the head motion of every trial, writing the face track of each trial to `<face tracks>`, and run the
   frozen method on the COHFACE videos with controlled translation, rotation, and brightness changes, written to
   `<perturbed dir>`:
   ```
   python evaluation/head_motion.py <COHFACE folder> <MAHNOB-HCI folder> <face tracks> 8 \
       <records>/cohface_frozen_window_rates_three_references.csv \
       <records>/mahnob_frozen_window_rates_three_references.csv \
       > <records>/head_motion_trials.csv
   evaluation/motion_perturbation.sh <COHFACE folder> <cohface bank>/priors_group.csv \
       <perturbed dir> <records> 8 translation:0.5 translation:1 translation:2 \
       rotation:0.25 rotation:0.5 rotation:1 brightness:0.01 brightness:0.02 brightness:0.05
   ```
3. Score COHFACE, then the MAHNOB-HCI participants, into the folder of the tables. The first MAHNOB-HCI command runs
   the chain on the development participants, which shows a missing file or a crash before any test score exists;
   the second scores the test participants. Each writes the rows of the article's tables and of the Supplementary
   tables, and `mahnob_test_scores.sh` also the agreement figures of the Supplementary:
   ```
   evaluation/cohface_scores.sh <records> <tables> <cohface dir>/data \
       <cohface runs>/frozen/final/data 8
   evaluation/mahnob_test_scores.sh <records> <tables> <mahnob dir>/data <mahnob bank>/data \
       <mahnob runs>/frozen/final/data <MAHNOB-HCI folder> 8 development
   evaluation/mahnob_test_scores.sh <records> <tables> <mahnob dir>/data <mahnob bank>/data \
       <mahnob runs>/frozen/final/data <MAHNOB-HCI folder> 8
   ```
4. Draw Figure 1 from the development session `mahnob_10` and Figure 2 from both datasets:
   ```
   python figures/architecture_figure.py <mahnob bank>/data/mahnob_10.pkl \
       <mahnob runs>/frozen/final/data/mahnob_10.pkl \
       <records>/mahnob_frozen_onset_window_rates_three_references.csv \
       <figures>/F1_architecture.pdf
   python figures/case_figure.py <records>/cohface_frozen_window_rates_three_references.csv \
       <cohface runs>/frozen/final/data <cohface bank>/data \
       <records>/mahnob_frozen_onset_window_rates_three_references.csv \
       <mahnob runs>/frozen/onset_final/data <mahnob bank>/data \
       evaluation/mahnob_subject_split.csv <figures>/F2_cases.pdf
   ```

## Other Videos

The extraction reads the folder layouts of COHFACE and MAHNOB-HCI, one class for each in `observations/datasets.py`
that lists the videos of the dataset and loads their belt signals. Another set of videos needs such a class; the
belt it loads is stored beside the estimates and read only by the evaluation. The chest strip, the torso box, and
the observations then follow from the video as in the steps above, the reliability priors are fitted on the bank of
the new videos themselves (`estimate/bank_priors.sh`), and `estimate/materialize.sh frozen` writes each trial's
estimates to `<run>/final/data`.

## Credits

The OF, DoF, and P1D observation operators originate in resPyre (<https://github.com/phuselab/resPyre>), described in
Boccignone et al., *Remote Respiration Measurement with RGB Cameras: A Review and Benchmark*, ACM Computing Surveys
58(5), 2025. Like resPyre, this code is released under the GNU General Public License v3.0.
