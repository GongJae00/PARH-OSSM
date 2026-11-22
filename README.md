<h1 align="center">Video-Based Respiratory Rate Estimation via<br>Multichannel Oscillator State-Space Fusion</h1>

<p align="center">
YoungJae Cho<sup>1,†</sup>, Do-Yup Kim<sup>2,†</sup>, Dae-Yeol Kim<sup>1,&#42;</sup>, Youngjun Kim<sup>3,&#42;</sup><br>
<sup>1</sup>University of Ulsan &nbsp; <sup>2</sup>Incheon National University &nbsp; <sup>3</sup>Changwon National University<br>
<sup>†</sup>Equal contribution &nbsp; <sup>&#42;</sup>Corresponding authors<br>
<i>Scientific Reports</i>, in revision
</p>

## What the Code Computes

Breathing moves the upper torso by about a pixel in an ordinary video, mixed with head motion, drift, and
changes of light. The code reads that movement through three video observations and estimates one latent
breathing oscillation from all of them with PARH-OSSM, the Physiology-Aligned Regime-Adaptive Harmonic
Oscillatory State-Space Model, a multichannel filter that weights each observation by a reliability computed
from the video alone. It returns a respiratory rate and a
waveform; no step reads the respiratory belt, which only the evaluation uses.

<br>
<p align="center"><img src="figures/F1_architecture.png" width="100%"></p>
<p align="center"><sub><b>PARH-OSSM.</b> (a) The chest region, placed from the shoulder landmarks of the first
frame, and the torso box below the face. (b) The three observations of one trial and each one's share of the
Kalman update; at the transient (grey line) the update moves to torso displacement. (c) The state-space model
and its smoothed state. (d) The rate track and the waveform.</sub></p>
<br>

1. **Observations.** Optical flow (OF) and the one-dimensional intensity profile (P1D) measure the vertical
   motion in the chest region; torso displacement is the vertical shift of a box below the face. The extraction
   computes six more signals, which the bank leaves out because each repeats one of these measurements or, for
   the frame-difference statistics, responds to light.
2. **Estimation.** OSSM-KF tracks each observation alone. PARH-OSSM filters the three together as views of one
   state: a breathing oscillation, its second harmonic, a baseline, and a residual. The noise variance of each
   observation is divided by a trust computed from its innovations, its agreement with the others, and how
   clearly the window shows one rate. The rate combines the phase of the smoothed oscillation with a
   reliability-weighted readout of the OSSM-KF tracks.
3. **Evaluation.** The rate is scored in 30-s windows against three belt rates (Hilbert phase, Welch peak, and
   breath count) and against a constant predictor. MAHNOB-HCI is split into 9 development and 18 test
   participants, the latter scored once after the method was fixed. The waveform is scored by its concordance
   with the belt within a ±2 s shift, against two nulls.

## Layout

```
observations/   video → chest region and torso box → nine signals
  image/          regions, motion operators, torso shift
  signals/        signal definitions and families
  trials/         per-trial reading and writing
estimate/       the bank and the two filters
  bank/           the three observations kept and the exclusions
  filters/        OSSM-KF and PARH-OSSM
  readout/        rate readout, observability, rate posterior, priors
  controls/       the reliability controls
evaluation/     belt references → window rates → tables
  reference/      belt rates, onset placement
  rates/          window rates, confirmatory tests
  waveform/       bounded-lag CCC and its nulls
  across_trials/  slope of trial means
  motion/         head motion, image perturbations
  reliability/    redundancy of the observations, control summaries
  tables/         rows of the article's tables
figures/        figure scripts
configs/        extraction settings, session lists, the MAHNOB-HCI split
tests/          tests that pin the filters, the references, the bank, the controls
```

| In the article | Code |
|---|---|
| Observations | `observations/image/chest_roi.py`, `chest_motion.py`, `torso_shift_cache.py` |
| Bank and exclusions | `estimate/bank/`, `estimate/bank_priors.sh` |
| OSSM-KF, PARH-OSSM, observation law | `estimate/filters/ossm_kf.py`, `parh_ossm.py` |
| Readout, reliability priors | `estimate/readout/` |
| Stages and reliability controls | `estimate/materialize.sh`, `estimate/controls/` |
| Reference rates | `evaluation/reference/` |
| Table 2 | `evaluation/main_tables.sh`, `evaluation/tables/main_tables.py` |
| Table 3 | `evaluation/waveform/` |
| Confirmatory tests | `evaluation/rates/confirmatory_test.py` |
| Each observation tracked alone (Results, large errors) | `evaluation/rates/single_track_errors.py` |
| Agreement across trials | `evaluation/across_trials/` |
| Error under motion | `evaluation/motion/`, `evaluation/motion_perturbation.sh` |
| Tables of the analyses that the article reports in its text (the script keeps its earlier name), and the parameter list | `evaluation/tables/supplementary_tables.py`, `parameter_table.py` |
| Figures | `figures/architecture_figure.py`, `case_figure.py`, `agreement_figure.py`, `rate_agreement_figure.py` |

A run is a folder of per-trial pickles. `frozen` is this repository's name for the reported run. The first four
rows name arguments of `estimate/materialize.sh <run>`; `with_dof`, `chest_only` and `w2` are names of output
folders that the scoring scripts read, made as the paragraph after the commands below says:

| Run | Method | In the article |
|---|---|---|
| `frozen` | PARH-OSSM | Tables 2 and 3 |
| `plain_multichannel`, `readout`, `observation_law`, `observability` | stages before the rate posterior | Table 2, lower block |
| `equal_filter`, `equal_readout`, `shuffled_readout` | reliability controls | Results text |
| `no_of`, `no_p1d`, `no_torso` | observations left out | code only |
| `with_dof` (folder) | DoF added back | code only |
| `chest_only`, `w2` (folders) | the submitted method on COHFACE and MAHNOB-HCI | Table 2 |

## Running

Python 3.10 with `requirements.txt`; `python -m pytest` runs the tests. COHFACE
(<https://www.idiap.ch/en/scientific-research/data/cohface>) and MAHNOB-HCI (<https://mahnob-db.eu/>) come from
their providers and are not redistributed. Commands run from the repository root; each `<...>` is a folder of
your choice.

COHFACE, after making `data.cfr.avi` beside each `data.avi`
(`ffmpeg -i data.avi -vsync cfr -r 20 -c:v mpeg4 -q:v 2 data.cfr.avi`):

```
python observations/run_shards.py --dataset COHFACE --dataset-root <COHFACE> \
    --config configs/cohface_first_frame_roi.json --work <cohface> --shards 8 --no-caches
estimate/bank_priors.sh COHFACE <cohface>/data <cohface bank> 8
for run in frozen plain_multichannel readout observation_law observability; do
    estimate/materialize.sh $run <cohface bank>/data <cohface bank>/priors_group.csv \
        <cohface runs>/$run 8
done
evaluation/window_rates.sh cohface <cohface>/data <cohface runs> <records> 8 - \
    frozen plain_multichannel readout observation_law observability
evaluation/bank_rates.sh cohface <cohface bank>/data <records> 8
```

MAHNOB-HCI uses the 488 sessions in `configs/evaluation/mahnob_common488_sessions.txt`, and
`window_rates.sh` scores both placements of the belt, from the stimulus onset and ending with the recording:

```
python observations/image/torso_shift_cache.py <MAHNOB-HCI> <torso cache> 8 \
    configs/evaluation/mahnob_common488_sessions.txt
python observations/run_shards.py --dataset MAHNOB --dataset-root <MAHNOB-HCI> \
    --config configs/mahnob_first_frame_roi.json --work <mahnob> --shards 8 \
    --trials configs/evaluation/mahnob_common488_sessions.txt \
    --torso-cache <torso cache> --no-caches
estimate/bank_priors.sh MAHNOB_tailaligned <mahnob>/data <mahnob bank> 8
for run in frozen plain_multichannel readout observation_law observability; do
    estimate/materialize.sh $run <mahnob bank>/data <mahnob bank>/priors_group.csv \
        <mahnob runs>/$run 8
done
evaluation/window_rates.sh mahnob <mahnob>/data <mahnob runs> <records> 8 <MAHNOB-HCI> \
    frozen plain_multichannel readout observation_law observability
evaluation/bank_rates.sh mahnob <mahnob bank>/data <records> 8
```

The other runs of the table above take the same `materialize.sh` call. The three folders run the `frozen`
settings on other inputs: `with_dof` and `chest_only` on a bank made by `bank_priors.sh ... with_dof` or
`... chest_only`, for example `estimate/materialize.sh frozen <that bank>/data <that bank>/priors_group.csv
<runs>/with_dof 8`, and `w2` on the submitted extraction (`configs/mahnob.json`) with the priors stored in
`estimate/readout/submitted_reliability_priors.csv.gz` in place of `priors_group.csv`, which were computed from
observation files that were not kept. Head motion and the image perturbations come
from `evaluation/motion/head_motion.py` and `evaluation/motion_perturbation.sh`. The scoring scripts then write
the table rows: the first MAHNOB-HCI call scores the development participants, the second the test participants.

```
evaluation/cohface_scores.sh <records> <tables> <cohface>/data \
    <cohface runs>/frozen/final/data 8
evaluation/mahnob_test_scores.sh <records> <tables> <mahnob>/data <mahnob bank>/data \
    <mahnob runs>/frozen/final/data <MAHNOB-HCI> 8 development
evaluation/mahnob_test_scores.sh <records> <tables> <mahnob>/data <mahnob bank>/data \
    <mahnob runs>/frozen/final/data <MAHNOB-HCI> 8
```

Figure 2 comes from `figures/case_figure.py` and Figure 3 from `figures/agreement_figure.py`, which reads the
trial-mean CSVs that `evaluation/across_trials/rate_agreement.py` writes. Figure 1 of the article, `figures/architecture_figure.py`,
draws the regions on a photograph of an author, which is not distributed. The overview at the top of this page is
a schematic of the same pipeline, not the article's Figure 1.

Another set of videos needs a class in `observations/datasets.py` that lists its videos and loads its belt;
the reliability priors are then fitted on its own bank with `estimate/bank_priors.sh`.

## Credits

The OF, DoF, and P1D operators originate in resPyre (<https://github.com/phuselab/resPyre>), described in
Boccignone et al., *Remote Respiration Measurement with RGB Cameras: A Review and Benchmark*, ACM Computing
Surveys 58(5), 2025. Like resPyre, this code is released under the GNU General Public License v3.0.
