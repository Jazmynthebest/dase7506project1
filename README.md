# DASE7506 MP1 Compact Hybrid Language Model

Complete-test BPB: **1.6017257331409482**. CPU FP32, four threads, protocol `7506-mp1-wt2-v2`.
Author: Huang Leyi. Student ID: 3036708725.
The report is `report/MP1_Report.pdf`. The frozen bundle is `inference_v3/`.

## Direct evaluation

Extract both archives into the same parent directory. They merge into `MP1_submission/`.
Use Python 3.12. Run from `MP1_submission/code/`:

```bash
python -m venv .venv
```

Activate `.venv/Scripts/Activate.ps1` on Windows or `source .venv/bin/activate` on Linux.

```bash
python -m pip install torch==2.7.1 --index-url https://download.pytorch.org/whl/cpu
python -m pip install -r requirements-experiment.txt
python -m unittest discover -s tests -v
python evaluate.py --checkpoint ../inference_v3/checkpoint.pt --device cpu --precision fp32 --threads 4 --split test --output reproduced_test.json
python verify_submission.py --checkpoint ../inference_v3/checkpoint.pt --output reproduced_contract.json
```

The evaluator imports `student_v3.py` and `student_v2.py` from code. Identical copies are also frozen in the bundle. The tokenizer there is a provenance copy; evaluation uses unchanged code/data. No retraining is needed. Only dependency installation requires internet.

## Training from scratch

All following commands run in code, with fresh run directories. They reproduce the selected ancestry and selection procedure. CPU library and hardware differences can cause small numerical variation; use the frozen checkpoint to reproduce the ranked result exactly.

The source archive includes a populated `code/runs/` directory containing experiment records. Before starting the training commands, preserve those records by renaming that directory once. From `code/`, use Windows PowerShell:

```powershell
Move-Item -LiteralPath runs -Destination runs_recorded
```

Or on Linux/macOS:

```bash
mv runs runs_recorded
```

Use an unused destination name if `runs_recorded` already exists. The trainers create new output directories and reject populated ones. Direct evaluation above does not require this rename. Retain the frozen `inference_v3/` bundle while retraining; relocate it only if you subsequently want to freeze the newly trained model.

```bash
python train.py --implementation model --device cpu --threads 4 --seed 17 --eval-every 300 --run-dir runs/baseline
python train_continue.py --checkpoint runs/baseline/checkpoint.pt --run-dir runs/matched_control --extra-targets 3613342 --learning-rate .0004 --seed 29 --eval-every 300
python train_v2.py --checkpoint runs/matched_control/checkpoint.pt --run-dir runs/v3_matched_control --steps 5200 --learning-rate .0004 --seed 73 --eval-every 200 --patience 999 --keep-snapshots
python average_checkpoints.py --checkpoints runs/v3_matched_control/step_4800.pt runs/v3_matched_control/step_5000.pt runs/v3_matched_control/step_5200.pt --output runs/v3_baseline_average_4800_5200/checkpoint.pt
python train_v2.py --checkpoint runs/v3_baseline_average_4800_5200/checkpoint.pt --run-dir runs/v3_small_continue --steps 2400 --learning-rate .00015 --seed 101 --eval-every 200 --patience 999 --keep-snapshots
python average_checkpoints.py --checkpoints runs/v3_small_continue/step_2000.pt runs/v3_small_continue/step_2200.pt runs/v3_small_continue/step_2400.pt --output runs/v3_small_average_2000_2400/checkpoint.pt
python build_memory_v2.py --output runs/v3_memory_context6_cap256.pt --max-order 6 --prune-by-context --trigram-cap 256
python select_v3_adaptive.py --checkpoint runs/v3_small_average_2000_2400/checkpoint.pt --memory runs/v3_memory_context6_cap256.pt --output-dir runs/v3_final_context6_fine --agreement-groups --memory-order 6 --memory-boundaries .02 .05 .1 .2 .35 .5 .7
python select_continuous.py --checkpoint runs/v3_final_context6_fine/checkpoint.pt --output-dir runs/v3_final_context6_fine_continuous --adaptive
python prepare_selected.py --checkpoint runs/v3_final_context6_fine_continuous/checkpoint.pt --output runs/v3_final_selected/checkpoint.pt
```

`prepare_selected.py` records dropout 0 explicitly to match original neural training. It changes metadata, not weights. In the recorded run the final continuation was paused and resumed with saved optimizer and RNG state; executing its full schedule above is equivalent.

## Control and ablation

```bash
python train_continue.py --checkpoint runs/v3_small_average_2000_2400/checkpoint.pt --run-dir runs/v3_final_matched_control --extra-targets 3613342 --learning-rate .00005 --seed 103 --eval-every 200 --save-last
python calibrate_control.py --checkpoint runs/v3_final_matched_control/last.pt --output control_calibration.json
python ablate_v3.py --checkpoint runs/v3_final_selected/checkpoint.pt --output ablation.json
```

The hybrid uses 75,702,942 neural targets plus 3,613,342 statistical-fit targets. The pure-neural control processes 79,316,284 neural targets. Averaging adds no exposure. Control validation BPB is 1.77036155; full hybrid validation BPB is 1.58733383. Ablations fix neural weights and do not recalibrate removed components.

## Freeze and resource measurement

For a new reproduction, rename or relocate the supplied inference_v3 directory before freezing; the freeze script refuses to overwrite a bundle.

```bash
python freeze_v3.py --checkpoint runs/v3_final_selected/checkpoint.pt
python measure.py --checkpoint runs/baseline/checkpoint.pt --output-dir measurements/baseline --split test --repeats 3
python measure.py --checkpoint ../inference_v3/checkpoint.pt --output-dir measurements/final --split test --repeats 3
```

Recorded evidence contains one paired complete-test measurement: final 37.741 s versus baseline 18.605 s, ratio 2.03; peak RAM 1.660 GiB; inference assets 48.400 MiB. The report makes no multi-run median claim. `inference_v3/FREEZE.json` binds inference files and configuration to their hashes. The fixed model, common.py, evaluate.py, baseline config, tokenizer and data remain unchanged.

## Evidence and costs

`code/evidence/v3/` contains frozen-result measurements, the matched-target comparison, fixed-weight ablations and correctness checks. `code/runs/` includes saved metadata, not intermediate weights or optimizer files. COSTS.json lists recorded training and search stage runtimes. These do not include all interrupted or unsaved work. The earlier recovery record documents at least 125 extra wide-model updates that are search cost, not selected ancestry. Older experiments support development history; the report summarizes the comparisons relevant to the final method.

## Acknowledgments

The course supplied the baseline and fixed benchmark. OpenAI Codex assisted with planning, implementation, experimental work and report drafting. Smoothing draws on Chen and Goodman (https://aclanthology.org/P96-1041/); the continuous cache adapts Grave, Joulin and Usunier (https://arxiv.org/abs/1612.04426). The implementation uses cosine-normalized hidden states and resets cache history per window. Dataset attribution and CC BY-SA 3.0 / GFDL notices are retained in code/README.md. All learning uses only supplied training text; no external pretrained weights are used.
