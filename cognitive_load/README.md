This folder contains a compact version of our cognitive load ML pipeline.

Contents
- train/run_experiments.py: CLI to run ML experiments per configuration/split, writes summary tables.
- train/train_ml_models.py: Feature engineering and classical ML training/evaluation.
- train/data_loader.py: Loads multimodal signals (remote/contact PPG/Resp, blink markers), builds inputs per config/split.
- train/utils.py: Experiment configs, constants, NASA‑TLX label creation, signal helpers.
- models/trainers.py, models/resnet1d.py: DL model definitions (optional; not required for ML-only runs).

Quick start
1) Install deps (Python 3.9+ recommended):
   pip install -r requirements.txt

2) Prepare data (not included). Set DATA_ROOT to your data path (default: ./data):
   export DATA_ROOT=/path/to/data
   Expected files under :
   - NASA_TLX.csv (present in the dataset)
   - CogPhys_all_Folds.pkl (present in the codebase under dataset/CogPhysFolds/)
   - crossval_rppg_waveforms.pickle (needs to be generated using the notebook from the main folder)
   - crossval_resp_waveforms.pickle (needs to be generated using the notebook from the main folder)
   - eos_norm_dict.pkl (needs to be generated using the eos_norm.py notebook in this folder)
   - GT.pkl (for contact signals - can be generated using the notebook from the main folder)

3) Run ML experiments (example: remote PPG + remote Resp + blink, split 0):
   python train/run_experiments.py --ml --exp remote_ppg_remote_resp_blink --split 0
   Repeat for splits 0..3. Logs will be in ./results/experiments_log_split{split}.txt

4) Aggregate and generate LaTeX (optional):
   python scripts/parse_cv_results.py
   Appends aggregated stats and a LaTeX table to ./cv_results.log

Notes
- To change which inputs are used, see experiment keys in train/utils.py (EXPERIMENT_CONFIGS).
- If you move data, prefer setting DATA_ROOT instead of editing code.
- Results/ and model artifacts are intentionally excluded.
