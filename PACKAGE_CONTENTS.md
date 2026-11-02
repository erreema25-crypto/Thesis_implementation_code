# Final thesis code package

This package contains the complete reported experiment and thesis source. It is
prepared for archival or GitHub upload; no upload has been performed.

## Primary deliverables

- `Improved_EKF_CNN_BiLSTM_Thesis.ipynb` — executed notebook (all cells passed).
- `Improved_EKF_CNN_BiLSTM_Thesis.html` — browsable notebook export.
- `gnss_thesis_pipeline.py` — reusable data, EKF, model, evaluation, and plotting logic.
- `run_thesis_experiment.py` — full non-interactive experiment, including five-seed robustness.
- `build_thesis_notebook.py` — reproducible notebook builder.
- `Thesis_Final_Departmental_Submission.pdf` — compiled 83-page thesis PDF.
- `thesis_latex/` — complete Overleaf/LaTeX source, bibliography, figures, and tables.

## Data and trained models

- `uav_navigation_dataset.csv` — supplied Kaggle-source CSV used to condition carriers.
- `outputs/data/kaggle_conditioned_attack_dataset.csv.gz` — complete generated
  semi-synthetic sample-level dataset (144 matched scenarios; 144,000 rows).
- `outputs/models/cnn_bilstm_with_ekf.pt` — trained proposed-model checkpoint and metadata.
- `outputs/models/cnn_bilstm_without_ekf.pt` — trained matched ablation checkpoint and metadata.
- `outputs/run_manifest.json` — exact configuration, environment, split IDs, thresholds,
  class multipliers, EKF description, and robustness seeds.

`outputs/tables/` contains all numerical evidence, including split provenance,
metrics, grouped confidence intervals, per-class/subtype results, five-seed runs,
Jacobian validation, GradientSHAP, permutation importance, and latency.
`outputs/figures/` contains the generated thesis figures; the central bar charts
and explainability/robustness figures also have vector PDF versions.

## Reproduction

```bash
python -m pip install -r requirements.txt
python run_thesis_experiment.py
python build_thesis_notebook.py
jupyter nbconvert --execute --to notebook --inplace \
  Improved_EKF_CNN_BiLSTM_Thesis.ipynb
```

The final reported split uses 34/7/7 independent Kaggle source groups. All
Normal, Spoofing, and Jamming variants from one source segment remain together.
The results are Kaggle-conditioned and semi-synthetic, not evidence of live RF
or deployment performance.

Before departmental submission, replace the bracketed author, student,
institution, school, supervisor, and date placeholders near the top of
`thesis_latex/main.tex`.

The final thesis chapter body is 15,003 words under a LaTeX-aware count that
excludes front matter, appendices, and bibliography. The bibliography contains
50 cited references. Both the exact vector architecture schematic and the new
high-resolution implementation-matched architecture image are included.
