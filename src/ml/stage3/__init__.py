"""Stage 3: fair supervised baselines, full test evaluation, temporal evaluation and Isolation Forest.

Run in this order:
    python -m src.ml.stage3.train_eval     # LR / RF / XGBoost, validation selection, test evaluation
    python -m src.ml.stage3.anomaly        # Isolation Forest + threshold, test evaluation
    python -m src.ml.stage3.temporal       # capture-day generalisation check
Artifacts: ml/models/stage3/, results: ml/results/stage3/.
"""
