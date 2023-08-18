# Evaluation Wiki POC

## Modules
- `plotting`: used to create any plots and tables from results
- `process_legal_results.py`: evaluate and plot results for court rulings predictions
- `top_k_prediction_eval.py`: used to compute metrics for model predictions

### Runnables
python -m evaluation.top_k_evaluation <key-of-run>
python -m evaluation.plotting.precomputed_plotting <key-of-run>