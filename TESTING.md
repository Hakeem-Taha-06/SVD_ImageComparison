# Testing Guide

## Run Tests

```bash
# Quick test (default: 4 configs)
python test_pipeline.py --dataset data/raw_data/dataset2_Faculty_entering

# Full sweep (25 configs)  
python test_pipeline.py --dataset data/raw_data/dataset2_Faculty_entering --full-sweep

# Custom
python test_pipeline.py --dataset data/raw_data/dataset2_Faculty_entering --ranks 10,20 --thresholds 0.1,0.15
```

## Results

Files in `results/<dataset_name>/`:
- `summary_metrics_*.csv` - Main results (FPS, reduction %, filter accuracy)
- `per_frame_log_*.csv` - Detailed per-frame data
- `baseline_metrics_*.csv` - Pure AI baseline
- `comparison_metrics_*.csv` - Improvements vs baseline
- `parameter_sweep_*.csv` - All configs table

## Metrics

**Filter Accuracy** = (TP + TN) / Total  
→ How well the filter decides which frames need AI

**Reduction %** = % of AI calls avoided  
**FPS** = Processing speed  
**Precision/Recall/F1** = Detection quality (needs ground truth)

## Ground Truth

```bash
# Generate template (saved to data/labels/)
python create_ground_truth_template.py --dataset data/raw_data/dataset2_Faculty_entering

# Edit CSV in data/labels/ to mark novel frames (is_novel=1)
# Then run tests
```

## Analyze

```bash
python analyze_results.py --results results/dataset2_Faculty_entering

# Or specify labels directly
python analyze_results.py --results results/dataset2_Faculty_entering --labels data/labels/dataset2_Faculty_entering_labels.csv
```
