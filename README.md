# SVD Novelty Detection - Testing Framework

Efficient video frame processing using SVD-based novelty detection to reduce AI inference calls.

## Quick Start

### Run Tests
```bash
# Basic test (4 configurations, ~30 seconds)
python test_pipeline.py --dataset data/raw_data/dataset2_Faculty_entering

# Full parameter sweep (25 configurations)
python test_pipeline.py --dataset data/raw_data/dataset2_Faculty_entering --full-sweep

# Custom parameters
python test_pipeline.py --dataset data/raw_data/dataset2_Faculty_entering --ranks 10,20,30 --thresholds 0.1,0.15,0.2

# Fast testing (mock AI, no real YOLO)
python test_pipeline.py --dataset data/raw_data/dataset2_Faculty_entering --mock-ai
```

### View Results
Results are saved in `results/<dataset_name>/`:
- `per_frame_log_*.csv` - Detailed per-frame metrics
- `summary_metrics_*.csv` - Aggregated results per configuration
- `baseline_metrics_*.csv` - Pure AI baseline performance
- `comparison_metrics_*.csv` - Filter vs baseline comparison
- `parameter_sweep_*.csv` - Overview of all configurations

### Analyze Results
```bash
python analyze_results.py --results results/dataset2_Faculty_entering
```

## Key Metrics

### Filter Accuracy
**Filter Accuracy = (TP + TN) / Total Frames**

Measures how well the SVD filter decides which frames need AI processing:
- **TP** (True Positive): Novel frame → Filter detected ✅
- **TN** (True Negative): Non-novel frame → Filter ignored ✅
- **FP** (False Positive): Non-novel frame → Filter detected ❌
- **FN** (False Negative): Novel frame → Filter missed ⚠️

### Performance Metrics
- **FPS**: Processing speed (frames per second)
- **Reduction %**: Percentage of AI calls avoided
- **CPU/Memory**: Resource usage
- **Precision/Recall/F1**: Detection quality

## Project Structure

```
data/
├── raw_data/              # Datasets (images/videos)
│   ├── dataset1_test/
│   ├── dataset2_Faculty_entering/
│   └── dataset3_evening_university_walk/
└── labels/                # Ground truth labels
    ├── dataset1_test_labels.csv
    └── dataset2_Faculty_entering_labels.csv

testing/                   # Core framework (7 modules)
├── baseline_runner.py     # Pure AI baseline
├── experiment_runner.py   # SVD filter experiments
├── ground_truth_loader.py # Load truth labels
├── metrics_calculator.py  # Compute metrics
├── csv_logger.py         # Save CSV outputs
├── system_monitor.py     # CPU/memory tracking
└── __init__.py

test_pipeline.py          # Main entry point ⭐
run_tests.py             # Test orchestrator
analyze_results.py       # Post-processing analysis
create_ground_truth_template.py  # Generate label templates
verify_setup.py          # Check installation
```

## Ground Truth Labels

Create labels to enable quality metrics (precision, recall, F1):

```bash
# Generate template (saved to data/labels/)
python create_ground_truth_template.py --dataset data/raw_data/dataset2_Faculty_entering

# Edit the CSV file in data/labels/ to mark novel frames (is_novel=1)
# Then run tests - quality metrics will be calculated automatically
```

Format (saved in `data/labels/<dataset_name>_labels.csv`):
```csv
frame_index,is_novel
0,0
1,0
12,1  ← Novel event
13,1
14,1
15,0
```

## Installation

```bash
pip install -r requirements_testing.txt
```

## Documentation

- **README.md** - This file (overview and quick start)
- **TESTING.md** - Testing guide (detailed commands and metrics)

## Example Results

With SVD filter (Rank=10, Threshold=0.15):
- **99.15% reduction** in AI calls (118 → 1)
- **31% faster** processing
- **94% CPU savings**
- **Filter Accuracy**: 86.55%

## License

See LICENSE file.