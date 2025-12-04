"""
Quick verification script to test the testing framework installation.
Runs a minimal test with mock AI to verify all components work.

Usage:
    python verify_setup.py
"""

import sys
import logging
from pathlib import Path

logging.basicConfig(
    level=logging.INFO,
    format='%(levelname)s: %(message)s'
)
logger = logging.getLogger(__name__)


def check_dependencies():
    """Check if required packages are installed."""
    logger.info("Checking dependencies...")
    
    required_packages = [
        ('numpy', 'numpy'),
        ('cv2', 'opencv-python'),
        ('psutil', 'psutil'),
        ('skimage', 'scikit-image'),
        ('matplotlib', 'matplotlib')
    ]
    
    missing = []
    for import_name, package_name in required_packages:
        try:
            __import__(import_name)
            logger.info(f"  ✓ {package_name}")
        except ImportError:
            logger.error(f"  ✗ {package_name} - NOT FOUND")
            missing.append(package_name)
    
    # YOLO is optional for mock tests
    try:
        from ultralytics import YOLO
        logger.info(f"  ✓ ultralytics (YOLO available)")
    except ImportError:
        logger.warning(f"  ⚠ ultralytics - NOT FOUND (mock AI will be used)")
    
    if missing:
        logger.error(f"\nMissing packages: {', '.join(missing)}")
        logger.error(f"Install with: pip install {' '.join(missing)}")
        return False
    
    logger.info("✓ All required dependencies installed\n")
    return True


def check_structure():
    """Check if directory structure is correct."""
    logger.info("Checking directory structure...")
    
    required_files = [
        'run_tests.py',
        'testing/__init__.py',
        'testing/baseline_runner.py',
        'testing/experiment_runner.py',
        'testing/ground_truth_loader.py',
        'testing/metrics_calculator.py',
        'testing/csv_logger.py',
        'testing/system_monitor.py',
        'ai/detection_ai_test1.py',
        'preprocessing/image_preprocessing_test1.py',
        'svd/svd_test2.py'
    ]
    
    missing = []
    for file_path in required_files:
        if Path(file_path).exists():
            logger.info(f"  ✓ {file_path}")
        else:
            logger.error(f"  ✗ {file_path} - NOT FOUND")
            missing.append(file_path)
    
    if missing:
        logger.error(f"\nMissing files: {', '.join(missing)}")
        return False
    
    logger.info("✓ All required files present\n")
    return True


def check_datasets():
    """Check if datasets are available."""
    logger.info("Checking datasets...")
    
    data_dir = Path('data')
    if not data_dir.exists():
        logger.warning("  ⚠ 'data' directory not found")
        logger.info("  Creating 'data' directory...")
        data_dir.mkdir()
        return False
    
    datasets = list(data_dir.glob('dataset*'))
    if not datasets:
        logger.warning("  ⚠ No datasets found in 'data' directory")
        logger.info("  Please add datasets to 'data' folder")
        return False
    
    for dataset in datasets:
        logger.info(f"  ✓ {dataset.name}")
    
    logger.info(f"✓ Found {len(datasets)} dataset(s)\n")
    return True


def run_quick_test():
    """Run a quick test with mock AI."""
    logger.info("Running quick verification test...")
    logger.info("  Using mock AI (no YOLO required)")
    logger.info("  Testing with minimal parameters\n")
    
    try:
        from testing import TestSuite
        
        # Find first available dataset
        data_dir = Path('data')
        datasets = list(data_dir.glob('dataset*'))
        
        if not datasets:
            logger.error("No datasets available for testing")
            return False
        
        dataset_path = str(datasets[0])
        logger.info(f"Using dataset: {dataset_path}\n")
        
        # Initialize test suite with mock AI
        suite = TestSuite(
            dataset_path=dataset_path,
            output_dir='results/verification_test',
            target_size=(120, 160),  # Small size for fast test
            use_mock_ai=True
        )
        
        # Run minimal test: 1 rank, 1 threshold
        logger.info("Starting test (this may take a minute)...\n")
        results = suite.run_full_test_suite(
            compression_ranks=[10],
            thresholds=[0.1]
        )
        
        # Verify output files were created
        output_dir = Path('results/verification_test')
        csv_files = list(output_dir.glob('*.csv'))
        
        logger.info("\n" + "=" * 60)
        logger.info("VERIFICATION TEST COMPLETE")
        logger.info("=" * 60)
        logger.info(f"✓ Generated {len(csv_files)} CSV files:")
        for csv_file in sorted(csv_files):
            logger.info(f"    {csv_file.name}")
        
        logger.info("\n✓ Testing framework is working correctly!")
        logger.info("\nNext steps:")
        logger.info("  1. Review generated CSV files in results/verification_test/")
        logger.info("  2. Create ground truth labels (see TESTING_README.md)")
        logger.info("  3. Run full tests: python run_tests.py --dataset data/your_dataset")
        logger.info("=" * 60 + "\n")
        
        return True
        
    except Exception as e:
        logger.error(f"\n✗ Test failed: {e}", exc_info=True)
        return False


def main():
    """Main verification workflow."""
    print("\n" + "=" * 60)
    print("SVD NOVELTY DETECTION TESTING FRAMEWORK")
    print("Setup Verification")
    print("=" * 60 + "\n")
    
    # Check dependencies
    if not check_dependencies():
        logger.error("\n⚠ Please install missing dependencies before continuing")
        sys.exit(1)
    
    # Check structure
    if not check_structure():
        logger.error("\n⚠ Directory structure is incomplete")
        sys.exit(1)
    
    # Check datasets
    has_datasets = check_datasets()
    
    # Run quick test
    if has_datasets:
        user_input = input("Run quick verification test? (y/n): ").strip().lower()
        if user_input == 'y':
            success = run_quick_test()
            sys.exit(0 if success else 1)
        else:
            logger.info("\nSkipping verification test")
            logger.info("Run manually with: python run_tests.py --dataset data/your_dataset --mock-ai")
    else:
        logger.warning("\n⚠ Cannot run verification test without datasets")
        logger.info("Add datasets to 'data' folder and run this script again")
    
    print("\n✓ Setup verification complete\n")


if __name__ == "__main__":
    main()
