from svd.svd_test1 import SVDImageComparator



if __name__ == "__main__":
    comparator = SVDImageComparator()
    
    results = comparator.compare_images('src/image1.jpg', 'src/image6.jpg', k=50)
    
    print(f"\n{'='*60}")
    print(f"SVD-BASED IMAGE SIMILARITY: {results['similarity_score']:.4f}")
    print(f"{'='*60}")
    
    print(f"\nDetailed SVD Metrics:")
    svd = results['svd_metrics']
    print(f"  Subspace Angles Similarity:  {svd['left_subspace_angles']:.4f}")
    print(f"  Subspace Projection:         {svd['left_subspace_projection']:.4f}")
    print(f"  Reconstruction Similarity:   {svd['reconstruction_similarity']:.4f}")
    print(f"  Singular Value Cosine:       {svd['singular_value_cosine']:.4f}")
    
    print(f"\nReference Metrics (for validation):")
    ref = results['reference_metrics']
    print(f"  SSIM:             {ref['ssim']:.4f}")
    print(f"  Pixel Similarity: {ref['pixel_similarity']:.4f}")
    
    print(f"\nInterpretation (SVD-based):")
    score = results['similarity_score']
    if score > 0.95:
        print("  → Images have nearly identical structure (SVD)")
    elif score > 0.85:
        print("  → Images have very similar structure (SVD)")
    elif score > 0.70:
        print("  → Images have moderately similar structure (SVD)")
    elif score > 0.50:
        print("  → Images have some structural similarities (SVD)")
    else:
        print("  → Images have different structure (SVD)")
    
    # Visualize
    comparator.visualize_comparison(k_values=[10, 50, 100])
    comparator.plot_singular_values(top_n=100)