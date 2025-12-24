"""
SVD-based reconstruction error computation for novelty detection.
"""

import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
import numpy as np
from typing import Optional
import logging

logger = logging.getLogger(__name__)


class SVDNoveltyDetector:
    """SVD-based reconstruction error computation for novelty detection."""
    
    def __init__(self, compression_rank: int = 20):
        """
        Initialize SVD detector.
        
        Args:
            compression_rank: Number of singular values to retain
        """
        self.compression_rank = compression_rank
    
    def compute_svd_reconstruction(self, frame: np.ndarray) -> tuple:
        """
        Compute SVD reconstruction of a frame.
        
        Uses compact SVD: frame ≈ U_k @ diag(S_k) @ V_k^T
        
        Args:
            frame: 2D grayscale image (height, width)
            
        Returns:
            Tuple of (reconstructed_frame, singular_values)
        """
        # Ensure 2D input
        if frame.ndim != 2:
            raise ValueError(f"Expected 2D frame, got shape {frame.shape}")
        
        # Compute SVD
        U, S, Vt = np.linalg.svd(frame, full_matrices=False)
        
        # Truncate to compression rank
        k = min(self.compression_rank, len(S))
        U_k = U[:, :k]
        S_k = S[:k]
        Vt_k = Vt[:k, :]
        
        # Reconstruct
        frame_reconstructed = U_k @ np.diag(S_k) @ Vt_k
        
        return frame_reconstructed, S_k
    
    def compute_reconstruction_error(self, frame: np.ndarray, 
                                     frame_reconstructed: np.ndarray) -> float:
        """
        Compute Frobenius norm reconstruction error.
        
        Error = ||frame - frame_reconstructed||_F / ||frame||_F
        (Normalized for scale-invariance)
        
        Args:
            frame: Original frame
            frame_reconstructed: SVD reconstructed frame
            
        Returns:
            Normalized reconstruction error (float)
        """
        diff = frame - frame_reconstructed
        error = np.linalg.norm(diff, 'fro')
        frame_norm = np.linalg.norm(frame, 'fro')
        
        if frame_norm == 0:
            return 0.0
        
        return error / frame_norm
    
    def compare_frames(self, current_frame: np.ndarray) -> float:
        """
        Compute reconstruction error for a single frame.
        
        Args:
            current_frame: Current frame to analyze
            
        Returns:
            Reconstruction error score
        """
        frame_reconstructed, _ = self.compute_svd_reconstruction(current_frame)
        error = self.compute_reconstruction_error(current_frame, frame_reconstructed)
        return error, frame_reconstructed
    
    def plot_reconstruction(self, frame: np.ndarray, frame_reconstructed: np.ndarray, 
                           error: float, save_path: Optional[str] = None) -> None:
        """
        Visualize original, reconstruction, and difference.
        
        Args:
            frame: Original frame
            frame_reconstructed: Reconstructed frame
            error: Reconstruction error value
            save_path: Optional path to save figure
        """
        fig = plt.figure(figsize=(14, 4))
        gs = GridSpec(1, 3, figure=fig, hspace=0.3, wspace=0.3)
        
        # Original
        ax1 = fig.add_subplot(gs[0, 0])
        im1 = ax1.imshow(frame, cmap='gray')
        ax1.set_title('Original Frame')
        ax1.axis('off')
        plt.colorbar(im1, ax=ax1)
        
        # Reconstruction
        ax2 = fig.add_subplot(gs[0, 1])
        im2 = ax2.imshow(frame_reconstructed, cmap='gray')
        ax2.set_title(f'SVD Reconstruction (rank={self.compression_rank})')
        ax2.axis('off')
        plt.colorbar(im2, ax=ax2)
        
        # Difference
        diff = np.abs(frame - frame_reconstructed)
        ax3 = fig.add_subplot(gs[0, 2])
        im3 = ax3.imshow(diff, cmap='hot')
        ax3.set_title(f'Reconstruction Error\nNorm = {error:.4f}')
        ax3.axis('off')
        plt.colorbar(im3, ax=ax3)
        
        plt.suptitle(f'SVD Novelty Detection (τ threshold)', fontsize=14, fontweight='bold')
        
        if save_path:
            plt.savefig(save_path, dpi=100, bbox_inches='tight')
            logger.info(f"Saved plot to {save_path}")
        else:
            plt.show()
        
        plt.close()
