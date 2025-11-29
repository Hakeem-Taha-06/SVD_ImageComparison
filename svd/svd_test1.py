import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
from typing import Tuple, Optional, Dict
from skimage.metrics import structural_similarity as ssim


class SVDImageComparator:
    """
    A class for comparing images using Singular Value Decomposition (SVD).
    This version properly uses SVD-based metrics as the primary comparison method.
    """
    
    def __init__(self):
        self.image1_array = None
        self.image2_array = None
        self.svd1 = None
        self.svd2 = None
        
    def load_image(self, image_path: str, grayscale: bool = True) -> np.ndarray:
        """Load an image from file and convert to numpy array."""
        img = Image.open(image_path)
        if grayscale:
            img = img.convert('L')
        return np.array(img, dtype=np.float64)
    
    def resize_images(self, img1: np.ndarray, img2: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """Resize images to match dimensions for comparison."""
        if img1.shape != img2.shape:
            target_shape = (
                min(img1.shape[0], img2.shape[0]),
                min(img1.shape[1], img2.shape[1])
            )
            
            img1_pil = Image.fromarray(img1.astype(np.uint8))
            img2_pil = Image.fromarray(img2.astype(np.uint8))
            
            img1_pil = img1_pil.resize((target_shape[1], target_shape[0]))
            img2_pil = img2_pil.resize((target_shape[1], target_shape[0]))
            
            img1 = np.array(img1_pil, dtype=np.float64)
            img2 = np.array(img2_pil, dtype=np.float64)
        
        return img1, img2
    
    def compute_svd(self, image_array: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Compute the Singular Value Decomposition of an image."""
        U, S, Vt = np.linalg.svd(image_array, full_matrices=False)
        return U, S, Vt
    
    def reconstruct_image(self, U: np.ndarray, S: np.ndarray, Vt: np.ndarray, 
                         k: Optional[int] = None) -> np.ndarray:
        """Reconstruct image using top k singular values."""
        if k is None:
            k = len(S)
        
        S_k = np.diag(S[:k])
        U_k = U[:, :k]
        Vt_k = Vt[:k, :]
        
        reconstructed = U_k @ S_k @ Vt_k
        return np.clip(reconstructed, 0, 255)
    
    def compare_singular_values(self, S1: np.ndarray, S2: np.ndarray, k: int) -> Dict[str, float]:
        """
        Compare singular values using multiple methods.
        Returns individual scores for analysis.
        """
        k = min(k, len(S1), len(S2))
        S1_k = S1[:k]
        S2_k = S2[:k]
        
        # Method 1: Cosine similarity
        S1_normalized = S1_k / np.linalg.norm(S1_k)
        S2_normalized = S2_k / np.linalg.norm(S2_k)
        cosine_sim = np.clip(np.dot(S1_normalized, S2_normalized), 0.0, 1.0)
        
        # Method 2: Energy ratio comparison
        total_energy_1 = np.sum(S1**2)
        total_energy_2 = np.sum(S2**2)
        energy_1 = np.sum(S1_k**2) / total_energy_1
        energy_2 = np.sum(S2_k**2) / total_energy_2
        energy_similarity = 1 - abs(energy_1 - energy_2)
        
        # Method 3: L2 distance between normalized singular values
        l2_dist = np.linalg.norm(S1_normalized - S2_normalized)
        l2_similarity = 1 - (l2_dist / np.sqrt(2))  # Normalize to [0, 1]
        
        return {
            'cosine': cosine_sim,
            'energy': energy_similarity,
            'l2': max(0, l2_similarity)
        }
    
    def compare_subspaces(self, U1: np.ndarray, U2: np.ndarray, k: int) -> Dict[str, float]:
        """
        Compare the subspaces spanned by top-k left singular vectors.
        This is crucial for SVD-based comparison!
        """
        k = min(k, U1.shape[1], U2.shape[1])
        U1_k = U1[:, :k]
        U2_k = U2[:, :k]
        
        # Method 1: Frobenius norm of projection matrix difference
        P1 = U1_k @ U1_k.T
        P2 = U2_k @ U2_k.T
        proj_diff = np.linalg.norm(P1 - P2, 'fro')
        max_proj_diff = np.sqrt(2 * U1.shape[0])  # Theoretical maximum
        proj_similarity = 1 - (proj_diff / max_proj_diff)
        
        # Method 2: Principal angles between subspaces
        cross = U1_k.T @ U2_k
        sigma = np.linalg.svd(cross, compute_uv=False)
        sigma = np.clip(sigma, 0, 1)
        principal_angles = np.arccos(sigma)
        
        # Convert angles to similarity (0 angle = identical subspaces)
        angle_similarity = 1 - (np.mean(principal_angles) / (np.pi / 2))
        
        return {
            'projection': max(0, proj_similarity),
            'principal_angles': max(0, angle_similarity),
            'angles_rad': principal_angles
        }
    
    def compare_right_subspaces(self, Vt1: np.ndarray, Vt2: np.ndarray, k: int) -> Dict[str, float]:
        """
        Compare the subspaces spanned by top-k right singular vectors.
        """
        k = min(k, Vt1.shape[0], Vt2.shape[0])
        V1_k = Vt1[:k, :].T  # Transpose to get V
        V2_k = Vt2[:k, :].T
        
        # Use same methods as left subspace comparison
        P1 = V1_k @ V1_k.T
        P2 = V2_k @ V2_k.T
        proj_diff = np.linalg.norm(P1 - P2, 'fro')
        max_proj_diff = np.sqrt(2 * V1_k.shape[0])
        proj_similarity = 1 - (proj_diff / max_proj_diff)
        
        return {
            'projection': max(0, proj_similarity)
        }
    
    def compare_reconstructions(self, U1: np.ndarray, S1: np.ndarray, Vt1: np.ndarray,
                               U2: np.ndarray, S2: np.ndarray, Vt2: np.ndarray,
                               k: int) -> float:
        """
        Compare images via their k-rank SVD approximations.
        This tests if low-rank approximations are similar.
        """
        recon1 = self.reconstruct_image(U1, S1, Vt1, k)
        recon2 = self.reconstruct_image(U2, S2, Vt2, k)
        
        mse = np.mean((recon1 - recon2) ** 2)
        max_mse = 255 ** 2
        return 1 - (mse / max_mse)
    
    def compute_svd_similarity(self, S1: np.ndarray, S2: np.ndarray,
                              U1: np.ndarray, U2: np.ndarray,
                              Vt1: np.ndarray, Vt2: np.ndarray,
                              k: int) -> Dict[str, float]:
        """
        Compute comprehensive SVD-based similarity metrics.
        This is the MAIN similarity computation for SVD testing.
        """
        # Compare singular values
        sv_metrics = self.compare_singular_values(S1, S2, k)
        
        # Compare left subspaces (row space)
        left_metrics = self.compare_subspaces(U1, U2, k)
        
        # Compare right subspaces (column space)
        right_metrics = self.compare_right_subspaces(Vt1, Vt2, k)
        
        # Compare reconstructions
        recon_similarity = self.compare_reconstructions(U1, S1, Vt1, U2, S2, Vt2, k)
        
        # Combine SVD metrics with appropriate weights
        # Subspace comparison is most important, then reconstruction, then singular values
        svd_similarity = (
            0.35 * left_metrics['principal_angles'] +  # Left subspace
            0.35 * left_metrics['projection'] +        # Projection distance
            0.15 * sv_metrics['cosine'] +              # Singular value similarity
            0.15 * recon_similarity                    # Reconstruction similarity
        )
        
        return {
            'svd_similarity': np.clip(svd_similarity, 0.0, 1.0),
            'singular_value_cosine': sv_metrics['cosine'],
            'singular_value_energy': sv_metrics['energy'],
            'singular_value_l2': sv_metrics['l2'],
            'left_subspace_angles': left_metrics['principal_angles'],
            'left_subspace_projection': left_metrics['projection'],
            'right_subspace_projection': right_metrics['projection'],
            'reconstruction_similarity': recon_similarity
        }
    
    def compute_ssim(self, img1: np.ndarray, img2: np.ndarray) -> float:
        """Compute SSIM for reference comparison."""
        img1_uint8 = np.clip(img1, 0, 255).astype(np.uint8)
        img2_uint8 = np.clip(img2, 0, 255).astype(np.uint8)
        return ssim(img1_uint8, img2_uint8, data_range=255)
    
    def compare_images(self, image1_path: str, image2_path: str, 
                      k: Optional[int] = None) -> dict:
        """
        Main method to compare two images using SVD.
        SVD metrics are primary; other metrics included for validation only.
        
        Args:
            image1_path: Path to first image
            image2_path: Path to second image
            k: Number of singular values to use (default: 50)
            
        Returns:
            Dictionary with SVD similarity as primary metric
        """
        # Load and prepare images
        self.image1_array = self.load_image(image1_path)
        self.image2_array = self.load_image(image2_path)
        self.image1_array, self.image2_array = self.resize_images(
            self.image1_array, self.image2_array
        )
        
        # Compute SVD
        U1, S1, Vt1 = self.compute_svd(self.image1_array)
        U2, S2, Vt2 = self.compute_svd(self.image2_array)
        self.svd1 = (U1, S1, Vt1)
        self.svd2 = (U2, S2, Vt2)
        
        if k is None:
            k = 50
        k = min(k, len(S1), len(S2))
        
        # PRIMARY: SVD-based similarity
        svd_metrics = self.compute_svd_similarity(S1, S2, U1, U2, Vt1, Vt2, k)
        
        # REFERENCE ONLY: Traditional metrics for validation
        ssim_score = self.compute_ssim(self.image1_array, self.image2_array)
        mse = np.mean((self.image1_array - self.image2_array) ** 2)
        pixel_similarity = 1 - (mse / (255 ** 2))
        
        return {
            # PRIMARY METRIC
            'similarity_score': svd_metrics['svd_similarity'],
            
            # Detailed SVD metrics
            'svd_metrics': svd_metrics,
            
            # Reference metrics (for validation/comparison)
            'reference_metrics': {
                'ssim': ssim_score,
                'pixel_similarity': pixel_similarity,
                'mse': mse
            },
            
            # Metadata
            'k_used': k,
            'image_shape': self.image1_array.shape,
            'total_singular_values': len(S1)
        }
    
    def visualize_comparison(self, k_values: list = [10, 50, 100]):
        """Visualize original images and reconstructions."""
        if self.svd1 is None or self.svd2 is None:
            raise ValueError("Must run compare_images first")
        
        U1, S1, Vt1 = self.svd1
        U2, S2, Vt2 = self.svd2
        
        fig, axes = plt.subplots(2, len(k_values) + 1, figsize=(15, 6))
        
        axes[0, 0].imshow(self.image1_array, cmap='gray')
        axes[0, 0].set_title('Image 1 (Original)')
        axes[0, 0].axis('off')
        
        axes[1, 0].imshow(self.image2_array, cmap='gray')
        axes[1, 0].set_title('Image 2 (Original)')
        axes[1, 0].axis('off')
        
        for idx, k in enumerate(k_values, 1):
            recon1 = self.reconstruct_image(U1, S1, Vt1, k)
            axes[0, idx].imshow(recon1, cmap='gray')
            axes[0, idx].set_title(f'Image 1 (k={k})')
            axes[0, idx].axis('off')
            
            recon2 = self.reconstruct_image(U2, S2, Vt2, k)
            axes[1, idx].imshow(recon2, cmap='gray')
            axes[1, idx].set_title(f'Image 2 (k={k})')
            axes[1, idx].axis('off')
        
        plt.tight_layout()
        plt.show()
    
    def plot_singular_values(self, top_n: int = 100):
        """Plot singular values of both images for comparison."""
        if self.svd1 is None or self.svd2 is None:
            raise ValueError("Must run compare_images first")
        
        _, S1, _ = self.svd1
        _, S2, _ = self.svd2
        
        plt.figure(figsize=(12, 5))
        
        plt.subplot(1, 2, 1)
        plt.plot(S1[:top_n], label='Image 1', marker='o', markersize=3)
        plt.plot(S2[:top_n], label='Image 2', marker='s', markersize=3)
        plt.xlabel('Index')
        plt.ylabel('Singular Value')
        plt.title('Singular Values Comparison')
        plt.legend()
        plt.grid(True, alpha=0.3)
        
        plt.subplot(1, 2, 2)
        plt.semilogy(S1[:top_n], label='Image 1', marker='o', markersize=3)
        plt.semilogy(S2[:top_n], label='Image 2', marker='s', markersize=3)
        plt.xlabel('Index')
        plt.ylabel('Singular Value (log scale)')
        plt.title('Singular Values (Log Scale)')
        plt.legend()
        plt.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.show()

