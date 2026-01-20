import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle, Circle, Ellipse
from scipy.ndimage import gaussian_filter
import os

# ============================================
# CONFIGURATION
# ============================================
OUTPUT_DIR = "./mri_images"
NUM_SLICES = 30
IMAGE_SIZE = 512  # Higher resolution for better quality

os.makedirs(OUTPUT_DIR, exist_ok=True)

# ============================================
# Generate Realistic Brain MRI Slice
# ============================================
def generate_realistic_brain_mri(slice_num, total_slices):
    """
    Generate a realistic brain MRI slice with anatomical features
    """
    img = np.zeros((IMAGE_SIZE, IMAGE_SIZE))
    
    # Calculate position in brain (0 = bottom, 1 = top)
    z_position = slice_num / total_slices
    
    # Create coordinate grids
    y, x = np.ogrid[:IMAGE_SIZE, :IMAGE_SIZE]
    center_y, center_x = IMAGE_SIZE // 2, IMAGE_SIZE // 2
    
    # === SKULL (outer boundary) ===
    skull_radius_y = 220
    skull_radius_x = 200
    skull_mask = ((y - center_y)**2 / skull_radius_y**2 + 
                  (x - center_x)**2 / skull_radius_x**2) <= 1
    img[skull_mask] = 40  # Dark gray for skull
    
    # === BRAIN TISSUE ===
    # Brain gets smaller near top and bottom
    brain_scale = 1.0 - 0.4 * abs(z_position - 0.5) * 2
    brain_radius_y = int(180 * brain_scale)
    brain_radius_x = int(165 * brain_scale)
    
    brain_mask = ((y - center_y)**2 / brain_radius_y**2 + 
                  (x - center_x)**2 / brain_radius_x**2) <= 1
    
    # White matter (lighter)
    img[brain_mask] = 160
    
    # Gray matter (darker, outer layer)
    gray_matter_outer = ((y - center_y)**2 / brain_radius_y**2 + 
                        (x - center_x)**2 / brain_radius_x**2) <= 1
    gray_matter_inner = ((y - center_y)**2 / (brain_radius_y*0.85)**2 + 
                        (x - center_x)**2 / (brain_radius_x*0.85)**2) <= 1
    gray_matter_mask = gray_matter_outer & ~gray_matter_inner
    img[gray_matter_mask] = 120
    
    # === VENTRICLES (dark fluid-filled spaces) ===
    if 0.35 < z_position < 0.75:
        # Left ventricle
        ventricle_left_y = center_y - 20
        ventricle_left_x = center_x - 40
        ventricle_size = 25 * (1 - abs(z_position - 0.55) * 2)
        
        left_vent = ((y - ventricle_left_y)**2 + (x - ventricle_left_x)**2) <= ventricle_size**2
        img[left_vent] = 20
        
        # Right ventricle
        ventricle_right_y = center_y - 20
        ventricle_right_x = center_x + 40
        right_vent = ((y - ventricle_right_y)**2 + (x - ventricle_right_x)**2) <= ventricle_size**2
        img[right_vent] = 20
    
    # === CORPUS CALLOSUM (connects hemispheres) ===
    if 0.4 < z_position < 0.7:
        cc_y_start = center_y - 30
        cc_y_end = center_y - 20
        cc_x_start = center_x - 50
        cc_x_end = center_x + 50
        
        corpus_callosum = ((y >= cc_y_start) & (y <= cc_y_end) & 
                          (x >= cc_x_start) & (x <= cc_x_end))
        img[corpus_callosum] = 180
    
    # === TUMOR/LESION (for clinical relevance) ===
    # Add a small abnormality in middle slices
    if 0.45 < z_position < 0.55:
        tumor_y = center_y + 40
        tumor_x = center_x - 50
        tumor_radius = 18
        
        tumor_mask = ((y - tumor_y)**2 + (x - tumor_x)**2) <= tumor_radius**2
        img[tumor_mask] = 255  # Bright white (hyperintense)
        
        # Add edema (swelling) around tumor
        edema_mask = ((y - tumor_y)**2 + (x - tumor_x)**2) <= (tumor_radius * 2)**2
        edema_mask = edema_mask & ~tumor_mask
        img[edema_mask] = np.maximum(img[edema_mask], 90)
    
    # === CEREBELLUM (back of brain, lower slices) ===
    if z_position < 0.4:
        cerebellum_y = center_y + 80
        cerebellum_radius = 60 * (0.4 - z_position) / 0.4
        cerebellum_mask = ((y - cerebellum_y)**2 + (x - center_x)**2) <= cerebellum_radius**2
        img[cerebellum_mask] = 140
    
    # === Add realistic noise and texture ===
    noise = np.random.randn(IMAGE_SIZE, IMAGE_SIZE) * 8
    img = img + noise
    
    # Apply Gaussian blur for realistic MRI appearance
    img = gaussian_filter(img, sigma=1.5)
    
    # Clip values
    img = np.clip(img, 0, 255)
    
    return img.astype(np.uint8)

# ============================================
# Generate Enhanced MRI Stack
# ============================================
def generate_enhanced_mri_stack():
    """
    Create a complete MRI stack with metadata overlay
    """
    print(f"Generating {NUM_SLICES} high-quality MRI slices...")
    
    for i in range(NUM_SLICES):
        # Generate brain slice
        brain_img = generate_realistic_brain_mri(i, NUM_SLICES)
        
        # Create figure with medical information overlay
        fig, ax = plt.subplots(figsize=(8, 8), facecolor='black')
        ax.imshow(brain_img, cmap='gray', aspect='auto')
        ax.axis('off')
        
        # Add DICOM-style annotations
        slice_position = f"Slice {i+1}/{NUM_SLICES}"
        z_position_mm = f"Z: {(i - NUM_SLICES//2) * 5} mm"
        
        # Top-left: Patient info
        ax.text(20, 30, "PATIENT: DEMO-001", 
                color='cyan', fontsize=10, family='monospace', weight='bold')
        ax.text(20, 50, "BRAIN AXIAL T1", 
                color='cyan', fontsize=9, family='monospace')
        ax.text(20, 70, slice_position, 
                color='white', fontsize=10, family='monospace', weight='bold')
        
        # Top-right: Technical info
        ax.text(IMAGE_SIZE - 20, 30, "3T MRI", 
                color='yellow', fontsize=9, family='monospace', ha='right')
        ax.text(IMAGE_SIZE - 20, 50, "TR: 500ms", 
                color='yellow', fontsize=8, family='monospace', ha='right')
        ax.text(IMAGE_SIZE - 20, 70, z_position_mm, 
                color='white', fontsize=9, family='monospace', ha='right')
        
        # Bottom-left: Orientation markers
        ax.text(20, IMAGE_SIZE - 20, "L", 
                color='red', fontsize=14, family='sans-serif', weight='bold')
        ax.text(IMAGE_SIZE - 40, IMAGE_SIZE - 20, "R", 
                color='red', fontsize=14, family='sans-serif', weight='bold')
        
        # Bottom-right: Scale bar
        scale_start_x = IMAGE_SIZE - 120
        scale_end_x = IMAGE_SIZE - 20
        scale_y = IMAGE_SIZE - 40
        ax.plot([scale_start_x, scale_end_x], [scale_y, scale_y], 
                color='white', linewidth=2)
        ax.text((scale_start_x + scale_end_x)/2, scale_y - 15, "50mm", 
                color='white', fontsize=8, ha='center', family='monospace')
        
        # Highlight tumor in certain slices
        if 13 <= i <= 16:
            ax.text(IMAGE_SIZE//2, 480, "LESION DETECTED", 
                   color='red', fontsize=12, family='sans-serif', 
                   weight='bold', ha='center',
                   bbox=dict(boxstyle='round', facecolor='black', alpha=0.7))
        
        plt.tight_layout(pad=0)
        
        # Save
        filename = os.path.join(OUTPUT_DIR, f"brain_slice_{i:03d}.png")
        plt.savefig(filename, facecolor='black', dpi=100, bbox_inches='tight', pad_inches=0.1)
        plt.close()
        
        if (i + 1) % 5 == 0:
            print(f"   Generated {i + 1}/{NUM_SLICES} slices")
    
    print(f"\nEnhanced MRI stack saved to: {OUTPUT_DIR}")

# ============================================
# Generate Sample CT Scan (Alternative)
# ============================================
def generate_ct_scan_slice(slice_num, total_slices):
    """
    Generate a CT scan style image (different from MRI)
    """
    img = np.zeros((IMAGE_SIZE, IMAGE_SIZE))
    
    z_position = slice_num / total_slices
    
    y, x = np.ogrid[:IMAGE_SIZE, :IMAGE_SIZE]
    center_y, center_x = IMAGE_SIZE // 2, IMAGE_SIZE // 2
    
    # Bone (very bright in CT)
    skull_radius_y = 220
    skull_radius_x = 200
    skull_thickness = 15
    
    skull_outer = ((y - center_y)**2 / skull_radius_y**2 + 
                   (x - center_x)**2 / skull_radius_x**2) <= 1
    skull_inner = ((y - center_y)**2 / (skull_radius_y-skull_thickness)**2 + 
                   (x - center_x)**2 / (skull_radius_x-skull_thickness)**2) <= 1
    
    skull_mask = skull_outer & ~skull_inner
    img[skull_mask] = 240  # Bone is very bright in CT
    
    # Brain tissue (medium gray)
    brain_scale = 1.0 - 0.3 * abs(z_position - 0.5) * 2
    brain_radius_y = int(180 * brain_scale)
    brain_radius_x = int(165 * brain_scale)
    
    brain_mask = ((y - center_y)**2 / brain_radius_y**2 + 
                  (x - center_x)**2 / brain_radius_x**2) <= 1
    img[brain_mask] = 100
    
    # CSF (darker)
    if 0.3 < z_position < 0.7:
        ventricle_size = 30 * (1 - abs(z_position - 0.5) * 2)
        left_vent = ((y - (center_y - 20))**2 + (x - (center_x - 40))**2) <= ventricle_size**2
        right_vent = ((y - (center_y - 20))**2 + (x - (center_x + 40))**2) <= ventricle_size**2
        img[left_vent | right_vent] = 10
    
    # Add CT noise
    noise = np.random.randn(IMAGE_SIZE, IMAGE_SIZE) * 12
    img = img + noise
    img = gaussian_filter(img, sigma=0.8)
    img = np.clip(img, 0, 255)
    
    return img.astype(np.uint8)

# ============================================
# MAIN
# ============================================
if __name__ == "__main__":
    print("=" * 60)
    print("Generating Clinical-Grade MRI Images")
    print("=" * 60)
    
    # Choose which to generate
    print("\nSelect image type:")
    print("1. Enhanced Brain MRI (with tumor/lesion)")
    print("2. CT Scan")
    
    choice = input("Enter choice (1 or 2, default=1): ").strip() or "1"
    
    if choice == "2":
        print("\nGenerating CT scan slices...")
        for i in range(NUM_SLICES):
            ct_img = generate_ct_scan_slice(i, NUM_SLICES)
            
            fig, ax = plt.subplots(figsize=(8, 8), facecolor='black')
            ax.imshow(ct_img, cmap='bone', aspect='auto')
            ax.axis('off')
            
            # CT annotations
            ax.text(20, 30, "CT BRAIN", color='cyan', fontsize=10, family='monospace', weight='bold')
            ax.text(20, 50, f"Slice {i+1}/{NUM_SLICES}", color='white', fontsize=10, family='monospace')
            ax.text(IMAGE_SIZE - 20, 30, "120 kV", color='yellow', fontsize=9, family='monospace', ha='right')
            
            plt.tight_layout(pad=0)
            filename = os.path.join(OUTPUT_DIR, f"brain_slice_{i:03d}.png")
            plt.savefig(filename, facecolor='black', dpi=100, bbox_inches='tight', pad_inches=0.1)
            plt.close()
            
            if (i + 1) % 5 == 0:
                print(f"   Generated {i + 1}/{NUM_SLICES} slices")
    else:
        generate_enhanced_mri_stack()
    
    print("\n" + "=" * 60)
    print("Medical imaging generation complete!")
    print("=" * 60)
    print("\nUse Case Examples:")
    print("   • SLIDE_RIGHT/LEFT: Navigate through brain slices")
    print("   • SLIDE_UP/DOWN: Zoom to examine lesion detail")
    print("   • GRASP/RELEASE: Rotate for different viewing angles")
    print("   • Perfect for sterile surgical planning!")
    print("=" * 60)
