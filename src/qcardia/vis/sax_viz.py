"""SAX-specific visualization utilities for multi-slice cardiac segmentation.

This module provides comprehensive visualization and quantitative analysis tools
specifically designed for short-axis (SAX) cardiac cine data, including:
- Total volume curves across all slices
- Per-slice volume heatmaps
- 3D animated rendering of the cardiac cycle
"""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation, PillowWriter
from matplotlib import colormaps
from pathlib import Path
import warnings
warnings.filterwarnings('ignore')

# Physical spacing constants (in mm)
PIXEL_SPACING_MM = 1.25  # In-plane resolution
SLICE_SPACING_MM = 10.0  # Between slices

# Structure colors matching standard cardiac visualization
STRUCTURE_COLORS = {
    0: (0, 0, 0),           # Background - black
    1: (0.8, 0.2, 0.2),     # LV - red
    2: (0.2, 0.8, 0.2),     # MYO - green  
    3: (0.2, 0.2, 0.8),     # RV - blue
}

STRUCTURE_NAMES = {
    1: "LV",
    2: "MYO",
    3: "RV"
}


def compute_sax_volume_curves(segmentation):
    """Compute total volume curves for all structures across all slices.
    
    Args:
        segmentation: (n_slices, n_frames, H, W) segmentation array
        
    Returns:
        dict: {structure_id: volume_curve_array} where each curve is (n_frames,) in mL
    """
    n_slices, n_frames, height, width = segmentation.shape
    
    # Voxel volume in mm³
    voxel_volume_mm3 = PIXEL_SPACING_MM * PIXEL_SPACING_MM * SLICE_SPACING_MM
    # Convert mm³ to mL (1 mL = 1000 mm³)
    voxel_volume_ml = voxel_volume_mm3 / 1000.0
    
    volume_curves = {}
    
    for structure_id in [1, 2, 3]:
        # Count voxels for this structure at each timepoint (summing across all slices)
        structure_mask = (segmentation == structure_id)
        voxel_counts = structure_mask.sum(axis=(0, 2, 3))  # Sum over slices, H, W
        
        # Convert to volume in mL
        volume_curves[structure_id] = voxel_counts * voxel_volume_ml
    
    return volume_curves


def compute_sax_volume_heatmap_data(segmentation):
    """Compute per-slice, per-frame volume data for heatmap visualization.
    
    Args:
        segmentation: (n_slices, n_frames, H, W) segmentation array
        
    Returns:
        dict: {structure_id: heatmap_array} where each array is (n_slices, n_frames) in mL
    """
    n_slices, n_frames, height, width = segmentation.shape
    
    # Voxel volume in mm³ (for a single slice)
    voxel_volume_mm3 = PIXEL_SPACING_MM * PIXEL_SPACING_MM * SLICE_SPACING_MM
    voxel_volume_ml = voxel_volume_mm3 / 1000.0
    
    heatmap_data = {}
    
    for structure_id in [1, 2, 3]:
        # Count voxels for this structure at each slice and timepoint
        structure_mask = (segmentation == structure_id)
        voxel_counts = structure_mask.sum(axis=(2, 3))  # Sum over H, W -> (n_slices, n_frames)
        
        # Convert to volume in mL
        heatmap_data[structure_id] = voxel_counts * voxel_volume_ml
    
    return heatmap_data


def plot_sax_total_volume_curves(volume_curves, chamber_type, model_name, output_path):
    """Plot total volume over time for all structures with ED/ES markers.
    
    Args:
        volume_curves: dict from compute_sax_volume_curves()
        chamber_type: str, e.g., "CINE_SAX"
        model_name: str, model identifier
        output_path: Path object for saving figure
    """
    fig, ax = plt.subplots(figsize=(12, 6))
    
    n_frames = len(volume_curves[1])
    frames = np.arange(n_frames)
    
    # Plot curves for each structure
    for structure_id in [1, 2, 3]:
        volumes = volume_curves[structure_id]
        color = STRUCTURE_COLORS[structure_id]
        name = STRUCTURE_NAMES[structure_id]
        
        ax.plot(frames, volumes, 'o-', color=color, linewidth=2, 
                markersize=4, label=name, alpha=0.8)
        
        # Mark ED (end-diastole = max volume) and ES (end-systole = min volume)
        if structure_id in [1, 3]:  # LV and RV chambers
            ed_idx = np.argmax(volumes)
            es_idx = np.argmin(volumes)
            
            ax.plot(ed_idx, volumes[ed_idx], 's', color=color, 
                   markersize=10, markeredgewidth=2, markeredgecolor='black',
                   markerfacecolor=color, alpha=0.9)
            ax.plot(es_idx, volumes[es_idx], 's', color=color,
                   markersize=10, markeredgewidth=2, markeredgecolor='black',
                   markerfacecolor='white', alpha=0.9)
            
            # Compute ejection fraction
            edv = volumes[ed_idx]
            esv = volumes[es_idx]
            ef = ((edv - esv) / edv * 100) if edv > 0 else 0
            
            # Add text annotation
            ax.text(0.02, 0.98 - (structure_id-1)*0.08, 
                   f'{name}: EDV={edv:.1f}mL, ESV={esv:.1f}mL, EF={ef:.1f}%',
                   transform=ax.transAxes, fontsize=10, 
                   verticalalignment='top',
                   bbox=dict(boxstyle='round', facecolor=color, alpha=0.3))
    
    ax.set_xlabel('Frame', fontsize=12, weight='bold')
    ax.set_ylabel('Volume (mL)', fontsize=12, weight='bold')
    ax.set_title(f'{chamber_type} - Total Volume Over Time\n{model_name}', 
                fontsize=14, weight='bold', pad=15)
    ax.legend(loc='upper right', fontsize=11, framealpha=0.95)
    ax.grid(True, alpha=0.3, linestyle='--')
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()


def plot_sax_volume_heatmaps(heatmap_data, chamber_type, model_name, output_path):
    """Plot volume heatmaps (slice × frame) for each structure.
    
    Args:
        heatmap_data: dict from compute_sax_volume_heatmap_data()
        chamber_type: str
        model_name: str
        output_path: Path object for saving figure
    """
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    
    for idx, structure_id in enumerate([1, 2, 3]):
        ax = axes[idx]
        data = heatmap_data[structure_id]
        name = STRUCTURE_NAMES[structure_id]
        
        # Create heatmap
        im = ax.imshow(data, aspect='auto', cmap='viridis', interpolation='nearest')
        
        # Add colorbar
        cbar = plt.colorbar(im, ax=ax)
        cbar.set_label('Volume (mL)', fontsize=10)
        
        # Labels
        ax.set_xlabel('Frame', fontsize=11, weight='bold')
        ax.set_ylabel('Slice (base → apex)', fontsize=11, weight='bold')
        ax.set_title(f'{name} Volume Distribution', fontsize=12, weight='bold', pad=10)
        
        # Set ticks
        n_slices, n_frames = data.shape
        ax.set_xticks(np.linspace(0, n_frames-1, min(10, n_frames)))
        ax.set_yticks(np.arange(0, n_slices, max(1, n_slices//10)))
    
    fig.suptitle(f'{chamber_type} - Volume Heatmaps (Slice × Frame)\n{model_name}',
                fontsize=14, weight='bold', y=1.02)
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()


def create_sax_3d_animation(segmentation, chamber_type, model_name, output_path, fps=10):
    """Create 3D animated visualization of SAX segmentation stack through cardiac cycle.
    
    Args:
        segmentation: (n_slices, n_frames, H, W) segmentation array
        chamber_type: str
        model_name: str
        output_path: Path object for saving GIF
        fps: int, frames per second for animation
    """
    from mpl_toolkits.mplot3d import Axes3D
    
    n_slices, n_frames, height, width = segmentation.shape
    
    # Create figure with 3D axis
    fig = plt.figure(figsize=(10, 10))
    ax = fig.add_subplot(111, projection='3d')
    
    # Physical spacing
    pixel_spacing = PIXEL_SPACING_MM
    slice_spacing = SLICE_SPACING_MM
    
    # Downsample for performance (voxels are computationally expensive)
    downsample_factor = 4
    
    # Find bounding box across all frames to center the view
    print(f"    Computing bounding box for centering...")
    all_structures = (segmentation > 0)  # Any structure
    z_coords, t_coords, y_coords, x_coords = np.where(all_structures)
    
    if len(z_coords) == 0:
        print(f"    Warning: No structures found in segmentation")
        return
    
    # Get bounds with some padding
    z_min, z_max = z_coords.min(), z_coords.max()
    y_min, y_max = y_coords.min(), y_coords.max()
    x_min, x_max = x_coords.min(), x_coords.max()
    
    # Add padding (10% on each side)
    padding = 0.1
    z_range = z_max - z_min
    y_range = y_max - y_min
    x_range = x_max - x_min
    
    z_pad = max(1, int(z_range * padding))
    y_pad = max(10, int(y_range * padding))
    x_pad = max(10, int(x_range * padding))
    
    z_min = max(0, z_min - z_pad)
    z_max = min(n_slices - 1, z_max + z_pad)
    y_min = max(0, y_min - y_pad)
    y_max = min(height - 1, y_max + y_pad)
    x_min = max(0, x_min - x_pad)
    x_max = min(width - 1, x_max + x_pad)
    
    print(f"    Bounding box: z=[{z_min}:{z_max}], y=[{y_min}:{y_max}], x=[{x_min}:{x_max}]")
    
    def animate_frame(frame_idx):
        ax.clear()
        
        # Get current frame data with cropping and downsampling
        frame_data_full = segmentation[z_min:z_max+1, frame_idx, y_min:y_max+1, x_min:x_max+1]
        frame_data = frame_data_full[::1, ::downsample_factor, ::downsample_factor]
        
        n_z, ds_height, ds_width = frame_data.shape
        
        # Create color array for voxels
        filled = np.zeros((n_z, ds_height, ds_width), dtype=bool)
        colors = np.zeros((n_z, ds_height, ds_width, 4), dtype=float)
        
        # Fill voxels for each structure
        for structure_id in [1, 2, 3]:
            mask = (frame_data == structure_id)
            filled |= mask
            
            # Set colors where this structure exists
            color = STRUCTURE_COLORS[structure_id] + (0.85,)  # Add alpha
            for i in range(4):
                colors[mask, i] = color[i]
        
        # Plot voxels
        if filled.any():
            ax.voxels(filled, facecolors=colors, edgecolors='gray', 
                     linewidth=0.1, alpha=0.95)
        
        # Set labels
        ax.set_xlabel('X (mm)', fontsize=12, labelpad=10)
        ax.set_ylabel('Y (mm)', fontsize=12, labelpad=10)
        ax.set_zlabel('Slice (mm)', fontsize=12, labelpad=10)
        
        # Set limits based on downsampled data
        ax.set_xlim(0, ds_width)
        ax.set_ylim(0, ds_height)
        ax.set_zlim(0, n_z)
        
        # Custom tick labels to show mm (accounting for the crop offset)
        x_ticks = ax.get_xticks()
        y_ticks = ax.get_yticks()
        z_ticks = ax.get_zticks()
        
        ax.set_xticklabels([f'{int((x_min + t * downsample_factor) * pixel_spacing)}' 
                           for t in x_ticks])
        ax.set_yticklabels([f'{int((y_min + t * downsample_factor) * pixel_spacing)}' 
                           for t in y_ticks])
        ax.set_zticklabels([f'{int((z_min + t) * slice_spacing)}' 
                           for t in z_ticks])
        
        # Title with frame number
        ax.set_title(f'{chamber_type} - 3D Cardiac Cycle\n{model_name}\nFrame {frame_idx + 1}/{n_frames}', 
                    fontsize=14, weight='bold', pad=25)
        
        # Fixed view angle with better perspective
        ax.view_init(elev=25, azim=45)
        
        # Add legend
        from matplotlib.patches import Patch
        legend_elements = [
            Patch(facecolor=STRUCTURE_COLORS[1], label='LV', alpha=0.85),
            Patch(facecolor=STRUCTURE_COLORS[2], label='MYO', alpha=0.85),
            Patch(facecolor=STRUCTURE_COLORS[3], label='RV', alpha=0.85)
        ]
        ax.legend(handles=legend_elements, loc='upper left', fontsize=11, framealpha=0.95)
        
        ax.set_facecolor('white')
        ax.grid(True, alpha=0.25)
        
        # Improve 3D appearance
        ax.xaxis.pane.fill = False
        ax.yaxis.pane.fill = False
        ax.zaxis.pane.fill = False
        ax.xaxis.pane.set_edgecolor('gray')
        ax.yaxis.pane.set_edgecolor('gray')
        ax.zaxis.pane.set_edgecolor('gray')
        ax.xaxis.pane.set_alpha(0.1)
        ax.yaxis.pane.set_alpha(0.1)
        ax.zaxis.pane.set_alpha(0.1)
        
        # Set aspect ratio to be equal
        ax.set_box_aspect([ds_width, ds_height, n_z * 0.8])  # Adjust z scale for better view
    
    # Create animation
    anim = FuncAnimation(fig, animate_frame, frames=n_frames, 
                        interval=1000//fps, repeat=True)
    
    # Save as GIF
    writer = PillowWriter(fps=fps)
    anim.save(output_path, writer=writer)
    plt.close()
    
    print(f"    3D animation saved with {n_frames} frames showing cardiac cycle")


def plot_all_sax_visualizations(segmentation, chamber_type, model_name, output_dir, fps=10):
    """Generate all SAX-specific visualizations.
    
    Args:
        segmentation: (n_slices, n_frames, H, W) segmentation array
        chamber_type: str
        model_name: str  
        output_dir: Path object for output directory
        fps: int, frames per second for 3D animation
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # 1. Total volume curves
    print(f"  Creating total volume curves...")
    volume_curves = compute_sax_volume_curves(segmentation)
    volume_plot_path = output_dir / "sax_total_volume_curves.png"
    plot_sax_total_volume_curves(volume_curves, chamber_type, model_name, volume_plot_path)
    
    # 2. Volume heatmaps  
    print(f"  Creating volume heatmaps...")
    heatmap_data = compute_sax_volume_heatmap_data(segmentation)
    heatmap_plot_path = output_dir / "sax_volume_heatmaps.png"
    plot_sax_volume_heatmaps(heatmap_data, chamber_type, model_name, heatmap_plot_path)
    
    # 3. 3D animation
    print(f"  Creating 3D animation (this may take a minute)...")
    animation_path = output_dir / "sax_3d_animation.gif"
    create_sax_3d_animation(segmentation, chamber_type, model_name, animation_path, fps=fps)
    
    print(f"    ✓ Volume curves: {volume_plot_path.name}")
    print(f"    ✓ Heatmaps: {heatmap_plot_path.name}")
    print(f"    ✓ 3D animation: {animation_path.name}")
