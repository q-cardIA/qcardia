"""
Utility functions for LA inference visualization and analysis.
"""

import numpy as np
from matplotlib import pyplot as plt
from matplotlib.colors import ListedColormap
from matplotlib.animation import FuncAnimation, PillowWriter
from pathlib import Path


# Pastel/matte colors for visualization
PASTEL_RED = (1, 0.3, 0.3)      # Matte red for LV
PASTEL_GREEN = (0.3, 0.8, 0.3)  # Matte green for MYO
PASTEL_BLUE = (0.3, 0.3, 1)     # Matte blue for RV


def get_custom_colormap():
    """
    Create custom colormap with pastel colors for segmentation visualization.
    
    Returns:
        ListedColormap: Custom colormap with transparent background and pastel colors.
    """
    colors = [(0, 0, 0, 0),           # 0: transparent (background)
              PASTEL_RED + (1,),      # 1: pastel red (LV)
              PASTEL_GREEN + (1,),    # 2: pastel green (myocardium)
              PASTEL_BLUE + (1,)]     # 3: pastel blue (RV)
    return ListedColormap(colors)


def create_static_segmentation_plot(input_image, pred_slice, chamber_type, mid_frame_idx, output_path):
    """
    Create static 3-panel segmentation visualization.
    
    Args:
        input_image: Original grayscale image
        pred_slice: Segmentation prediction
        chamber_type: Name of chamber (e.g., "CINE_2CH")
        mid_frame_idx: Frame index being displayed
        output_path: Path to save the output image
    """
    cmap_custom = get_custom_colormap()
    pred_slice = pred_slice.astype(np.uint8)
    
    fig, axes = plt.subplots(1, 3, figsize=(16, 5))
    fig.patch.set_facecolor('white')
    
    # Panel 1: Original image
    axes[0].imshow(input_image, cmap='gray')
    axes[0].set_title('Original Image', fontsize=12, color='white')
    axes[0].axis('off')
    axes[0].set_facecolor('black')
    
    # Panel 2: Segmentation with black background
    axes[1].imshow(pred_slice, cmap=cmap_custom, vmin=0, vmax=3, interpolation='nearest')
    axes[1].set_title(f'Segmentation\n(Red=LV, Green=MYO, Blue=RV)', fontsize=12, color='white')
    axes[1].axis('off')
    axes[1].set_facecolor('black')
    
    # Panel 3: Overlay
    axes[2].imshow(input_image, cmap='gray')
    pred_masked = np.ma.masked_where(pred_slice == 0, pred_slice)
    axes[2].imshow(pred_masked, cmap=cmap_custom, alpha=0.5, vmin=0, vmax=3, interpolation='nearest')
    axes[2].set_title('Overlay', fontsize=12, color='white')
    axes[2].axis('off')
    axes[2].set_facecolor('black')
    
    plt.suptitle(f"{chamber_type} - Segmentation Results (Frame {mid_frame_idx})", fontsize=14)
    plt.tight_layout()
    plt.savefig(output_path, bbox_inches='tight', dpi=100, facecolor='white', pad_inches=0.1)
    plt.close()


def create_segmentation_gif(cine_segmentation, pixel_array, chamber_type, num_frames, output_path):
    """
    Create animated GIF of segmentation across all cardiac phases.
    
    Args:
        cine_segmentation: Full segmentation array (slices, frames, H, W)
        pixel_array: Original image frames
        chamber_type: Name of chamber
        num_frames: Number of temporal frames
        output_path: Path to save the GIF
    """
    cmap_custom = get_custom_colormap()
    seg_shape = cine_segmentation.shape
    
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    fig.patch.set_facecolor('black')
    
    def animate_frame(frame_idx):
        # Clear all axes
        for ax in axes:
            ax.clear()
            ax.axis('off')
            ax.set_facecolor('black')
        
        # Get frame-specific data
        if len(seg_shape) == 4:
            if seg_shape[1] == num_frames:
                frame_pred = cine_segmentation[0, frame_idx, :, :]
            else:
                frame_pred = cine_segmentation[0, :, :, frame_idx]
        else:
            frame_pred = cine_segmentation[0]
        
        # Get corresponding input
        if isinstance(pixel_array, list):
            frame_input = pixel_array[frame_idx] if frame_idx < len(pixel_array) else pixel_array[0]
        else:
            frame_input = pixel_array[frame_idx] if frame_idx < pixel_array.shape[0] else pixel_array[0]
        
        frame_pred = frame_pred.astype(np.uint8)
        
        # Panel 1: Original
        axes[0].imshow(frame_input, cmap='gray')
        axes[0].set_title(f'Original image', fontsize=14, color='white')
        
        # Panel 2: Segmentation with black background
        axes[1].imshow(frame_pred, cmap=cmap_custom, vmin=0, vmax=3, interpolation='nearest')
        axes[1].set_title('Segmentation', fontsize=14, color='white')
        
        # Panel 3: Overlay
        axes[2].imshow(frame_input, cmap='gray')
        frame_masked = np.ma.masked_where(frame_pred == 0, frame_pred)
        axes[2].imshow(frame_masked, cmap=cmap_custom, alpha=0.5, vmin=0, vmax=3, interpolation='nearest')
        axes[2].set_title('Overlay', fontsize=14, color='white')
        
        fig.suptitle(f"{chamber_type} - Frame {frame_idx}/{num_frames}", fontsize=16, color='white', y=0.98)
    
    # Create animation
    anim = FuncAnimation(fig, animate_frame, frames=num_frames, interval=100, repeat=True)
    
    # Save as GIF with tight margins on bottom/left/right, extra room on top for suptitle
    writer = PillowWriter(fps=10)
    plt.subplots_adjust(top=0.88, bottom=0.02, left=0.02, right=0.98, wspace=0.05)
    anim.save(output_path, writer=writer, savefig_kwargs={'facecolor': 'black', 'pad_inches': 0.02})
    plt.close()


def plot_volume_curves(lv_vol, myo_vol, rv_vol, lv_ef, rv_ef, chamber_type, output_path):
    """
    Plot LV, MYO, and RV volume curves together.
    
    Args:
        lv_vol: LV volume curve
        myo_vol: Myocardium volume curve
        rv_vol: RV volume curve
        lv_ef: LV ejection fraction (percentage)
        rv_ef: RV ejection fraction (percentage)
        chamber_type: Name of chamber
        output_path: Path to save the plot
    """
    fig, ax = plt.subplots(1, 1, figsize=(10, 6))
    
    frames = np.arange(len(lv_vol))
    ax.plot(frames, lv_vol, marker='o', linewidth=2.5, color=PASTEL_RED, label='LV', markersize=6)
    ax.plot(frames, myo_vol, marker='s', linewidth=2.5, color=PASTEL_GREEN, label='Myocardium', markersize=6)
    ax.plot(frames, rv_vol, marker='^', linewidth=2.5, color=PASTEL_BLUE, label='RV', markersize=6)
    
    ax.set_xlabel('Frame', fontsize=12)
    ax.set_ylabel('Volume (mL)', fontsize=12)
    ax.set_title(f"{chamber_type} - Cardiac Volume Curves\n(LV EF: {lv_ef:.1f}% | RV EF: {rv_ef:.1f}%)", fontsize=14)
    ax.legend(loc='best', fontsize=11)
    ax.grid(True, alpha=0.3, linestyle='--')
    
    plt.tight_layout()
    plt.savefig(output_path, bbox_inches='tight', dpi=120)
    plt.close()


def plot_marker_points(input_image, pred_masked, lv_centers, rv_centers, rv_insertions, 
                       mid_frame_idx, chamber_type, cmap_custom, output_path):
    """
    Visualize anatomical marker points on segmentation.
    
    Args:
        input_image: Original image
        pred_masked: Masked segmentation for overlay
        lv_centers: LV center points
        rv_centers: RV center points
        rv_insertions: RV insertion points
        mid_frame_idx: Frame index to visualize
        chamber_type: Name of chamber
        cmap_custom: Custom colormap
        output_path: Path to save visualization
    """
    fig, ax = plt.subplots(1, 1, figsize=(8, 8))
    ax.imshow(input_image, cmap='gray')
    
    # Overlay segmentation
    ax.imshow(pred_masked, cmap=cmap_custom, alpha=0.4, vmin=0, vmax=3, interpolation='nearest')
    
    # Plot markers if available
    if lv_centers and len(lv_centers) > 0:
        lv_pts = lv_centers[0] if isinstance(lv_centers[0], list) else lv_centers
        if lv_pts and len(lv_pts) > 0:
            lv_pt = lv_pts[mid_frame_idx] if mid_frame_idx < len(lv_pts) else lv_pts[0]
            if lv_pt[0] > 0 and lv_pt[1] > 0:
                ax.plot(lv_pt[1], lv_pt[0], 'r*', markersize=20, markeredgecolor='white', 
                       markeredgewidth=2, label='LV Center')
    
    if rv_centers and len(rv_centers) > 0:
        rv_pts = rv_centers[0] if isinstance(rv_centers[0], list) else rv_centers
        if rv_pts and len(rv_pts) > 0:
            rv_pt = rv_pts[mid_frame_idx] if mid_frame_idx < len(rv_pts) else rv_pts[0]
            if rv_pt[0] > 0 and rv_pt[1] > 0:
                ax.plot(rv_pt[1], rv_pt[0], 'b*', markersize=20, markeredgecolor='white', 
                       markeredgewidth=2, label='RV Center')
    
    if rv_insertions and len(rv_insertions) > 0:
        rv_ins = rv_insertions[0] if isinstance(rv_insertions[0], list) else rv_insertions
        if rv_ins and len(rv_ins) > 0:
            ins_pt = rv_ins[mid_frame_idx] if mid_frame_idx < len(rv_ins) else rv_ins[0]
            if len(ins_pt) == 2 and ins_pt[0][0] > 0:
                ax.plot([ins_pt[0][0], ins_pt[1][0]], [ins_pt[0][1], ins_pt[1][1]], 
                       'yo', markersize=12, markeredgecolor='white', markeredgewidth=2, 
                       label='RV Insertions')
    
    ax.set_title(f"{chamber_type} - Anatomical Markers (Frame {mid_frame_idx})", fontsize=14)
    ax.legend(loc='upper right', fontsize=10)
    ax.axis('off')
    
    plt.tight_layout()
    plt.savefig(output_path, bbox_inches='tight', dpi=120)
    plt.close()


def compute_ejection_fraction(volume_curve):
    """
    Compute ejection fraction from volume curve.
    
    Args:
        volume_curve: Array of volumes across cardiac cycle
        
    Returns:
        float: Ejection fraction as percentage
    """
    ed_vol = volume_curve[0]
    es_time = np.argmin(volume_curve)
    es_vol = volume_curve[es_time]
    
    if ed_vol > 0:
        return ((ed_vol - es_vol) / ed_vol) * 100
    else:
        return 0.0
