#!/usr/bin/env python3
"""
Process FOA Recording - Generate Power Maps

This script processes a First Order Ambisonics (FOA) audio recording
and generates power maps showing the spatial distribution of sound sources.

Usage:
    python process_foa_recording.py <input_file> [options]

Example:
    python process_foa_recording.py recording.wav --mode MUSIC --output powermap.png

The input file should be a 4-channel FOA recording in ACN ordering (W, Y, Z, X).
"""

import argparse
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation, FFMpegWriter
import soundfile as sf
from pathlib import Path
from tqdm import tqdm
from powermap_foa import PowermapFOA, PowermapMode


def load_foa_audio(filepath, target_sr=None):
    """
    Load FOA audio file.
    
    Parameters
    ----------
    filepath : str
        Path to audio file (should be 4 channels)
    target_sr : float, optional
        If provided, resample to this sample rate
        
    Returns
    -------
    audio : np.ndarray
        Audio data, shape (4, n_samples)
    sample_rate : float
        Sample rate in Hz
    """
    print(f"Loading audio file: {filepath}")
    
    # Load audio
    audio, sample_rate = sf.read(filepath, always_2d=True)
    
    print(f"  - Original sample rate: {sample_rate} Hz")
    print(f"  - Audio shape: {audio.shape}")
    print(f"  - Duration: {audio.shape[0] / sample_rate:.2f} seconds")
    
    # Check channel count
    if audio.shape[1] != 4:
        raise ValueError(f"Expected 4 channels (FOA), got {audio.shape[1]} channels")
    
    # Transpose to (channels, samples)
    audio = audio.T
    
    # Resample if needed
    if target_sr is not None and sample_rate != target_sr:
        print(f"  - Resampling from {sample_rate} Hz to {target_sr} Hz...")
        from scipy import signal as sp_signal
        
        resampled = []
        for ch in range(4):
            # Calculate resampling ratio
            ratio = target_sr / sample_rate
            n_samples_new = int(audio.shape[1] * ratio)
            
            # Resample
            resampled_ch = sp_signal.resample(audio[ch], n_samples_new)
            resampled.append(resampled_ch)
        
        audio = np.array(resampled)
        sample_rate = target_sr
        print(f"  - New shape: {audio.shape}")
    
    print(f"  ✓ Audio loaded successfully")
    return audio, sample_rate


def process_audio_frames(audio, analyzer, frame_size, hop_size, show_progress=True):
    """
    Process audio in overlapping frames and generate power maps.
    
    Parameters
    ----------
    audio : np.ndarray
        Input audio, shape (4, n_samples)
    analyzer : PowermapFOA
        Powermap analyzer instance
    frame_size : int
        Frame size in samples
    hop_size : int
        Hop size in samples
    show_progress : bool
        Whether to show progress bar
        
    Returns
    -------
    powermaps : list
        List of powermap arrays
    timestamps : np.ndarray
        Timestamp for each powermap in seconds
    """
    n_samples = audio.shape[1]
    n_frames = (n_samples - frame_size) // hop_size + 1
    
    powermaps = []
    timestamps = []
    
    iterator = range(n_frames)
    if show_progress:
        iterator = tqdm(iterator, desc="Processing frames", unit="frame")
    
    for i in iterator:
        start_idx = i * hop_size
        end_idx = start_idx + frame_size
        
        if end_idx > n_samples:
            break
        
        # Extract frame
        frame = audio[:, start_idx:end_idx]
        
        # Process frame
        analyzer.process_frame(frame)
        
        # Get powermap
        _, pmap = analyzer.get_powermap(normalize=True)
        
        powermaps.append(pmap.copy())
        timestamps.append(start_idx / analyzer.sample_rate)
    
    return powermaps, np.array(timestamps)


def plot_static_powermap(analyzer, powermap, title="FOA Powermap", output_file=None):
    """
    Create a static powermap visualization.
    
    Parameters
    ----------
    analyzer : PowermapFOA
        Powermap analyzer instance
    powermap : np.ndarray
        Powermap values
    title : str
        Plot title
    output_file : str, optional
        If provided, save figure to this file
    """
    fig = plt.figure(figsize=(14, 6))
    
    # Update analyzer's powermap for image generation
    analyzer.pmap = powermap
    analyzer.pmap_ready = True
    
    # Plot 1: 2D powermap
    ax1 = plt.subplot(121)
    image = analyzer.get_powermap_image(width=720, aspect_ratio=2.0)
    extent = [-180, 180, -90, 90]
    im1 = ax1.imshow(image, extent=extent, origin='lower', cmap='hot', aspect='auto')
    plt.colorbar(im1, ax=ax1, label='Normalized Power')
    ax1.set_xlabel('Azimuth (degrees)')
    ax1.set_ylabel('Elevation (degrees)')
    ax1.set_title('2D Powermap')
    ax1.grid(True, alpha=0.3, color='white', linewidth=0.5)
    
    # Add cardinal directions
    ax1.axvline(0, color='cyan', linewidth=1, linestyle='--', alpha=0.5)
    ax1.axhline(0, color='cyan', linewidth=1, linestyle='--', alpha=0.5)
    ax1.text(0, 92, 'Front', ha='center', color='cyan', fontsize=10, weight='bold')
    ax1.text(180, 0, 'Back', ha='center', va='center', color='cyan', fontsize=8)
    ax1.text(-90, 0, 'Left', ha='center', va='center', color='cyan', fontsize=8)
    ax1.text(90, 0, 'Right', ha='center', va='center', color='cyan', fontsize=8)
    
    # Plot 2: Horizontal plane slice
    ax2 = plt.subplot(122)
    
    grid_dirs, _ = analyzer.get_powermap(normalize=True)
    
    # Extract horizontal plane
    ele_threshold = 15
    mask = np.abs(grid_dirs[:, 1]) < ele_threshold
    azi_h = grid_dirs[mask, 0]
    pmap_h = powermap[mask]
    
    # Sort by azimuth
    sort_idx = np.argsort(azi_h)
    azi_h = azi_h[sort_idx]
    pmap_h = pmap_h[sort_idx]
    
    ax2.plot(azi_h, pmap_h, 'r-', linewidth=2)
    ax2.fill_between(azi_h, pmap_h, alpha=0.3, color='red')
    ax2.set_xlabel('Azimuth (degrees)')
    ax2.set_ylabel('Normalized Power')
    ax2.set_title('Horizontal Plane Slice (elevation ≈ 0°)')
    ax2.grid(True, alpha=0.3)
    ax2.set_xlim(-180, 180)
    ax2.set_ylim(0, 1.05)
    
    # Mark cardinal directions
    ax2.axvline(0, color='cyan', linestyle='--', alpha=0.5, linewidth=1)
    ax2.axvline(-90, color='cyan', linestyle=':', alpha=0.3, linewidth=1)
    ax2.axvline(90, color='cyan', linestyle=':', alpha=0.3, linewidth=1)
    
    plt.suptitle(title, fontsize=14, weight='bold')
    plt.tight_layout()
    
    if output_file:
        print(f"Saving figure to: {output_file}")
        plt.savefig(output_file, dpi=150, bbox_inches='tight')
        print("  ✓ Figure saved")
    
    return fig


def plot_temporal_powermap(analyzer, powermaps, timestamps, output_file=None):
    """
    Create a temporal powermap visualization showing evolution over time.
    
    Parameters
    ----------
    analyzer : PowermapFOA
        Powermap analyzer instance
    powermaps : list
        List of powermap arrays
    timestamps : np.ndarray
        Timestamps for each powermap
    output_file : str, optional
        If provided, save figure to this file
    """
    # Extract horizontal plane data
    grid_dirs, _ = analyzer.get_powermap(normalize=True)
    ele_threshold = 15
    mask = np.abs(grid_dirs[:, 1]) < ele_threshold
    azi_indices = mask
    
    # Create time-azimuth matrix
    n_frames = len(powermaps)
    azi_values = grid_dirs[mask, 0]
    
    # Sort by azimuth
    sort_idx = np.argsort(azi_values)
    azi_sorted = azi_values[sort_idx]
    
    # Build matrix
    temporal_map = np.zeros((len(azi_sorted), n_frames))
    for i, pmap in enumerate(powermaps):
        pmap_h = pmap[mask]
        temporal_map[:, i] = pmap_h[sort_idx]
    
    # Plot
    fig = plt.figure(figsize=(14, 8))
    
    # Plot 1: Temporal evolution
    ax1 = plt.subplot(211)
    extent = [timestamps[0], timestamps[-1], -180, 180]
    im = ax1.imshow(temporal_map, extent=extent, origin='lower', 
                   aspect='auto', cmap='hot', interpolation='bilinear')
    plt.colorbar(im, ax=ax1, label='Normalized Power')
    ax1.set_xlabel('Time (seconds)')
    ax1.set_ylabel('Azimuth (degrees)')
    ax1.set_title('Temporal Power Map Evolution (Horizontal Plane)')
    ax1.grid(True, alpha=0.3, color='white', linewidth=0.5)
    
    # Add azimuth markers
    ax1.axhline(0, color='cyan', linestyle='--', alpha=0.5, linewidth=1)
    ax1.axhline(-90, color='cyan', linestyle=':', alpha=0.3, linewidth=0.5)
    ax1.axhline(90, color='cyan', linestyle=':', alpha=0.3, linewidth=0.5)
    
    # Plot 2: Average powermap
    ax2 = plt.subplot(212)
    avg_pmap = np.mean(temporal_map, axis=1)
    ax2.plot(azi_sorted, avg_pmap, 'r-', linewidth=2, label='Time-averaged')
    ax2.fill_between(azi_sorted, avg_pmap, alpha=0.3, color='red')
    ax2.set_xlabel('Azimuth (degrees)')
    ax2.set_ylabel('Average Normalized Power')
    ax2.set_title('Time-Averaged Powermap (Horizontal Plane)')
    ax2.grid(True, alpha=0.3)
    ax2.set_xlim(-180, 180)
    ax2.legend()
    
    # Mark cardinal directions
    ax2.axvline(0, color='cyan', linestyle='--', alpha=0.5, linewidth=1, label='Front')
    ax2.axvline(-90, color='green', linestyle=':', alpha=0.5, linewidth=1, label='Left')
    ax2.axvline(90, color='blue', linestyle=':', alpha=0.5, linewidth=1, label='Right')
    ax2.legend(loc='upper right')
    
    plt.tight_layout()
    
    if output_file:
        print(f"Saving temporal figure to: {output_file}")
        plt.savefig(output_file, dpi=150, bbox_inches='tight')
        print("  ✓ Figure saved")
    
    return fig


def create_animation(analyzer, powermaps, timestamps, output_file, fps=10):
    """
    Create an animated powermap video.
    
    Parameters
    ----------
    analyzer : PowermapFOA
        Powermap analyzer instance
    powermaps : list
        List of powermap arrays
    timestamps : np.ndarray
        Timestamps for each powermap
    output_file : str
        Output video file path
    fps : int
        Frames per second
    """
    print(f"Creating animation: {output_file}")
    print(f"  - Number of frames: {len(powermaps)}")
    print(f"  - FPS: {fps}")
    
    fig, ax = plt.subplots(figsize=(12, 6))
    
    extent = [-180, 180, -90, 90]
    
    # Initialize with first frame
    analyzer.pmap = powermaps[0]
    analyzer.pmap_ready = True
    image_data = analyzer.get_powermap_image(width=720, aspect_ratio=2.0)
    
    im = ax.imshow(image_data, extent=extent, origin='lower', 
                  cmap='hot', aspect='auto', vmin=0, vmax=1)
    plt.colorbar(im, ax=ax, label='Normalized Power')
    ax.set_xlabel('Azimuth (degrees)')
    ax.set_ylabel('Elevation (degrees)')
    ax.grid(True, alpha=0.3, color='white', linewidth=0.5)
    
    # Add cardinal directions
    ax.axvline(0, color='cyan', linewidth=1, linestyle='--', alpha=0.5)
    ax.axhline(0, color='cyan', linewidth=1, linestyle='--', alpha=0.5)
    
    title = ax.text(0.5, 1.05, '', transform=ax.transAxes, 
                   ha='center', fontsize=12, weight='bold')
    
    def update(frame):
        analyzer.pmap = powermaps[frame]
        analyzer.pmap_ready = True
        image_data = analyzer.get_powermap_image(width=720, aspect_ratio=2.0)
        im.set_array(image_data)
        title.set_text(f'FOA Powermap - Time: {timestamps[frame]:.2f} s')
        return im, title
    
    anim = FuncAnimation(fig, update, frames=len(powermaps), 
                        interval=1000/fps, blit=True)
    
    # Save animation
    writer = FFMpegWriter(fps=fps, bitrate=2000)
    anim.save(output_file, writer=writer)
    
    plt.close(fig)
    print(f"  ✓ Animation saved")


def main():
    parser = argparse.ArgumentParser(
        description='Process FOA recording and generate power maps',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Basic usage with 24kHz recording
  python process_foa_recording.py recording_24k.wav
  
  # Specify output file and mode
  python process_foa_recording.py recording.wav --mode MVDR --output result.png
  
  # Create animation
  python process_foa_recording.py recording.wav --animate --anim-output powermap.mp4
  
  # Process with custom parameters
  python process_foa_recording.py recording.wav --frame-size 2048 --grid-res 5 --sources 2
        """
    )
    
    # Required arguments
    parser.add_argument('input', type=str, help='Input FOA audio file (4 channels)')
    
    # Processing parameters
    parser.add_argument('--sample-rate', type=int, default=24000,
                       help='Target sample rate in Hz (default: 24000)')
    parser.add_argument('--frame-size', type=int, default=1024,
                       help='Analysis frame size in samples (default: 1024)')
    parser.add_argument('--hop-size', type=int, default=128,
                       help='STFT hop size in samples (default: 128)')
    parser.add_argument('--grid-res', type=int, default=3,
                       help='Spherical grid resolution (default: 3)')
    
    # Algorithm parameters
    parser.add_argument('--mode', type=str, default='MUSIC',
                       choices=['PWD', 'MVDR', 'MUSIC', 'MUSIC_LOG', 'MINNORM', 'MINNORM_LOG'],
                       help='Powermap computation mode (default: MUSIC)')
    parser.add_argument('--sources', type=int, default=1,
                       help='Expected number of sources (default: 1)')
    parser.add_argument('--smoothing', type=float, default=0.25,
                       help='Temporal smoothing coefficient [0-1] (default: 0.25)')
    parser.add_argument('--threshold', type=float, default=1e-6,
                       help='Signal power threshold for silence detection (default: 1e-6)')
    
    # Output options
    parser.add_argument('--output', type=str, default=None,
                       help='Output image file for static powermap')
    parser.add_argument('--temporal', action='store_true',
                       help='Generate temporal evolution plot')
    parser.add_argument('--temporal-output', type=str, default=None,
                       help='Output file for temporal plot')
    parser.add_argument('--animate', action='store_true',
                       help='Create animated powermap video')
    parser.add_argument('--anim-output', type=str, default=None,
                       help='Output video file for animation')
    parser.add_argument('--fps', type=int, default=10,
                       help='Animation frames per second (default: 10)')
    
    # Time selection
    parser.add_argument('--start', type=float, default=0.0,
                       help='Start time in seconds (default: 0.0)')
    parser.add_argument('--duration', type=float, default=None,
                       help='Duration to process in seconds (default: entire file)')
    
    args = parser.parse_args()
    
    print("=" * 70)
    print("FOA POWERMAP GENERATOR")
    print("=" * 70)
    
    # Load audio
    try:
        audio, sample_rate = load_foa_audio(args.input, target_sr=args.sample_rate)
    except Exception as e:
        print(f"Error loading audio: {e}")
        return 1
    
    # Extract time segment if specified
    if args.start > 0 or args.duration is not None:
        start_sample = int(args.start * sample_rate)
        if args.duration is not None:
            end_sample = start_sample + int(args.duration * sample_rate)
        else:
            end_sample = audio.shape[1]
        
        audio = audio[:, start_sample:end_sample]
        print(f"Extracted segment: {args.start:.2f}s to {end_sample/sample_rate:.2f}s")
    
    # Create analyzer
    print("\nInitializing powermap analyzer...")
    analyzer = PowermapFOA(
        sample_rate=sample_rate,
        frame_size=args.frame_size,
        hop_size=args.hop_size,
        grid_resolution=args.grid_res
    )
    
    # Configure
    analyzer.mode = PowermapMode[args.mode]
    analyzer.n_sources = args.sources
    analyzer.pmap_avg_coeff = args.smoothing
    analyzer.set_power_threshold(args.threshold)
    
    print(f"  - Mode: {analyzer.mode.name}")
    print(f"  - Number of sources: {analyzer.n_sources}")
    print(f"  - Grid directions: {analyzer.n_dirs}")
    print(f"  - Temporal smoothing: {analyzer.pmap_avg_coeff}")
    print(f"  - Power threshold: {analyzer.power_threshold:.2e}")
    
    # Process audio
    print("\nProcessing audio...")
    powermaps, timestamps = process_audio_frames(
        audio, analyzer, args.frame_size, args.hop_size
    )
    
    print(f"  ✓ Generated {len(powermaps)} powermaps")
    print(f"  - Time span: {timestamps[0]:.2f}s to {timestamps[-1]:.2f}s")
    
    # Generate outputs
    print("\nGenerating visualizations...")
    
    # Static powermap (last frame or average)
    avg_powermap = np.mean(powermaps, axis=0)
    
    output_file = args.output
    if output_file is None:
        input_path = Path(args.input)
        output_file = input_path.stem + '_powermap.png'
    
    title = f"FOA Powermap ({args.mode}) - {Path(args.input).name}"
    plot_static_powermap(analyzer, avg_powermap, title=title, output_file=output_file)
    
    # Temporal plot
    if args.temporal:
        temporal_file = args.temporal_output
        if temporal_file is None:
            temporal_file = Path(args.input).stem + '_temporal.png'
        plot_temporal_powermap(analyzer, powermaps, timestamps, output_file=temporal_file)
    
    # Animation
    if args.animate:
        anim_file = args.anim_output
        if anim_file is None:
            anim_file = Path(args.input).stem + '_powermap.mp4'
        
        try:
            create_animation(analyzer, powermaps, timestamps, anim_file, fps=args.fps)
        except Exception as e:
            print(f"Error creating animation: {e}")
            print("Make sure FFmpeg is installed for video export")
    
    # Show plots if no animation
    if not args.animate:
        print("\nDisplaying plots (close windows to exit)...")
        plt.show()
    
    print("\n" + "=" * 70)
    print("PROCESSING COMPLETE")
    print("=" * 70)
    
    return 0


if __name__ == "__main__":
    import sys
    try:
        sys.exit(main())
    except KeyboardInterrupt:
        print("\n\nInterrupted by user")
        sys.exit(1)
    except Exception as e:
        print(f"\nError: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
