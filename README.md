# First Order Ambisonics (FOA) Powermap Generator

A Python implementation of the powermap generator from the SPARTA VST plugin suite, specifically for First Order Ambisonics (FOA).

## Overview

This implementation provides spatial audio analysis and visualization tools for FOA (B-format) audio signals. It generates activity maps showing the direction and intensity of sound sources in a 3D sound field.

## Based On

- **SPARTA**: Spatial Audio Real-Time Applications
  - Repository: https://github.com/leomccormack/SPARTA
  - License: GPLv3
  
- **Spatial Audio Framework (SAF)**
  - Repository: https://github.com/leomccormack/Spatial_Audio_Framework
  - License: ISC

## Reference

McCormack, L., Delikaris-Manias, S. and Pulkki, V., 2017.
"Parametric acoustic camera for real-time sound capture, analysis and tracking."
In Proceedings of the 20th International Conference on Digital Audio Effects (DAFx-17), pp. 412-419.

## Features

### Silence Detection and Thresholding (New!)

The implementation now includes **automatic silence detection** to prevent spurious artifacts during quiet portions of recordings:

- **Power thresholding**: Automatically suppresses powermap when signal energy is below threshold
- **No 45° artifacts**: Prevents numerical instabilities from creating phantom sources during silence
- **Clean visualizations**: Maps go fully dark when there's no signal
- **Configurable threshold**: Adjust sensitivity based on recording conditions

See [SILENCE_DETECTION.md](SILENCE_DETECTION.md) for detailed information.

### Powermap Modes

1. **PWD** (Plane Wave Decomposition)
   - Activity map based on hypercardioid beamformers
   - Fast computation, good for real-time applications
   
2. **MVDR** (Minimum Variance Distortionless Response)
   - Adaptive beamforming approach
   - Better noise rejection than PWD
   
3. **MUSIC** (Multiple Signal Classification)
   - Subspace method for high-resolution DOA estimation
   - Excellent for resolving multiple sources
   - Requires knowing the number of sources
   
4. **MinNorm** (Minimum Norm)
   - Alternative subspace method
   - Similar to MUSIC but different constraint
   
5. **Logarithmic variants** (MUSIC_LOG, MINNORM_LOG)
   - Enhanced dynamic range visualization

### Ambisonic Format Support

- **Channel Ordering**:
  - ACN (Ambisonic Channel Number) - modern standard
  - FuMa (Furse-Malham) - legacy format
  
- **Normalization**:
  - N3D (Full 3D normalization)
  - SN3D (Schmidt semi-normalization) - modern standard
  - FuMa (Furse-Malham) - legacy format

## Installation

### Requirements

```bash
pip install numpy scipy matplotlib
```

### Files

- `powermap_foa.py` - Main implementation
- `powermap_example.py` - Usage examples and visualizations
- `README.md` - This file

## Quick Start

```python
from powermap_foa import PowermapFOA, PowermapMode
import numpy as np

# Create analyzer
analyzer = PowermapFOA(
    sample_rate=48000,
    frame_size=1024,
    hop_size=128,
    grid_resolution=2
)

# Configure
analyzer.mode = PowermapMode.MUSIC
analyzer.n_sources = 1
analyzer.pmap_avg_coeff = 0.25
analyzer.set_power_threshold(1e-6)  # Set silence threshold

# Process FOA audio (4 channels: W, Y, Z, X in ACN ordering)
# Shape: (4, frame_size)
foa_audio = np.random.randn(4, 1024) * 0.1
analyzer.process_frame(foa_audio)

# Get results
grid_directions, powermap = analyzer.get_powermap(normalize=True)

# Check if signal was detected
if analyzer.get_current_power() > analyzer.power_threshold:
    # Find peak direction
    peak_idx = np.argmax(powermap)
    azimuth, elevation = grid_directions[peak_idx]
    print(f"Peak at: azimuth={azimuth:.1f}°, elevation={elevation:.1f}°")
else:
    print("Silence detected - no source localization")
```

## Usage Examples

### Example 1: Single Source Analysis

```python
from powermap_foa import PowermapFOA, PowermapMode
import numpy as np

# Create test FOA signal from specific direction (45° azimuth, 0° elevation)
def create_foa_signal(azimuth_deg, elevation_deg, sample_rate, n_samples):
    azi_rad = np.deg2rad(azimuth_deg)
    ele_rad = np.deg2rad(elevation_deg)
    
    t = np.linspace(0, n_samples / sample_rate, n_samples)
    signal = np.sin(2 * np.pi * 1000 * t)  # 1 kHz tone
    
    foa = np.zeros((4, n_samples))
    foa[0, :] = signal * 1.0  # W
    foa[1, :] = signal * np.sqrt(3) * np.cos(ele_rad) * np.sin(azi_rad)  # Y
    foa[2, :] = signal * np.sqrt(3) * np.sin(ele_rad)  # Z
    foa[3, :] = signal * np.sqrt(3) * np.cos(ele_rad) * np.cos(azi_rad)  # X
    
    return foa * 0.5

# Analyze
analyzer = PowermapFOA(sample_rate=48000, frame_size=1024)
analyzer.mode = PowermapMode.MUSIC

audio = create_foa_signal(45, 0, 48000, 1024)
analyzer.process_frame(audio)

grid_dirs, pmap = analyzer.get_powermap()
```

### Example 2: Multiple Sources

```python
analyzer = PowermapFOA(sample_rate=48000, frame_size=1024)
analyzer.mode = PowermapMode.MUSIC
analyzer.n_sources = 2  # Tell MUSIC to expect 2 sources

# Create two sources and mix
source1 = create_foa_signal(-45, 0, 48000, 1024)
source2 = create_foa_signal(60, 20, 48000, 1024)
audio = source1 * 0.6 + source2 * 0.4

analyzer.process_frame(audio)
```

### Example 3: Real-time Processing

```python
import sounddevice as sd

analyzer = PowermapFOA(sample_rate=48000, frame_size=1024)
analyzer.mode = PowermapMode.MVDR

def audio_callback(indata, frames, time, status):
    # indata shape: (frames, 4) for 4 channels
    foa_audio = indata.T  # Transpose to (4, frames)
    analyzer.process_frame(foa_audio)
    
    if analyzer.pmap_ready:
        grid_dirs, pmap = analyzer.get_powermap()
        # Update visualization here

# Start audio stream
stream = sd.InputStream(
    channels=4,
    samplerate=48000,
    blocksize=1024,
    callback=audio_callback
)
stream.start()
```

### Example 4: Visualization

```python
import matplotlib.pyplot as plt

# Get 2D image representation
image = analyzer.get_powermap_image(width=720, aspect_ratio=2.0)

plt.figure(figsize=(12, 6))
plt.imshow(image, extent=[-180, 180, -90, 90], 
          origin='lower', cmap='hot', aspect='auto')
plt.colorbar(label='Power')
plt.xlabel('Azimuth (degrees)')
plt.ylabel('Elevation (degrees)')
plt.title('FOA Powermap')
plt.show()
```

## API Reference

### PowermapFOA Class

#### Constructor

```python
PowermapFOA(sample_rate=48000, frame_size=1024, hop_size=128, grid_resolution=9)
```

**Parameters:**
- `sample_rate` (float): Audio sample rate in Hz
- `frame_size` (int): Analysis frame size in samples
- `hop_size` (int): STFT hop size in samples
- `grid_resolution` (int): Spherical grid density (higher = more directions)

#### Properties

- `mode` (PowermapMode): Analysis mode (PWD, MVDR, MUSIC, etc.)
- `channel_order` (ChannelOrder): ACN or FuMa
- `norm_type` (NormType): N3D, SN3D, or FuMa
- `n_sources` (int): Number of sources (for MUSIC/MinNorm)
- `cov_avg_coeff` (float): Covariance averaging [0..1]
- `pmap_avg_coeff` (float): Powermap temporal averaging [0..1]
- `pmap_eq` (np.ndarray): Per-band EQ weights

#### Methods

**process_frame(audio)**
- Process a frame of FOA audio
- `audio`: np.ndarray, shape (4, frame_size)
- Returns: bool (True if powermap ready)

**get_powermap(normalize=True)**
**get_powermap_image(width=720, aspect_ratio=2.0)**
- Get powermap as 2D image
- Returns: np.ndarray, shape (height, width)

**set_power_threshold(threshold)**
- Set minimum signal power threshold for silence detection
- `threshold`: float, typical values 1e-7 to 1e-4

**get_current_power()**
- Get current frame's signal power level
- Returns: float, current power2) [azimuth, elevation] in degrees
  - `pmap`: np.ndarray, shape (n_dirs,) power values

**get_powermap_image(width=720, aspect_ratio=2.0)**
- Get powermap as 2D image
- Returns: np.ndarray, shape (height, width)

## FOA Channel Format

The implementation expects FOA audio in **ACN** (Ambisonic Channel Number) ordering with **SN3D** normalization by default (configurable):

| Channel | Component | Description |
|---------|-----------|-------------|
| 0 | W | Omnidirectional (pressure) |
| 1 | Y | Left-Right (figure-8 pointing left) |
| 2 | Z | Up-Down (figure-8 pointing up) |
| 3 | X | Front-Back (figure-8 pointing front) |

### Alternative Formats

For **FuMa** (B-format):
```python
analyzer.channel_order = ChannelOrder.FUMA
analyzer.norm_type = NormType.FUMA
```

FuMa ordering: W, X, Y, Z

## Performance Tips

1. **Grid Resolution**: Lower `grid_resolution` for faster processing
   - Resolution 1-2: Fast, good for real-time
   - Resolution 5-9: Slow, better spatial resolution

2. **Frame Size**: Larger frames = better frequency resolution but more latency
   - 512-1024: Good balance
   - 2048+: High resolution, high latency

3. **Averaging**: Higher `pmap_avg_coeff` = smoother but slower response
   - 0.0: No averaging, instant response
   - 0.3-0.5: Smooth, good for visualization
   - 0.8+: Very smooth, slow response

4. **Mode Selection**:
   - **PWD**: Fastest, good baseline
   - **MVDR**: Moderate speed, better than PWD
## Troubleshooting

### Spurious 45° artifacts during silence
- **Solution**: Use power thresholding (enabled by default)
- Adjust threshold: `analyzer.set_power_threshold(1e-6)`
- See [SILENCE_DETECTION.md](SILENCE_DETECTION.md) for details

### No peaks detected

1. **First Order Only**: This implementation is optimized for FOA (1st order)
   - Higher orders would require modifications to SH basis functions
   
2. **Spatial Resolution**: FOA has limited spatial resolution (~90° beamwidth)
   - Cannot resolve sources closer than ~60° apart reliably
   
3. **MUSIC/MinNorm**: Require knowing the number of sources
   - Performance degrades if n_sources is incorrect

## Troubleshooting

### No peaks detected
- Check input signal level (should be audible amplitude)
- Verify channel ordering matches your source
- Try different modes (MUSIC vs PWD)
- Increase `pmap_avg_coeff` for smoothing

### Wrong direction detected
- Verify azimuth convention (0° = front)
- Check channel order (ACN vs FuMa)
- Check normalization (SN3D vs N3D vs FuMa)
- Ensure signal is properly encoded in FOA

### Poor resolution
- Increase `grid_resolution`
- Try MUSIC mode with correct n_sources
- Consider that FOA has inherent resolution limits

## Advanced Usage

### Custom Frequency Weighting

```python
# Boost high frequencies
analyzer.pmap_eq[analyzer.n_bands//2:] = 1.5

# Suppress low frequencies
analyzer.pmap_eq[:analyzer.n_bands//4] = 0.5
```

### Batch Processing

```python
# Process audio file
import soundfile as sf

audio, sr = sf.read('foa_recording.wav')  # Shape: (n_samples, 4)
audio = audio.T  # Transpose to (4, n_samples)

frame_size = 1024
n_frames = len(audio[0]) // frame_size

powermaps = []
for i in range(n_frames):
    frame = audio[:, i*frame_size:(i+1)*frame_size]
    analyzer.process_frame(frame)
    _, pmap = analyzer.get_powermap()
    powermaps.append(pmap)

powermaps = np.array(powermaps)  # Shape: (n_frames, n_dirs)
```

## License

This Python implementation follows the licensing of the original SPARTA project:
- ISC License (permissive, similar to MIT/BSD)

## Contributing

This is a faithful Python port of the C implementation. For improvements or bug reports, please ensure compatibility with the original SPARTA behavior.

## References

1. McCormack, L., Delikaris-Manias, S. and Pulkki, V., 2017. "Parametric acoustic camera for real-time sound capture, analysis and tracking." DAFx-17.

2. McCormack, L. and Politis, A., 2019. "SPARTA & COMPASS: Real-time implementations of linear and parametric spatial audio reproduction and processing methods." AES Conference on Immersive and Interactive Audio.

3. Schmidt, R., 1986. "Multiple emitter location and signal parameter estimation." IEEE Transactions on Antennas and Propagation.

4. Pulkki, V., 2007. "Spatial sound reproduction with directional audio coding." Journal of the Audio Engineering Society.

## Use of AI Tools

During development, AI-based tools (Claude Opus 4.5) were used to assist with:
- Code scaffolding and translation from SPARTA VST Plugins
- Documentation drafting
- Debugging and testing ideas

All generated content was reviewed, modified, and tested carefully. 

## Contact

For questions about the original SPARTA implementation:
- GitHub: https://github.com/leomccormack/SPARTA
- Author: Leo McCormack

For questions about this Python port:
- Open an issue in your repository
