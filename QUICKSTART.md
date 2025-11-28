# Quick Start Guide - FOA Powermap Generator

## Installation

```bash
# Install dependencies
pip install numpy scipy matplotlib

# Download files
# - powermap_foa.py
# - demo_simple.py (optional)
# - powermap_example.py (optional)
```

## 30-Second Example

```python
from powermap_foa import PowermapFOA, PowermapMode
import numpy as np

# Create analyzer
pm = PowermapFOA(sample_rate=48000, frame_size=1024)
pm.mode = PowermapMode.MUSIC

# Create test FOA audio (4 channels: W, Y, Z, X)
audio = np.random.randn(4, 1024) * 0.1

# Process
pm.process_frame(audio)

# Get result
directions, powermap = pm.get_powermap()
peak_idx = np.argmax(powermap)
azi, ele = directions[peak_idx]
print(f"Peak at: {azi:.1f}° azimuth, {ele:.1f}° elevation")
```

## Run Examples

```bash
# Simple demo with visualization
python demo_simple.py

# Comprehensive examples
python powermap_example.py

# Run tests
python test_powermap_foa.py
```

## Common Use Cases

### 1. Analyze Recorded FOA Audio

```python
import soundfile as sf
from powermap_foa import PowermapFOA, PowermapMode

# Load FOA recording (4 channels)
audio, sr = sf.read('recording.wav')  # Shape: (n_samples, 4)
audio = audio.T  # Transpose to (4, n_samples)

# Setup
pm = PowermapFOA(sample_rate=sr, frame_size=1024)
pm.mode = PowermapMode.MVDR

# Process frames
frame_size = 1024
for i in range(0, len(audio[0]) - frame_size, frame_size):
    frame = audio[:, i:i+frame_size]
    pm.process_frame(frame)
    dirs, pmap = pm.get_powermap()
    # Analyze pmap...
```

### 2. Real-Time Monitoring

```python
import sounddevice as sd
from powermap_foa import PowermapFOA, PowermapMode

pm = PowermapFOA(sample_rate=48000, frame_size=512)
pm.mode = PowermapMode.PWD  # Fast mode

def callback(indata, frames, time, status):
    foa = indata.T  # (4, frames)
    pm.process_frame(foa)
    if pm.pmap_ready:
        dirs, pmap = pm.get_powermap()
        # Update display...

stream = sd.InputStream(channels=4, samplerate=48000, 
                       blocksize=512, callback=callback)
stream.start()
```

### 3. Multiple Source Detection

```python
from powermap_foa import PowermapFOA, PowermapMode

pm = PowermapFOA(sample_rate=48000, frame_size=1024)
pm.mode = PowermapMode.MUSIC
pm.n_sources = 2  # Expecting 2 sources

# Process audio with multiple sources
pm.process_frame(audio)
dirs, pmap = pm.get_powermap(normalize=True)

# Find peaks
threshold = 0.5
peaks = np.where(pmap > threshold)[0]
peak_directions = dirs[peaks]
print(f"Found {len(peaks)} peaks")
```

### 4. Visualization

```python
import matplotlib.pyplot as plt

# Get 2D image
image = pm.get_powermap_image(width=720, aspect_ratio=2.0)

# Display
plt.figure(figsize=(12, 6))
plt.imshow(image, extent=[-180, 180, -90, 90], 
          origin='lower', cmap='hot', aspect='auto')
plt.colorbar(label='Power')
plt.xlabel('Azimuth (°)')
plt.ylabel('Elevation (°)')
plt.title('FOA Powermap')
plt.show()
```

## Configuration Tips

### For Best Accuracy
```python
pm = PowermapFOA(
    sample_rate=48000,
    frame_size=2048,        # Larger frame
    grid_resolution=5       # Higher resolution
)
pm.mode = PowermapMode.MUSIC
pm.n_sources = 1
pm.pmap_avg_coeff = 0.1   # Less averaging
```

### For Real-Time Performance
```python
pm = PowermapFOA(
    sample_rate=48000,
    frame_size=512,         # Smaller frame
    grid_resolution=2       # Lower resolution
)
pm.mode = PowermapMode.PWD  # Fastest mode
pm.pmap_avg_coeff = 0.5     # More smoothing
```

### For Visualization
```python
pm = PowermapFOA(sample_rate=48000, frame_size=1024, grid_resolution=3)
pm.mode = PowermapMode.MVDR
pm.pmap_avg_coeff = 0.4     # Smooth animation
```

## Troubleshooting

### No Peak Detected
- Check audio level (should be audible)
- Verify channel order (ACN vs FuMa)
- Try different mode (PWD is most robust)

### Wrong Direction
- Check channel order: `pm.channel_order = ChannelOrder.FUMA`
- Check normalization: `pm.norm_type = NormType.FUMA`
- Verify FOA encoding is correct

### Performance Issues
- Lower `grid_resolution` (2 or 3)
- Smaller `frame_size` (512)
- Use PWD or MVDR mode
- Consider Numba/Cython for speedup

## FOA Format Reference

### ACN Ordering (Default)
```
Channel 0: W (omnidirectional)
Channel 1: Y (left-right)
Channel 2: Z (up-down)
Channel 3: X (front-back)
```

### SN3D Normalization (Default)
```
W: 1.0
Y, Z, X: √3
```

### Azimuth Convention
```
0° = Front
90° = Left
180° = Back
270° = Right
```

### Elevation Convention
```
+90° = Up
0° = Horizontal
-90° = Down
```

## Next Steps

1. Try `demo_simple.py` to see basic usage
2. Explore `powermap_example.py` for advanced features
3. Read `README.md` for complete documentation
4. Run `test_powermap_foa.py` to verify installation

## Support

- Original SPARTA: https://github.com/leomccormack/SPARTA
- FOA Resources: http://www.ambisonic.net/
- Spatial Audio: https://en.wikipedia.org/wiki/Ambisonics
