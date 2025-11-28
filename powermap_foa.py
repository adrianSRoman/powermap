"""
First Order Ambisonics (FOA) Powermap Generator

This is a Python implementation of the powermap generator from the SPARTA
VST plugin suite, specifically for first-order ambisonics (FOA).

Based on the C implementation from:
https://github.com/leomccormack/SPARTA
https://github.com/leomccormack/Spatial_Audio_Framework

Reference:
McCormack, L., Delikaris-Manias, S. and Pulkki, V., 2017.
"Parametric acoustic camera for real-time sound capture, analysis and tracking."
In Proceedings of the 20th International Conference on Digital Audio Effects (DAFx-17)

Author: Python implementation
License: ISC (matching original SPARTA license)
"""

import numpy as np
from scipy import signal
from scipy.linalg import eigh, svd
from enum import Enum
from typing import Tuple, Optional


class PowermapMode(Enum):
    """Available powermap/activity-map computation modes"""
    PWD = 1          # Plane Wave Decomposition (hypercardioid beamformers)
    MVDR = 2         # Minimum Variance Distortionless Response
    CROPAC_LCMV = 3  # Cross-Pattern Coherence LCMV
    MUSIC = 4        # Multiple Signal Classification
    MUSIC_LOG = 5    # MUSIC with logarithmic output
    MINNORM = 6      # Minimum Norm subspace method
    MINNORM_LOG = 7  # MinNorm with logarithmic output


class ChannelOrder(Enum):
    """Ambisonic channel ordering conventions"""
    ACN = 1   # Ambisonic Channel Number (modern standard)
    FUMA = 2  # Furse-Malham (legacy, FOA only)


class NormType(Enum):
    """Ambisonic normalization conventions"""
    N3D = 1   # Full 3D normalization
    SN3D = 2  # Schmidt semi-normalization (modern standard)
    FUMA = 3  # Furse-Malham (legacy, FOA only)


class PowermapFOA:
    """
    First Order Ambisonics Powermap Generator
    
    Analyzes FOA audio signals to generate spatial activity maps showing
    sound source directions and intensities.
    """
    
    def __init__(
        self,
        sample_rate: float = 48000.0,
        frame_size: int = 1024,
        hop_size: int = 128,
        grid_resolution: int = 9
    ):
        """
        Initialize the FOA powermap analyzer.
        
        Parameters
        ----------
        sample_rate : float
            Audio sample rate in Hz
        frame_size : int
            Analysis frame size in samples
        hop_size : int
            STFT hop size in samples
        grid_resolution : int
            Spherical grid resolution (higher = more directions)
        """
        self.sample_rate = sample_rate
        self.frame_size = frame_size
        self.hop_size = hop_size
        self.n_fft = frame_size
        
        # FOA is 1st order: (N+1)^2 = 4 channels (W, X, Y, Z)
        self.sh_order = 1
        self.n_channels = 4
        
        # Settings
        self.mode = PowermapMode.MUSIC
        self.channel_order = ChannelOrder.ACN
        self.norm_type = NormType.SN3D
        self.n_sources = 1
        
        # Averaging coefficients
        self.cov_avg_coeff = 0.0  # Covariance matrix averaging [0..1]
        self.pmap_avg_coeff = 0.25  # Powermap averaging [0..1]
        
        # Signal detection threshold
        self.power_threshold = 1e-6  # Minimum signal power to generate powermap
        self.current_power = 0.0  # Current frame power (for diagnostics)
        
        # Generate spherical scanning grid
        self.grid_dirs_deg, self.grid_dirs_rad = self._generate_grid(grid_resolution)
        self.n_dirs = len(self.grid_dirs_deg)
        
        # Compute real spherical harmonic basis functions at grid points
        self.Y_grid = self._compute_sh_basis(self.grid_dirs_rad)
        self.Y_grid_complex = self.Y_grid.astype(np.complex64)
        
        # Initialize STFT
        self.n_bands = hop_size + 5  # Hybrid filterbank bands
        self.time_slots = frame_size // hop_size
        
        # Initialize buffers
        self.input_buffer = np.zeros((self.n_channels, frame_size))
        self.buffer_idx = 0
        
        # Covariance matrices per frequency band
        self.Cx = np.zeros((self.n_bands, self.n_channels, self.n_channels), 
                           dtype=np.complex64)
        
        # Powermap output
        self.pmap = np.zeros(self.n_dirs)
        self.prev_pmap = np.zeros(self.n_dirs)
        self.pmap_ready = False
        
        # EQ per band (for frequency-dependent weighting)
        self.pmap_eq = np.ones(self.n_bands)
        
    def _generate_grid(self, resolution: int) -> Tuple[np.ndarray, np.ndarray]:
        """
        Generate a spherical scanning grid using an icosahedron subdivision.
        
        For simplicity, we use a uniform angular grid. For better results,
        the original uses geodesic polyhedron subdivisions.
        
        Returns
        -------
        dirs_deg : np.ndarray
            Grid directions in degrees, shape (n_dirs, 2) [azimuth, elevation]
        dirs_rad : np.ndarray
            Grid directions in radians, shape (n_dirs, 2) [azimuth, elevation]
        """
        # Create a reasonably uniform spherical grid
        n_azi = 36 * resolution
        n_ele = 18 * resolution + 1
        
        azimuths = np.linspace(-180, 180, n_azi, endpoint=True)
        elevations = np.linspace(-90, 90, n_ele, endpoint=True)
        
        # Create mesh grid
        azi_grid, ele_grid = np.meshgrid(azimuths, elevations)
        
        dirs_deg = np.stack([azi_grid.flatten(), ele_grid.flatten()], axis=1)
        dirs_rad = np.deg2rad(dirs_deg)
        
        return dirs_deg, dirs_rad
    
    def _compute_sh_basis(self, dirs_rad: np.ndarray) -> np.ndarray:
        """
        Compute real spherical harmonic basis functions for first order.
        
        Parameters
        ----------
        dirs_rad : np.ndarray
            Directions in radians, shape (n_dirs, 2) [azimuth, elevation]
            
        Returns
        -------
        Y : np.ndarray
            SH basis matrix, shape (4, n_dirs)
            ACN ordering: [W, Y, Z, X] = [0, -1, 0, 1] (degree, order)
        """
        n_dirs = len(dirs_rad)
        Y = np.zeros((4, n_dirs))
        
        azi = dirs_rad[:, 0]
        ele = dirs_rad[:, 1]
        
        # Real spherical harmonics for 1st order (SN3D normalization)
        # These are the standard real SH basis functions for FOA
        # ACN ordering: [W, Y, Z, X] corresponding to [Y_0^0, Y_1^-1, Y_1^0, Y_1^1]
        
        # Order 0 (W) - omnidirectional
        Y[0, :] = 1.0
        
        # Order 1 (SN3D normalization - no sqrt(3) factor)
        # Y_1^-1 (Y channel in ACN) - left/right
        Y[1, :] = np.cos(ele) * np.sin(azi)
        
        # Y_1^0 (Z channel in ACN) - up/down
        Y[2, :] = np.sin(ele)
        
        # Y_1^1 (X channel in ACN) - front/back
        Y[3, :] = np.cos(ele) * np.cos(azi)
        
        return Y
    
    def _convert_channel_order(self, audio: np.ndarray, 
                                from_order: ChannelOrder) -> np.ndarray:
        """Convert channel ordering convention."""
        if from_order == ChannelOrder.ACN:
            return audio
        elif from_order == ChannelOrder.FUMA:
            # FUMA: [W, X, Y, Z] -> ACN: [W, Y, Z, X]
            return audio[[0, 2, 3, 1], :]
        return audio
    
    def _convert_normalization(self, audio: np.ndarray,
                                from_norm: NormType) -> np.ndarray:
        """Convert normalization convention to SN3D (internal format)."""
        if from_norm == NormType.SN3D:
            # Already in SN3D, no conversion needed
            return audio
        elif from_norm == NormType.N3D:
            # N3D to SN3D conversion factors for FOA
            factors = np.array([1.0, 1.0/np.sqrt(3), 1.0/np.sqrt(3), 1.0/np.sqrt(3)])
            return audio * factors[:, np.newaxis]
        elif from_norm == NormType.FUMA:
            # FUMA to SN3D conversion factors
            # FUMA W has sqrt(2) factor, and XYZ are already SN3D-like
            factors = np.array([np.sqrt(2), 1.0, 1.0, 1.0])
            return audio * factors[:, np.newaxis]
        return audio
    
    def _stft_analysis(self, audio: np.ndarray) -> np.ndarray:
        """
        Perform STFT analysis on input audio.
        
        Parameters
        ----------
        audio : np.ndarray
            Input audio, shape (n_channels, frame_size)
            
        Returns
        -------
        stft : np.ndarray
            STFT coefficients, shape (n_channels, n_bands, time_slots)
        """
        window = signal.windows.hann(self.n_fft)
        
        stft_data = []
        for ch in range(self.n_channels):
            f, t, Zxx = signal.stft(
                audio[ch],
                fs=self.sample_rate,
                window=window,
                nperseg=self.n_fft,
                noverlap=self.n_fft - self.hop_size,
                nfft=self.n_fft
            )
            stft_data.append(Zxx)
        
        stft = np.array(stft_data)  # shape: (n_channels, n_freqs, n_times)
        
        # Trim to match expected band count
        if stft.shape[1] > self.n_bands:
            stft = stft[:, :self.n_bands, :]
            
        return stft
    
    def _compute_covariance(self, stft: np.ndarray) -> None:
        """
        Update covariance matrices from STFT data.
        
        Parameters
        ----------
        stft : np.ndarray
            STFT coefficients, shape (n_channels, n_bands, time_slots)
        """
        for band in range(min(self.n_bands, stft.shape[1])):
            # Get SH coefficients for this band: (n_channels, time_slots)
            X_band = stft[:, band, :]
            
            # Compute covariance: Cx = X @ X^H / time_slots
            new_Cx = (X_band @ X_band.conj().T) / self.time_slots
            
            # Exponential averaging
            self.Cx[band] = (self.cov_avg_coeff * self.Cx[band] + 
                            (1.0 - self.cov_avg_coeff) * new_Cx)
    
    def _generate_pwd_map(self, Cx_grouped: np.ndarray) -> np.ndarray:
        """
        Generate Plane Wave Decomposition (PWD) powermap.
        
        This uses hypercardioid beamformers pointing at each grid direction.
        
        Parameters
        ----------
        Cx_grouped : np.ndarray
            Grouped covariance matrix, shape (n_channels, n_channels)
            
        Returns
        -------
        pmap : np.ndarray
            Powermap values, shape (n_dirs,)
        """
        pmap = np.zeros(self.n_dirs)
        
        for i in range(self.n_dirs):
            # Steering vector (SH basis at this direction)
            a = self.Y_grid_complex[:, i]
            
            # PWD: P(theta) = a^H * Cx * a
            pmap[i] = np.real(a.conj().T @ Cx_grouped @ a)
        
        return np.maximum(pmap, 0)
    
    def _generate_mvdr_map(self, Cx_grouped: np.ndarray, 
                           reg_param: float = 8.0) -> np.ndarray:
        """
        Generate MVDR (Minimum Variance Distortionless Response) powermap.
        
        Parameters
        ----------
        Cx_grouped : np.ndarray
            Grouped covariance matrix, shape (n_channels, n_channels)
        reg_param : float
            Regularization parameter for matrix inversion
            
        Returns
        -------
        pmap : np.ndarray
            Powermap values, shape (n_dirs,)
        """
        pmap = np.zeros(self.n_dirs)
        
        # Add diagonal loading for regularization
        trace = np.trace(Cx_grouped).real
        Cx_reg = Cx_grouped + (trace / self.n_channels / reg_param) * np.eye(self.n_channels)
        
        try:
            Cx_inv = np.linalg.inv(Cx_reg)
        except np.linalg.LinAlgError:
            return pmap
        
        for i in range(self.n_dirs):
            a = self.Y_grid_complex[:, i]
            
            # MVDR: P(theta) = 1 / (a^H * Cx^-1 * a)
            denom = np.real(a.conj().T @ Cx_inv @ a)
            if denom > 1e-10:
                pmap[i] = 1.0 / denom
        
        return np.maximum(pmap, 0)
    
    def _generate_music_map(self, Cx_grouped: np.ndarray, 
                            use_log: bool = False) -> np.ndarray:
        """
        Generate MUSIC (Multiple Signal Classification) powermap.
        
        Parameters
        ----------
        Cx_grouped : np.ndarray
            Grouped covariance matrix, shape (n_channels, n_channels)
        use_log : bool
            If True, return log of the values
            
        Returns
        -------
        pmap : np.ndarray
            Powermap values, shape (n_dirs,)
        """
        pmap = np.zeros(self.n_dirs)
        
        # Eigendecomposition
        eigenvalues, eigenvectors = eigh(Cx_grouped)
        
        # Sort eigenvalues in descending order
        idx = eigenvalues.argsort()[::-1]
        eigenvalues = eigenvalues[idx]
        eigenvectors = eigenvectors[:, idx]
        
        # Noise subspace (eigenvectors corresponding to smallest eigenvalues)
        n_signal = min(self.n_sources, self.n_channels - 1)
        noise_subspace = eigenvectors[:, n_signal:]
        
        # Compute MUSIC spectrum
        for i in range(self.n_dirs):
            a = self.Y_grid_complex[:, i]
            
            # Project onto noise subspace
            projection = noise_subspace @ noise_subspace.conj().T @ a
            norm_sq = np.abs(np.vdot(projection, projection))
            
            # MUSIC: P(theta) = 1 / ||P_N * a||^2
            if norm_sq > 1e-10:
                pmap[i] = 1.0 / norm_sq
        
        if use_log:
            pmap = np.log10(pmap + 1e-10)
            
        return np.maximum(pmap, 0)
    
    def _generate_minnorm_map(self, Cx_grouped: np.ndarray,
                              use_log: bool = False) -> np.ndarray:
        """
        Generate Minimum Norm powermap.
        
        Parameters
        ----------
        Cx_grouped : np.ndarray
            Grouped covariance matrix, shape (n_channels, n_channels)
        use_log : bool
            If True, return log of the values
            
        Returns
        -------
        pmap : np.ndarray
            Powermap values, shape (n_dirs,)
        """
        pmap = np.zeros(self.n_dirs)
        
        # Eigendecomposition
        eigenvalues, eigenvectors = eigh(Cx_grouped)
        
        # Sort eigenvalues in descending order
        idx = eigenvalues.argsort()[::-1]
        eigenvalues = eigenvalues[idx]
        eigenvectors = eigenvectors[:, idx]
        
        # Noise subspace
        n_signal = min(self.n_sources, self.n_channels - 1)
        noise_subspace = eigenvectors[:, n_signal:]
        
        # Minimum norm vector (first element = 1, minimum norm constraint)
        try:
            # Create constraint matrix [1, 0, 0, ...]^T
            e1 = np.zeros((self.n_channels, 1), dtype=np.complex64)
            e1[0] = 1.0
            
            # Compute pseudo-inverse of noise subspace with constraint
            G = noise_subspace @ noise_subspace.conj().T
            G_inv = np.linalg.pinv(G)
            c = G_inv @ e1
            c = c / np.sqrt(np.vdot(c, c))  # Normalize
            
            # Compute spectrum
            for i in range(self.n_dirs):
                a = self.Y_grid_complex[:, i]
                val = np.abs(np.vdot(a, c))**2
                if val > 1e-10:
                    pmap[i] = 1.0 / val
        except:
            pass
        
        if use_log:
            pmap = np.log10(pmap + 1e-10)
            
        return np.maximum(pmap, 0)
    
    def _generate_powermap(self) -> np.ndarray:
        """
        Generate powermap based on current mode and covariance data.
        
        Returns
        -------
        pmap : np.ndarray
            Powermap values, shape (n_dirs,)
        """
        # Group covariance matrices across frequency bands with EQ weighting
        Cx_grouped = np.zeros((self.n_channels, self.n_channels), dtype=np.complex64)
        
        for band in range(self.n_bands):
            weight = self.pmap_eq[band] * 1e3  # Scale factor from original
            Cx_grouped += self.Cx[band] * weight
        
        # Compute signal power from covariance matrix
        trace = np.trace(Cx_grouped).real
        self.current_power = trace / self.n_channels  # Average power across channels
        
        # Check if signal power exceeds threshold
        if self.current_power < self.power_threshold:
            # Silence detected - return zero powermap
            return np.zeros(self.n_dirs)
        
        # Additional check for numerical stability
        if trace < 1e-8:
            return np.zeros(self.n_dirs)
        
        # Generate powermap based on selected mode
        if self.mode == PowermapMode.PWD:
            pmap = self._generate_pwd_map(Cx_grouped)
        elif self.mode == PowermapMode.MVDR:
            pmap = self._generate_mvdr_map(Cx_grouped)
        elif self.mode == PowermapMode.MUSIC:
            pmap = self._generate_music_map(Cx_grouped, use_log=False)
        elif self.mode == PowermapMode.MUSIC_LOG:
            pmap = self._generate_music_map(Cx_grouped, use_log=True)
        elif self.mode == PowermapMode.MINNORM:
            pmap = self._generate_minnorm_map(Cx_grouped, use_log=False)
        elif self.mode == PowermapMode.MINNORM_LOG:
            pmap = self._generate_minnorm_map(Cx_grouped, use_log=True)
        else:
            pmap = self._generate_pwd_map(Cx_grouped)
        
        return pmap
    
    def process_frame(self, audio: np.ndarray) -> bool:
        """
        Process a frame of FOA audio data.
        
        Parameters
        ----------
        audio : np.ndarray
            FOA input audio, shape (4, frame_size)
            Expected ordering: ACN [W, Y, Z, X]
            Expected normalization: SN3D (can be configured)
            
        Returns
        -------
        ready : bool
            True if a new powermap is ready
        """
        # Convert channel ordering and normalization to internal format (ACN, N3D)
        audio = self._convert_channel_order(audio, self.channel_order)
        audio = self._convert_normalization(audio, self.norm_type)
        
        # Perform STFT
        stft = self._stft_analysis(audio)
        
        # Update covariance matrices
        self._compute_covariance(stft)
        
        # Generate powermap
        pmap = self._generate_powermap()
        
        # Temporal averaging (only if there's signal)
        if self.current_power >= self.power_threshold:
            self.pmap = (1.0 - self.pmap_avg_coeff) * pmap + \
                        self.pmap_avg_coeff * self.prev_pmap
            self.prev_pmap = self.pmap.copy()
        else:
            # During silence, zero out the powermap
            self.pmap = np.zeros(self.n_dirs)
            # Optionally decay previous powermap to avoid long tails
            self.prev_pmap *= 0.5
        
        self.pmap_ready = True
        return True
    
    def get_powermap(self, normalize: bool = True) -> Tuple[np.ndarray, np.ndarray]:
        """
        Get the current powermap.
        
        Parameters
        ----------
        normalize : bool
            If True, normalize powermap to [0, 1] range
            
        Returns
        -------
        grid_dirs : np.ndarray
            Grid directions in degrees, shape (n_dirs, 2) [azimuth, elevation]
        pmap : np.ndarray
            Powermap values, shape (n_dirs,)
        """
        if not self.pmap_ready:
            return self.grid_dirs_deg, np.zeros(self.n_dirs)
        
        pmap = self.pmap.copy()
        
        if normalize and pmap.max() > pmap.min():
            pmap = (pmap - pmap.min()) / (pmap.max() - pmap.min())
        
        return self.grid_dirs_deg, pmap
    
    def set_power_threshold(self, threshold: float) -> None:
        """
        Set the minimum signal power threshold for powermap generation.
        
        Parameters
        ----------
        threshold : float
            Minimum power level. Frames below this will produce zero powermap.
            Typical values: 1e-6 to 1e-4 depending on recording conditions.
        """
        self.power_threshold = max(0.0, threshold)
    
    def get_current_power(self) -> float:
        """
        Get the current frame's signal power.
        
        Returns
        -------
        power : float
            Current signal power level
        """
        return self.current_power
    
    def get_powermap_image(self, width: int = 720, 
                          aspect_ratio: float = 2.0) -> np.ndarray:
        """
        Get powermap as a 2D image array.
        
        Parameters
        ----------
        width : int
            Image width in pixels
        aspect_ratio : float
            Aspect ratio (width/height)
            
        Returns
        -------
        image : np.ndarray
            Powermap image, shape (height, width)
        """
        height = int(width / aspect_ratio)
        
        # Create 2D grid
        azi_img = np.linspace(-180, 180, width)
        ele_img = np.linspace(-90, 90, height)
        azi_grid, ele_grid = np.meshgrid(azi_img, ele_img)
        
        # Interpolate powermap to image grid
        from scipy.interpolate import griddata
        
        _, pmap = self.get_powermap(normalize=True)
        
        image = griddata(
            self.grid_dirs_deg,
            pmap,
            (azi_grid, ele_grid),
            method='linear',
            fill_value=0
        )
        
        return image


# Example usage
if __name__ == "__main__":
    # Create powermap analyzer
    analyzer = PowermapFOA(
        sample_rate=48000,
        frame_size=1024,
        hop_size=128,
        grid_resolution=2
    )
    
    # Configure settings
    analyzer.mode = PowermapMode.MUSIC
    analyzer.n_sources = 2
    analyzer.pmap_avg_coeff = 0.25
    
    # Simulate FOA audio (4 channels: W, Y, Z, X in ACN)
    # This creates a simple test signal from the front
    duration = 1.0
    n_samples = int(analyzer.sample_rate * duration)
    
    # Create a signal from azimuth=0°, elevation=0° (front)
    # In FOA: strong W and X components
    audio_foa = np.zeros((4, analyzer.frame_size))
    t = np.linspace(0, analyzer.frame_size / analyzer.sample_rate, analyzer.frame_size)
    test_signal = np.sin(2 * np.pi * 1000 * t)  # 1 kHz tone
    
    audio_foa[0, :] = test_signal * 0.5  # W (omnidirectional)
    audio_foa[3, :] = test_signal * 0.5  # X (front-back)
    
    # Process frame
    analyzer.process_frame(audio_foa)
    
    # Get powermap
    grid_dirs, pmap = analyzer.get_powermap()
    
    # Find peak direction
    peak_idx = np.argmax(pmap)
    peak_dir = grid_dirs[peak_idx]
    
    print(f"Powermap generated with {len(pmap)} directions")
    print(f"Peak direction: azimuth={peak_dir[0]:.1f}°, elevation={peak_dir[1]:.1f}°")
    print(f"Peak value: {pmap[peak_idx]:.3f}")
