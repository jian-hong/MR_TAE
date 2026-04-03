"""
Unit tests for data generation.
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

import pytest
import torch
import numpy as np

from mr_tae_fusion.config import Config, SignalConfig, NoiseConfig
from mr_tae_fusion.data import (
    generate_dep, generate_dop, generate_pd_signal,
    generate_wgn, generate_impulsive_noise, generate_composite_noise,
    add_noise_at_snr, CurriculumNoiseScheduler,
    PDSignalDataset
)


class TestPulseGenerators:
    """Test pulse generation functions."""
    
    @pytest.fixture
    def time_vector(self):
        """Create a time vector."""
        return np.linspace(0, 2e-6, 2001)
    
    def test_generate_dep(self, time_vector):
        """Test DEP generation."""
        pulse = generate_dep(
            time_vector,
            amplitude=5.0,
            tau1=50e-9,
            tau2=5e-9,
            t_start=1e-6
        )
        
        assert pulse.shape == time_vector.shape
        assert np.max(np.abs(pulse)) > 0  # Pulse should have amplitude
        
        # Pulse should be zero before start time
        before_start = time_vector < 0.9e-6
        assert np.allclose(pulse[before_start], 0, atol=1e-10)
    
    def test_generate_dop(self, time_vector):
        """Test DOP generation."""
        pulse = generate_dop(
            time_vector,
            amplitude=5.0,
            tau=30e-9,
            fc=50e6,
            t_start=1e-6
        )
        
        assert pulse.shape == time_vector.shape
        
        # Should have oscillating pattern (sign changes)
        after_start = time_vector > 1e-6
        if np.any(after_start):
            signs = np.sign(pulse[after_start])
            assert np.any(signs > 0) and np.any(signs < 0)  # Oscillation
    
    def test_generate_pd_signal_types(self):
        """Test all signal types generate successfully."""
        for signal_type in ['A', 'B', 'C', 'D', 'E', 'F']:
            t, signal, mask = generate_pd_signal(signal_type)
            
            assert len(t) == len(signal) == len(mask)
            assert signal.dtype == np.float64
            assert mask.dtype == np.int64
    
    def test_invalid_signal_type(self):
        """Test invalid signal type raises error."""
        with pytest.raises(ValueError):
            generate_pd_signal('Invalid')


class TestNoiseGenerators:
    """Test noise generation functions."""
    
    def test_generate_wgn(self):
        """Test white Gaussian noise."""
        noise = generate_wgn(1000, amplitude=0.1)
        
        assert len(noise) == 1000
        assert np.abs(np.std(noise) - 0.1) < 0.05  # Close to target std
    
    def test_generate_impulsive(self):
        """Test impulsive noise is sparse."""
        noise = generate_impulsive_noise(
            10000,
            probability=0.01,
            scale=1.0
        )
        
        # Should be sparse
        nonzero_fraction = np.count_nonzero(noise) / len(noise)
        assert nonzero_fraction < 0.05  # Less than 5% non-zero
        
        # Should have some high amplitude spikes
        assert np.max(np.abs(noise)) > 0.1
    
    def test_add_noise_at_snr(self):
        """Test SNR control."""
        signal = np.random.randn(1000)
        noise = np.random.randn(1000)
        
        target_snr = -10.0
        noisy, actual_snr = add_noise_at_snr(signal, noise, target_snr)
        
        # Actual SNR should be close to target
        assert np.abs(actual_snr - target_snr) < 1.0


class TestCurriculumScheduler:
    """Test curriculum learning scheduler."""
    
    def test_phase_progression(self):
        """Test phases progress correctly."""
        config = NoiseConfig()
        scheduler = CurriculumNoiseScheduler(config)
        
        # Phase 1 at start
        snr_range = scheduler.get_snr_range(0, 100)
        assert snr_range == config.snr_range_phase1
        
        # Phase 2 at middle
        snr_range = scheduler.get_snr_range(30, 100)
        assert snr_range == config.snr_range_phase2
        
        # Phase 3 at end
        snr_range = scheduler.get_snr_range(60, 100)
        assert snr_range == config.snr_range_phase3
    
    def test_noise_type_progression(self):
        """Test noise type changes with phase."""
        scheduler = CurriculumNoiseScheduler()
        
        # Phase 1: WGN only
        assert scheduler.get_noise_type(5, 100) == 'wgn'
        
        # Phase 2: WGN + impulsive
        assert scheduler.get_noise_type(30, 100) == 'wgn_impulsive'
        
        # Phase 3: Composite
        assert scheduler.get_noise_type(60, 100) == 'composite'


class TestPDSignalDataset:
    """Test PyTorch Dataset."""
    
    @pytest.fixture
    def config(self):
        return Config()
    
    def test_dataset_length(self, config):
        """Test dataset reports correct length."""
        dataset = PDSignalDataset(config, num_samples=100)
        assert len(dataset) == 100
    
    def test_getitem_returns_dict(self, config):
        """Test __getitem__ returns expected keys."""
        dataset = PDSignalDataset(config, num_samples=10)
        sample = dataset[0]
        
        assert 'noisy' in sample
        assert 'clean' in sample
        assert 'mask' in sample
        assert 'snr' in sample
        assert 'type' in sample
    
    def test_getitem_shapes(self, config):
        """Test returned tensors have correct shapes."""
        dataset = PDSignalDataset(config, num_samples=10)
        sample = dataset[0]
        
        # Noisy and clean should be (1, L)
        assert sample['noisy'].dim() == 2
        assert sample['noisy'].shape[0] == 1
        
        # Mask should be (L,)
        assert sample['mask'].dim() == 1
        
        # Lengths should match
        assert sample['noisy'].shape[1] == sample['mask'].shape[0]
    
    def test_curriculum_update(self, config):
        """Test epoch update for curriculum."""
        dataset = PDSignalDataset(config, num_samples=10)
        
        dataset.update_epoch(0)
        info_early = dataset.get_curriculum_info()
        
        dataset.update_epoch(90)
        info_late = dataset.get_curriculum_info()
        
        # Phase should change
        assert info_early['phase'] != info_late['phase']
    
    def test_reproducibility(self, config):
        """Test reproducibility with same seed."""
        dataset1 = PDSignalDataset(config, num_samples=10, seed=42)
        dataset2 = PDSignalDataset(config, num_samples=10, seed=42)
        
        sample1 = dataset1[0]
        sample2 = dataset2[0]
        
        assert torch.allclose(sample1['clean'], sample2['clean'])


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
