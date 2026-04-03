# Examiner Q&A Prep

## AE Physics

AE pulses propagate through oil and tank structures with attenuation from spreading, damping, and impedance mismatch at boundaries.

## AE Sensor Installation and Acquisition

Sensors require stable coupling and consistent mounting pressure; field coupling and tank geometry introduce additional dispersion and reflection not present in simple lab setups.

## AE vs Electrical Detection

AE is stronger against EMI but can be weaker for deeply embedded sources; electrical methods are often more sensitive but more EMI-prone.

## Simulated Data Limitations

Synthetic pulses + mathematical noise improve controllability but do not fully model structural reverberation and propagation physics.

## WGAN Overfitting Concern and Mitigation

Mitigation implemented as two targeted modules:
- noise-only WGAN (`wgan_noise_aug.py`)
- conditional pulse-shape WGAN (`wgan_pulse_aug.py`)

Both enforce gradient penalty, spectral normalization, feature matching, and MMD-based early-stopping design constraints.

## Source Location Ambiguity

Current denoiser/classifier stack is optimized for type discrimination, not explicit localization regression.

## Mechanical vs PD Noise Discrimination

Current pipeline separates PD and synthetic corruption in training; future robustness should include more field-collected mechanical transients.
