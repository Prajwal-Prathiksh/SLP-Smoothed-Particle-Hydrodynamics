# Taylor Green Vortex Simulation

## PySPH Testing

* 00 - `test.py --openmp --scheme wcsph --pfreq 30`

* 01 - `test.py --openmp --scheme iisph --pfreq 5`

* 02 - `test.py --openmp --scheme wcsph --delta-sph --pfreq 30`

* 03 - `Elliptical-Drop-PySPH.py --openmp` - Gaussian Kernel, IsothermalEOS, LaminarViscosityDeltaSPH, Spatial_Acceleration | Correct Order

* 04 - `Elliptical-Drop-PySPH.py --openmp` - QuinticSpline Kernel, IsothermalEOS, LaminarViscosityDeltaSPH, Spatial_Acceleration | Correct Order