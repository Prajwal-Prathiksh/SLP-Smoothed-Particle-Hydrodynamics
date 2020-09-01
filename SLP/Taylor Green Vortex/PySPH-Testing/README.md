# Taylor Green Vortex Simulation

## PySPH Testing

* 00 - `pysph run taylor_green --openmp --nx 50 --pfreq 100 --scheme wcsph --delta-sph`


* 01 - `pysph run taylor_green --openmp --nx 50 --pfreq 100 --scheme wcsph --delta-sph --perturb 0.2`

* 02 - `Taylor_Green_3.py --openmp --pfreq 100` - nx = 50 | perturb = 0

* 03 - `Taylor_Green_3.py --openmp --pfreq 100` - nx = 50 | perturb = 0.2

* 04 - `Taylor_Green-PySPH.py --openmp --pfreq 100` - nx = 50 | perturb = 0 | c0 = 10.0 | IsothermalEOS, Spatial_Acceleration

* 05 - `Taylor_Green-PySPH.py --openmp --pfreq 100` - nx = 50 | perturb = 0 | c0 = 10.0 | IsothermalEOS, Spatial_Acceleration, LaminarViscosityDeltaSPHPreStep | QuinticSpline | PECIntegrator

* 06 - `Taylor_Green-PySPH.py --openmp --pfreq 100` - nx = 50 | perturb = 0 | c0 = 10.0 | IsothermalEOS, Spatial_Acceleration, LaminarViscosityDeltaSPHPreStep - Correct Order | QuinticSpline | PECIntegrator

* 07 - `Taylor_Green-PySPH.py --openmp --pfreq 100` - nx = 50 | perturb = 0.2 | c0 = 10.0 | IsothermalEOS, Spatial_Acceleration, LaminarViscosityDeltaSPHPreStep - Correct Order | QuinticSpline | PECIntegrator

* 08 - `pysph run taylor_green --openmp --nx 50 --pfreq 100 --scheme edac --perturb 0.2`

* 09 - `pysph run taylor_green --openmp --nx 50 --pfreq 100 --scheme edac `

* 10 - `Taylor_Green-PySPH.py --openmp --pfreq 100` - nx = 50 | perturb = 0.2 | c0 = 10.0 | TaitEOS, Spatial_Acceleration, LaminarViscosityDeltaSPHPreStep - Correct Order | QuinticSpline | PECIntegrator

* 11 - `Taylor_Green-PySPH.py --openmp --pfreq 100` - nx = 30 | perturb = 0.2 | c0 = 10.0 | IsothermalEOS, Spatial_Acceleration, LaminarViscosityDeltaSPHPreStep - Correct Order | QuinticSpline | PECIntegrator

* 12 - `Taylor_Green_3.py --openmp --pfreq 100` - nx = 30 | perturb = 0.2

* 13 - `pysph run taylor_green --openmp --nx 30 --pfreq 100 --scheme wcsph --delta-sph --perturb 0.2`

* 14 - `pysph run taylor_green --openmp --nx 30 --pfreq 100 --scheme edac --perturb 0.2`

* 15 - `Taylor_Green-PySPH.py --openmp --pfreq 100` - nx = 30 | perturb = 0.2 | c0 = 10.0 | IsothermalEOS, Spatial_Acceleration, LaminarViscosityDeltaSPHPreStep - Correct Order | WendlandQuintic, hdx = 2 | PECIntegrator

* 16- `Taylor_Green-PySPH.py --openmp --pfreq 100` - nx = 30 | perturb = 0.2 | c0 = 10.0 | IsothermalEOS, Spatial_Acceleration, LaminarViscosityDeltaSPHPreStep - Correct Order | WendlandQuintic, hdx = 1.33 | PECIntegrator

* 17- `Taylor_Green-PySPH.py --openmp --pfreq 100` - nx = 30 | perturb = 0.2 | c0 = 10.0 | IsothermalEOS, Spatial_Acceleration, LaminarViscosityDeltaSPHPreStep - Correct Order | WendlandQuintic, hdx = 1.33 | PECIntegrator(fluid = DPSPHStep())




* check - **Same as 00**    