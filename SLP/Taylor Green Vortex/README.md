# Taylor Green Vortex Simulation

Contains the implementation of the Taylor-Green Vortex problem using the δ+ SPH scheme.

#### The files are as follows:
1. Taylor_Green.py:
    * QuinticSpline - kernel
    * EulerIntegrator
    * ContinuityEquation_DPSPH

1. Taylor_Green_2.py:
    * QuinticSpline - kernel
    * EulerIntegrator
    * ContinuityEquation_RDGC_DPSPH

1. Taylor_Green_3.py:
    * WendlandQuintic - kernel
    * EulerIntegrator - with PST correction
    * ContinuityEquation_RDGC_DPSPH
    * ParticleShiftingTechnique