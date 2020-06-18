###########################################################################
# IMPORTS
###########################################################################

# PySPH sph imports
from pysph.sph.equation import Equation, Group, MultiStageEquations
from pysph.sph.integrator import Integrator, IntegratorStep

################################################################################
# Delta_Plus - SPH Sceheme: Integrator
################################################################################

class EulerIntegrator_DPSPH(Integrator):
    def one_timestep(self, t, dt):
        # Evaluate Stage1
        self.compute_accelerations(0)
        self.stage1()
        self.update_domain()
        self.do_post_stage(dt, 1)

        # Evaluate Stage2 - PST Correction
        self.compute_accelerations(1, update_nnps=True)
        self.stage2()
        self.update_domain()
        self.do_post_stage(dt, 2)

class EulerStep_DPSPH(IntegratorStep):
    """Fast but inaccurate integrator. Use this for testing"""
    def stage1(self, d_idx, d_u, d_v, d_w, d_au, d_av, d_aw, d_x, d_y,
                  d_z, d_rho, d_arho, dt):
        d_u[d_idx] += dt*d_au[d_idx]
        d_v[d_idx] += dt*d_av[d_idx]
        d_w[d_idx] += dt*d_aw[d_idx]

        d_x[d_idx] += dt*d_u[d_idx]
        d_y[d_idx] += dt*d_v[d_idx]
        d_z[d_idx] += dt*d_w[d_idx]

        d_rho[d_idx] += dt*d_arho[d_idx]

    def stage2(self, d_idx, d_DX, d_DY, d_x, d_y):
        r"""
            Particle Shifting Technique correction
        """
        d_x[d_idx] += d_DX[d_idx]
        d_y[d_idx] += d_DY[d_idx]