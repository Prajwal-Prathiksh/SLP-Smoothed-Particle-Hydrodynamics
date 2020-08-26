###########################################################################
# IMPORTS
###########################################################################

# PySPH sph imports
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


class TransportVelocityStep_DPSPH(IntegratorStep):
    """Integrator defined in 'A transport velocity formulation for
    smoothed particle hydrodynamics', 2013, JCP, 241, pp 292--307

    For a predictor-corrector style of integrator, this integrator
    should operate only in PEC mode.

    """
    def initialize(self, d_rho0, d_idx, d_rho):
        d_rho0[d_idx] = d_rho[d_idx]

    def stage1(self, d_idx, d_u, d_v, d_w, d_au, d_av, d_aw, d_uhat, d_auhat, d_vhat,
                  d_avhat, d_what, d_awhat, d_x, d_y, d_z, dt, d_rho0, d_arho, d_rho):
        dtb2 = 0.5*dt

        # velocity update eqn (14)
        d_u[d_idx] += dtb2*d_au[d_idx]
        d_v[d_idx] += dtb2*d_av[d_idx]
        d_w[d_idx] += dtb2*d_aw[d_idx]

        # advection velocity update eqn (15)
        d_uhat[d_idx] = d_u[d_idx] + dtb2*d_auhat[d_idx]
        d_vhat[d_idx] = d_v[d_idx] + dtb2*d_avhat[d_idx]
        d_what[d_idx] = d_w[d_idx] + dtb2*d_awhat[d_idx]

        # position update eqn (16)
        d_x[d_idx] += dt*d_uhat[d_idx]
        d_y[d_idx] += dt*d_vhat[d_idx]
        d_z[d_idx] += dt*d_what[d_idx]

        d_rho[d_idx] = d_rho0[d_idx] + dtb2 * d_arho[d_idx]

    def stage2(self, d_idx, d_u, d_v, d_w, d_au, d_av, d_aw, d_vmag2, dt, d_rho0, d_arho, d_rho):
        dtb2 = 0.5*dt

        # corrector update eqn (17)
        d_u[d_idx] += dtb2*d_au[d_idx]
        d_v[d_idx] += dtb2*d_av[d_idx]
        d_w[d_idx] += dtb2*d_aw[d_idx]

        # magnitude of velocity squared
        d_vmag2[d_idx] = (d_u[d_idx]*d_u[d_idx] + d_v[d_idx]*d_v[d_idx] +
                          d_w[d_idx]*d_w[d_idx])
        
        d_rho[d_idx] = d_rho0[d_idx] + dt * d_arho[d_idx]