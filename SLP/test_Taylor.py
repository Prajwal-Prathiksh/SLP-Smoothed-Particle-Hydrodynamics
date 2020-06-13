# PySPH base imports
from pysph.base.utils import get_particle_array_wcsph
from pysph.base.kernels import CubicSpline

#PySPH solver imports
from pysph.solver.application import Application
from pysph.solver.solver import Solver

#PySPH sph imports
from pysph.sph.equation import Equation, Group
from pysph.sph.integrator import Integrator, IntegratorStep

# Additional import
import numpy as np
pi = np.pi


class TaitEOS(Equation):
    ########### CHNAGED
    def __init__(self, dest, sources, c0, rho0=1000.0):
        self.rho0 = rho0
        self.c0 = c0
        self.c0_2 = c0*c0

        super(TaitEOS, self).__init__(dest, sources)

    def loop(self, d_idx, d_rho, d_p):
        
        rhoi = d_rho[d_idx]

        # Equation of state
        d_p[d_idx] = self.c0_2 * (rhoi - self.rho0)


class ContinuityEquation(Equation):
    ############## DELTA
    def __init__(self, dest, sources, delta, c0, H):

        self.delta = delta
        self.c0 = c0
        self.H = H

        # Calculate constant
        self.CONST = delta * H * c0

        super(ContinuityEquation, self).__init__(dest, sources)

    def initialize(self, d_idx, d_arho):
        d_arho[d_idx] = 0.0

    def loop(self, d_idx, s_idx, d_arho, d_rho, VIJ, DWIJ, s_m, s_rho, XIJ, R2IJ, EPS):
        
        rhoi = d_rho[d_idx]
        rhoj = s_rho[s_idx]

        # Volume element
        Vj = s_m[s_idx]/rhoj

        vijdotDWij = VIJ[0]*DWIJ[0] + VIJ[1]*DWIJ[1] + VIJ[2]*DWIJ[2]
        xjidotDWij = -1.0 * (XIJ[0]*DWIJ[0] + XIJ[1]*DWIJ[1] + XIJ[2]*DWIJ[2]) # Multipled with -1 to convert XIJ to XJI
        
        # psi_ij
        psi_ij = 2 * (rhoj - rhoi)

        # Continuity density term
        tmp1 = rhoi * vijdotDWij

        # Dissipative diffusive term
        tmp2 = self.CONST * psi_ij * xjidotDWij / (R2IJ + EPS) # NOTE: R2JI = R2IJ

        
        d_arho[d_idx] += (tmp1 + tmp2) * Vj
        ###############################################
        #d_arho[d_idx] += s_m[s_idx]*vijdotDWij


class MomentumEquation(Equation):
    def __init__(self, dest, sources, dim, mu, fx=0.0, fy=0.0, fz=0.0):
        r'''
        Parameters
        ----------
        c0 : float
            reference speed of sound
        alpha : float
            produces a shear and bulk viscosity
        beta : float
            used to handle high Mach number shocks
        gx : float
            body force per unit mass along the x-axis
        '''

        self.alpha =0.2
        self.beta = 0.1
        self.gy = 9
        self.EPS = 1e-4

        self.dim = dim
        self.mu = mu
        self.fx = fx
        self.fy = fy
        self.fz = fz

        # Calculate constant
        self.K = 2 * (dim + 2)

        self.CONST = self.K * mu

        super(MomentumEquation, self).__init__(dest, sources)

    def initialize(self, d_idx, d_au, d_av, d_aw):
        d_au[d_idx] = 0.0
        d_av[d_idx] = 0.0
        d_aw[d_idx] = 0.0

    def loop(self, d_idx, s_idx, d_rho, s_rho, DWIJ, s_m, VIJ, XIJ, R2IJ, d_au, d_av, d_aw, d_p, s_p, EPS):

        rhoj = s_rho[s_idx]
        Vj = s_m[s_idx] / rhoj

        Pi = d_p[d_idx]
        Pj = s_p[s_idx]

        vjidotxji = VIJ[0]*XIJ[0] + VIJ[1]*XIJ[1] + VIJ[2]*XIJ[2]
        
        # F_ij
        if Pi < 0.0:
            Fij = Pi - Pj
        else:
            Fij = (Pi + Pj) * -1.0

        # pi_ij
        pi_ij  = vjidotxji / (R2IJ + EPS)

        tmp = (Fij + self.CONST * pi_ij) * Vj 

        # Accelerations
        d_au[d_idx] += tmp*DWIJ[0]
        d_av[d_idx] += tmp*DWIJ[1]
        d_aw[d_idx] += tmp*DWIJ[2]

    def post_loop(self, d_idx, d_au, d_av, d_aw, d_rho):

        rhoi = d_rho[d_idx] #Mome
        
        d_au[d_idx] = d_au[d_idx] / rhoi + self.fx
        d_av[d_idx] = d_av[d_idx] / rhoi + self.fy
        d_aw[d_idx] = d_aw[d_idx] / rhoi + self.fz



class XSPHCorrection(Equation):
    def __init__(self, dest, sources, eps=0.5):
        r'''
        Parameters
        ----------
        eps : float
            constant
        '''

        self.eps = eps
        super(XSPHCorrection, self).__init__(dest, sources)

    def initialize(self, d_idx, d_ax, d_ay):
        d_ax[d_idx] = 0.0
        d_ay[d_idx] = 0.0

    def loop(self, d_idx, s_idx, s_m, d_ax, d_ay, WIJ, RHOIJ1, VIJ):

        tmp = self.eps * s_m[s_idx] * WIJ * RHOIJ1

        d_ax[d_idx] += tmp * VIJ[0]
        d_ay[d_idx] += tmp * VIJ[1]

    def post_loop(self, d_idx, d_ax, d_ay, d_u, d_v):
        d_ax[d_idx] += d_u[d_idx]
        d_ay[d_idx] += d_v[d_idx]


class EulerIntegrator(Integrator):
    def one_timestep(self, t, dt):
        self.initialize()
        self.compute_accelerations()
        self.stage1()
        self.do_post_stage(dt, 1)
        

class EulerStep(IntegratorStep):
    def initialize(self):
        pass

    def stage1(self, d_idx, d_u, d_v, d_au, d_av, d_x, d_y,
                  d_rho, d_arho, dt=0.0):
        d_u[d_idx] += dt*d_au[d_idx]
        d_v[d_idx] += dt*d_av[d_idx]
       
        d_x[d_idx] += dt*d_u[d_idx]
        d_y[d_idx] += dt*d_v[d_idx]

        d_rho[d_idx] += dt*d_arho[d_idx]

class testTaylor (Application):
    def initialize(self):
        self.dx = 0.05 
        self.rho0 = 1000
        self.vol = self.dx * self.dx
        self.m0 = self.rho0 * self.vol
        self.hdx = 1.3
        self.h0 = self.hdx * self.dx
        self.c0 = 15

    def create_particles(self):
        '''
        Set up particle arrays
        '''
        dx = self.dx
        # Fluid
        x0, y0 = np.mgrid[dx/2: pi-dx/2:dx, dx/2: pi-dx/2:dx]
        u0 = np.sin(x0)*np.cos(y0)
        v0 = -1.0 * np.cos(x0)*np.sin(y0)


        pa_fluid = get_particle_array_wcsph(
            name='fluid', x = x0, y=y0, u=u0, v=v0,  rho = self.rho0, m=self.m0, h=self.h0)

        return [pa_fluid]
        
    def create_solver(self):
        '''
        Set up the kernel and integrator to be used
        '''

        kernel = CubicSpline(dim=2)

        integrator = EulerIntegrator(fluid = EulerStep())

        dt = 1e-6 # Time-steps and final time
        tf = dt*10

        solver = Solver(
            kernel=kernel, dim = 2, integrator=integrator,dt = dt, tf = tf
        )
        return solver

    def create_equations(self):
        '''
        Set up the SPH equations
        '''
        rho0 = 1000
        c0 = 70
        g = -9.81
        h = 1.3*0.05
        equations = [
            Group(
                equations=[ 
                    # Equation of State
                    TaitEOS(dest='fluid', sources=None, rho0=self.rho0, c0 = self.c0),
                ], real=False
            ),
            Group(
                equations=[
                    # Continuity Equation
                    ContinuityEquation(dest='fluid', sources=['fluid'], delta=0.1, c0=self.c0, H = self.h0),

                    # Momentum Equation
                    MomentumEquation(dest='fluid', sources=['fluid'], dim=2, mu=1),

                    # Position step
                    XSPHCorrection (dest='fluid', sources=['fluid'], eps = 0.5)
                ], real= True
            )
        ]

        return equations


if __name__ == '__main__':
    app = testTaylor()
    app.run()