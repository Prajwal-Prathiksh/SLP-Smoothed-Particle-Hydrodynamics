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

# Import Delta_Plus - SPH Equations
from Delta_Plus_SPH import EOS_DeltaPlus_SPH, MomentumEquation_DeltaPlus_SPH, ContinuityEquation_DeltaPlus_SPH
##NOTE: git checkout b139a3ba 

class EulerIntegrator(Integrator):
    def one_timestep(self, t, dt):
        self.initialize()
        self.compute_accelerations()
        self.stage1()
        #self.stage2()
        self.do_post_stage(dt,1)

class EulerStep(IntegratorStep):
    def initialize(self):
        pass

    def stage1(self, d_idx, d_u, d_v, d_au, d_av, d_x, d_y, d_rho, d_arho, dt=0.0):
        d_u[d_idx] += dt*d_au[d_idx]
        d_v[d_idx] += dt*d_av[d_idx]
       
        d_x[d_idx] += dt*d_u[d_idx]
        d_y[d_idx] += dt*d_v[d_idx]

        d_rho[d_idx] += dt*d_arho[d_idx]

    '''def stage2(self, d_idx, d_x, d_y):
 
        if d_x[d_idx] > pi:
            d_x[d_idx] = pi - d_x[d_idx]

        if d_y[d_idx] > pi:
            d_y[d_idx] = pi - d_y[d_idx]'''

class Taylor_Green(Application):
    def initialize(self):
        self.dx = 1.0/50.0
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
        x0, y0 = np.mgrid[dx/2: 1-dx/2:dx, dx/2: 1-dx/2:dx]
        v0 = np.sin(2*pi*x0)*np.cos(2*pi*y0)
        u0 = -1.0 * np.cos(2*pi*x0)*np.sin(2*pi*y0)


        pa_fluid = get_particle_array_wcsph(
            name='fluid', x = x0, y=y0, u=u0, v=v0,  rho = self.rho0, m=self.m0, h=self.h0)

        return [pa_fluid]

    def create_solver(self):

        kernel = CubicSpline(dim=2)
        
        integrator = EulerIntegrator(fluid = EulerStep())

        dt = 1e-5
        tf = 1

        solver = Solver(
            kernel=kernel, dim=2, integrator=integrator, dt=dt, tf=tf, pfreq=50,
        )
        return solver

    def create_equations(self):
        equations = [
            Group(
                equations=[
                    EOS_DeltaPlus_SPH(dest='fluid', sources=['fluid'],rho0=self.rho0, c0=self.c0)
                ], real=False       
            ),

    
            Group(
                equations=[
                    ContinuityEquation_DeltaPlus_SPH(dest='fluid', sources=['fluid'], delta=0.1, c0=self.c0,H=self.h0), 
                    MomentumEquation_DeltaPlus_SPH(dest='fluid', sources=['fluid'], dim=2, mu=1)
                ],
                    real=True
            )
        ]

        return equations

if __name__ == '__main__':
    app = Taylor_Green()
    app.run()