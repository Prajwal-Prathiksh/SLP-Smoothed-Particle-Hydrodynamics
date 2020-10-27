# Include path
from os import error
import sys
sys.path.insert(1, '/home/prajwal/Desktop/Winter_Project/SLP-Smoothed-Particle-Hydrodynamics')
sys.path.insert(1, 'E:\IIT Bombay - Miscellaneous\Winter Project\SLP-Smoothed-Particle-Hydrodynamics')

# PyPSH Equations Import
from pysph.sph.equation import Equation, Group
from pysph.base.kernels import WendlandQuintic

# PySPH solver imports
from pysph.solver.application import Application
from pysph.solver.solver import Solver

# Math Imports 
from math import ceil, floor, sqrt
import numpy as np

# SPH Equation Imports
from pysph.sph.integrator import EulerIntegrator

from SLP.dpsph.integrator import RK4Step, RK4Integrator, EulerStep
from SLP.dpsph.utils import get_particle_array_RK4


class circular_velocity(Equation):

    def __init__(self, dest, sources, R=1.0):
        r'''
        Parameters:
        -----------
        R: Radius of the circle
        '''
        self.R = R
        super(circular_velocity, self).__init__(dest, sources)

    def loop(self, d_idx, d_x, d_y, d_ax, d_ay):
        d_ax[d_idx] = -1.0*d_y[d_idx]
        d_ay[d_idx] = d_x[d_idx]

class circular_acceleration(Equation):

    def __init__(self, dest, sources, R=1.0):
        r'''
        Parameters:
        -----------
        R: Radius of the circle
        '''
        self.R = R
        super(circular_acceleration, self).__init__(dest, sources)

    def loop(self, d_idx, d_x, d_y, d_au, d_av):
        d_au[d_idx] = -1.0*d_x[d_idx]
        d_av[d_idx] = -1.0*d_y[d_idx]

################################################################################
# Test Integrator - Application
################################################################################
class Test_Integrator(Application):
    def add_user_options(self, group):
        group.add_argument(
            "--R", action="store", type=float, dest="R", default=1.0,
            help="Radius of the circular orbit"
        )
        group.add_argument(
            "--N", action="store", type=float, dest="N", default=50,
            help="Number of Time-steps"
        )
        group.add_argument(
            "--INT", action="store", type=str, dest="INT", default='eul',
            help="Set integrator ('eul' = Euler, 'rk4' = RK-4)"
        )

    def consume_user_options(self):
        '''
        Initialize simulation paramters
        '''
        self.R = self.options.R
        self.tf = np.pi*6.0
        self.dt = self.tf/self.options.N
        self.pfreq = 1
        self.INT = self.options.INT

    def create_particles(self):
        pa = get_particle_array_RK4(
            x=self.R, y=0.0, u=0, v=self.R, m=1.0, h=1.0, name='fluid'
        )
        print('dt: ', self.dt)
        return [pa]
    
    def create_solver(self):
        '''
        Define solver
        '''
        kernel = WendlandQuintic(dim=2)

        if self.INT == 'eul':
            integrator = EulerIntegrator(fluid = EulerStep())
        elif self.INT =='rk4':
            integrator = RK4Integrator(fluid = RK4Step())
        else:
            raise Exception('Invalid integrator argument')

        solver = Solver(
            kernel=kernel, dim=2, integrator=integrator, dt=self.dt, tf=self.tf, 
            pfreq=self.pfreq
        )

        return solver

    def create_equations(self):

        equations = [
            Group(equations=[
                circular_velocity(dest='fluid', sources=None, R=self.R),
                circular_acceleration(dest='fluid', sources=None, R=self.R),
            ], real=True),
        ]

        return equations

    ###### Post processing
    def post_process(self, info_fname):
        import os
        info = self.read_info(info_fname)
        if len(self.output_files) == 0:
            return

        from pysph.solver.utils import iter_output

        files = self.output_files
        t, xa, ya = [], [], []
        for sd, array in iter_output(files, 'fluid'):
            _t = sd['t']
            t.append(_t)
            x, y = array.get('x', 'y')

            xa.append(x[0])
            ya.append(y[0])

        t = np.array(t)
        xa = np.array(xa)
        ya = np.array(ya)

        fname = os.path.join(self.output_dir, 'results.npz')
        np.savez(fname, t=t, x=xa, y=ya)

        import matplotlib
        
        matplotlib.use('Agg')

        from matplotlib import pyplot as plt
        plt.clf()
        plt.plot(t, xa, label='x')
        plt.plot(t, ya, label='y')
        plt.plot(t, self.R*np.cos(t), label='exact x')
        plt.plot(t, self.R*np.sin(t), label='exact y')
        plt.xlabel('t')
        plt.legend()
        fig = os.path.join(self.output_dir, "position.png")
        plt.savefig(fig, dpi=300)

################################################################################
# Main Code
################################################################################
if __name__ == '__main__':
    app = Test_Integrator()
    app.run()
    app.post_process(app.info_filename)


    