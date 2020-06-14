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
import os
pi = np.pi

# Import Delta_Plus - SPH Equations
from Delta_Plus_SPH import EOS_DeltaPlus_SPH, MomentumEquation_DeltaPlus_SPH, ContinuityEquation_DeltaPlus_SPH
##NOTE: git checkout b139a3ba 


def exact_solution(U, b, t, x, y):
    factor = U * np.exp(b*t)

    u = -np.cos(2*pi*x) * np.sin(2*pi*y)
    v = np.sin(2*pi*x) * np.cos(2*pi*y)
    p = -0.25 * (np.cos(4*pi*x) + np.cos(4*pi*y))

    return factor * u, factor * v, factor * factor * p

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
        self.nx = 50.0
        self.re = 100.0
        self.U = 1.0
        self.L = 1.0
        self.rho0 = 1000.0

        self.c0 = 15.0
        self.hdx = 1.3

        self.nu = self.L * self.U/self.re
        self.mu = self.nu * self.rho0
        self.dx = self.L/self.nx
        
        
        self.vol = self.dx * self.dx
        self.m0 = self.rho0 * self.vol
        
        self.h0 = self.hdx * self.dx
        

        dt_cfl = 0.25 * self.h0 / (self.c0 + self.U)
        dt_viscous = 0.125 * self.h0**2 / self.nu
        dt_force = 0.25 * 1.0

        self.dt = min(dt_cfl, dt_viscous, dt_force)
        self.tf = 2.0

    def create_particles(self):
        '''
        Set up particle arrays
        '''
        dx = self.dx
        L = self.L
        # Fluid
        x0, y0 = np.mgrid[dx/2: L-dx/2:dx, dx/2: L-dx/2:dx]
        
        b = -8.0-pi*pi/self.re

        u0, v0, p0 = exact_solution(U=self.U, b=b, t=0, x=x0, y=y0)


        pa_fluid = get_particle_array_wcsph(
            name='fluid', x = x0, y=y0, u=u0, v=v0, p=p0, rho = self.rho0, m=self.m0, h=self.h0)

        return [pa_fluid]

    def create_solver(self):

        kernel = CubicSpline(dim=2)
        
        integrator = EulerIntegrator(fluid = EulerStep())

        solver = Solver(
            kernel=kernel, dim=2, integrator=integrator, dt=self.dt, tf=self.tf, pfreq=100,
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
                    MomentumEquation_DeltaPlus_SPH(dest='fluid', sources=['fluid'], dim=2, mu=self.mu)
                ],
                    real=True
            )
        ]

        return equations

    # Post processing
    def _get_post_process_props(self, array):
        """
        Return x, y, u, v
        """
        x, y, u, v = array.get('x', 'y', 'u', 'v')
        
        return x, y, u, v

    def post_process(self, info_fname):

        info = self.read_info(info_fname)
        if len(self.output_files) == 0:
            return

        from pysph.solver.utils import iter_output
        decay_rate = -8.0 * pi**2 / self.re

        files = self.output_files
        t, ke, ke_ex, decay, linf, l1 = [], [], [], [], [], []
        for sd, array in iter_output(files, 'fluid'):
            _t = sd['t']
            t.append(_t)
            x, y, u, v = self._get_post_process_props(array)
            u_e, v_e, p_e = exact_solution(self.U, decay_rate, _t, x, y)
            vmag2 = u**2 + v**2
            vmag = np.sqrt(vmag2)
            ke.append(0.5 * np.sum(self.m0 * vmag2))
            vmag2_e = u_e**2 + v_e**2
            vmag_e = np.sqrt(vmag2_e)
            ke_ex.append(0.5 * np.sum(self.m0 * vmag2_e))

            vmag_max = vmag.max()
            decay.append(vmag_max)
            theoretical_max = self.U * np.exp(decay_rate * _t)
            linf.append(abs((vmag_max - theoretical_max) / theoretical_max))

            l1_err = np.average(np.abs(vmag - vmag_e))
            avg_vmag_e = np.average(np.abs(vmag_e))
            # scale the error by the maximum velocity.
            l1.append(l1_err / avg_vmag_e)


        t, ke, ke_ex, decay, l1, linf = list(map(
            np.asarray, (t, ke, ke_ex, decay, l1, linf))
        )
        decay_ex = self.U * np.exp(decay_rate * t)
        fname = os.path.join(self.output_dir, 'results.npz')
        np.savez(
            fname, t=t, ke=ke, ke_ex=ke_ex, decay=decay, linf=linf, l1=l1, decay_ex=decay_ex
        )

        import matplotlib
        matplotlib.use('Agg')

        from matplotlib import pyplot as plt
        plt.clf()
        plt.semilogy(t, decay_ex, label="exact")
        plt.semilogy(t, decay, label="computed")
        plt.xlabel('t')
        plt.ylabel('max velocity')
        plt.legend()
        fig = os.path.join(self.output_dir, "decay.png")
        plt.savefig(fig, dpi=300)

        plt.clf()
        plt.plot(t, linf)
        plt.xlabel('t')
        plt.ylabel(r'$L_\infty$ error')
        fig = os.path.join(self.output_dir, "linf_error.png")
        plt.savefig(fig, dpi=300)

        plt.clf()
        plt.plot(t, l1, label="error")
        plt.xlabel('t')
        plt.ylabel(r'$L_1$ error')
        fig = os.path.join(self.output_dir, "l1_error.png")
        plt.savefig(fig, dpi=300)


if __name__ == '__main__':
    app = Taylor_Green()
    app.run()
    app.post_process(app.info_filename)