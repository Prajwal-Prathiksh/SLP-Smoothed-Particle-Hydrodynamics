###########################################################################
# IMPORTS
###########################################################################

# PySPH base imports
from pysph.base.kernels import (
    WendlandQuintic, QuinticSpline, Gaussian, CubicSpline
)
from pysph.base.nnps import DomainManager
from pysph.base.utils import (
    get_particle_array_wcsph, get_particle_array_tvf_solid
)

# PySPH solver imports
from pysph.solver.application import Application
from pysph.solver.solver import Solver

# PySPH sph imports
from pysph.sph.equation import Equation, Group
from pysph.sph.wc.edac import ComputeAveragePressure
from pysph.sph.basic_equations import SummationDensity
from pysph.sph.wc.transport_velocity import (
    StateEquation, SetWallVelocity, SolidWallPressureBC, 
    MomentumEquationPressureGradient, MomentumEquationViscosity, 
    SolidWallNoSlipBC, MomentumEquationArtificialStress
)

# SPH Integrator Imports
from pysph.sph.integrator import EulerIntegrator, PECIntegrator
from pysph.sph.integrator_step import (
    EulerStep, WCSPHStep, TransportVelocityStep
)


# Numpy import
from numpy import cos, sin, exp, pi
import numpy as np
import os

# Include path
import sys
#sys.path.insert(1, '/home/prajwal/Desktop/Winter_Project/SLP-Smoothed-Particle-Hydrodynamics')
sys.path.insert(1, 'E:\\IIT Bombay\\Winter Project - 2019\\SLP-Smoothed-Particle-Hydrodynamics')

# SPH Equation Imports
from SLP.dpsph.equations import (
    PST_PreStep_1, PST_PreStep_2, PST, AverageSpacing
)
from SLP.dpsph.integrator import DPSPHStep, TransportVelocityStep_DPSPH

### EOS
from pysph.sph.basic_equations import IsothermalEOS 
from pysph.sph.wc.basic import TaitEOS

### Momentum Equation
from SLP.dpsph.governing_equations import LaminarViscosityDeltaSPHPreStep   
from pysph.sph.wc.viscosity import LaminarViscosityDeltaSPH
from pysph.sph.wc.basic import  MomentumEquation, MomentumEquationDeltaSPH

### Continuity Equation
from pysph.sph.wc.transport_velocity import ContinuityEquation
from pysph.sph.wc.basic import (
    ContinuityEquationDeltaSPHPreStep, ContinuityEquationDeltaSPH
)

### Position Equation
from SLP.dpsph.governing_equations import Spatial_Acceleration
from pysph.sph.basic_equations import XSPHCorrection

### Kernel Correction
from pysph.sph.wc.kernel_correction import (
    GradientCorrection, GradientCorrectionPreStep
)

################################################################################
# CODE
################################################################################

L = 1.0
Umax = 1.0
c0 = 10 * Umax
rho0 = 1.0
p0 = c0 * c0 * rho0

# Numerical setup
hdx = 1.0

################################################################################
# Tayloy Green Vortex - Application
################################################################################
class Cavity(Application):
    def add_user_options(self, group):
        group.add_argument(
            "--nx", action="store", type=int, dest="nx", default=50,
            help="Number of points along x direction."
        )
        group.add_argument(
            "--re", action="store", type=float, dest="re", default=100,
            help="Reynolds number (defaults to 100)."
        )
        self.n_avg = 5
        group.add_argument(
            "--n-vel-avg", action="store", type=int, dest="n_avg",
            default=None,
            help="Average velocities over these many saved timesteps."
        )

    def consume_user_options(self):
        '''
        Initialize simulation paramters
        '''
        self.nx = self.options.nx
        if self.options.n_avg is not None:
            self.n_avg = self.options.n_avg

        self.dx = L / self.nx
        self.re = self.options.re
        h0 = hdx * self.dx
        self.nu = Umax * L / self.re
        dt_cfl = 0.25 * h0 / (c0 + Umax)
        dt_viscous = 0.125 * h0**2 / self.nu
        dt_force = 1.0
        self.tf = 10.0
        self.dt = min(dt_cfl, dt_viscous, dt_force)
        self.h0 = h0
        self.rho0 = rho0
        self.p0 = p0
        self.c0 = c0

        self.PSR_Rh = 0.05
        self.PST_R_coeff = 0.2 #1e-4
        self.PST_n_exp = 4.0 #3.0
        self.PST_Uc0 = 1.0
        self.PST_boundedFlow = True
        
    def create_particles(self):
        dx = self.dx
        ghost_extent = 5 * dx
        # create all the particles
        _x = np.arange(-ghost_extent - dx / 2, L + ghost_extent + dx / 2, dx)
        x, y = np.meshgrid(_x, _x)
        x = x.ravel()
        y = y.ravel()

        # sort out the fluid and the solid
        indices = []
        for i in range(x.size):
            if ((x[i] > 0.0) and (x[i] < L)):
                if ((y[i] > 0.0) and (y[i] < L)):
                    indices.append(i)

        # create the arrays
        solid = get_particle_array_tvf_solid(name='solid', x=x, y=y)

        # remove the fluid particles from the solid
        fluid = solid.extract_particles(indices)
        fluid.set_name('fluid')
        solid.remove_particles(indices)

        disp_vals = (self.re, self.dt, self.dx)
        print("Lid driven cavity :: Re = %d, dt = %g, dx = %g" % disp_vals)

        # setup the particle properties
        volume = dx * dx

        # mass is set to get the reference density of rho0
        fluid.m[:] = volume * rho0
        solid.m[:] = volume * rho0
        # Set a reference rho also, some schemes will overwrite this with a
        # summation density.
        fluid.rho[:] = rho0
        solid.rho[:] = rho0

        # smoothing lengths
        fluid.h[:] = hdx * dx
        solid.h[:] = hdx * dx

        # imposed horizontal velocity on the lid
        solid.u[:] = 0.0
        solid.v[:] = 0.0

        for i in range(solid.get_number_of_particles()):
            if solid.y[i] > L:
                solid.u[i] = Umax

        # volume is set as dx^2
        fluid.V[:] = 1. / volume
        solid.V[:] = 1. / volume

        tv_props_fluid = [
            'uhat', 'vhat', 'what', 'auhat', 'avhat', 'awhat', 
            'vmag2', 'V', 'arho'
        ]
        for i in tv_props_fluid:
            fluid.add_property(i)  

        tv_props_solid = [
            'u0', 'v0', 'w0', 'V', 'wij', 'ax', 'ay', 'az',
            'uf', 'vf', 'wf', 'ug', 'vg', 'wg', 'arho'
        ]
        for i in tv_props_solid:
            solid.add_property(i)

        add_props = [
            'rho0', 'x0', 'y0', 'z0', 'lmda', 'gradlmda', 'DX', 'DY', 'DZ', 
            'DRh'
            ]
        for i in add_props:
            fluid.add_property(i)
            solid.add_property(i)
        fluid.add_property('m_mat', stride=9)
        fluid.add_property('gradrho', stride=3)
        solid.add_property('m_mat', stride=9)
        solid.add_property('gradrho', stride=3)


        fluid.set_output_arrays(
            ['x', 'y', 'z', 'u', 'v', 'w', 'rho', 'p', 'h', 'm', 'au', 'av',
            'aw', 'V', 'vmag2', 'pid', 'gid', 'tag', 'lmda', 'DRh', 'DX', 'DY', 
            'DZ']
        )
        solid.set_output_arrays(
            ['x', 'y', 'z', 'u', 'v', 'w', 'rho', 'p', 'h', 'm', 'V', 'pid', 
            'gid', 'tag']
        )

        fluid.lmda[:] = 1.0
        fluid.DX[:] = 0.0
        fluid.DY[:] = 0.0
        fluid.DY[:] = 0.0
        fluid.DRh[:] = 0.0

        
        return [fluid, solid]

    def create_solver(self):
        '''
        Define solver
        '''
        kernel = QuinticSpline(dim=2)

        integrator = PECIntegrator(fluid=TransportVelocityStep_DPSPH()) 
        #PECIntegrator(fluid=TransportVelocityStep())

        solver = Solver(
            kernel=kernel, dim=2, integrator=integrator, dt=self.dt, tf=self.tf, 
            pfreq=500
        )

        return solver

    def create_equations(self):
        '''
        Set-up governing equations
        '''

        '''
        equations = [
            Group(equations=[
                SummationDensity(dest='fluid', sources=['fluid', 'solid'])
            ], real=False),

            Group(equations=[
                StateEquation(dest='fluid', sources=None, p0=self.p0, rho0=self.rho0), 
                SetWallVelocity(dest='solid', sources=['fluid'])
            ], real=False),

            Group(equations=[
                SolidWallPressureBC(dest='solid', sources=['fluid'], rho0=self.rho0, p0=self.p0)
            ], real=False),

            Group(equations=[
                MomentumEquationPressureGradient(dest='fluid', sources=['fluid','solid'], pb=self.p0), 
                MomentumEquationViscosity(dest='fluid', sources=['fluid'], nu=self.nu), 
                SolidWallNoSlipBC(dest='fluid', sources=['solid'], nu=self.nu), 
                MomentumEquationArtificialStress(dest='fluid', sources=['fluid'])
            ], real=True)
        ]   
        '''        
        equations = [
            Group(equations=[
                GradientCorrectionPreStep(dest='fluid', sources=['fluid', 'solid'], dim=2),
                GradientCorrection(dest='fluid', sources=['fluid', 'solid'], dim=2, tol=0.1),
                ContinuityEquationDeltaSPHPreStep(dest='fluid', sources=['fluid', 'solid']),
                PST_PreStep_1(dest='fluid', sources=['fluid', 'solid'], dim=2, boundedFlow=self.PST_boundedFlow),
                PST_PreStep_2(dest='fluid', sources=['fluid', 'solid'], dim=2, H=self.h0, boundedFlow=self.PST_boundedFlow),
                PST(dest='fluid', sources=['fluid', 'solid'], dim=2, H=self.h0, dt=self.dt, dx=self.dx, Uc0=self.PST_Uc0, Rh=self.PSR_Rh, saveAllDRh=True, R_coeff=self.PST_R_coeff, n_exp=self.PST_n_exp, boundedFlow=self.PST_boundedFlow),
            ], real=False),

            Group(equations=[
                ContinuityEquation(dest='fluid', sources=['fluid', 'solid']),                 
                ContinuityEquationDeltaSPH(damp_fac=0.2*0.5, dest='fluid', sources=['fluid', 'solid'], c0=self.c0, delta=0.1),
                #SummationDensity(dest='fluid', sources=['fluid', 'solid'])
            ], real=False),

            Group(equations=[
                IsothermalEOS(dest='fluid', sources=['fluid'], rho0=self.rho0, c0=self.c0, p0=0.0),
                SetWallVelocity(dest='solid', sources=['fluid'])
            ], real=False),

            Group(equations=[
                SolidWallPressureBC(b=1.01, dest='solid', sources=['fluid'], rho0=self.rho0, p0=self.p0)
            ], real=False),

            Group(equations=[
                MomentumEquationPressureGradient(dest='fluid', sources=['fluid','solid'], pb=self.p0),  
                LaminarViscosityDeltaSPHPreStep(dest='fluid', sources=['fluid']),
                LaminarViscosityDeltaSPH(dest='fluid', sources=['fluid'], dim=2, rho0=self.rho0, nu=self.nu),
                #MomentumEquationViscosity(dest='fluid', sources=['fluid'], nu=self.nu), 
                SolidWallNoSlipBC(dest='fluid', sources=['solid'], nu=self.nu), 
                #MomentumEquationArtificialStress(dest='fluid', sources=['fluid']),  
            ], real=True), 
        ]       

        return equations

    def post_process(self, info_fname):
        try:
            import matplotlib
            matplotlib.use('Agg')
            from matplotlib import pyplot as plt
        except ImportError:
            print("Post processing requires matplotlib.")
            return
        if self.rank > 0:
            return
        info = self.read_info(info_fname)
        if len(self.output_files) == 0:
            return
        t, ke = self._plot_ke_history()
        x, ui, vi, ui_c, vi_c = self._plot_velocity()
        res = os.path.join(self.output_dir, "results.npz")
        np.savez(res, t=t, ke=ke, x=x, u=ui, v=vi, u_c=ui_c, v_c=vi_c)

    def _plot_ke_history(self):
        from pysph.tools.pprocess import get_ke_history
        from matplotlib import pyplot as plt
        t, ke = get_ke_history(self.output_files, 'fluid')
        plt.clf()
        plt.plot(t, ke)
        plt.xlabel('t')
        plt.ylabel('Kinetic energy')
        fig = os.path.join(self.output_dir, "ke_history.png")
        plt.savefig(fig, dpi=300)
        return t, ke

    def _plot_velocity(self):
        from pysph.tools.interpolator import Interpolator
        from pysph.solver.utils import load
        from pysph.examples.ghia_cavity_data import get_u_vs_y, get_v_vs_x
        # interpolated velocities
        _x = np.linspace(0, 1, 101)
        xx, yy = np.meshgrid(_x, _x)

        # take the last solution data
        fname = self.output_files[-1]
        data = load(fname)
        tf = data['solver_data']['t']
        interp = Interpolator(list(data['arrays'].values()), x=xx, y=yy)
        ui = np.zeros_like(xx)
        vi = np.zeros_like(xx)

        # Average out the velocities over the last n_avg timesteps
        for fname in self.output_files[-self.n_avg:]:
            data = load(fname)
            tf = data['solver_data']['t']
            interp.update_particle_arrays(list(data['arrays'].values()))
            _u = interp.interpolate('u')
            _v = interp.interpolate('v')
            _u.shape = 101, 101
            _v.shape = 101, 101
            ui += _u
            vi += _v

        ui /= self.n_avg
        vi /= self.n_avg

        # velocity magnitude
        self.vmag = vmag = np.sqrt(ui**2 + vi**2)
        import matplotlib.pyplot as plt

        f = plt.figure()

        plt.streamplot(
            xx, yy, ui, vi, density=(2, 2),  # linewidth=5*vmag/vmag.max(),
            color=vmag
        )
        plt.xlim(0, 1)
        plt.ylim(0, 1)
        plt.colorbar()
        plt.xlabel('$x$')
        plt.ylabel('$y$')
        plt.title('Streamlines at %s seconds' % tf)
        fig = os.path.join(self.output_dir, 'streamplot.png')
        plt.savefig(fig, dpi=300)

        f = plt.figure()

        ui_c = ui[:, 50]
        vi_c = vi[50]

        s1 = plt.subplot(211)
        s1.plot(ui_c, _x, label='Computed')

        y, data = get_u_vs_y()
        if self.re in data:
            s1.plot(data[self.re], y, 'o', fillstyle='none',
                    label='Ghia et al.')
        s1.set_xlabel(r'$v_x$')
        s1.set_ylabel(r'$y$')
        s1.legend()

        s2 = plt.subplot(212)
        s2.plot(_x, vi_c, label='Computed')
        x, data = get_v_vs_x()
        if self.re in data:
            s2.plot(x, data[self.re], 'o', fillstyle='none',
                    label='Ghia et al.')
        s2.set_xlabel(r'$x$')
        s2.set_ylabel(r'$v_y$')
        s2.legend()

        fig = os.path.join(self.output_dir, 'centerline.png')
        plt.savefig(fig, dpi=300)
        return _x, ui, vi, ui_c, vi_c

################################################################################
# Main Code
################################################################################
if __name__ == '__main__':
    app = Cavity()
    app.run()
    app.post_process(app.info_filename)