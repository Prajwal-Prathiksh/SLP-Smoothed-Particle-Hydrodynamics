###########################################################################
# IMPORTS
###########################################################################

# PySPH base imports
from pysph.base.kernels import (
    WendlandQuintic, QuinticSpline, Gaussian, CubicSpline
)
from pysph.base.nnps import DomainManager
from pysph.base.utils import get_particle_array_wcsph, get_particle_array

# PySPH solver imports
from pysph.solver.application import Application
from pysph.solver.solver import Solver

# PySPH sph imports
from pysph.sph.equation import Equation, Group, MultiStageEquations
from pysph.sph.wc.edac import ComputeAveragePressure
from pysph.sph.basic_equations import SummationDensity

from pysph.sph.wc.gtvf import (
    ContinuityEquationGTVF, CorrectDensity, MomentumEquationPressureGradient,
    MomentumEquationArtificialStress, MomentumEquationViscosity, 
)

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
from pysph.sph.wc.gtvf import GTVFIntegrator, GTVFStep


# Numpy import
from numpy import cos, sin, exp, pi
import numpy as np
import os

# Include path
import sys
sys.path.insert(1, '/home/prajwal/Desktop/Winter_Project/SLP-Smoothed-Particle-Hydrodynamics')
sys.path.insert(1, 'E:\IIT Bombay\Winter Project - 2019\SLP-Smoothed-Particle-Hydrodynamics')

# SPH Equation Imports
from SLP.dpsph.equations import PST_PreStep_1, PST_PreStep_2, PST, AverageSpacing
from SLP.dpsph.integrator import DPSPHStep, TransportVelocityStep_DPSPH

### EOS
from pysph.sph.basic_equations import IsothermalEOS 
from pysph.sph.wc.basic import TaitEOS

### Momentum Equation
from SLP.dpsph.governing_equations import LaminarViscosityDeltaSPHPreStep   
from pysph.sph.wc.viscosity import LaminarViscosityDeltaSPH, LaminarViscosity
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

# Exact Solution - Taylor Green Vortex
def exact_solution(U, b, t, x, y):
    factor = U * exp(b*t)

    u = -cos(2*pi*x) * sin(2*pi*y)
    v = sin(2*pi*x) * cos(2*pi*y)
    p = -0.25 * (cos(4*pi*x) + cos(4*pi*y))

    return factor * u, factor * v, factor * factor * p

################################################################################
# Tayloy Green Vortex - Application
################################################################################
class Taylor_Green(Application):
    def add_user_options(self, group):
        group.add_argument(
            "--init", action="store", type=str, default=None,
            help="Initialize particle positions from given file."
        )
        group.add_argument(
            "--perturb", action="store", type=float, dest="perturb", default=0,
            help="Random perturbation of initial particles as a fraction "
            "of dx (setting it to zero disables it, the default)."
        )
        group.add_argument(
            "--nx", action="store", type=int, dest="nx", default=50,
            help="Number of points along x direction. (default 50)"
        )
        group.add_argument(
            "--re", action="store", type=float, dest="re", default=100,
            help="Reynolds number (defaults to 100)."
        )
        group.add_argument(
            "--hdx", action="store", type=float, dest="hdx", default=1.0,
            help="Ratio h/dx."
        )
        group.add_argument(
            "--pb-factor", action="store", type=float, dest="pb_factor",
            default=1.0,
            help="Use fraction of the background pressure (default: 1.0)."
        )
        corrections = ['', 'mixed', 'gradient', 'crksph']
        group.add_argument(
            "--kernel-correction", action="store", type=str,
            dest='kernel_correction',
            default='', help="Type of Kernel Correction", choices=corrections
        )
        group.add_argument(
            "--remesh", action="store", type=int, dest="remesh", default=0,
            help="Remeshing frequency (setting it to zero disables it)."
        )
        remesh_types = ['m4', 'sph']
        group.add_argument(
            "--remesh-eq", action="store", type=str, dest="remesh_eq",
            default='m4', choices=remesh_types,
            help="Remeshing strategy to use."
        )
        group.add_argument(
            "--shift-freq", action="store", type=int, dest="shift_freq",
            default=0,
            help="Particle position shift frequency.(set zero to disable)."
        )
        shift_types = ['simple', 'fickian']
        group.add_argument(
            "--shift-kind", action="store", type=str, dest="shift_kind",
            default='simple', choices=shift_types,
            help="Use of fickian shift in positions."
        )
        group.add_argument(
            "--shift-parameter", action="store", type=float,
            dest="shift_parameter", default=None,
            help="Constant used in shift, range for 'simple' is 0.01-0.1"
            "range 'fickian' is 1-10."
        )
        group.add_argument(
            "--shift-correct-vel", action="store_true",
            dest="correct_vel", default=False,
            help="Correct velocities after shifting (defaults to false)."
        )
        group.add_argument(
            "--tvf-correct", action="store_true", default=False,
            dest="tvf_correct", help="TVF correction"
        )

    def consume_user_options(self):
        '''
        Initialize simulation paramters
        '''
        self.U = 1.0
        self.L = 1.0
        self.rho0 = 1.0
        self.c0 = 10 * self.U
        self.p0 = self.c0**2 * self.rho0
        self.hdx = 2.0

        self.nx = self.options.nx
        self.perturb = self.options.perturb # Perturbation factor
        self.re = self.options.re

        self.dx = dx = self.L / self.nx
        self.volume = dx * dx
        self.hdx = self.options.hdx
        self.kernel_correction = self.options.kernel_correction

        # Calculate simulation parameters
        self.nu = self.L * self.U/self.re
        self.mu = self.nu * self.rho0
        self.dx = self.L/self.nx
        
        
        self.vol = self.dx * self.dx
        self.m0 = self.rho0 * self.vol        
        self.h0 = self.hdx * self.dx
        
        # Simulation time-step parameters
        dt_cfl = 0.25 * self.h0 / (self.c0 + self.U)
        dt_viscous = 0.125 * self.h0**2 / self.nu
        dt_force = 0.25 * 1.0

        self.dt = min(dt_cfl, dt_viscous, dt_force)
        self.tf = 2.0

        self.PSR_Rh = 0.05
        self.PST_R_coeff = 0.2 #1e-4
        self.PST_n_exp = 4.0 #3.0
        self.PST_Uc0 = 10.0
        self.PST_boundedFlow = True

        self.TVF_correction = self.options.tvf_correct

        # Print parameters
        print('dx : ', self.dx)
        print('dt : ', self.dt)

    def create_particles(self):
        '''
        Set up particle arrays
        '''
        dx = self.dx
        L = self.L
        _x = np.arange(dx / 2, L, dx)
        x, y = np.meshgrid(_x, _x)

        if self.options.perturb > 0:
            np.random.seed(1)
            factor = dx * self.options.perturb
            x += np.random.random(x.shape) * factor
            y += np.random.random(x.shape) * factor
        
        m = self.volume * self.rho0
        h = self.hdx * dx
        re = self.options.re
        b = -8.0*pi*pi / re
        u0, v0, p0 = exact_solution(U=self.U, b=b, t=0, x=x, y=y)
        color0 = cos(2*pi*x) * cos(4*pi*y)

        # create the arrays
        fluid = get_particle_array(name='fluid', x=x, y=y, m=m, h=h, u=u0,
                                   v=v0, rho=self.rho0, p=p0, color=color0)

        fluid.add_property('m_mat', stride=9)
        fluid.add_property('gradrho', stride=3)
        fluid.add_property('gradlmda', stride=3)
        #fluid.add_property('gradvhat', stride=9)
        #fluid.add_property('sigma', stride=9)
        #fluid.add_property('asigma', stride=9)
        
        add_props = [
            'lmda', 'rho0', 'u0', 'v0', 'w0', 'x0', 'y0', 'z0', 'ax', 'ay', 
            'az', 'DRh', 'DY', 'DX', 'DZ', 'uhat', 'vhat', 'what', 'auhat', 
            'avhat', 'awhat', 'vmag2', 'V', 'arho', 'vmag', 
            #'rhodiv', 'p0', 'arho0'
        ]
        for i in add_props:
            fluid.add_property(i)

        fluid.set_output_arrays(
            ['x', 'y', 'z', 'u', 'v', 'w', 'rho', 'p', 'h', 'm', 'vmag', 
            'vmag2', 'pid', 'gid', 'tag', 'DRh', 'lmda']
        )

        # setup the particle properties
        volume = dx * dx

        # mass is set to get the reference density of rho0
        fluid.V[:] = 1. / volume
        fluid.lmda[:] = 1.0

        return [fluid]

    def create_domain(self):
        '''
        Set-up periodic boundary
        '''
        L = self.L
        return DomainManager(
            xmin=0, xmax=L, ymin=0, ymax=L, 
            periodic_in_x=True, periodic_in_y=True
        )

    def create_solver(self):
        '''
        Define solver
        '''
        kernel = QuinticSpline(dim=2) 

        integrator = PECIntegrator(fluid = DPSPHStep())
        
        if self.TVF_correction == True:
            integrator = PECIntegrator(fluid = TransportVelocityStep_DPSPH())
            #integrator = GTVFIntegrator(fluid=GTVFStep())

        solver = Solver(
            kernel=kernel, dim=2, integrator=integrator, dt=self.dt, tf=self.tf, 
            pfreq=100
        )

        return solver

    def create_equations(self):
        '''
        Set-up governing equations
        '''

        equations = [
            Group(equations=[
                IsothermalEOS(dest='fluid', sources=['fluid'], rho0=self.rho0, c0=self.c0, p0=0.0),
                GradientCorrectionPreStep(dest='fluid', sources=['fluid'], dim=2),
            ], real=False),

            Group(equations=[
                GradientCorrection(dest='fluid', sources=['fluid'], dim=2, tol=0.1),
                ContinuityEquationDeltaSPHPreStep(dest='fluid', sources=['fluid']),
                #PST_PreStep_1(dest='fluid', sources=['fluid'], dim=2, boundedFlow=self.PST_boundedFlow),
                #PST_PreStep_2(dest='fluid', sources=['fluid'], dim=2, H=self.h0, boundedFlow=self.PST_boundedFlow),
            ], real=False),

            Group(equations=[
                PST(dest='fluid', sources=['fluid'], dim=2, H=self.h0, dt=self.dt, dx=self.dx, Uc0=self.PST_Uc0, Rh=self.PSR_Rh, saveAllDRh=True, R_coeff=self.PST_R_coeff, n_exp=self.PST_n_exp, boundedFlow=self.PST_boundedFlow),
                ContinuityEquation(dest='fluid', sources=['fluid']),                 
                ContinuityEquationDeltaSPH(dest='fluid', sources=['fluid'], c0=self.c0, delta=0.1),
                LaminarViscosityDeltaSPHPreStep(dest='fluid', sources=['fluid']),
                #LaminarViscosity(dest='fluid', sources=['fluid'], nu=self.nu),
                LaminarViscosityDeltaSPH(dest='fluid', sources=['fluid'], dim=2, rho0=self.rho0, nu=self.nu),
                Spatial_Acceleration(dest='fluid', sources=['fluid']),
            ], real=True),
        ]

        if self.TVF_correction == True: 
            equations = [
                Group(equations=[
                    GradientCorrectionPreStep(dest='fluid', sources=['fluid'], dim=2),
                    GradientCorrection(dest='fluid', sources=['fluid'], dim=2, tol=0.1),
                    ContinuityEquationDeltaSPHPreStep(dest='fluid', sources=['fluid']),
                    #PST_PreStep_1(dest='fluid', sources=['fluid'], dim=2, boundedFlow=self.PST_boundedFlow),
                    #PST_PreStep_2(dest='fluid', sources=['fluid'], dim=2, H=self.h0, boundedFlow=self.PST_boundedFlow),
                ], real=False),

                Group(equations=[
                    PST(dest='fluid', sources=['fluid'], dim=2, H=self.h0, dt=self.dt, dx=self.dx, Uc0=self.PST_Uc0, Rh=self.PSR_Rh, saveAllDRh=True, R_coeff=self.PST_R_coeff, n_exp=self.PST_n_exp, boundedFlow=self.PST_boundedFlow),
                    ContinuityEquation(dest='fluid', sources=['fluid']),                 
                    ContinuityEquationDeltaSPH(dest='fluid', sources=['fluid'], c0=self.c0, delta=0.1),
                ], real=True),

                Group(equations=[
                    StateEquation(dest='fluid', sources=None, p0=self.p0, rho0=self.rho0),
                    #IsothermalEOS(dest='fluid', sources=['fluid'], rho0=self.rho0, c0=self.c0, p0=0.0),
                ], real=False),

                Group(equations=[
                    MomentumEquationPressureGradient(dest='fluid', sources=['fluid'], pb=self.p0),  
                    #MomentumEquationViscosity(dest='fluid', sources=['fluid'], nu=self.nu), 
                    #MomentumEquationArtificialStress(dest='fluid', sources=['fluid']),
                    LaminarViscosityDeltaSPHPreStep(dest='fluid', sources=['fluid']),
                    #LaminarViscosity(dest='fluid', sources=['fluid'], nu=self.nu),
                    LaminarViscosityDeltaSPH(dest='fluid', sources=['fluid'], dim=2, rho0=self.rho0, nu=self.nu),
                      
                ], real=True),
            ]

        return equations

    ###### Post processing
    def _get_post_process_props(self, array):
        """Return x, y, m, u, v, p.
        """
        if 'pavg' not in array.properties or \
           'pavg' not in array.output_property_arrays:
            self._add_extra_props(array)
            sph_eval = self._get_sph_evaluator(array)
            sph_eval.update_particle_arrays([array])
            sph_eval.evaluate()

        x, y, m, u, v, p, pavg = array.get(
            'x', 'y', 'm', 'u', 'v', 'p', 'pavg'
        )
        return x, y, m, u, v, p - pavg

    def _add_extra_props(self, array):
        extra = ['pavg', 'nnbr']
        for prop in extra:
            if prop not in array.properties:
                array.add_property(prop)
        array.add_output_arrays(extra)

    def _get_sph_evaluator(self, array):
        if not hasattr(self, '_sph_eval'):
            from pysph.tools.sph_evaluator import SPHEvaluator
            equations = [
                ComputeAveragePressure(dest='fluid', sources=['fluid'])
            ]
            dm = self.create_domain()
            sph_eval = SPHEvaluator(
                arrays=[array], equations=equations, dim=2,
                kernel=QuinticSpline(dim=2), domain_manager=dm
            )
            self._sph_eval = sph_eval
        return self._sph_eval

    def post_process(self, info_fname):
        info = self.read_info(info_fname)
        if len(self.output_files) == 0:
            return

        from pysph.solver.utils import iter_output
        decay_rate = -8.0 * pi**2 / self.re
        U = self.U

        files = self.output_files
        t, ke, ke_ex, decay, linf, l1, p_l1 = [], [], [], [], [], [], []
        for sd, array in iter_output(files, 'fluid'):
            _t = sd['t']
            t.append(_t)
            x, y, m, u, v, p = self._get_post_process_props(array)
            u_e, v_e, p_e = exact_solution(U, decay_rate, _t, x, y)
            vmag2 = u**2 + v**2
            vmag = np.sqrt(vmag2)
            ke.append(0.5 * np.sum(m * vmag2))
            vmag2_e = u_e**2 + v_e**2
            vmag_e = np.sqrt(vmag2_e)
            ke_ex.append(0.5 * np.sum(m * vmag2_e))

            vmag_max = vmag.max()
            decay.append(vmag_max)
            theoretical_max = U * np.exp(decay_rate * _t)
            linf.append(abs((vmag_max - theoretical_max) / theoretical_max))

            l1_err = np.average(np.abs(vmag - vmag_e))
            avg_vmag_e = np.average(np.abs(vmag_e))
            # scale the error by the maximum velocity.
            l1.append(l1_err / avg_vmag_e)

            p_e_max = np.abs(p_e).max()
            p_error = np.average(np.abs(p - p_e)) / p_e_max
            p_l1.append(p_error)

        t, ke, ke_ex, decay, l1, linf, p_l1 = list(map(
            np.asarray, (t, ke, ke_ex, decay, l1, linf, p_l1))
        )
        decay_ex = U * np.exp(decay_rate * t)
        fname = os.path.join(self.output_dir, 'results.npz')
        np.savez(
            fname, t=t, ke=ke, ke_ex=ke_ex, decay=decay, linf=linf, l1=l1,
            p_l1=p_l1, decay_ex=decay_ex
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

        plt.clf()
        plt.plot(t, p_l1, label="error")
        plt.xlabel('t')
        plt.ylabel(r'$L_1$ error for $p$')
        fig = os.path.join(self.output_dir, "p_l1_error.png")
        plt.savefig(fig, dpi=300)

    def customize_output(self):
        self._mayavi_config('''
        b = particle_arrays['fluid']
        b.scalar = 'vmag'
        ''')
        
################################################################################
# Main Code
################################################################################
if __name__ == '__main__':
    app = Taylor_Green()
    app.run()
    app.post_process(app.info_filename)


