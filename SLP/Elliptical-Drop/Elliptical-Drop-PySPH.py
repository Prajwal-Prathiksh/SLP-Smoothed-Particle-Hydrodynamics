###########################################################################
# IMPORTS
###########################################################################

# PySPH base imports
from pysph.base.kernels import WendlandQuintic, QuinticSpline, Gaussian
from pysph.base.nnps import DomainManager
from pysph.base.utils import get_particle_array_wcsph

# PySPH solver imports
from pysph.solver.application import Application
from pysph.solver.solver import Solver

# PySPH sph imports
from pysph.sph.equation import Equation, Group
from pysph.sph.wc.edac import ComputeAveragePressure

# SPH Integrator Imports
from pysph.sph.integrator import EulerIntegrator, PECIntegrator, EPECIntegrator
from pysph.sph.integrator_step import EulerStep, WCSPHStep, TransportVelocityStep

# Numpy import
from numpy import ones_like, mgrid, sqrt
import numpy as np
import os

# Include path
import sys
sys.path.insert(1, '/home/prajwal/Desktop/Winter_Project/SLP-Smoothed-Particle-Hydrodynamics')

# SPH Equation Imports
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
def _derivative(x, t):
    A, a = x
    Anew = A*A*(a**4 - 1)/(a**4 + 1)
    anew = -a*A
    return np.array((Anew, anew))


def _scipy_integrate(y0, tf, dt):
    from scipy.integrate import odeint
    result = odeint(_derivative, y0, [0.0, tf])
    return result[-1]


def _numpy_integrate(y0, tf, dt):
    t = 0.0
    y = y0
    while t <= tf:
        t += dt
        y += dt*_derivative(y, t)
    return y


def exact_solution(tf=0.0075, dt=1e-6, n=101):
    """Exact solution for the locus of the circular patch.

    n is the number of points to find the result at.

    Returns the semi-minor axis, A, pressure, x, y.

    Where x, y are the points corresponding to the ellipse.
    """
    import numpy

    y0 = np.array([100.0, 1.0])

    try:
        from scipy.integrate import odeint
    except ImportError:
        Anew, anew = _numpy_integrate(y0, tf, dt)
    else:
        Anew, anew = _scipy_integrate(y0, tf, dt)

    dadt = _derivative([Anew, anew], tf)[0]
    po = 0.5*-anew**2 * (dadt - Anew**2)

    theta = numpy.linspace(0, 2*numpy.pi, n)

    return anew, Anew, po, anew*numpy.cos(theta), 1/anew*numpy.sin(theta)

################################################################################
# EllipticalDrop - Application
################################################################################
class EllipticalDrop(Application):
    def initialize(self):
        '''
        Initialize simulation paramters
        '''

        # Simulation Parameters
        self.nx = 40.0
        self.co = 1400.0
        self.ro = 1.0
        self.hdx = 1.3
        self.dx = 0.025
        self.alpha = 0.1
        self.dx = 1.0/self.nx
        self.rho0 = self.ro
        self.c0 = self.co
        self.h0 = self.dx * self.hdx
        self.dt = 0.25*self.hdx*self.dx/(141 + self.co)
        dx = self.dx
        hdx = self.hdx
        co = self.co
        ro = self.ro
        self.mu = ro*self.alpha*hdx*dx*co/8.0
        self.nu = self.mu / self.rho0

        self.tf = 0.0076

    def create_particles(self):
        """Create the circular patch of fluid."""
        dx = self.dx
        hdx = self.hdx
        co = self.co
        ro = self.ro
        name = 'fluid'
        x, y = mgrid[-1.05:1.05+1e-4:dx, -1.05:1.05+1e-4:dx]
        x = x.ravel()
        y = y.ravel()

        m = ones_like(x)*dx*dx*ro
        h = ones_like(x)*hdx*dx
        rho = ones_like(x) * ro
        u = -100*x
        v = 100*y

        # remove particles outside the circle
        indices = []
        for i in range(len(x)):
            if sqrt(x[i]*x[i] + y[i]*y[i]) - 1 > 1e-10:
                indices.append(i)

        pa = get_particle_array_wcsph(x=x, y=y, m=m, rho=rho, h=h, u=u, v=v,
                                name=name)
        pa.remove_particles(indices)

        print("Elliptical drop :: %d particles"
              % (pa.get_number_of_particles()))
        print("Effective viscosity: rho*alpha*h*c/8 = %s" % self.mu)
        
        pa.add_property('m_mat', stride=9)
        pa.add_property('gradrho', stride=3)

        add_props = ['rho0', 'u0', 'v0', 'w0', 'x0', 'y0', 'z0', 'ax', 'ay', 'az']
        for i in add_props:
            pa.add_property(i)
        return [pa]

    def create_solver(self):
        '''
        Define solver
        '''

        kernel = QuinticSpline(dim=2)#Gaussian(dim=2)
        
        integrator = EPECIntegrator(fluid = WCSPHStep())
        solver = Solver(
            kernel=kernel, dim=2, integrator=integrator, dt=self.dt, tf=self.tf, 
            pfreq=30
        )

        return solver

    def create_equations(self):
        '''
        Set-up governing equations
        '''
        '''
        equations = [
            Group(equations=[
                TaitEOS(dest='fluid', sources=None, rho0=self.rho0, c0=self.c0, gamma=7.0,
                p0=0.0)
            ],
            real=False),
            Group(equations=[
                GradientCorrectionPreStep(dest='fluid', sources=['fluid'], dim=2)
            ],
            real=False),
            Group(equations=[
                GradientCorrection(dest='fluid', sources=['fluid'], dim=2, tol=0.1), 
                ContinuityEquationDeltaSPHPreStep(dest='fluid', sources=['fluid'])
            ],
            real=True),
            Group(equations=[
                ContinuityEquation(dest='fluid', sources=['fluid']), 
                ContinuityEquationDeltaSPH(dest='fluid', sources=['fluid'], c0=self.c0,
                delta=0.1), 
                MomentumEquation(dest='fluid', sources=['fluid'], c0=self.c0,
                alpha=0.0, beta=0.0), 
                MomentumEquationDeltaSPH(dest='fluid', sources=['fluid'], rho0=self.rho0,
                c0=self.c0, alpha=0.1), 
                XSPHCorrection(dest='fluid', sources=['fluid'], eps=0.5)
            ],
            real=True)
        ]
        '''
        equations = [
            Group(equations=[
                IsothermalEOS(dest='fluid', sources=['fluid'], rho0=self.rho0, c0=self.c0, p0=0.0),
                GradientCorrectionPreStep(dest='fluid', sources=['fluid'], dim=2)
            ],
            real=False),

            Group(equations=[
                #LaminarViscosityDeltaSPHPreStep(dest='fluid', sources=['fluid']),
                MomentumEquation(dest='fluid', sources=['fluid'], c0=self.c0, alpha=0.0, beta=0.0), 
                #MomentumEquationDeltaSPH(dest='fluid', sources=['fluid'], rho0=self.rho0, c0=self.c0, alpha=0.1), 
                LaminarViscosityDeltaSPH(dest='fluid', sources=['fluid'], dim=2, rho0=self.rho0, nu=self.nu), 
                #XSPHCorrection(dest='fluid', sources=['fluid'], eps=0.5),
                Spatial_Acceleration(dest='fluid', sources=['fluid']),
                ContinuityEquation(dest='fluid', sources=['fluid']),
                GradientCorrection(dest='fluid', sources=['fluid'], dim=2, tol=0.1), 
                ContinuityEquationDeltaSPHPreStep(dest='fluid', sources=['fluid']),                 
                ContinuityEquationDeltaSPH(dest='fluid', sources=['fluid'], c0=self.c0,
                delta=0.1), 
            ],
            real=True),
        ]


        return equations

    def _make_final_plot(self):
        try:
            import matplotlib
            matplotlib.use('Agg')
            from matplotlib import pyplot as plt
        except ImportError:
            print("Post processing requires matplotlib.")
            return
        last_output = self.output_files[-1]
        from pysph.solver.utils import load
        data = load(last_output)
        pa = data['arrays']['fluid']
        tf = data['solver_data']['t']
        a, A, po, xe, ye = exact_solution(tf)
        print("At tf=%s" % tf)
        print("Semi-major axis length (exact, computed) = %s, %s"
              % (1.0/a, max(pa.y)))
        plt.plot(xe, ye, label='exact')
        plt.scatter(pa.x, pa.y, marker='.', label='particles')
        plt.ylim(-2, 2)
        plt.xlim(plt.ylim())
        plt.title("Particles at %s secs" % tf)
        plt.xlabel('x')
        plt.ylabel('y')
        plt.legend()
        fig = os.path.join(self.output_dir, "comparison.png")
        plt.savefig(fig, dpi=300)
        print("Figure written to %s." % fig)
        
    def _compute_results(self):
        from pysph.solver.utils import iter_output
        from collections import defaultdict
        data = defaultdict(list)
        for sd, array in iter_output(self.output_files, 'fluid'):
            _t = sd['t']
            data['t'].append(_t)
            m, u, v, x, y = array.get('m', 'u', 'v', 'x', 'y')
            vmag2 = u**2 + v**2
            data['ke'].append(0.5*np.sum(m*vmag2))
            data['xmax'].append(x.max())
            data['ymax'].append(y.max())
            a, A, po, _xe, _ye = exact_solution(_t, n=0)
            data['minor'].append(a)
            data['major'].append(1.0/a)
            data['xe'].append(_xe)
            data['ye'].append(_ye)
            data['y'].append(y)
            data['x'].append(x)


        for key in data:
            data[key] = np.asarray(data[key])
        fname = os.path.join(self.output_dir, 'results.npz')
        np.savez(fname, **data)

    def post_process(self, info_file_or_dir):
        if self.rank > 0:
            return
        self.read_info(info_file_or_dir)
        if len(self.output_files) == 0:
            return
        self._compute_results()
        self._make_final_plot()


if __name__ == '__main__':
    app = EllipticalDrop()
    app.run()
    app.post_process(app.info_filename)
