r"""
:math:`\delta^+` SPH
#####################
References
-----------
    .. [Sun2018] P. N. Sun, A. Colagrossi, and A. M. Zhang, “Numerical 
        simulation of the self-propulsive motion of a fishlike swimming foil 
        using the δ+-SPH model,” Theor. Appl. Mech. Lett., vol. 8, no. 2, 
        pp. 115–125, 2018, doi: 10.1016/j.taml.2018.02.007.

    .. [Sun2017] P. N. Sun, A. Colagrossi, S. Marrone, and A. M. Zhang, “The 
        δplus-SPH model: Simple procedures for a further improvement of the SPH 
        scheme,” Comput. Methods Appl. Mech. Eng., vol. 315, pp. 25–49, Mar. 
        2017, doi: 10.1016/j.cma.2016.10.028.
"""
###########################################################################
# PARTICLE ARRAY
###########################################################################
from pysph.base.utils import get_particle_array

### `\delta^+` Fluid Particle Array
def get_particle_array_DeltaPlus_fluid(constants=None, **props):
    """Returns a fluid particle array for the :math:`\delta^+` - SPH Scheme

        This sets the default properties to be::

            ['x', 'y', 'z', 'u', 'v', 'w', 'm', 'h', 'rho', 'p', 'au', 'av', 
            'aw', 'gid', 'pid', 'tag', 'ax', 'ay', 'az', 'arho', 'x0', 'y0', 
            'u0', 'v0', 'rho0', 'xstar', 'ystar', 'ustar', 'vstar', 'rhostar',
            'vmag','vmag2', 'V', 'lmda', 'DX', 'DY', 'DZ', 'Dmag',]

        Parameters:
        -----------
        constants : dict
            Dictionary of constants

        Other Parameters
        ----------------
        props : dict
            Additional keywords passed are set as the property arrays.

        See Also
        --------
        get_particle_array
    """
    # Properties required for :math:`\delta^+` - SPH Scheme
    deltaPlus_props = [
            'ax', 'ay', 'az', 'arho',
            'x0', 'y0', 'u0', 'v0', 'rho0',
            'xstar', 'ystar', 'ustar', 'vstar', 'rhostar',
            'vmag','vmag2', 'V',
            'lmda', 'DX', 'DY', 'DZ', 'Dmag',
    ]

    pa = get_particle_array(
        constants=constants, additional_props=deltaPlus_props, **props
    )

    # Additional properties required for free surfaces
    pa.add_property('m_mat', stride=9)
    pa.add_property('gradlmda', stride=3)
    pa.add_property('gradrho', stride=3)

    # default property arrays to save out
    pa.set_output_arrays([
        'x', 'y', 'z', 'u', 'v', 'w', 'rho', 'p', 'h', 'm', 
        'vmag', 'vmag2', 'lmda', 'DX', 'DY', 'DZ', 'Dmag',
        'pid', 'gid', 'tag',
    ])
    return pa

### `\delta^+` Solid Particle Array
def get_particle_array_DeltaPlus_solid(constants=None, **props):
    """Returns a solid particle array for the :math:`\delta^+` - SPH Scheme

        This sets the default properties to be::

            ['x', 'y', 'z', 'u', 'v', 'w', 'm', 'h', 'rho', 'p', 'au', 'av', 
            'aw', 'gid', 'pid', 'tag', 'lmda', 'V', 'wij', 'wij2', 'ug', 'vf', 
            'wg', 'uf', 'vg', 'wf', ]

        Parameters:
        -----------
        constants : dict
            Dictionary of constants

        Other Parameters
        ----------------
        props : dict
            Additional keywords passed are set as the property arrays.

        See Also
        --------
        get_particle_array
    """
    # Properties required for :math:`\delta^+` - SPH Scheme
    deltaPlus_props = [
        'lmda', 'V', 'wij', 'wij2',
        'ug', 'vf', 'wg', 'uf', 'vg', 'wf', 
    ]

    pa = get_particle_array(
        constants=constants, additional_props=deltaPlus_props, **props
    )

    # Additional properties required
    pa.add_property('m_mat', stride=9)
    pa.add_property('gradlmda', stride=3)
    pa.add_property('gradrho', stride=3)

    # default property arrays to save out
    pa.set_output_arrays([
        'x', 'y', 'z', 'u', 'v', 'w', 'rho', 'p', 'h', 
        'pid', 'gid', 'tag'
    ])
    return pa


###########################################################################
# EQUATIONS & RESPECTIVE IMPORTS
###########################################################################
from pysph.sph.equation import Equation, MultiStageEquations, Group
from math import sqrt
from textwrap import dedent
from compyle.api import declare

### Kernel Correction------------------------------------------------------
from pysph.sph.wc.kernel_correction import (
    GradientCorrectionPreStep, GradientCorrection, 
)

### Equation of State------------------------------------------------------
from pysph.sph.basic_equations import IsothermalEOS 

### Continuity Equation----------------------------------------------------
from pysph.sph.wc.basic import (
    ContinuityEquationDeltaSPHPreStep, ContinuityEquationDeltaSPH
)
from pysph.sph.wc.transport_velocity import ContinuityEquation

### Momentum Equation------------------------------------------------------
from pysph.sph.wc.viscosity import LaminarViscosityDeltaSPH

class LaminarViscosityDeltaSPHPreStep(Equation):
    r""" *Momentum equation defined by the :math:`\delta^+` SPH scheme*

        ..math::
            \frac{D\mathbf{u_i}}{Dt}=\frac{1}{\rho_i}\sum_j\Big(F_{ij}\nabla_i 
            W_{ij}V_j\big)+\mathbf{f_i}

        where,

        ..math::
            F_{ij}=\begin{cases}
                -(p_j+p_i), & p_i\geq 0\\ 
                -(p_j-p_i), & p_i<0
                \end{cases}

        References:
        -----------

        .. [Sun2018] P. N. Sun, A. Colagrossi, and A. M. Zhang, “Numerical 
            simulation of the self-propulsive motion of a fishlike swimming foil 
            using the δ+-SPH model,” Theor. Appl. Mech. Lett., vol. 8, no. 2, 
            pp. 115–125, 2018, doi: 10.1016/j.taml.2018.02.007.

        Parameters:
        -----------
        fx : float, Default = 0.0
            Body-force in x-axis

        fy : float, Default = 0.0
            Body-force in y-axis

        fz : float, Default = 0.0
            Body-force in z-axis
    """

    def __init__(self, dest, sources, fx=0.0, fy=0.0, fz=0.0):
        r"""
            Parameters:
            -----------
            fx : float, Default = 0.0
                Body-force in x-axis

            fy : float, Default = 0.0
                Body-force in y-axis

            fz : float, Default = 0.0
                Body-force in z-axis
            """
        self.fx = fx
        self.fy = fy
        self.fz = fz

        super(LaminarViscosityDeltaSPHPreStep, self).__init__(dest, sources)

    def initialize(self, d_idx, d_au, d_av, d_aw):
        d_au[d_idx] = 0.0
        d_av[d_idx] = 0.0
        d_aw[d_idx] = 0.0

    def loop(
        self, d_idx, s_idx, s_rho, DWIJ, s_m, d_au, d_av,  d_aw, d_p, s_p, d_rho
    ):

        rhoi = d_rho[d_idx]         
        Vj = s_m[s_idx]/s_rho[s_idx]

        Pi = d_p[d_idx]
        Pj = s_p[s_idx]
        
        # F_ij
        if Pi < 0.0:
            Fij = -1.0*(Pj - Pi)
        else:
            Fij = -1.0*(Pi + Pj)

        fac = Fij*Vj/rhoi

        # Accelerations
        d_au[d_idx] += fac*DWIJ[0]
        d_av[d_idx] += fac*DWIJ[1]
        d_aw[d_idx] += fac*DWIJ[2]

### Position Equation------------------------------------------------------
class Spatial_Acceleration(Equation):
    r""" *Spatial Acceleration*

        ..math::
            \frac{D\mathbf{r_i}}{Dt}=\mathbf{u_i}
    """
    def __init__(self, dest, sources):
        super(Spatial_Acceleration, self).__init__(dest, sources)

    def initialize(self, d_idx, d_ax, d_ay, d_az, d_u, d_v, d_w):
        d_ax[d_idx] = d_u[d_idx]
        d_ay[d_idx] = d_v[d_idx]
        d_az[d_idx] = d_w[d_idx]

### Boundary Conditions----------------------------------------------------
from pysph.sph.wc.transport_velocity import (
    SetWallVelocity, SolidWallNoSlipBC
)

class EvaluateNumberDensity(Equation):
    def __init__(self, dest, sources, hij_fac=0.5):
        self.hij_fac = hij_fac
        super(EvaluateNumberDensity, self).__init__(dest, sources)

    def initialize(self, d_idx, d_wij, d_wij2):
        d_wij[d_idx] = 0.0
        d_wij2[d_idx] = 0.0

    def loop(self, d_idx, d_wij, d_wij2, XIJ, HIJ, RIJ, SPH_KERNEL):
        wij = SPH_KERNEL.kernel(XIJ, RIJ, HIJ)
        wij2 = SPH_KERNEL.kernel(XIJ, RIJ, self.hij_fac*HIJ)
        d_wij[d_idx] += wij
        d_wij2[d_idx] += wij2

class SetPressureSolid(Equation):
    def __init__(self, dest, sources, rho0, p0, b=1.0, gx=0.0, gy=0.0, gz=0.0,
                 hg_correction=True, hij_fac=0.5):
        
        self.rho0 = rho0
        self.gx = gx
        self.gy = gy
        self.gz = gz
        self.p0 = p0
        self.b = b
        self.hij_fac = hij_fac
        #self.hg_correction = hg_correction
        super(SetPressureSolid, self).__init__(dest, sources)

    def initialize(self, d_idx, d_p):
        d_p[d_idx] = 0.0

    def loop(self, d_idx, s_idx, d_p, s_p, s_rho,
             d_au, d_av, d_aw, XIJ, RIJ, HIJ, d_wij2, SPH_KERNEL):

        # numerator of Eq. (27) ax, ay and az are the prescribed wall
        # accelerations which must be defined for the wall boundary
        # particle
        wij = SPH_KERNEL.kernel(XIJ, RIJ, HIJ)
        wij2 = SPH_KERNEL.kernel(XIJ, RIJ, self.hij_fac*HIJ)

        gdotxij = (self.gx - d_au[d_idx])*XIJ[0] + \
            (self.gy - d_av[d_idx])*XIJ[1] + \
            (self.gz - d_aw[d_idx])*XIJ[2]

        if d_wij2[d_idx] > 1e-14:
            d_p[d_idx] += s_p[s_idx]*wij2 + s_rho[s_idx]*gdotxij*wij2
        else:
            d_p[d_idx] += s_p[s_idx]*wij + s_rho[s_idx]*gdotxij*wij

    def post_loop(self, d_idx, d_wij, d_wij2, d_p, d_rho):
        # extrapolated pressure at the ghost particle
        if d_wij2[d_idx] > 1e-14:
            d_p[d_idx] /= d_wij2[d_idx]
        elif d_wij[d_idx] > 1e-14:
            d_p[d_idx] /= d_wij[d_idx]

        # update the density from the pressure Eq. (28)
        d_rho[d_idx] = self.rho0 * (d_p[d_idx]/self.p0 + self.b)

### Particle Shifting Technique (PST)--------------------------------------
class PST_PreStep_1(Equation):
    r"""**PST_PreStep_1**

        See :class:`pysph.sph.wc.kernel_correction.GradientCorrectionPreStep`

        Calculates the minimum eigenvalue :math:`\lambda_i` of the 
        :math:`\mathbb{L}` matrix, only if the flow has free surfaces

        Parameters:
        -----------
        dim : integer
            Number of dimensions

        boundedFlow : boolean
            If True, flow has free surface/s

        Notes:
        ------ 
        This equation needs to be called after `GradientCorrectionPreStep` has 
        computed the :math:`\mathbb{L}` matrix
    """
    def __init__(self, dest, sources, dim, boundedFlow):
        r'''
            Parameters:
            -----------
            dim : integer
                Number of dimensions

            boundedFlow : boolean,
                If True, flow has free surface/s
        '''                   
        self.dim = dim
        self.boundedFlow = boundedFlow

        super(PST_PreStep_1, self).__init__(dest, sources)

    def _cython_code_(self):
        code = dedent("""
        cimport cython
        from pysph.base.linalg3 cimport eigen_decomposition
        """)
        return code

    def initialize(self, d_idx, d_lmda):
        d_lmda[d_idx] = 1.0 # Initialize \lambda_i

    def loop(self, d_idx, d_m_mat, d_lmda):

        i, j, n = declare('int', 3)        
        n = self.dim

        ## Matrix and vector declarations ##

        # Matrix of Eigenvectors (columns)
        R = declare('matrix((3,3))')
        # Eigenvalues
        V = declare('matrix((3,))')

        # L-Matrix
        L = declare('matrix((3,3))')

        if self.boundedFlow == False:
            for i in range(n):
                for j in range(n):
                    L[i][j] = d_m_mat[9 * d_idx + 3 * i + j]

            if n == 2:
                L[2][2] = 99999.0

            # compute the principle stresses
            eigen_decomposition(L, R, cython.address(V[0]))

            lmda = 1.0
            for i in range(n):
                if V[i] < lmda and V[i] >= 0.0:
                    lmda = V[i]

            d_lmda[d_idx] = lmda

class PST_PreStep_2(Equation):
    r"""**PST_PreStep_2**

        See :class:`pysph.sph.wc.kernel_correction.GradientCorrectionPreStep` 
        and :class:`pysph.sph.wc.kernel_correction.GradientCorrection` which
        multiples the matrix :math:`\mathbb{L}` with :math:`\nabla W_{ij}`

        Calculates :math:`\langle\nabla\lambda_i\rangle`


        ..math::
            \langle\nabla\lambda_i\rangle=\sum_j(\lambda_j-\lambda_i)\otimes
            \mathbb{L}_i\nabla_i W_{ij}V_j

        Parameters:
        -----------
        boundedFlow : boolean
            If True, flow has free surface/s

        Notes:
        ------ 
        This equation needs to be grouped in the same group as 
        `GradientCorrection` and `PST_PreStep_1`, since it needs to be 
        pre-multiplied with :math:`\mathbb{L}` and requires the pre-calculated
        :math:`\lambda_i` value
        """
    def __init__(self, dest, sources, boundedFlow):
        r'''
            Parameters:
            -----------
            boundedFlow : boolean
                If True, flow has free surface/s
        '''       
        self.boundedFlow = boundedFlow
        
        super(PST_PreStep_2, self).__init__(dest, sources)

    def initialize(self, d_idx, d_gradlmda):
        d_gradlmda[d_idx*3 + 0] = 0.0
        d_gradlmda[d_idx*3 + 1] = 0.0
        d_gradlmda[d_idx*3 + 2] = 0.0

    def loop(self, d_idx, s_idx, d_lmda, s_lmda, s_rho, s_m, d_gradlmda, DWIJ):

        if self.boundedFlow == False:
            lmdai = d_lmda[d_idx]
            lmdaj = s_lmda[s_idx]

            Vj = s_m[s_idx]/s_rho[s_idx]

            fac = (lmdai - lmdaj) * Vj

            d_gradlmda[d_idx*3 + 0] += fac * DWIJ[0]
            d_gradlmda[d_idx*3 + 1] += fac * DWIJ[1]
            d_gradlmda[d_idx*3 + 2] += fac * DWIJ[2]

class PST(Equation):
    r"""**Particle-Shifting Technique employed (:math:`\delta^+` - SPH Scheme)**

        ..math::
            \mathbf{r_i}^\ast=\mathbf{r_i}+\delta\mathbf{\hat{r_i}}

            \delta \mathbf{r_i} = \frac{-U_{max}h\Delta t}{2}\sum_i \Bigg[1 + R\Bigg(\frac{W_{ij}}{W(\Delta)}\Bigg)^n\Bigg]\nabla \mathbf{W_{ij}} \Bigg(\frac{m_j}{\rho_i + \rho_j}\Bigg)
        
        where, 

        ..math::
            \delta\mathbf{\hat{r_i}}=\begin{cases}
                0  & ,\lambda_i\in[0,0.4) \\
                (\mathbb{I}-\mathbf{n_i}\otimes\mathbf{n_i})\delta\mathbf{r_i} 
                & ,\lambda_i\in[0.4, 0.75] \\
                \delta\mathbf{r_i} & ,\lambda_i\in(0.75,1]
            \end{cases}

        ..math::
            \delta\mathbf{r_i}=\frac{-\Delta t c_o(2h)^2}{h_i}.\sum_j\bigg[1+R
            \bigg(\frac{W_{ij}}{W(\Delta p)}\bigg)^n\bigg]\nabla_i W_{ij}\bigg(
            \frac{m_j}{\rho_i+\rho_j}\bigg)

        ..math::
            \mathbf{n_i}=\frac{\langle\nabla\lambda_i\rangle}{|\langle\nabla
            \lambda_i\rangle|}

        ..math::
            \langle\nabla\lambda_i\rangle=\sum_j(\lambda_j-\lambda_i)\otimes
            \mathbb{L}_i\nabla_i W_{ij}V_j

        ..math::
            \lambda_i=\text{min}\big(\text{eigenvalue}(\mathbb{L_i^{-1}})\big)

        ..math::
            \mathbb{L_i}=\bigg[\sum_j\mathbf{r_{ji}}\otimes\nabla_i W_{ij}V_j
            \bigg]^{-1}


        References:
        -----------
        .. [Sun2017] Sun, P. N., et al. “The δ p l u s -SPH Model: Simple
        Procedures for a Further Improvement of the SPH Scheme.” Computer 
        Methods in Applied Mechanics and Engineering, vol. 315, Mar. 2017, pp. 
        25–49. DOI.org (Crossref), doi:10.1016/j.cma.2016.10.028.

        .. [Sun2019] Sun, P. N., et al. “A Consistent Approach to Particle 
        Shifting in the δ - Plus -SPH Model.” Computer Methods in Applied 
        Mechanics and Engineering, vol. 348, May 2019, pp. 912–34. DOI.org 
        (Crossref), doi:10.1016/j.cma.2019.01.045.

        .. [Monaghan2000] Monaghan, J. J. “SPH without a Tensile Instability.” 
        Journal of Computational Physics, vol. 159, no. 2, Apr. 2000, pp. 
        290–311. DOI.org (Crossref), doi:10.1006/jcph.2000.6439.
        
        Parameters:
        -----------
        H : float
            Kernel smoothing length (:math:`h`)

        dt : float
            Time step (:math:`\Delta t`)

        dx : float
            Initial particle spacing (:math:`\Delta x`)

        Uc0 : float
            :math:`\frac{U}{c_o}` value of the system

        boundedFlow : boolean
            If True, flow has free surface/s

        R_coeff : float, default = 0.2
            Artificial pressure coefficient

        n_exp : float, default = 4
            Artificial pressure exponent

        Rh : float, default = 0.05
            Maximum :math:`\frac{|\delta r_i|}{h}` value allowed during 
            particle shifting 
            (Note: :math:`\delta r_i = 0` if :math:`\frac{|\delta r_i|}{h}>R_h`)
    """
    def __init__(
        self, dest, sources, H, dt, dx, Uc0, boundedFlow, R_coeff=0.2, n_exp=4.0,
        max_Dmag=0.05,
    ):        
        self.R_coeff = R_coeff
        self.n_exp = n_exp
        self.H = H
        self.dx = dx
        self.Uc0 = Uc0
        self.max_Dmag = max_Dmag
        self.boundedFlow = boundedFlow
        self.eps = H**2 * 0.01
        self.CONST = -0.5*dt*H 

        super(PST, self).__init__(dest, sources)

    def initialize(self, d_idx, d_DX, d_DY, d_DZ, d_Dmag):
        d_DX[d_idx] = 0.0
        d_DY[d_idx] = 0.0
        d_DZ[d_idx] = 0.0
        d_Dmag[d_idx] = 0.0

    def loop(
        self, d_idx, d_rho, d_DX, d_DY, d_DZ, s_idx, s_rho, s_m, XIJ, DWIJ, WIJ,
        SPH_KERNEL
    ):
        rhoi = d_rho[d_idx]
        rhoj = s_rho[s_idx]
        mj = s_m[s_idx]

        # Calculate Kernel values
        w_delta_s = SPH_KERNEL.kernel(XIJ, self.dx, self.H)

        ################################################################
        # Calculate \delta r_i
        ################################################################

        # Calcuate fij
        fij = self.R_coeff * pow((WIJ/(self.eps+w_delta_s)), self.n_exp)

        # Calcuate multiplicative factor
        fac = (1.0 + fij)*(mj/(rhoi+rhoj+self.eps))

        # Sum \delta r_i
        d_DX[d_idx] += self.CONST * self.Uc0 * fac * DWIJ[0]
        d_DY[d_idx] += self.CONST * self.Uc0 * fac * DWIJ[1]
        d_DZ[d_idx] += self.CONST * self.Uc0 * fac * DWIJ[2] 

    def post_loop(self, d_idx, d_lmda, d_DX, d_DY, d_DZ, d_Dmag):

        lmdai = d_lmda[d_idx]
        if self.boundedFlow == True or lmdai > 0.75:

            mag = sqrt(d_DX[d_idx]*d_DX[d_idx] + d_DY[d_idx]*d_DY[d_idx] + d_DZ[d_idx]*d_DZ[d_idx]) / self.dx
            d_Dmag[d_idx] = mag

            if mag > self.max_Dmag:
                # Check norm condition and correct the values
                d_DX[d_idx] = d_DX[d_idx] * self.max_Dmag / mag
                d_DY[d_idx] = d_DY[d_idx] * self.max_Dmag / mag
                d_DZ[d_idx] = d_DZ[d_idx] * self.max_Dmag / mag
                d_Dmag[d_idx] = self.max_Dmag


###########################################################################
# INTEGRATOR
###########################################################################
from pysph.sph.integrator import Integrator, IntegratorStep

### Runge-Kutta Fourth-Order Integrator------------------------------------
class RK4Integrator(Integrator):
    def one_timestep(self, t, dt):
        #Initialise `y_{n}` properties
        self.stage0()

        #Store computed `k_1` properties, and subsequently calculate `Y_2`
        self.compute_accelerations(update_nnps=True)
        self.stage1()
        self.update_domain()

        #Store computed `k_2` properties, and subsequently calculate `Y_3`
        self.compute_accelerations(update_nnps=True)
        self.stage2()
        self.update_domain()

        #Store computed `k_3` properties, and subsequently calculate `Y_4`
        self.compute_accelerations(update_nnps=True)
        self.stage3()
        self.update_domain()

        #Store computed `k_4` properties, and subsequently calculate `y_{n+1}`
        self.compute_accelerations(update_nnps=True)
        self.stage4()
        self.update_domain()

        self.compute_accelerations(index=1, update_nnps=True)
        self.stage5()
        self.update_domain()

### Runge-Kutta Fourth-Order Integrator Step-------------------------------
class RK4Step(IntegratorStep):
    def stage0(
        self, d_idx, d_x, d_y, d_u, d_v, d_rho,
        d_x0, d_y0, d_u0, d_v0, d_rho0,
    ):
        #Initialise `y_{n}` properties
        d_x0[d_idx] = d_x[d_idx]
        d_y0[d_idx] = d_y[d_idx]
        d_u0[d_idx] = d_u[d_idx]
        d_v0[d_idx] = d_v[d_idx]
        d_rho0[d_idx] = d_rho[d_idx]

        #print('0', d_x[d_idx], d_y[d_idx], d_u[d_idx], d_v[d_idx], d_rho[d_idx])

    def stage1(
        self, d_idx, d_x, d_y, d_u, d_v, d_rho,
        d_x0, d_y0, d_u0, d_v0, d_rho0,
        d_ax, d_ay, d_au, d_av, d_arho,
        d_xstar, d_ystar, d_ustar, d_vstar, d_rhostar,
        dt,
    ):
        dtby2 = dt*0.5
        
        #Store computed `k_1` properties
        d_xstar[d_idx] = d_ax[d_idx]
        d_ystar[d_idx] = d_ay[d_idx]
        d_ustar[d_idx] = d_au[d_idx]
        d_vstar[d_idx] = d_av[d_idx]
        d_rhostar[d_idx] = d_arho[d_idx]
        
        #Calculate `Y_2`
        d_x[d_idx] = d_x0[d_idx] + d_ax[d_idx]*dtby2
        d_y[d_idx] = d_y0[d_idx] + d_ay[d_idx]*dtby2
        d_u[d_idx] = d_u0[d_idx] + d_au[d_idx]*dtby2
        d_v[d_idx] = d_v0[d_idx] + d_av[d_idx]*dtby2
        d_rho[d_idx] = d_rho0[d_idx] + d_arho[d_idx]*dtby2

    def stage2(
        self, d_idx, d_x, d_y, d_u, d_v, d_rho,
        d_x0, d_y0, d_u0, d_v0, d_rho0,
        d_ax, d_ay, d_au, d_av, d_arho,
        d_xstar, d_ystar, d_ustar, d_vstar, d_rhostar,
        dt,
    ):
        dtby2 = dt*0.5
        
        #Store computed `k_2` properties
        d_xstar[d_idx] += 2.0*d_ax[d_idx]
        d_ystar[d_idx] += 2.0*d_ay[d_idx]
        d_ustar[d_idx] += 2.0*d_au[d_idx]
        d_vstar[d_idx] += 2.0*d_av[d_idx]
        d_rhostar[d_idx] += 2.0*d_arho[d_idx]
        
        #Calculate `Y_3`
        d_x[d_idx] = d_x0[d_idx] + d_ax[d_idx]*dtby2
        d_y[d_idx] = d_y0[d_idx] + d_ay[d_idx]*dtby2
        d_u[d_idx] = d_u0[d_idx] + d_au[d_idx]*dtby2
        d_v[d_idx] = d_v0[d_idx] + d_av[d_idx]*dtby2
        d_rho[d_idx] = d_rho0[d_idx] + d_arho[d_idx]*dtby2

    def stage3(
        self, d_idx, d_x, d_y, d_u, d_v, d_rho,
        d_x0, d_y0, d_u0, d_v0, d_rho0,
        d_ax, d_ay, d_au, d_av, d_arho,
        d_xstar, d_ystar, d_ustar, d_vstar, d_rhostar,
        dt,
    ):
        #Store computed `k_3` properties
        d_xstar[d_idx] += 2.0*d_ax[d_idx]
        d_ystar[d_idx] += 2.0*d_ay[d_idx]
        d_ustar[d_idx] += 2.0*d_au[d_idx]
        d_vstar[d_idx] += 2.0*d_av[d_idx]
        d_rhostar[d_idx] += 2.0*d_arho[d_idx]
        
        #Calculate `Y_4`
        d_x[d_idx] = d_x0[d_idx] + d_ax[d_idx]*dt
        d_y[d_idx] = d_y0[d_idx] + d_ay[d_idx]*dt
        d_u[d_idx] = d_u0[d_idx] + d_au[d_idx]*dt
        d_v[d_idx] = d_v0[d_idx] + d_av[d_idx]*dt
        d_rho[d_idx] = d_rho0[d_idx] + d_arho[d_idx]*dt

    def stage4(
        self, d_idx, d_x, d_y, d_u, d_v, d_rho,
        d_x0, d_y0, d_u0, d_v0, d_rho0,
        d_ax, d_ay, d_au, d_av, d_arho,
        d_xstar, d_ystar, d_ustar, d_vstar, d_rhostar,
        d_vmag2, d_vmag, dt,
    ):
        dtby6 = dt/6.0
        #Store computed `k_4` properties
        d_xstar[d_idx] += d_ax[d_idx]
        d_ystar[d_idx] += d_ay[d_idx]
        d_ustar[d_idx] += d_au[d_idx]
        d_vstar[d_idx] += d_av[d_idx]
        d_rhostar[d_idx] += d_arho[d_idx]
        
        #Calculate `y_{n+1}`
        d_x[d_idx] = d_x0[d_idx] + d_xstar[d_idx]*dtby6
        d_y[d_idx] = d_y0[d_idx] + d_ystar[d_idx]*dtby6
        d_u[d_idx] = d_u0[d_idx] + d_ustar[d_idx]*dtby6
        d_v[d_idx] = d_v0[d_idx] + d_vstar[d_idx]*dtby6
        d_rho[d_idx] = d_rho0[d_idx] + d_rhostar[d_idx]*dtby6

        # magnitude of velocity squared
        d_vmag2[d_idx] = (d_u[d_idx]*d_u[d_idx] + d_v[d_idx]*d_v[d_idx])

        d_vmag[d_idx] = sqrt(d_vmag2[d_idx])

    def stage5(
        self, d_idx, d_x, d_y,
        d_DX, d_DY,
    ):

        # PST Corrections
        d_x[d_idx] += d_DX[d_idx]
        d_y[d_idx] += d_DY[d_idx]


###########################################################################
# SCHEME
###########################################################################
from pysph.sph.scheme import Scheme

### `\delta^+` SPH Scheme
class DeltaPlusScheme(Scheme):
    def __init__(
        self, fluids, solids, dim, rho0, c0, nu, p0, hdx, dx, h0, dt, 
        PST_boundedFlow=True, max_Dmag=0.05,
    ):
        self.fluids = fluids
        self.solids = solids
        self.solver = None
        self.rho0 = rho0
        self.c0 = c0
        self.p0 = p0
        self.nu = nu
        self.dim = dim
        self.hdx = hdx
        self.dx = dx
        self.dt=dt
        self.h0 = h0
        self.PST_boundedFlow = PST_boundedFlow
        self.max_Dmag = max_Dmag

    def add_user_options(self, group):
        group.add_argument(
            "--pst-dmag", action="store", type=float, dest="max_Dmag", default=0.05,
            help="Maximum permissible norm of the correction from PST (`Dmag`) => [Rh = (delta r_i)/(delta x_i)]"
        )

    def consume_user_options(self, options):
        vars = ['max_Dmag']
        data = dict((var, self._smart_getattr(options, var))
                    for var in vars)
        self.configure(**data)

    def get_timestep(self, cfl=1.0):
        dt_cfl = cfl * self.h0/self.c0
        dt_viscous = 0.125 * self.h0**2/self.nu
        dt_force = 1.0

        return min(dt_cfl, dt_viscous, dt_force)

    def configure_solver(self, kernel=None, integrator_cls=None,
                         extra_steppers=None, **kw):
        from pysph.base.kernels import WendlandQuintic
        from pysph.solver.solver import Solver

        if kernel is None:
            kernel = WendlandQuintic(dim=self.dim)
        steppers = {}
        if extra_steppers is not None:
            steppers.update(extra_steppers)

        step_cls = RK4Step
        for fluid in self.fluids:
            if fluid not in steppers:
                steppers[fluid] = step_cls()

        cls = integrator_cls if integrator_cls is not None else RK4Integrator
        integrator = cls(**steppers)

        if 'dt' not in kw:
            kw['dt'] = self.get_timestep()
        self.solver = Solver(
            dim=self.dim, integrator=integrator, kernel=kernel, **kw
        )

    def get_equations(self):
        all = self.fluids + self.solids
        
        stage0 = []
        eq1 = []
        if self.solids:
            for solid in self.solids:
                eq1.append(EvaluateNumberDensity(dest=solid, sources=self.fluids))

        for fluid in self.fluids:
            eq1.append(GradientCorrectionPreStep(dest=fluid, sources=all, dim=self.dim))
            eq1.append(GradientCorrection(dest=fluid, sources=all, dim=self.dim))
            eq1.append(ContinuityEquationDeltaSPHPreStep(dest=fluid, sources=all))
        stage0.append(Group(equations=eq1, real=False))

        eq2 = []    
        for fluid in self.fluids:
            eq2.append(ContinuityEquation(dest=fluid, sources=all))
            eq2.append(ContinuityEquationDeltaSPH(dest=fluid, sources=all, c0=self.c0))
        stage0.append(Group(equations=eq2, real=False))

        eq3 = []
        for fluid in self.fluids:
            eq3.append(IsothermalEOS(dest=fluid, sources=None, rho0=self.rho0, c0=self.c0, p0=0.0))
        stage0.append(Group(equations=eq3, real=False))

        if self.solids:
            eq4 = []
            for solid in self.solids:
                eq4.append(SetWallVelocity(dest=solid, sources=self.fluids))
                eq4.append(SetPressureSolid(dest=solid, sources=self.fluids, rho0=self.rho0, p0=self.p0))
            stage0.append(Group(equations=eq4, real=False))

        eq5 = []
        for fluid in self.fluids:
            eq5.append(LaminarViscosityDeltaSPHPreStep(dest=fluid, sources=self.fluids))
            eq5.append(LaminarViscosityDeltaSPH(dest=fluid, sources=self.fluids, dim=self.dim, rho0=self.rho0, nu=self.nu))
            if self.solids:
                eq5.append(SolidWallNoSlipBC(dest=fluid, sources=self.solids, nu=self.nu))
            eq5.append(Spatial_Acceleration(dest=fluid, sources=self.fluids))
        stage0.append(Group(equations=eq5, real=True))

        stage1 = []
        eq6 = []
        for fluid in self.fluids:
            eq6.append(GradientCorrectionPreStep(dest=fluid, sources=all, dim=self.dim))
            eq6.append(GradientCorrection(dest=fluid, sources=all, dim=self.dim))
            eq6.append(PST_PreStep_1(dest=fluid, sources=all, dim=self.dim, boundedFlow=self.PST_boundedFlow))
            eq6.append(PST_PreStep_2(dest=fluid, sources=all, boundedFlow=self.PST_boundedFlow))
        stage1.append(Group(equations=eq6, real=False))

        eq7 = []
        for fluid in self.fluids:
            eq7.append(PST(dest=fluid, sources=all, H=self.h0, dt=self.dt, dx=self.dx, Uc0=self.c0, boundedFlow=self.PST_boundedFlow, max_Dmag=self.max_Dmag))
        stage1.append(Group(equations=eq7, real=False))

        return MultiStageEquations([stage0,stage1])

    def setup_properties(self, particles, clean=True):
        temp = ['m_mat', 'gradlmda', 'gradrho']

        particle_arrays = dict([(p.name, p) for p in particles])
        dummy = get_particle_array_DeltaPlus_fluid(name='junk')
        props = list(dummy.properties.keys())   
        for item in temp:
            props.remove(item)
        output_props = dummy.output_property_arrays
        for fluid in self.fluids:
            pa = particle_arrays[fluid]
            self._ensure_properties(pa, props, clean)
            pa.set_output_arrays(output_props)
            pa.add_property('m_mat', stride=9)
            pa.add_property('gradlmda', stride=3)
            pa.add_property('gradrho', stride=3)

        dummy = get_particle_array_DeltaPlus_solid(name='junk')
        props = list(dummy.properties.keys())
        for item in temp:
            props.remove(item)
        output_props = dummy.output_property_arrays
        for solid in self.solids:
            pa = particle_arrays[solid]
            self._ensure_properties(pa, props, clean)
            pa.set_output_arrays(output_props)
            pa.add_property('m_mat', stride=9)
            pa.add_property('gradlmda', stride=3)
            pa.add_property('gradrho', stride=3)