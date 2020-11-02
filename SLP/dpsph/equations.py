###########################################################################
# IMPORTS
###########################################################################

# PyPSH Equations Import
from pysph.sph.equation import Equation

from pysph.base.reduce_array import parallel_reduce_array, serial_reduce_array

# Miscellaneous Import
from textwrap import dedent
from compyle.api import declare

# Math import
from math import sqrt, pow

################################################################################
# Particle Shifting Technique
################################################################################

class AverageSpacing(Equation):
    r"""*Average particle spacing in the neighbourhood of the :math:`i^{th}` 
        particle*

        ..math::
            \Delta p_i = \frac{1}{N_i}\sum_j |\mathbf{r}_{ij}|

        where, :math:`N_i = ` number of neighbours of the :math:`i^{th}` particle

        References:
        -----------
        .. [Monaghan2000] Monaghan, J. J. “SPH without a Tensile Instability.” 
        Journal of Computational Physics, vol. 159, no. 2, Apr. 2000, pp. 
        290–311. DOI.org (Crossref), doi:10.1006/jcph.2000.6439.

        Parameters:
        -----------
        dim : integer
            Number of dimensions
    """
    def __init__(self, dest, sources, dim):
        r"""
        Parameters:
        -----------
        dim : integer
            Number of dimensions
        """
        self.dim = dim
        super(AverageSpacing, self).__init__(dest, sources)

    def initialize(self, d_idx, d_delta_s):
        d_delta_s[d_idx] = 0.0

    def loop_all(
        self, d_idx, d_delta_s, d_x, d_y, d_z, s_x, s_y, s_z, NBRS, N_NBRS
    ):
        i = declare('int')
        s_idx = declare('long')
        xij = declare('matrix(3)')
        rij = 0.0
        sum = 0.0

        for i in range(N_NBRS):
            s_idx = NBRS[i]
            xij[0] = d_x[d_idx] - s_x[s_idx]
            xij[1] = d_y[d_idx] - s_y[s_idx]
            xij[2] = d_z[d_idx] - s_z[s_idx]
            rij = sqrt(xij[0]*xij[0] + xij[1]*xij[1] + xij[2]*xij[2])
            
            sum += rij
    
        d_delta_s[d_idx] += sum/N_NBRS

class PST_PreStep_1(Equation):
    r"""**PST_PreStep_1**

        Parameters:
        -----------
        dim : integer
            Number of dimensions

        boundedFlow : boolean
            If True, flow has free surface/s
    """
    def __init__(self, dest, sources, dim, boundedFlow):
        r'''
            Parameters:
            -----------
            dim : integer
                Number of dimensions

            boundedFlow : boolean, default = False
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
        multiples the matrix :math:`L_a` with :math:`\nabla W_{ij}`

        Parameters:
        -----------
        H : float
            Kernel smoothing length (:math:`h`)

        dim : integer
            Number of dimensions

        boundedFlow : boolean
            If True, flow has free surface/s

        Notes:
        ------ 
        This equation needs to be grouped in the same group as 
        `GradientCorrection` and `PST_PreStep_1`, since it needs to be 
        pre-multiplied with :math:`L_a` and requires the pre-calculated
        :math:`\lambda_i` value
        """
    def __init__(self, dest, sources, H, dim, boundedFlow):
        r'''
            Parameters:
            -----------
            H : float
                Kernel smoothing length (:math:`h`)

            dim : integer
                Number of dimensions

            boundedFlow : boolean
                If True, flow has free surface/s
        '''       
            
        self.H = H
        self.dim = dim
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
        Rh=0.05,
    ):
        r'''
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

            R_coeff : float, default = 0.2
                Artificial pressure coefficient

            n_exp : float, default = 4
                Artificial pressure exponent

            Rh : float, default = 0.05
                Maximum :math:`\frac{|\delta r_i|}{h}` value allowed during 
                particle shifting 
                (Note: :math:`\delta r_i = 0` if :math:`\frac{|\delta r_i|}{h}>R_h`)

            boundedFlow : boolean, default = False
                If True, flow has free surface/s
        '''       
        
        self.R_coeff = R_coeff
        self.n_exp = n_exp
        self.H = H
        self.dx = dx
        self.Uc0 = Uc0
        self.Rh = Rh
        self.boundedFlow = boundedFlow
        self.eps = H**2 * 0.01
        self.CONST = -0.5*dt*H 

        super(PST, self).__init__(dest, sources)

    def initialize(self, d_idx, d_DX, d_DY, d_DZ, d_DRh):
        d_DX[d_idx] = 0.0
        d_DY[d_idx] = 0.0
        d_DZ[d_idx] = 0.0
        d_DRh[d_idx] = 0.0

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

    def post_loop(self, d_idx, d_lmda, d_DX, d_DY, d_DZ, d_DRh):

        lmdai = d_lmda[d_idx]
        if self.boundedFlow == True or lmdai > 0.75:

            rh = sqrt(d_DX[d_idx]*d_DX[d_idx] + d_DY[d_idx]*d_DY[d_idx] + d_DZ[d_idx]*d_DZ[d_idx]) / self.H
            d_DRh[d_idx] = rh

            if rh > self.Rh:
                # Check Rh condition and correct the values
                d_DX[d_idx] = d_DX[d_idx] * self.Rh / rh
                d_DY[d_idx] = d_DY[d_idx] * self.Rh / rh
                d_DZ[d_idx] = d_DZ[d_idx] * self.Rh / rh
                d_DRh[d_idx] = self.Rh

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