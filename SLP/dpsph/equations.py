###########################################################################
# IMPORTS
###########################################################################

# PyPSH Equations Import
from pysph.sph.equation import Equation

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
    r"""*Particle-Shifting Technique employed in
        Delta_plus SPH scheme*

        ..math::
            \mathbf{r_i}^\ast=\mathbf{r_i}+\delta\mathbf{\hat{r_i}}
        
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

        dim : integer
            Number of dimensions

        cfl : float, default = 0.5
            CFL value

        Uc0 : float. default = 15.0
            :math:`\frac{U}{c_o}` value of the system

        R_coeff : float, default = 0.2
            Artificial pressure coefficient

        n_exp : float, default = 4
            Artificial pressure exponent

        Rh : float, default = 0.075
            Maximum :math:`\frac{|\delta r_i|}{h}` value allowed during Particle
            shifting 
            (Note: :math:`\delta r_i = 0` if :math:`\frac{|\delta r_i|}{h}>R_h`)
    """
    def __init__(self, dest, sources, dim):
        r'''
            Parameters:
            -----------
            dim : integer
                Number of dimensions

        '''       
            
        self.dim = dim
        super(PST_PreStep_1, self).__init__(dest, sources)

    def _cython_code_(self):
        code = dedent("""
        cimport cython
        from pysph.base.linalg3 cimport eigen_decomposition
        """)
        return code

    def initialize(self, d_idx, d_lmda):
        d_lmda[d_idx] = 0.0 # Initialize \lambda1

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

        for i in range(n):
            for j in range(n):
                L[i][j] = d_m_mat[9 * d_idx + 3 * i + j]

        if n == 2:
            L[2][2] = 99999.0

        # compute the principle stresses
        eigen_decomposition(L, R, cython.address(V[0]))

        lmda = V[0]
        for i in range(1, n):
            if V[i] < lmda:
                lmda = V[i]

        d_lmda[d_idx] = lmda


class PST_PreStep_2(Equation):
    def __init__(self, dest, sources, dim, H):
        r'''
            Parameters:
            -----------
            dim : integer
                Number of dimensions

        '''       
            
        self.dim = dim
        self.H = H
        super(PST_PreStep_2, self).__init__(dest, sources)

    def initialize(self, d_idx, d_gradlmda):
        d_gradlmda[d_idx*3 + 0] = 0.0
        d_gradlmda[d_idx*3 + 1] = 0.0
        d_gradlmda[d_idx*3 + 2] = 0.0
    def loop_all(
        self, d_idx, d_x, d_y, d_z, d_lmda, d_gradlmda, s_x, s_y, s_z, s_rho, 
        s_m, s_lmda, SPH_KERNEL, NBRS, N_NBRS
    ):
        n, i, j = declare('int', 3)
        n = self.dim
        s_idx = declare('long')
        xij = declare('matrix(3)')
        dwij = declare('matrix(3)')

        lmdai = d_lmda[d_idx]
        for i in range(N_NBRS):
            s_idx = NBRS[i]

            Vj = s_m[s_idx]/s_rho[s_idx]
            lmdaj = s_lmda[s_idx]

            lmdaij = (lmdai - lmdaj)
            xij[0] = d_x[d_idx] - s_x[s_idx]
            xij[1] = d_y[d_idx] - s_y[s_idx]
            xij[2] = d_z[d_idx] - s_z[s_idx]
            rij = sqrt(xij[0]*xij[0] + xij[1]*xij[1] + xij[2]*xij[2])

            # Calculate Kernel gradient value
            SPH_KERNEL.gradient(xij, rij, self.H, dwij)

            # Grad Lambda value
            for j in range(n):
                d_gradlmda[d_idx*3 + j] += lmdaij * dwij[j] * Vj

class PST(Equation):
    r"""*Particle-Shifting Technique employed in
        Delta_plus SPH scheme*

        ..math::
            \mathbf{r_i}^\ast=\mathbf{r_i}+\delta\mathbf{\hat{r_i}}
        
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

        dim : integer
            Number of dimensions

        cfl : float, default = 0.5
            CFL value

        Uc0 : float. default = 15.0
            :math:`\frac{U}{c_o}` value of the system

        R_coeff : float, default = 0.2
            Artificial pressure coefficient

        n_exp : float, default = 4
            Artificial pressure exponent

        Rh : float, default = 0.075
            Maximum :math:`\frac{|\delta r_i|}{h}` value allowed during Particle
            shifting 
            (Note: :math:`\delta r_i = 0` if :math:`\frac{|\delta r_i|}{h}>R_h`)
    """
    def __init__(
        self, dest, sources, H, dim, cfl=0.5, Uc0=15.0, R_coeff=0.2, n_exp=4.0,
        Rh=0.115, saveAllDRh = False
    ):
        r'''
            Parameters:
            -----------
            H : float
                Kernel smoothing length (:math:`h`)

            dim : integer
                Number of dimensions

            cfl : float, default = 0.5
                CFL value

            Uc0 : float. default = 15.0
                :math:`\frac{U}{c_o}` value of the system

            R_coeff : float, default = 0.2
                Artificial pressure coefficient

            n_exp : float, default = 4
                Artificial pressure exponent

            Rh : float, default = 0.115
                Maximum :math:`\frac{|\delta r_i|}{h}` value allowed during Particle
                shifting 
                (Note: :math:`\delta r_i = 0` if :math:`\frac{|\delta r_i|}{h}>R_h`)
        '''       
        
        self.dim = dim
        self.R_coeff = R_coeff
        self.n_exp = n_exp
        self.H = H
        self.cfl = cfl
        self.Uc0 = Uc0
        self.Rh = Rh
        self.saveAllDRh = saveAllDRh

        self.CONST = (-self.cfl/self.Uc0)*4.0*H*H

        super(PST, self).__init__(dest, sources)

    def initialize(self, d_idx, d_DX, d_DY, d_DZ, d_DRh):
        d_DX[d_idx] = 0.0
        d_DY[d_idx] = 0.0
        d_DZ[d_idx] = 0.0
        d_DRh[d_idx] = 0.0

    def loop_all(
        self, d_idx, d_x, d_y, d_z, d_rho, d_delta_s, d_DX, d_DY, d_DZ, d_DRh, 
        d_lmda, d_gradlmda, s_x, s_y, s_z, s_rho, s_m, SPH_KERNEL, NBRS, 
        N_NBRS, EPS
    ):
        n, i, j, k = declare('int', 4)
        n = self.dim
        s_idx = declare('long')
        xij = declare('matrix(3)')
        dwij = declare('matrix(3)')
        ni = declare('matrix(3)')
        deltaR = declare('matrix(3)')
        res = declare('matrix(3)')
        gradlmda_i = declare('matrix(3)')
        M = declare('matrix(3,3)')

        for j in range(3):
            deltaR[j] = 0.0
            gradlmda_i[j] = d_gradlmda[d_idx*3 + j]

        rhoi = d_rho[d_idx]
        lmdai = d_lmda[d_idx]

        if lmdai > 0.4:
            ##################
            # Case - 2 & 3
            ##################

            # Calculate W(\delta s) value
            delta_s = d_delta_s[d_idx]
            w_delta_s = SPH_KERNEL.kernel(xij, delta_s, self.H) 

            for i in range(N_NBRS):
                s_idx = NBRS[i]

                rhoj = s_rho[s_idx]
                mj = s_m[s_idx]

                xij[0] = d_x[d_idx] - s_x[s_idx]
                xij[1] = d_y[d_idx] - s_y[s_idx]
                xij[2] = d_z[d_idx] - s_z[s_idx]
                rij = sqrt(xij[0]*xij[0] + xij[1]*xij[1] + xij[2]*xij[2])

                # Calculate Kernel values
                wij = SPH_KERNEL.kernel(xij, rij, self.H)
                SPH_KERNEL.gradient(xij, rij, self.H, dwij)

                ################################################################
                # Calculate \delta r_i
                ################################################################

                # Calcuate fij
                fij = self.R_coeff * pow((wij/(EPS+w_delta_s)), self.n_exp)

                # Calcuate multiplicative factor
                fac = (1.0 + fij)*(mj/(rhoi+rhoj+EPS))

                # Sum \delta r_i
                for j in range(n):
                    deltaR[j] += self.CONST*fac*dwij[j]

            if lmdai > 0.75:
                ##################
                # Case - 3
                ##################
                rh = sqrt(deltaR[0]*deltaR[0] + deltaR[1]*deltaR[1] + deltaR[2]*deltaR[2])/self.H

                if rh > self.Rh:
                    # Check Rh condition
                    #d_DX[d_idx] = 0.0
                    #d_DY[d_idx] = 0.0
                    #d_DZ[d_idx] = 0.0
                    if self.saveAllDRh == True:    
                        d_DRh[d_idx] = rh
                else:
                    d_DX[d_idx] = deltaR[0]
                    d_DY[d_idx] = deltaR[1]
                    d_DZ[d_idx] = deltaR[2]
                    d_DRh[d_idx] = rh
            
            elif lmdai <= 0.75:
                ##################
                # Case - 2
                ##################

                ni_norm = sqrt(gradlmda_i[0]*gradlmda_i[0] + gradlmda_i[1]*gradlmda_i[1] + gradlmda_i[2]*gradlmda_i[2])

                for j in range(n):
                    ni[j] = gradlmda_i[j] / (ni_norm + EPS)
                    res[j] = 0.0

                for j in range(n):
                    for k in range(n):
                        M[j][k] = - ni[j]*ni[k]
                        if j == k:
                            M[j][k] += 1.0


                for j in range(n):
                    for k in range(n):
                        res[j] += M[j][k] * deltaR[k]

                for j in range(n):
                    deltaR[j] = res[j]

                rh = sqrt(deltaR[0]*deltaR[0] + deltaR[1]*deltaR[1] + deltaR[2]*deltaR[2])/self.H

                if rh > self.Rh:
                    # Check Rh condition
                    #d_DX[d_idx] = 0.0
                    #d_DY[d_idx] = 0.0
                    #d_DZ[d_idx] = 0.0
                    if self.saveAllDRh == True:    
                        d_DRh[d_idx] = ni_norm#rh
                else:
                    d_DX[d_idx] = deltaR[0]
                    d_DY[d_idx] = deltaR[1]
                    d_DZ[d_idx] = deltaR[2]
                    d_DRh[d_idx] = rh

        else:
            ##################
            # Case - 1
            ##################

            d_DX[d_idx] = 0.0
            d_DY[d_idx] = 0.0
            d_DZ[d_idx] = 0.0
            d_DRh[d_idx] = 0.0