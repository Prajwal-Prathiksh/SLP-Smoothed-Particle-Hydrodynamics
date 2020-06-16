# PyPSH Equations Import
from pysph.sph.equation import Equation, Group

# Miscellaneous Import
from textwrap import dedent

#######################################
# Delta_Plus - SPH Sceheme
#######################################

class EOS_DeltaPlus_SPH(Equation):
    r"""*Equation of state defined by Delta_plus SPH*
        
        ..math::
            p_i = c_0^2(\rho_i - \rho_0)

        References:
        ----------
        .. [Sun2018b] Sun, Pengnan, et al. “An Accurate and Efficient SPH 
        Modeling of the Water Entry of Circular Cylinders.” Applied Ocean 
        Research, vol. 72, Mar. 2018, pp. 60–75. DOI.org (Crossref), 
        doi:10.1016/j.apor.2018.01.004.

        .. [Sun2018a] Sun, Peng-Nan, et al. “Numerical Simulation of the 
        Self-Propulsive Motion of a Fishlike Swimming Foil Using the δ + -SPH 
        Model.” Theoretical and Applied Mechanics Letters, vol. 8, no. 2, Mar. 
        2018, pp. 115–25. DOI.org (Crossref), doi:10.1016/j.taml.2018.02.007.
    """
    def __init__(self, dest, sources, rho0, c0):
        r'''
        Parameters:
        -----------
        rho0 : float
            Reference density of fluid (:math:`\rho_0`)
        c0 : float
            Maximum speed of sound expected in the system (:math:`c0`)
        '''
        self.rho0 = rho0
        self.c0 = c0
        self.c0_2 = c0*c0
        super(EOS_DeltaPlus_SPH, self).__init__(dest, sources)

    def loop(self, d_idx, d_rho, d_p):
        rhoi = d_rho[d_idx]

        # Equation of state
        d_p[d_idx] = self.c0_2 * (rhoi - self.rho0)

class ContinuityEquation_DeltaPlus_SPH(Equation):   
    r""" *Continuity equation with diffusive 
        terms defined by the Delta_plus SPH scheme*

        ..math::
            \frac{D\rho_i}{Dt} = \sum_j\rho_i\mathbf{u_{ij}}.\nabla_i W_{ij} V_j
            +\delta h c_o \Psi_{ij}\frac{\mathbf{r_{ji}}.\nabla_i W_{ij}}{|
            \mathbf{r_{ji}|^2}}V_j
        
        where,

        ..math::
            \Psi_{ij} = 2( \rho_j - \rho_i )

        References:
        -----------
        .. [Antuono2010] Antuono, M., et al. “Free-Surface Flows Solved by Means
        of SPH Schemes with Numerical Diffusive Terms.” Computer Physics 
        Communications, vol. 181, no. 3, Mar. 2010, pp. 532–49. DOI.org 
        (Crossref), doi:10.1016/j.cpc.2009.11.002.

        .. [Sun2018a] Sun, Peng-Nan, et al. “Numerical Simulation of the 
        Self-Propulsive Motion of a Fishlike Swimming Foil Using the δ + -SPH 
        Model.” Theoretical and Applied Mechanics Letters, vol. 8, no. 2, Mar. 
        2018, pp. 115–25. DOI.org (Crossref), doi:10.1016/j.taml.2018.02.007.
        """
    
    def __init__(self, dest, sources, delta, c0, H):
        r'''
        Parameters:
        -----------
        delta : float
            Density diffusion parameter (:math: `\delta`)
        c0 : float
            Maximum speed of sound expected in the system (:math:`c0`)
        H : float
            Kernel smoothing length (:math:`h`)
        '''

        self.delta = delta
        self.c0 = c0
        self.H = H

        # Calculate constant
        self.CONST = delta * H * c0

        super(ContinuityEquation_DeltaPlus_SPH, self).__init__(dest, sources)

    def initialize(self, d_idx, d_arho):
        d_arho[d_idx] = 0.0

    def loop(
        self, d_idx, s_idx, d_arho, d_rho, VIJ, DWIJ, s_m, s_rho, XIJ, R2IJ, 
        EPS
    ):
        
        rhoi = d_rho[d_idx]
        rhoj = s_rho[s_idx]

        # Volume element
        Vj = s_m[s_idx]/rhoj

        vijdotDWij = VIJ[0]*DWIJ[0] + VIJ[1]*DWIJ[1] + VIJ[2]*DWIJ[2]
        xjidotDWij = -1.0 * (XIJ[0]*DWIJ[0] + XIJ[1]*DWIJ[1] + XIJ[2]*DWIJ[2]) 
        # Multipled with -1 to convert XIJ to XJI


        # psi_ij
        psi_ij = 2 * (rhoj - rhoi)

        # Continuity density term
        tmp1 = rhoi * vijdotDWij

        # Dissipative diffusive term
        tmp2 = self.CONST * psi_ij * xjidotDWij / (R2IJ + EPS) 
        # NOTE: R2JI = R2IJ, since norm is symmertric

        d_arho[d_idx] +=  (tmp1 + tmp2) * Vj

class MomentumEquation_DeltaPlus_SPH(Equation):
    r""" *Momentum equation defined by the Delta_plus SPH scheme*

        ..math::
            \frac{D\mathbf{u_i}}{Dt}=\frac{1}{\rho_i}\sum_j\Big(F_{ij}\nabla_i 
            W_{ij}V_j+K\mu\pi_{ij}\nabla_i W_{ij}V_j\big)+\mathbf{f_i}

        where,

        ..math::
            F_{ij}=\begin{cases}
                -(p_j+p_i), & p_i\geq 0\\ 
                -(p_j-p_i), & p_i<0
                \end{cases}

        ..math::
            K = 2 (\text{dim} + 2)

        ..math::
            \pi_{ij}=\frac{\mathbf{u_{ji}}.\mathbf{r_{ji}}}{|\mathbf{r_{ji}}|^2}

        References:
        -----------
        .. [Bouscasse2013] Bouscasse, B., et al. “Nonlinear Water Wave 
        Interaction with Floating Bodies in SPH.” Journal of Fluids and
        Structures, vol. 42, Oct. 2013, pp. 112–29. DOI.org (Crossref), 
        doi:10.1016/j.jfluidstructs.2013.05.010.


        .. [Sun2018a] Sun, Peng-Nan, et al. “Numerical Simulation of the 
        Self-Propulsive Motion of a Fishlike Swimming Foil Using the δ + -SPH 
        Model.” Theoretical and Applied Mechanics Letters, vol. 8, no. 2, Mar. 
        2018, pp. 115–25. DOI.org (Crossref), doi:10.1016/j.taml.2018.02.007.
    """
    def __init__(self, dest, sources, dim, mu, fx=0.0, fy=0.0, fz=0.0):
        r"""
        Parameters:
        -----------
        dim : integer
            Number of dimensions
        mu : float
            Dynamic viscosity of the fluid (:math:`\mu = \rho_o \nu`)
        fx : float
            Body-force in x-axis
        fy : float
            Body-force in y-axis
        fz : float
            Body-force in z-axis
        """
        self.dim = dim
        self.mu = mu
        self.fx = fx
        self.fy = fy
        self.fz = fz

        # Calculate constant
        self.K = 2 * (dim + 2)

        self.CONST = self.K * mu

        super(MomentumEquation_DeltaPlus_SPH, self).__init__(dest, sources)

    def initialize(self, d_idx, d_au, d_av, d_aw):
        d_au[d_idx] = 0.0
        d_av[d_idx] = 0.0
        d_aw[d_idx] = 0.0

    def loop(
        self, d_idx, s_idx, d_rho, s_rho, DWIJ, s_m, VIJ, XIJ, R2IJ, d_au, d_av,
        d_aw, d_p, s_p, EPS
    ):
        
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
                
        rhoi = d_rho[d_idx]
        
        d_au[d_idx] = d_au[d_idx] / rhoi + self.fx
        d_av[d_idx] = d_av[d_idx] / rhoi + self.fy
        d_aw[d_idx] = d_aw[d_idx] / rhoi + self.fz

class RenormalizationTensor2D_DeltaPlus_SPH(Equation):
    r""" *Renormaliztion Tensor as defined by the Delta_plus SPH scheme for the 
        2D case*

        ..math::
            \mathbb{L}_{i}=\bigg[\sum_j\mathbf{r_{ji}}\otimes\nabla_i W_{ij}V_j\bigg]^{-1}

        ..math::
            \lambda_i=\text{min}\big(\text{eigenvalue}(\mathbb{L}_i^{-1})\big)

        References:
        -----------
        .. [Sun2017] Sun, P. N., et al. “The δ p l u s -SPH Model: Simple
        Procedures for a Further Improvement of the SPH Scheme.” Computer 
        Methods in Applied Mechanics and Engineering, vol. 315, Mar. 2017, pp. 
        25–49. DOI.org (Crossref), doi:10.1016/j.cma.2016.10.028.

        .. [Marrone2010] Marrone, S., et al. “Fast Free-Surface Detection and Level-
        Set Function Definition in SPH Solvers.” Journal of Computational Physics, 
        vol. 229, no. 10, May 2010, pp. 3652–63. DOI.org (Crossref), 
        doi:10.1016/j.jcp.2010.01.019.
    """
    def __init__(self, dest, sources, dim):
        r"""
        Parameters:
        -----------
        dim : integer
            Number of dimensions
        """
        if self.dim != 2:
            raise ValueError("Dimension must be 2!")
        else:
            self.dim = dim

        super(RenormalizationTensor2D_DeltaPlus_SPH, self).__init__(dest, sources)
    
    
    def _cython_code_(self):
        r"""
        Import eigen_decomposition function
        """

        code = dedent("""
        cimport cython
        from pysph.base.linalg3 cimport eigen_decomposition
        """)
        return code
    
    def initialize(self, d_idx, d_L00, d_L01, d_L10, d_L11, d_lambda):

        d_L00[d_idx] = 0.0
        d_L01[d_idx] = 0.0
        d_L10[d_idx] = 0.0
        d_L11[d_idx] = 0.0

        d_lambda = 0.0 # Initialize \lambda_i

    def loop(
        self, d_idx, s_idx, XIJ, DWIJ, s_m, s_rho, d_L00, d_L01, d_L10, d_L11
    ):
        r"""
        Computes the renormalization tensor
        
        Paramaters:
        -----------
        d_Lxx : DoubleArray
            Components of the renormalized tensor
        """

        rhoj = s_rho[s_idx]
        mj = s_m[s_idx]
        Vj = mj/rhoj

        # Tensor product
        ### (-1) multiplied because XJI = -1.0*XIJ
        a00 = -1.0*XIJ[0]*DWIJ[0]
        a01 = -1.0*XIJ[0]*DWIJ[1]
        a10 = -1.0*XIJ[1]*DWIJ[0]
        a11 = -1.0*XIJ[1]*DWIJ[1]

        # Sum Renormalization tensor
        d_L00 += a00*Vj
        d_L01 += a01*Vj
        d_L10 += a10*Vj
        d_L11 += a11*Vj

    def post_loop(self, d_idx, d_L00, d_L01, d_L10, d_L11, d_lambda, EPS):

        # Matrix of Eigenvectors (columns)
        eig_vect = declare('matrix((2,2))')

        # Eigenvalues
        eig_val = declare('matrix((2,))')

        # Renormalization tensor
        Li = declare('matrix((2,2))')

        # Initialize tensor
        Li[0][0] = d_L00[d_idx]
        Li[0][1] = d_L01[d_idx]
        Li[1][0] = d_L10[d_idx]
        Li[1][1] = d_L11[d_idx]
                
                
        # Calculate determinant
        Det = Li[0][0]*Li[1][1] - Li[0][1]*Li[1][0]
        Det = Det + EPS # Correction if determinant zero

        # Store the inverse of the tensor
        d_L00[d_idx] = Li[1][1]/Det
        d_L01[d_idx] = -1.0*Li[0][1]/Det
        d_L10[d_idx] = -1.0*Li[1][0]/Det
        d_L11[d_idx] = Li[0][0]/Det

        # Compute eigenvalues
        eigen_decomposition(Li, eig_vect, cython.address(eig_val[0]))

        # Store lambda_i with the smaller eigenvalue
        if eig_val[0] <= eig_val[1]:
            d_lambda[d_idx] = eig_val[0]
        else:
            d_lambda[d_idx] = eig_val[1]

class ContinuityEquation_RDGC_DeltaPlus_SPH(Equation):
    r""" *Continuity equation with diffusive terms and the renormalized density 
        gradient correction (RDGC) term  defined by the Delta_plus SPH scheme*

        ..math::
            \frac{D\rho_i}{Dt} = \sum_j\rho_i\mathbf{u_{ij}}.\nabla_i W_{ij}V_j
            +\delta h c_o \Psi_{ij}\frac{\mathbf{r_{ji}}.\nabla_i W_{ij}}{|
            \mathbf{r_{ji}|^2}}V_j

        where, 
        
        ..math::
            \Psi_{ij}=2(\rho_j-\rho_i)-(\langle\nabla\rho\rangle_i^L+\langle
            \nabla\rho\rangle_j L).\mathbf{r_{ji}}

        ..math::
            \langle\nabla\rho\rangle_i^L=\sum_j(\rho_j-\rho_i)\mathbb{L}_{i}.
            \nabla_i W_{ij}V_j

        References:
        -----------
        .. [Marrone2011] Marrone, S., et al. “δ-SPH Model for Simulating Violent
            Impact Flows.” Computer Methods in Applied Mechanics and 
            Engineering, vol. 200, no. 13–16, Mar. 2011, pp. 1526–42. DOI.org 
            (Crossref), doi:10.1016/j.cma.2010.12.016.

        .. [Bouscasse2013] Bouscasse, B., et al. “Nonlinear Water Wave 
            Interaction with Floating Bodies in SPH.” Journal of Fluids and 
            Structures, vol. 42, Oct. 2013, pp. 112–29. DOI.org (Crossref), 
            doi:10.1016/j.jfluidstructs.2013.05.010.
        """
    
    def __init__(self, dest, sources, delta, c0, H):
        r'''
        Parameters:
        -----------
        delta : float
            Density diffusion parameter (:math: `\delta`)
        c0 : float
            Maximum speed of sound expected in the system (:math:`c0`)
        H : float
            Kernel smoothing length (:math:`h`)
        '''

        self.delta = delta
        self.c0 = c0
        self.H = H

        # Calculate constant
        self.CONST = delta * H * c0

        super(ContinuityEquation_RDGC_DeltaPlus_SPH, self).__init__(dest, sources)

    def initialize(self, d_idx, d_arho):
        d_arho[d_idx] = 0.0

    def loop(
        self, d_idx, s_idx, d_arho, d_rho, VIJ, DWIJ, s_m, s_rho, XIJ, R2IJ, 
        EPS
    ):
        
        rhoi = d_rho[d_idx]
        rhoj = s_rho[s_idx]

        # Volume element
        Vj = s_m[s_idx]/rhoj

        vijdotDWij = VIJ[0]*DWIJ[0] + VIJ[1]*DWIJ[1] + VIJ[2]*DWIJ[2]
        xjidotDWij = -1.0 * (XIJ[0]*DWIJ[0] + XIJ[1]*DWIJ[1] + XIJ[2]*DWIJ[2]) 
        # Multipled with -1 to convert XIJ to XJI

        # psi_ij
        psi_ij = 2 * (rhoj - rhoi)

        # Continuity density term
        tmp1 = rhoi * vijdotDWij

        # Dissipative diffusive term
        tmp2 = self.CONST * psi_ij * xjidotDWij / (R2IJ + EPS) 
        # NOTE: R2JI = R2IJ, since norm is symmetric

        d_arho[d_idx] +=  (tmp1 + tmp2) * Vj

class ParticleShiftingTechnique(Equation):
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

        .. [Marrone2011] Marrone, S., et al. “δ-SPH Model for Simulating Violent
        Impact Flows.” Computer Methods in Applied Mechanics and Engineering, 
        vol. 200, no. 13–16, Mar. 2011, pp. 1526–42. DOI.org (Crossref), 
        doi:10.1016/j.cma.2010.12.016.
    """
    def __init__(self, dest, sources, R_coeff, n_exp, c0, H, dt):
        r'''
        Parameters:
        -----------
        R_coeff : float
            Artificial pressure coefficient
        n_exp : float
            Artificial pressure exponent
        c0 : float
            Maximum speed of sound expected in the system (:math:`c0`)
        H : float
            Kernel smoothing length (:math:`h`)
        dt : float
            Time step of integrator
        '''       
        self.R_coeff = R_coeff
        self.n_exp = n_exp
        self.c0 = c0
        self.H = H
        self.dt = dt

        self.CONST = -1.0*dt*c0*4*H*H

        super(ParticleShiftingTechnique, self).__init__(dest, sources)

    def initialize(self, d_idx, d_del_x, d_del_y):
        d_del_x[d_idx] = 0.0
        d_del_y[d_idx] = 0.0

    def loop(self, d_idx, s_idx, d_del_x, d_del_y, d_rho, s_rho, s_m, DWIJ, WIJ):

        #######################################
        #### Code is currently being written!!
        #######################################
        pass

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

    def initialize(self, d_idx, d_delta_p):
        d_delta_p[d_idx] = 0.0

    def loop_all(self, d_idx, d_delta_p, d_x, d_y, d_z, s_x, s_y, s_z, NBRS, N_NBRS):
        
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
        
        d_delta_p[d_idx] = sum/N_NBRS            
        #sum = sum/N_NBRS
        #d_delta_p[d_idx] += sum

class RDGC_DeltaPlus_SPH(Equation):
    r"""*The renormalized density gradient correction (RDGC) term  defined by 
        the Delta_plus SPH scheme*

        ..math::
            \langle\nabla\rho\rangle_i^L=\sum_j(\rho_j-\rho_i)\mathbb{L}_{i}.
            \nabla_i W_{ij}V_j

        References:
        -----------
        .. [Marrone2011] Marrone, S., et al. “δ-SPH Model for Simulating Violent
            Impact Flows.” Computer Methods in Applied Mechanics and 
            Engineering, vol. 200, no. 13–16, Mar. 2011, pp. 1526–42. DOI.org 
            (Crossref), doi:10.1016/j.cma.2010.12.016.
    """

    def __init__(self, dest, sources, dim):
        r"""
        Parameters:
        -----------
        dim : integer
            Number of dimensions
        """
        if self.dim != 2:
            raise ValueError("Dimension must be 2!")
        else:
            self.dim = dim

        super(RDGC_DeltaPlus_SPH, self).__init__(dest, sources)

    def initialize(self, d_idx, d_gradRho):
        d_gradRho[d_idx] = 0.0

    def loop(self, d_idx, s_idx, d_gradRho):
        pass