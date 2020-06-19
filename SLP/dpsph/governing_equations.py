###########################################################################
# IMPORTS
###########################################################################

# PyPSH Equations Import
from pysph.sph.equation import Equation

################################################################################
# Delta_Plus - SPH Sceheme: Governing equations
################################################################################

class EOS_DPSPH(Equation):
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

        Parameters:
        -----------
        rho0 : float
            Reference density of fluid (:math:`\rho_0`)

        c0 : float
            Maximum speed of sound expected in the system (:math:`c0`)
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
        super(EOS_DPSPH, self).__init__(dest, sources)

    def loop(self, d_idx, d_rho, d_p):
        rhoi = d_rho[d_idx]

        # Equation of state
        d_p[d_idx] = self.c0_2 * (rhoi - self.rho0)

class ContinuityEquation_DPSPH(Equation):   
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
        
        Parameters:
        -----------
        delta : float
            Density diffusion parameter (:math: `\delta`)

        c0 : float
            Maximum speed of sound expected in the system (:math:`c0`)

        H : float
            Kernel smoothing length (:math:`h`)

        dim : integer
            Number of dimensions
        """
    
    def __init__(self, dest, sources, delta, c0, H, dim):
        r'''
        Parameters:
        -----------
        delta : float
            Density diffusion parameter (:math: `\delta`)

        c0 : float
            Maximum speed of sound expected in the system (:math:`c0`)

        H : float
            Kernel smoothing length (:math:`h`)

        dim : integer
            Number of dimensions
        '''

        self.delta = delta
        self.c0 = c0
        self.H = H
        self.dim = dim

        # Calculate constant
        self.CONST = delta * H * c0

        super(ContinuityEquation_DPSPH, self).__init__(dest, sources)

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

class MomentumEquation_DPSPH(Equation):
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

        Parameters:
        -----------
        dim : integer
            Number of dimensions

        mu : float
            Dynamic viscosity of the fluid (:math:`\mu = \rho_o \nu`)

        fx : float, default = 0.0
            Body-force in x-axis

        fy : float, default = 0.0
            Body-force in y-axis

        fz : float, default = 0.0
            Body-force in z-axis
    """
    def __init__(self, dest, sources, dim, mu, fx=0.0, fy=0.0, fz=0.0):
        r"""
            Parameters:
            -----------
            dim : integer
                Number of dimensions

            mu : float
                Dynamic viscosity of the fluid (:math:`\mu = \rho_o \nu`)

            fx : float, Default = 0.0
                Body-force in x-axis

            fy : float, Default = 0.0
                Body-force in y-axis

            fz : float, Default = 0.0
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

        super(MomentumEquation_DPSPH, self).__init__(dest, sources)

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
            Fij = Pj - Pi
        else:
            Fij = (Pi + Pj)

        # pi_ij
        pi_ij  = vjidotxji / (R2IJ + EPS)

        tmp = (-1.0*Fij + self.CONST * pi_ij) * Vj 

        # Accelerations
        d_au[d_idx] += tmp*DWIJ[0]
        d_av[d_idx] += tmp*DWIJ[1]
        d_aw[d_idx] += tmp*DWIJ[2]

    def post_loop(self, d_idx, d_au, d_av, d_aw, d_rho):
                
        rhoi = d_rho[d_idx]
        
        d_au[d_idx] = d_au[d_idx] / rhoi + self.fx
        d_av[d_idx] = d_av[d_idx] / rhoi + self.fy
        d_aw[d_idx] = d_aw[d_idx] / rhoi + self.fz

class ContinuityEquation_RDGC_DPSPH(Equation):
    r"""*Continuity equation with diffusive terms and the renormalized density 
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

        Parameters:
        -----------
        delta : float
            Density diffusion parameter (:math: `\delta`)

        c0 : float
            Maximum speed of sound expected in the system (:math:`c0`)

        H : float
            Kernel smoothing length (:math:`h`)

        dim : integer
            Number of dimensions
        """
    
    def __init__(self, dest, sources, delta, c0, H, dim):
        r'''
        Parameters:
        -----------
        delta : float
            Density diffusion parameter (:math: `\delta`)

        c0 : float
            Maximum speed of sound expected in the system (:math:`c0`)

        H : float
            Kernel smoothing length (:math:`h`)
            
        dim : integer
            Number of dimensions
        '''
        
        if dim != 2:
            raise ValueError("ContinuityEquation_RDGC_DPSPH - Dimension must be 2!")
        else:
            self.dim = dim

        self.delta = delta
        self.c0 = c0
        self.H = H

        # Calculate constant
        self.CONST = delta * H * c0

        super(ContinuityEquation_RDGC_DPSPH, self).__init__(dest, sources)

    def initialize(self, d_idx, d_arho):
        d_arho[d_idx] = 0.0

    def loop(
        self, d_idx, s_idx, d_arho, d_rho, VIJ, DWIJ, s_m, s_rho, XIJ, R2IJ, EPS,
        d_grad_rho1, d_grad_rho2, s_grad_rho1, s_grad_rho2
    ):
        
        rhoi = d_rho[d_idx]
        rhoj = s_rho[s_idx]

        # Volume element
        Vj = s_m[s_idx]/rhoj

        vijdotDWij = VIJ[0]*DWIJ[0] + VIJ[1]*DWIJ[1] + VIJ[2]*DWIJ[2]
        xjidotDWij = -1.0 * (XIJ[0]*DWIJ[0] + XIJ[1]*DWIJ[1] + XIJ[2]*DWIJ[2]) 
        # Multipled with -1 to convert XIJ to XJI

        # RDGC Term
        rdgcTerm = (d_grad_rho1[d_idx] + s_grad_rho1[s_idx])*XIJ[0] + ( d_grad_rho2[d_idx] + s_grad_rho2[s_idx] )*XIJ[1]
                
        # psi_ij
        psi_ij = 2 * (rhoj - rhoi) + rdgcTerm # Since XJI = -XIJ

        # Continuity density term
        tmp1 = rhoi * vijdotDWij

        # Dissipative diffusive term
        tmp2 = self.CONST * psi_ij * xjidotDWij / (R2IJ + EPS) 
        # NOTE: R2JI = R2IJ, since norm is symmetric

        d_arho[d_idx] +=  (tmp1 + tmp2) * Vj