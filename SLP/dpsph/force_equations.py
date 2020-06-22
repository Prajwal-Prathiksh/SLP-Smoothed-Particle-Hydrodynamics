###########################################################################
# IMPORTS
###########################################################################

# PyPSH Equations Import
from pysph.sph.equation import Equation

################################################################################
# Fluid-Solid Coupling: Equations to evaluate the forces & torques on a solid
################################################################################

class NetForce_Solid(Equation):
    r"""*Equation to calculate the net global force on solids by the liquid*

        ..math::
            \mathbf{F}_{f-s} = \sum_i \Big[ \sum_j [-(p_i + p_j) + \mu\pi_{ij}]
            \nabla_i W_{ij}V_i V_j \Big]

        where, :math:`i \in` fluid, and :math:`j \in` solid

        References:
        ----------
        .. [Bouscasse2013] Bouscasse, B., et al. “Nonlinear Water Wave 
        Interaction with Floating Bodies in SPH.” Journal of Fluids and
        Structures, vol. 42, Oct. 2013, pp. 112–29. DOI.org (Crossref), 
        doi:10.1016/j.jfluidstructs.2013.05.010.

        Parameters:
        -----------
        dim : integer
            Number of dimensions

        mu : float
            Dynamic viscosity of the fluid (:math:`\mu = \rho_o \nu`)
    """
    def __init__(self, dest, sources, dim, mu):
        r"""
            Parameters:
            -----------
            dim : integer
                Number of dimensions

            mu : float
                Dynamic viscosity of the fluid (:math:`\mu = \rho_o \nu`)
        """
        self.dim = dim
        self.mu = mu

        super(NetForce_Solid, self).__init__(dest, sources)

    def initialize(self, d_idx, )