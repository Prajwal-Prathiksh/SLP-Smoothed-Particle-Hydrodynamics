###########################################################################
# IMPORTS
###########################################################################

# PySPH base imports
from pysph.base.utils import get_particle_array

###########################################################################
# Delta_Plus - SPH Sceheme: Particle array
###########################################################################

def get_particle_array_dpsph(constants=None, **props):
    """Return a particle array for the Delta_Plus - SPH formulation.

        This sets the defualt properties to be::

            ['x', 'y', 'z', 'u', 'v', 'w', 'rho', 'p', 'm', 'h', 'L00', 'L01', 
            'L10', 'L11', 'lmda', 'delta_p', 'grad_rho1', 'grad_rho2', 
            'DX', 'DY', 'DRh', 'arho', 'au', 'av', 'aw', 'gid', 'pid', 'tag']

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

    dpsph_props = [
        'rho0', 'x0', 'y0', 'z0', 'lmda', 'DX', 'DY', 'DZ', 'DRh', 'vmag', 
        'vmag2', 'arho'
    ]

    pa = get_particle_array(
        constants=constants, additional_props=dpsph_props, **props
    )

    pa.add_property('m_mat', stride=9)
    pa.add_property('gradrho', stride=3)
    pa.add_property('gradlmda', stride=3)

    # default property arrays to save out.
    pa.set_output_arrays([
        'x', 'y', 'z', 'u', 'v', 'w', 'rho', 'p', 'h', 'm', 'DRh', 'vmag', 
        'vmag2', 'lmda', 'pid', 'gid', 'tag', 
    ])

    return pa

def get_particle_array_RK4(constants=None, **props):

    rk4_props = [
        'rho',
        'ax', 'ay', 'au', 'av', 'arho',
        'x0', 'y0', 'u0', 'v0', 'rho0',
        'xstar', 'ystar', 'ustar', 'vstar', 'rhostar',
        'vmag', 'vmag2',
    ]

    pa = get_particle_array(
        constants=constants, additional_props=rk4_props, **props
    )

    # default property arrays to save out.
    pa.set_output_arrays([
        'x', 'y','z', 'u', 'v', 'w', 'rho', 'vmag', 'vmag2', 
        'pid', 'gid', 'tag', 
    ])

    return pa