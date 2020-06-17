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
            'd_x', 'd_y', 'arho', 'au', 'av', 'aw', 'gid', 'pid', 'tag']

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
        'arho', 'L00', 'L01', 'L10', 'L11', 'lmda', 'delta_p', 'grad_rho1', 
        'grad_rho2', 'd_x', 'd_y'
    ]

    pa = get_particle_array(
        constants=constants, additional_props=dpsph_props, **props
    )

    # default property arrays to save out.
    pa.set_output_arrays([
        'x', 'y', 'z', 'u', 'v', 'w', 'rho', 'p', 'm', 'h', 'lmda', 'delta_p',
        'd_x', 'd_y', 'pid', 'gid'
    ])

    return pa