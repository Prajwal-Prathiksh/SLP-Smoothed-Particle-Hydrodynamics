from pysph.sph.scheme import Scheme


class DeltaPlusScheme(Scheme):
    def __init__(self, fluids, solids, dim,):
        """
        Parameters
        ----------

        fluids: list
            a list with names of fluid particle arrays
        solids: list
            a list with names of solid (or boundary) particle arrays
        dim: int
            dimensionality of the problem
        """
        self.fluids = fluids
        self.solids = solids
        self.dim = dim
        self.solver = None

    def get_equations(self):
        pass

    def setup_properties(self, particles, clean=True):
        pass

    def configure_solver(self, kernel=None, integrator_cls=None,
                         extra_steppers=None, **kw):
        """Configure the solver to be generated.

        Parameters
        ----------

        kernel : Kernel instance.
            Kernel to use, if none is passed a default one is used.
        integrator_cls : pysph.sph.integrator.Integrator
            Integrator class to use, use sensible default if none is
            passed.
        extra_steppers : dict
            Additional integration stepper instances as a dict.
        **kw : extra arguments
            Any additional keyword args are passed to the solver instance.
        """
        pass