
""" A base estimator for use with minimum message length. """

__all__ = ["Estimator", "log_kappa"]

import logging
import numpy as np

logger = logging.getLogger(__name__)


# TODO: nomenclature in Wallace varies between N and D for dimension size
def log_kappa(D):
    """
    Return an approximation of the logarithm of the lattice constant 
    :math:`\kappa_D` using the relationship:

    .. math::

        I_1(x) \approx - \log\frac{h(\theta')}{\sqrt{F(\theta')}} 
                       - \log{f(x|\theta')}
                       - (D/2)\log{2\pi}
                       + \frac{1}{2}\log{(\pi{}D)}
                       - 1

    The expected error on :math:`\kappa_D` is less than 0.1 nit.  See Sections
    5.2.12 and 3.3.4 of Wallace (2005) for more details.

    :param D:
        The number of dimensions.
    """

    return  - 0.5 * D * np.log(2 * np.pi) \
            + 0.5 * np.log(np.pi * D) \
            - 1


class Estimator(object):

    """
    A base abstract class for minimum message length estimators.

    :param data:
        The data that will be fit by the estimator.
    """

    def __init__(self, quantum=0.1, **kwargs):
        self._quantum = quantum
        return None


    def __repr__(self):
        """
        Return a summary representation of the Estimator.
        """
        # The precision to which we display the message length will be
        # determined from the quantum.
        precision = int(np.ceil(np.max([0, -np.log10(self.quantum)])))
        return  "<{name} with {D} dimensions, {N} data, "\
                "and message length {L:.{precision}f} nits>".format(
                    name=self.__class__.__name__, D=self.dimensions, 
                    N=len(self.data), L=self.message_length,
                    precision=precision)


    @property
    def data(self):
        """
        Return the data that will be fit by the estimator.
        """
        return self._data


    @property
    def quantum(self):
        """
        Return the quantum of the data (i.e., the precision on data).
        """
        return self._quantum


    @property
    def weights(self):
        """
        Return the weights for this estimator.
        """
        raise NotImplementedError(
            "weights are not implemented yet")
        return self._weights


    @property
    def message_length(self):
        """
        Return the total message length.
        """
        return (- self.log_prior \
                + 0.5 * self.log_fisher \
                - self.log_data \
                + log_kappa(self.dimensions))
    

    @property
    def dimensions(self):
        """
        Return the number of dimensions in this estimator.
        """
        return len(self.parameter_names)


    def _set_parameter_values(self):
        """
        Set the parameter values to the Estimator.
        """

        values = self.estimate_parameters()
        for parameter_name, value in zip(self.parameter_names, values):
            setattr(self, "_{}".format(parameter_name), value)
        return values
        

    @property
    def parameter_names(self):
        """
        Return the names of the parameters in this estimator.
        """
        raise NotImplementedError("the parameter_names property should "
                                  "be defined in the Estimator sub-class")


    @property
    def log_prior(self):
        """
        Return the logarithm of the prior density.
        """
        raise NotImplementedError("the log_prior property should "
                                  "be defined in the Estimator sub-class")


    @property
    def log_fisher(self):
        """
        Return the logarithm of the determinant of the Fisher matrix.
        """
        raise NotImplementedError("the log_fisher property should "
                                  "be defined in the Estimator sub-class")


    @property
    def log_data(self):
        """
        Return the logarithm of the data.
        """
        raise NotImplementedError("the log_data property should "
                                  "be defined in the Estimator sub-class")
