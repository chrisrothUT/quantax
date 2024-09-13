from typing import Union, Optional, Tuple
from numbers import Number
import jax
import jax.numpy as jnp
import jax.tree_util as jtu
from ..utils import to_global_array


@jtu.register_pytree_node_class
class SamplerStatus:
    """
    The status of the sampler. This class is jittable, and there are 4 attributes.

    spins:
        The current spin configurations stored in the sampler

    wave_function:
        The wave_function of the current spin configurations

    prob:
        The probability of the current spin configurations

    propose_prob:
        The probability of proposing the current spin configuration
    """

    def __init__(
        self,
        spins: Optional[jax.Array] = None,
        wave_function: Optional[jax.Array] = None,
        prob: Optional[jax.Array] = None,
        propose_prob: Optional[jax.Array] = None,
    ):
        self.spins = spins
        self.wave_function = wave_function
        self.prob = prob
        self.propose_prob = propose_prob

    def tree_flatten(self) -> Tuple:
        children = (self.spins, self.wave_function, self.prob, self.propose_prob)
        aux_data = None
        return (children, aux_data)

    @classmethod
    def tree_unflatten(cls, aux_data, children):
        return cls(*children)


@jtu.register_pytree_node_class
class Samples:
    r"""
    The samples generated by the sampler.
    This class is jittable, and there are 3 attributes.

    spins:
        The spin configurations

    wave_function:
        The wave_function of the spin configurations

    reweight_factor:
        According to

        .. math::

            \left< x \right>_p = \frac{\sum_s p_s x_s}{\sum_s p_s}
            = \frac{\sum_s q_s x_s p_s/q_s}{\sum_s q_s p_s/q_s}
            = \frac{\left< x p/q \right>_q}{\left< p/q \right>_q},

        the expectation value with probability distribution p can be computed from
        samples with a different probability distribution q.

        The reweighting factor is defined as

        .. math::
            r_s = \frac{p_s/q_s}{\left< p/q \right>_q},

        so that :math:`\left< x \right>_p = \left< r x \right>_q`.

        Usually, :math:`p_s = |\psi(s)|^2` is the target probability,
        and :math:`q_s` can be chosen as :math:`|\psi(s)|^n` or computed from a
        helper neural network. In the former case,
        :math:`r_s = \frac{|\psi_s|^{2-n}}{\left< |\psi|^{2-n} \right>}`
    """

    def __init__(
        self,
        spins: jax.Array,
        wave_function: jax.Array,
        reweight: Union[float, jax.Array] = 2.0,
    ):
        """
        :param spins:
            The spin configurations

        :param wave_function:
            The wave_function of the spin configurations

        :param reweight:
            Either a number :math:`n` specifying the reweighting probability :math:`|\psi(s)|^n`,
            or the unnormalized reweighting factor :math:`r'_s = p_s/q_s`
        """
        self.spins = to_global_array(spins)
        self.wave_function = to_global_array(wave_function)
        if isinstance(reweight, Number) or reweight.size == 1:
            reweight_factor = jnp.abs(self.wave_function) ** (2 - reweight)
            self.reweight_factor = reweight_factor / jnp.mean(reweight_factor)
        else:
            self.reweight_factor = to_global_array(reweight)

    @property
    def nsamples(self) -> int:
        return self.spins.shape[0]

    def tree_flatten(self) -> Tuple:
        children = (self.spins, self.wave_function, self.reweight_factor)
        aux_data = None
        return (children, aux_data)

    @classmethod
    def tree_unflatten(cls, aux_data, children):
        return cls(*children)

    def __getitem__(self, idx):
        return Samples(
            self.spins[idx], self.wave_function[idx], self.reweight_factor[idx]
        )
