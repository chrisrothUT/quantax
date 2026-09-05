from typing import Optional
from jaxtyping import Key
import jax
import jax.numpy as jnp
import equinox as eqx
from .modules import NoGradLayer, RawInputLayer
from ..symmetry import Symmetry, TransND, Identity
from ..global_defs import get_lattice, get_subkeys, get_sites
from ..utils import _triangularb_circularpad
from ..sites import TriangularB, SquareB
from jax import random as jr
import math
from typing import Any, Callable
import equinox as eqx
import numpy as np
from jax import lax


class ReshapeConv(NoGradLayer):
    """
    Reshape the input to the shape suitable for convolutional layers.

    A fock state in Quantax is usually givne by a 1D array with entries +1/-1.
    This layer reshape it to `~quantax.sites.Lattice.shape`.
    """

    dtype: jnp.dtype = eqx.field(static=True)

    def __init__(self, dtype: jnp.dtype = jnp.float32):
        """
        :param dtype:
            Convert the input to the given data type, by default ``float32``.
        """
        super().__init__()
        self.dtype = dtype

    def __call__(self, x: jax.Array, *, key: Optional[Key] = None) -> jax.Array:
        lattice = get_lattice()
        shape = lattice.shape
        if lattice.is_fermion:
            shape = (shape[0] * 2,) + shape[1:]
        x = x.reshape(shape)
        x = x.astype(self.dtype)
        return x


class ConvSymmetrize(NoGradLayer, RawInputLayer):
    """
    Symmetrize the output of a convolutional network according to the given symmetry.
    """

    symm: Symmetry = eqx.field(static=True)

    def __init__(self, symm: Optional[Symmetry] = None):
        """
        :param symm:
            The symmetry used for symmetrization, by default
            `~quantax.symmetry.TransND` with sectors 0.
            If `~quantax.symmetry.Identity` is given, the layer won't symmetrize its
            output.
        """
        super().__init__()
        if symm is None:
            symm = TransND()
        self.symm = symm

    def __call__(self, x: jax.Array, s: jax.Array) -> jax.Array:
        if self.symm is Identity():
            return x

        x = x.reshape(-1, self.symm.nsymm).mean(axis=0)
        x = self.symm.symmetrize(x, s)

        return x


class SymmetryBreakingLayerEmbedding(eqx.Module):

    jast_inds: jax.Array
    reverse: jax.Array
    sub_inds: jax.Array
    W: jax.Array
    pow_two_array: jax.Array
    dtype: jnp.dtype

    def __init__(self, jast_inds, sub_inds, dtype = jnp.float64):
        
        self.jast_inds = jast_inds
        self.reverse = jnp.argsort(jast_inds)
        self.sub_inds = sub_inds
        self.dtype = dtype

        embedding_size = jnp.power(2,2*len(sub_inds))

        self.pow_two_array = jnp.power(2,2*len(sub_inds))[None,None]

        self.W = jr.normal(get_subkeys(), (embedding_size,2*len(sub_inds)), dtype=dtype)

    def __call__(self,x):
        x = x.reshape(2,-1)
        x = x[:,self.jast_inds].reshape(2,-1,len(self.sub_inds)).transpose(1,0,2).reshape(-1,2*len(self.sub_inds))

        inds = jnp.sum(jax.nn.relu(x)*self.pow_two_array,-1)

        x = self.W[inds]

        x = x.reshape(-1,2,len(self.sub_inds)).transpose(1,0,2)

        x = x.reshape(2,-1)[:,self.reverse]

        return x.ravel()

class SymmetryBreakingLayer(eqx.Module):

    jast_inds: jax.Array
    sub_inds: jax.Array
    reverse: jax.Array
    W: jax.Array
    dtype: jnp.dtype

    def __init__(self, jast_inds, sub_inds, features, dtype = jnp.float64):
        
        self.jast_inds = jast_inds
        self.sub_inds = sub_inds

        self.reverse = jnp.argsort(jast_inds)
        self.dtype = dtype
        mat_size = len(sub_inds)*features
        
        self.W = jr.normal(get_subkeys(), (mat_size,mat_size), dtype=dtype)

    def __call__(self,x):
        return jax.vmap(self.fwd, in_axes=1,out_axes=1)(x)

    def fwd(self,x):

        #features point group spin symm
        N = get_sites().N

        x = x.reshape(-1,2,N)

        x = x[:,:,self.jast_inds]
        x = x.reshape(x.shape[0],x.shape[1],-1,len(self.sub_inds)).transpose(0,1,3,2)

        features, n_spins, n_broken, _ = x.shape  

        mat_size = features*n_spins*n_broken

        x = x.reshape(mat_size,-1)

        x = self.W @ x  / mat_size**0.5

        x = x.reshape(features,n_spins,n_broken,-1).transpose(0,1,3,2)

        x = x.reshape(features,n_spins,-1)

        return x[:,:,self.reverse]
  


class Gconv(eqx.Module):

    weight: jax.Array
    idxarray: jax.Array

    def __init__(self, out_features, in_features, idxarray, npoint, layer0, key, spin_parity, dtype: jnp.dtype = jnp.float32):

        if layer0 == True:
            if spin_parity == 1 or spin_parity == -1:
                npoint = 2
            else:
                npoint = 1
            
            in_features = 2*in_features//npoint

            nelems = npoint*idxarray.shape[-1]
            idxarray = idxarray[:,:npoint] % nelems
            scale = (1/(in_features*nelems))**0.5
        else:
            nelems = npoint*idxarray.shape[-1]
            scale = (2/(in_features*nelems))**0.5

        self.weight = jax.random.normal(key, [out_features,in_features,nelems],dtype=dtype)*jnp.asarray([scale],dtype=dtype)

        self.idxarray = idxarray 

        super().__init__() 

    def __call__(self,x):
        
        lattice = get_lattice()

        x = x.reshape(1,-1,*lattice.shape[1:])

        weight = self.weight[...,self.idxarray]

        if weight.shape[-1] == 9:
            weight = weight.reshape(*weight.shape[:-1],3,3)
            if isinstance(lattice, SquareB):
                x = jax.vmap(_triangularb_circularpad)(x)
            else:
                x = jnp.concatenate((x[:,:,-1:],x,x[:,:,:1]),axis=-2)
                x = jnp.concatenate((x[:,:,:,-1:],x,x[:,:,:,:1]),axis=-1)
        elif weight.shape[-1] == 15:
            weight = weight.reshape(*weight.shape[:-1],5,3)
            x = jnp.concatenate((x[:,:,-2:],x,x[:,:,:2]),axis=-2)
            x = jnp.concatenate((x[:,:,:,-1:],x,x[:,:,:,:1]),axis=-1)
        elif weight.shape[-1] == 7:
            zeros = jnp.zeros_like(weight[...,:1])
            weight = jnp.concatenate((zeros,weight,zeros),-1)
            weight = weight.reshape(*weight.shape[:-1],3,3)
            x = jax.vmap(_triangularb_circularpad)(x)
        elif weight.shape[-1] == 18:
            weight = weight.reshape(*weight.shape[:-1],2,3,3)
            x = jnp.concatenate((x[:,:,-1:],x),axis=-3)
            x = jnp.concatenate((x[:,:,:,-1:],x,x[:,:,:,:1]),axis=-2)
            x = jnp.concatenate((x[:,:,:,:,-1:],x,x[:,:,:,:,:1]),axis=-1)

        if lattice.ndim == 2:
            weight = weight.transpose(0,2,1,3,4,5)
            weight = weight.reshape(weight.shape[0]*weight.shape[1],-1,weight.shape[4],weight.shape[5])

            x = x.astype(weight.dtype)
        
            return jax.lax.conv(x,weight,(1,1),'Valid')
        else:
            weight = weight.transpose(0,2,1,3,4,5,6)
            weight = weight.reshape(weight.shape[0]*weight.shape[1],-1,weight.shape[4],weight.shape[5],weight.shape[6])

            x = x.astype(weight.dtype)
            
            return jax.lax.conv(x,weight,(1,1,1),'Valid')

def zeros(key, shape, dtype):
    return jnp.zeros(shape, dtype)


def default_equivariant_initializer(key, shape, dtype):
    fan_in = np.prod(shape[1:])
    std = 1.0 / math.sqrt(fan_in)
    return std * jax.random.normal(key, shape, dtype)


class DenseSymmFFT(eqx.Module):
    # trainable arrays
    kernel: jax.Array

    # static fields
    space_group: Any = eqx.field(static=True)
    features: int = eqx.field(static=True)
    shape: tuple[int, ...] = eqx.field(static=True)
    mask: Any = eqx.field(static=True)
    precision: Any = eqx.field(static=True)

    n_cells: int = eqx.field(static=True)
    n_symm: int = eqx.field(static=True)
    n_point: int = eqx.field(static=True)
    sites_per_cell: int = eqx.field(static=True)
    mapping: Any = eqx.field(static=True)
    kernel_indices: Any = eqx.field(static=True)

    def __init__(
        self,
        space_group,
        features: int,
        in_features: int,
        shape: tuple[int, ...],
        *,
        key,
        mask=None,
        param_dtype=jnp.float32,
        precision=None,
        kernel_init: Callable = default_equivariant_initializer,
    ):
        sg = np.asarray(space_group)

        self.space_group = space_group
        self.features = features
        self.shape = tuple(shape)
        self.mask = mask
        self.precision = precision

        self.n_cells = int(np.prod(np.asarray(shape)))
        self.n_symm = len(sg)
        self.n_point = self.n_symm // self.n_cells
        self.sites_per_cell = sg.shape[1] // self.n_cells

        if mask is not None:
            mask_arr = np.asarray(mask.wrapped if hasattr(mask, "wrapped") else mask)
            (self.kernel_indices,) = np.nonzero(mask_arr)
            kernel_shape = (features, in_features, len(self.kernel_indices))
        else:
            self.kernel_indices = None
            kernel_shape = (
                features,
                in_features,
                self.n_cells * self.sites_per_cell,
            )

        # maps kernel site dimension to:
        # (sites_per_cell, n_point, *shape)
        self.mapping = (
            sg[:, ::self.n_cells]
            .reshape(self.n_cells, self.n_point, self.sites_per_cell)
            .transpose(2, 1, 0)
            .reshape(self.sites_per_cell, self.n_point, *self.shape)
        )

        k1, k2 = jax.random.split(key)

        self.kernel = kernel_init(k1, kernel_shape, param_dtype)

    def __call__(self, x: jax.Array) -> jax.Array:
        """
        Input shape:
            (..., in_features, n_sites)

        Output shape:
            (..., features, n_symm)
        """
        if x.ndim < 2:
            x = x[None]

        in_features = x.shape[0]

        x = x.reshape(in_features, self.sites_per_cell, *self.shape)

        if self.kernel_indices is not None:
            kernel_full = jnp.zeros(
                (
                    self.features,
                    in_features,
                    self.n_cells * self.sites_per_cell,
                ),
                dtype=self.kernel.dtype,
            )
            kernel = kernel_full.at[:, :, self.kernel_indices].set(self.kernel)
        else:
            kernel = self.kernel

        # promote manually
        dtype = jnp.result_type(x, kernel)
        x = x.astype(dtype)
        kernel = kernel.astype(dtype)

        # Expand kernel to:
        # (features, in_features, sites_per_cell, n_point, *shape)
        
        kernel = kernel[..., self.mapping]

        x = jnp.fft.fftn(x, s=self.shape).reshape(*x.shape[:2], self.n_cells)

        kernel = jnp.fft.fftn(kernel, s=self.shape).reshape(
            *kernel.shape[:4], self.n_cells
        )

        x = lax.dot_general(
            x,
            kernel,
            (((0, 1), (1, 2)), ((2,), (4,))),
            precision=self.precision,
        )

        x = x.transpose(1, 2, 0)
        x = x.reshape(*x.shape[:2], *self.shape)

        x = jnp.fft.ifftn(x, s=self.shape).reshape(*x.shape[:2], self.n_cells)
        
        x = x.transpose(0, 2, 1)

        x = x.reshape(self.features, self.n_symm)

        return x.real if not jnp.issubdtype(x.dtype, jnp.complexfloating) else x
    

class DenseEquivariantFFT(eqx.Module):
    # trainable
    kernel: jax.Array

    # static
    product_table: Any = eqx.field(static=True)
    features: int = eqx.field(static=True)
    shape: tuple[int, ...] = eqx.field(static=True)
    mask: Any = eqx.field(static=True)
    precision: Any = eqx.field(static=True)

    n_symm: int = eqx.field(static=True)
    n_cells: int = eqx.field(static=True)
    n_point: int = eqx.field(static=True)
    mapping: Any = eqx.field(static=True)
    kernel_indices: Any = eqx.field(static=True)

    def __init__(
        self,
        product_table,
        features: int,
        in_features: int,
        shape: tuple[int, ...],
        *,
        key,
        mask=None,
        param_dtype=jnp.float32,
        precision=None,
        kernel_init: Callable = default_equivariant_initializer,
    ):
        pt = np.asarray(product_table)

        self.product_table = product_table
        self.features = features
        self.shape = tuple(shape)
        self.mask = mask
        self.precision = precision

        self.n_symm = len(pt)
        self.n_cells = int(np.prod(np.asarray(shape)))
        self.n_point = self.n_symm // self.n_cells

        if mask is not None:
            mask_arr = np.asarray(mask.wrapped if hasattr(mask, "wrapped") else mask)
            (self.kernel_indices,) = np.nonzero(mask_arr)
            kernel_shape = (features, in_features, len(self.kernel_indices))
        else:
            self.kernel_indices = None
            kernel_shape = (
                features,
                in_features,
                self.n_point * self.n_cells,
            )

        # maps kernel group dimension to:
        # (n_point_in, n_point_out, *shape)
        self.mapping = (
            pt[: self.n_point]
            .reshape(self.n_point, self.n_cells, self.n_point)
            .transpose(0, 2, 1)
            .reshape(self.n_point, self.n_point, *self.shape)
        )

        self.kernel = kernel_init(key, kernel_shape, param_dtype)

    def __call__(self, x: jax.Array) -> jax.Array:
        """
        Input:
            x.shape == (..., in_features, n_symm)

        Output:
            y.shape == (..., features, n_symm)
        """

        in_features = x.shape[0]

        x = x.reshape(in_features, self.n_cells, self.n_point)
        x = x.transpose(0, 2, 1)
        x = x.reshape(*x.shape[:-1], *self.shape)

        if self.kernel_indices is not None:
            kernel_full = jnp.zeros(
                (
                    self.features,
                    in_features,
                    self.n_point * self.n_cells,
                ),
                dtype=self.kernel.dtype,
            )
            kernel = kernel_full.at[:, :, self.kernel_indices].set(self.kernel)
        else:
            kernel = self.kernel

        dtype = jnp.result_type(x, kernel)
        x = x.astype(dtype)
        kernel = kernel.astype(dtype)

        # kernel:
        # (features, in_features, n_point_in, n_point_out, *shape)
        kernel = kernel[..., self.mapping]

        x = jnp.fft.fftn(x, s=self.shape).reshape(*x.shape[:2], self.n_cells)

        kernel = jnp.fft.fftn(kernel, s=self.shape).reshape(
            *kernel.shape[:4], self.n_cells
        )

        x = lax.dot_general(
            x,
            kernel,
            (((0, 1), (1, 2)), ((2,), (4,))),
            precision=self.precision,
        )

        x = x.transpose(1, 2, 0)
        x = x.reshape(*x.shape[:2], *self.shape)

        x = jnp.fft.ifftn(x, s=self.shape).reshape(*x.shape[:2], self.n_cells)
        x = x.transpose(0, 2, 1)
        x = x.reshape(self.features, self.n_symm)

        if jnp.can_cast(x.dtype, dtype):
            return x
        else:
            return x.real