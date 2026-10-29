"""Check large-gauge hopping phases and physical many-fermion translations.

T_a c†_r T_a^-1 = c†_(r+a); define T_a|Psi> = exp(i q_a)|Psi>.
H(theta+2pi e_a) = U_a H(theta) U_a†,
U_a = exp(i 2pi/L_a sum_r r_a n_r).
Hence T_a U_a = exp(-i 2pi N/L_a) U_a T_a.
"""

import json
from pathlib import Path
import numpy as np
import jax
import jax.numpy as jnp
import quantax as qtx
from conditional_pfaffian import ConditionalPfaffian


def translate(spins, coords, extent, axis):
    """Translate occupied modes positively and return their Fock permutation sign."""
    target = coords.copy()
    target[:, axis] = (target[:, axis] + 1) % extent[axis]
    perm = np.ravel_multi_index(target.T, tuple(extent))
    modes = np.r_[perm, perm + len(coords)]
    occupied = np.flatnonzero(spins == 1)
    translated_occupied = modes[occupied]
    inversions = np.triu(
        translated_occupied[:, None] > translated_occupied[None, :], 1
    ).sum()
    result = np.empty_like(spins)
    result[modes] = spins
    return result, (-1.0) ** int(inversions)


def main():
    jax.config.update("jax_enable_x64", True)
    qtx.set_default_dtype(jnp.complex128)
    qtx.set_random_seed(3232)
    lat = qtx.sites.Square(6, particle_type="spinful_fermion", Nparticles=(15, 15))
    model = ConditionalPfaffian()
    # Exercise the symmetric projection away from its initial zero value.
    import equinox as eqx

    model = eqx.tree_at(
        lambda m: m.symmetric_projection.weight,
        model,
        jax.random.normal(
            qtx.get_subkeys(),
            model.symmetric_projection.weight.shape,
            dtype=jnp.float32,
        )
        * 0.01,
    )
    evaluate = jax.jit(lambda s, c: model(s, c))
    coords = np.asarray(lat.xyz_from_index[:, 1:])
    extent = np.asarray(lat.shape[1:])
    spins = np.asarray(qtx.utils.rand_states(8))
    flux = jnp.array([0.7, 1.3])
    errors, gauge_errors, hopping_errors = [], [], []
    for s in spins:
        occupation = (s[: lat.Nsites] + s[lat.Nsites :] + 2) / 2
        psi = evaluate(jnp.asarray(s), flux)
        for axis in range(2):
            translated, sign = translate(s, coords, extent, axis)
            other = evaluate(jnp.asarray(translated), flux)
            ratio = complex(other.sign / psi.sign * jnp.exp(other.logabs - psi.logabs))
            errors.append(abs(sign * ratio - 1))
            theta = 2 * np.pi / extent[axis]
            u = np.exp(1j * theta * (coords[:, axis] @ occupation))
            translated_n = (translated[: lat.Nsites] + translated[lat.Nsites :] + 2) / 2
            ut = np.exp(1j * theta * (coords[:, axis] @ translated_n))
            # Coefficient ratio after translation is inverse to the state eigenvalue.
            gauge_errors.append(abs(ut / u - np.exp(1j * theta * 30)))
            # All forward bonds, including wraps: U changes hopping by exp(i 2pi/L).
            for r in coords:
                rp = r.copy()
                rp[axis] = (rp[axis] + 1) % extent[axis]
                hopping_errors.append(
                    abs(np.exp(1j * theta * (rp[axis] - r[axis])) - np.exp(1j * theta))
                )
    assert max(errors) < 1e-5, max(errors)
    assert max(gauge_errors) < 1e-12
    assert max(hopping_errors) < 1e-12
    report = dict(
        particles=30,
        lattice=[6, 6],
        translation_convention="T cdag_r T^-1 = cdag_(r+a)",
        eigenvalue_convention="T_a Psi = exp(i q_a) Psi",
        gauge_convention="H(theta+2pi e_a)=U_a H(theta) U_a^dagger",
        momentum_shift_per_twist=-2 * np.pi * 30 / 6,
        momentum_shift_mod_2pi=0.0,
        max_base_translation_error=max(errors),
        max_gauge_translation_error=max(gauge_errors),
        max_hopping_phase_error=max(hopping_errors),
        configurations=len(spins),
        conclusion="A 2pi charge twist preserves q=0 at N=30,L=6; neither +/-pi/3 is the mapped sector.",
    )
    print(json.dumps(report, indent=2))
    Path("gauge_momentum_validation.json").write_text(
        json.dumps(report, indent=2) + "\n"
    )


if __name__ == "__main__":
    main()
