sites
=====

``sites`` contains the basic information of the system, including spatial positions 
and the particle type.

.. currentmodule:: quantax.sites

.. note::
    Currently Quantax only supports spin-1/2 systems, spinless fermion systems, 
    and spin-1/2 fermion systems. See `~quantax.PARTICLE_TYPE`.
    `FlavorLattice` and `TriangularFlavors` encode additional internal flavors
    as blocks of spinless modes, with flavor-preserving neighbor graphs.

.. warning::
    Unlike other NQS packages, in Quantax the spatial geometry and hilbert space 
    information provided by ``Sites`` is a **global constant**.
    There should only be one ``Sites`` defined in the beginning of a 
    program and kept fixed for the whole program.

Main classes
------------

.. autosummary::
    :nosignatures:
    :toctree:

    Sites
    Lattice
    FlavorLattice


Common lattices
---------------

.. autosummary::
    :toctree:

    Grid
    Chain
    Square
    Cube
    Pyrochlore
    Triangular
    TriangularB
    TriangularFlavors
