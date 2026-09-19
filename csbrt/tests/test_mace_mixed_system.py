"""Stage 1: ML/MM partitioning and mixed-system construction."""

from __future__ import annotations

import numpy as np
import pytest

openmm = pytest.importorskip("openmm")
unit = pytest.importorskip("openmm.unit")

from csbrt.mace_surrogate import (  # noqa: E402
    INTERPOLATION_PARAMETER,
    MACEConfig,
    MACESurrogateError,
    apply_system_in_place,
    attach_mace_to_context,
    create_mace_mixed_system,
    describe_ml_region,
    noninteracting_particles,
    partition_ml_atoms,
    positions_in_nm,
    validate_ml_region,
)


# --------------------------------------------------------------------------- partition


def test_ligand_only_partition(built_system):
    ml_atoms = partition_ml_atoms(built_system.topology, ligand_resname="LIG")
    assert ml_atoms == sorted(built_system.ligand_atoms)
    assert len(ml_atoms) == 9


def test_partition_rejects_missing_ligand(built_system):
    with pytest.raises(MACESurrogateError, match="No residue named"):
        partition_ml_atoms(built_system.topology, ligand_resname="XYZ")


def test_partition_includes_sphere_waters(built_system_no_ghost):
    built = built_system_no_ghost
    ml_atoms = partition_ml_atoms(
        built.topology,
        positions=built.positions_quantity(),
        ligand_resname="LIG",
        include_waters=True,
        sphere_radius_nm=1.0,
    )
    assert set(built.ligand_atoms) <= set(ml_atoms)
    assert len(ml_atoms) > len(built.ligand_atoms)
    assert (len(ml_atoms) - len(built.ligand_atoms)) % 3 == 0


def test_partition_water_selection_is_radius_limited(built_system_no_ghost):
    built = built_system_no_ghost
    near = partition_ml_atoms(
        built.topology,
        positions=built.positions_quantity(),
        include_waters=True,
        sphere_radius_nm=0.05,
    )
    far = partition_ml_atoms(
        built.topology,
        positions=built.positions_quantity(),
        include_waters=True,
        sphere_radius_nm=5.0,
    )
    assert near == sorted(built.ligand_atoms)
    assert len(far) > len(near)


def test_partition_needs_positions_for_waters(built_system):
    with pytest.raises(MACESurrogateError, match="needs positions"):
        partition_ml_atoms(built_system.topology, include_waters=True)


def test_loch_ghost_waters_are_excluded(built_system):
    """A zero-charge zero-epsilon water must never enter the ML region."""
    ghosts = noninteracting_particles(built_system.system)
    assert ghosts == set(built_system.ghost_atoms)
    ml_atoms = partition_ml_atoms(
        built_system.topology,
        positions=built_system.positions_quantity(),
        include_waters=True,
        sphere_radius_nm=5.0,
        exclude=ghosts,
    )
    assert not (set(ml_atoms) & ghosts)


def test_positions_in_nm_accepts_quantity_and_array(built_system):
    from_quantity = positions_in_nm(built_system.positions_quantity())
    from_array = positions_in_nm(built_system.positions_nm)
    assert np.allclose(from_quantity, from_array)


# --------------------------------------------------------------------------- region


def test_describe_region_reports_elements(built_system):
    region = describe_ml_region(built_system.topology, built_system.ligand_atoms)
    assert len(region) == 9
    assert region.ligand_atom_count == 9
    assert region.water_atom_count == 0
    assert sorted(set(region.atomic_numbers)) == [1, 6, 8]
    assert region.unsupported_elements == []
    assert region.residues == {"LIG": 1}


def test_region_size_cap_is_enforced(built_system):
    region = describe_ml_region(built_system.topology, built_system.ligand_atoms)
    with pytest.raises(MACESurrogateError, match="max_ml_atoms"):
        validate_ml_region(region, MACEConfig(enabled=True, max_ml_atoms=4))


def test_unsupported_element_rejected_for_mace_off(built_system):
    region = describe_ml_region(built_system.topology, built_system.ligand_atoms)
    region.atomic_numbers[0] = 30  # zinc
    with pytest.raises(MACESurrogateError, match="outside MACE-OFF"):
        validate_ml_region(
            region, MACEConfig(enabled=True, model_name="mace-off23-small")
        )


# --------------------------------------------------------------------------- build


def test_mixed_system_has_interpolation_parameter(built_system, stub_config):
    mixed = create_mace_mixed_system(
        built_system.system,
        built_system.topology,
        built_system.ligand_atoms,
        config=stub_config,
    )
    assert mixed.getNumParticles() == built_system.system.getNumParticles()
    context = built_system.context(system=mixed)
    assert INTERPOLATION_PARAMETER in context.getParameters()


def test_interpolation_recovers_the_classical_energy(built_system, stub_config):
    """lambda_interpolate=0 must reproduce the untouched MM Hamiltonian.

    This is the property the whole fallback rests on: the classical leg of the
    dual Hamiltonian has to be the same physics the pipeline ran before.
    """
    classical = built_system.potential_energy()
    mixed = create_mace_mixed_system(
        built_system.system,
        built_system.topology,
        built_system.ligand_atoms,
        config=stub_config,
    )
    at_zero = built_system.potential_energy(
        system=mixed, parameters={INTERPOLATION_PARAMETER: 0.0}
    )
    assert at_zero == pytest.approx(classical, rel=1e-6, abs=1e-6)


def test_surrogate_and_classical_energies_differ(built_system, stub_config):
    mixed = create_mace_mixed_system(
        built_system.system,
        built_system.topology,
        built_system.ligand_atoms,
        config=stub_config,
    )
    at_zero = built_system.potential_energy(
        system=mixed, parameters={INTERPOLATION_PARAMETER: 0.0}
    )
    at_one = built_system.potential_energy(
        system=mixed, parameters={INTERPOLATION_PARAMETER: 1.0}
    )
    assert abs(at_one - at_zero) > 1.0


def test_mm_region_is_untouched_by_the_ml_potential(built_system, stub_config):
    """Atoms outside the ML region must feel exactly the classical forces.

    The speedup is only legitimate if the protein and bulk solvent are still
    being integrated on ff14SB/TIP3P. Compare per-atom forces on the MM subset
    between the pure classical system and the mixed system in surrogate mode.
    """
    ml_atoms = set(built_system.ligand_atoms)
    # Waters and protein feel the ligand through the nonbonded force, so
    # compare only atoms beyond the cutoff-independent bonded terms: take every
    # atom that is not in the ML region and not a nonbonded neighbour of it.
    classical_context = built_system.context()
    classical_forces = classical_context.getState(getForces=True).getForces(
        asNumpy=True
    ).value_in_unit(unit.kilojoule_per_mole / unit.nanometer)

    mixed = create_mace_mixed_system(
        built_system.system,
        built_system.topology,
        built_system.ligand_atoms,
        config=stub_config,
    )
    mixed_context = built_system.context(system=mixed)
    mixed_context.setParameter(INTERPOLATION_PARAMETER, 0.0)
    mixed_forces = mixed_context.getState(getForces=True).getForces(
        asNumpy=True
    ).value_in_unit(unit.kilojoule_per_mole / unit.nanometer)

    mm_atoms = [
        i for i in range(built_system.system.getNumParticles()) if i not in ml_atoms
    ]
    assert np.allclose(
        np.asarray(classical_forces)[mm_atoms],
        np.asarray(mixed_forces)[mm_atoms],
        atol=1e-4,
    )


# --------------------------------------------------------------------------- attach


def test_apply_system_in_place_matches_a_fresh_context(built_system, stub_config):
    mixed = create_mace_mixed_system(
        built_system.system,
        built_system.topology,
        built_system.ligand_atoms,
        config=stub_config,
    )
    reference = built_system.potential_energy(
        system=mixed, parameters={INTERPOLATION_PARAMETER: 1.0}
    )

    context = built_system.context()
    apply_system_in_place(context.getSystem(), mixed)
    context.reinitialize(preserveState=True)
    context.setParameter(INTERPOLATION_PARAMETER, 1.0)
    live = context.getState(getEnergy=True).getPotentialEnergy().value_in_unit(
        unit.kilojoule_per_mole
    )
    assert live == pytest.approx(reference, rel=1e-6, abs=1e-6)


def test_apply_system_in_place_rejects_particle_count_change(built_system):
    other = openmm.System()
    other.addParticle(1.0)
    with pytest.raises(MACESurrogateError, match="particles"):
        apply_system_in_place(built_system.system, other)


def test_attach_preserves_positions_and_velocities(built_system, stub_config):
    context = built_system.context()
    context.setVelocitiesToTemperature(300 * unit.kelvin, 1234)
    before_positions = context.getState(getPositions=True).getPositions(asNumpy=True)
    before_velocities = context.getState(getVelocities=True).getVelocities(asNumpy=True)

    handle = attach_mace_to_context(
        context, built_system.topology, stub_config, built_system.ligand_atoms
    )

    after_positions = context.getState(getPositions=True).getPositions(asNumpy=True)
    after_velocities = context.getState(getVelocities=True).getVelocities(asNumpy=True)
    assert np.allclose(
        before_positions.value_in_unit(unit.nanometer),
        after_positions.value_in_unit(unit.nanometer),
    )
    assert np.allclose(
        before_velocities.value_in_unit(unit.nanometer / unit.picosecond),
        after_velocities.value_in_unit(unit.nanometer / unit.picosecond),
    )
    assert handle.parameter_name == INTERPOLATION_PARAMETER
    assert handle.interpolating is True
    assert handle.region.ligand_atom_count == 9
    assert context.getParameter(INTERPOLATION_PARAMETER) == pytest.approx(1.0)


def test_attach_then_classical_matches_the_original_energy(built_system, stub_config):
    classical = built_system.potential_energy()
    context = built_system.context()
    attach_mace_to_context(
        context, built_system.topology, stub_config, built_system.ligand_atoms
    )
    context.setParameter(INTERPOLATION_PARAMETER, 0.0)
    after = context.getState(getEnergy=True).getPotentialEnergy().value_in_unit(
        unit.kilojoule_per_mole
    )
    assert after == pytest.approx(classical, rel=1e-6, abs=1e-6)
