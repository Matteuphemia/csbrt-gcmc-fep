"""Unit conversions shared by the MACE surrogate modules.

MACE speaks eV and eV/A. OpenMM speaks kJ/mol and kJ/(mol nm). Amber/SOMD2
report kcal/mol. Every threshold in this package is stored in MACE's native
units and converted at the boundary, so a threshold quoted from the literature
(Kulichenko 2023, Duignan 2024: sigma_F ~ 0.05 eV/A) can be typed into the
config unchanged.

The CODATA 2022 value 1 eV = 96.48533212331 kJ/mol is used throughout; every
other constant is derived from it so the set cannot drift out of sync.
"""

from __future__ import annotations

# Base conversions.
EV_TO_KJ_PER_MOL = 96.48533212331
KCAL_TO_KJ = 4.184
ANGSTROM_TO_NM = 0.1

# Derived. Keep these expressions rather than literals: a hand-typed
# eV/A -> kJ/(mol nm) constant was previously out by the 10 A/nm factor, which
# silently multiplied the force threshold by ten.
NM_TO_ANGSTROM = 1.0 / ANGSTROM_TO_NM
EV_TO_KCAL_PER_MOL = EV_TO_KJ_PER_MOL / KCAL_TO_KJ
EV_PER_ANGSTROM_TO_KJ_PER_MOL_NM = EV_TO_KJ_PER_MOL / ANGSTROM_TO_NM
EV_PER_ANGSTROM_TO_KCAL_PER_MOL_ANG = EV_TO_KCAL_PER_MOL
KJ_PER_MOL_TO_EV = 1.0 / EV_TO_KJ_PER_MOL
KCAL_PER_MOL_TO_EV = 1.0 / EV_TO_KCAL_PER_MOL

#: Accepted spellings for the energy units a caller may hand to the UQ monitor.
ENERGY_UNITS_TO_EV = {
    "ev": 1.0,
    "kcal_per_mol": KCAL_PER_MOL_TO_EV,
    "kj_per_mol": KJ_PER_MOL_TO_EV,
}

#: Accepted spellings for force units.
FORCE_UNITS_TO_EV_PER_ANGSTROM = {
    "ev_per_ang": 1.0,
    "kcal_per_mol_ang": 1.0 / EV_PER_ANGSTROM_TO_KCAL_PER_MOL_ANG,
    "kj_per_mol_nm": 1.0 / EV_PER_ANGSTROM_TO_KJ_PER_MOL_NM,
}


def energy_to_ev(value, units: str):
    """Convert an energy (scalar or array) from ``units`` to eV."""
    try:
        factor = ENERGY_UNITS_TO_EV[units]
    except KeyError:
        raise ValueError(
            f"Unknown energy unit {units!r}; expected one of "
            f"{sorted(ENERGY_UNITS_TO_EV)}"
        ) from None
    return value * factor


def force_to_ev_per_angstrom(value, units: str):
    """Convert a force (scalar or array) from ``units`` to eV/A."""
    try:
        factor = FORCE_UNITS_TO_EV_PER_ANGSTROM[units]
    except KeyError:
        raise ValueError(
            f"Unknown force unit {units!r}; expected one of "
            f"{sorted(FORCE_UNITS_TO_EV_PER_ANGSTROM)}"
        ) from None
    return value * factor
