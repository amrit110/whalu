"""Species code → common name / scientific name mappings for multispecies_whale."""

from dataclasses import dataclass


@dataclass(frozen=True)
class Species:
    code: str
    common: str
    scientific: str
    emoji: str = ""


# Matches the multispecies_whale model class list
REGISTRY: dict[str, Species] = {
    "Bm": Species("Bm", "Blue whale", "Balaenoptera musculus", "🐋"),
    "Bp": Species("Bp", "Fin whale", "Balaenoptera physalus", "🐳"),
    "Mn": Species("Mn", "Humpback whale", "Megaptera novaeangliae", "🐳"),
    "Ba": Species("Ba", "Minke whale", "Balaenoptera acutorostrata", ""),
    "Bs": Species("Bs", "Sei whale", "Balaenoptera borealis", ""),
    "Be": Species("Be", "Bryde's whale", "Balaenoptera edeni", ""),
    "Eg": Species("Eg", "North Atlantic right whale", "Eubalaena glacialis", ""),
    "Oo": Species("Oo", "Orca", "Orcinus orca", "🐬"),
    # Call-type classes (not a species, but returned by the model)
    "Upcall": Species("Upcall", "Right whale upcall", "", ""),
    "Gunshot": Species("Gunshot", "Right whale gunshot", "", ""),
    "Call": Species("Call", "Generic call", "", ""),
    "Echolocation": Species("Echolocation", "Echolocation", "", ""),
    "Whistle": Species("Whistle", "Whistle", "", ""),
}


def display_name(code: str) -> str:
    """Return a human-readable species label for CLI output.

    For species with a scientific name, returns ``"<emoji> <Common name>"``
    (emoji omitted when blank). Falls back to the raw code for unknown
    codes and call-type classes that have no scientific name.

    Parameters
    ----------
    code : str
        NOAA Perch species code (e.g. ``"Bm"``) or call-type label.

    Returns
    -------
    str
        Display label, or the original code if not found in the registry.
    """
    s = REGISTRY.get(code)
    if s and s.scientific:
        return f"{s.emoji} {s.common}".strip()
    return code


def scientific_name(code: str) -> str:
    """Return the scientific name for a species code, or an empty string.

    Parameters
    ----------
    code : str
        NOAA Perch species code (e.g. ``"Mn"``).

    Returns
    -------
    str
        Binomial scientific name, or ``""`` for unknown codes and
        call-type classes.
    """
    s = REGISTRY.get(code)
    return s.scientific if s else ""
