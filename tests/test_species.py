"""Unit tests for whalu.species."""

import pytest

from whalu.species import REGISTRY, Species, display_name, scientific_name

WHALE_CODES = ["Bm", "Bp", "Mn", "Ba", "Bs", "Be", "Eg", "Oo"]
CALL_CODES = ["Upcall", "Gunshot", "Call", "Echolocation", "Whistle"]


class TestSpeciesDataclass:
    """Tests for the Species dataclass."""

    def test_fields_accessible(self):
        s = Species("Bm", "Blue whale", "Balaenoptera musculus", "🐋")
        assert s.code == "Bm"
        assert s.common == "Blue whale"
        assert s.scientific == "Balaenoptera musculus"
        assert s.emoji == "🐋"

    def test_emoji_defaults_to_empty_string(self):
        s = Species("Xx", "Unknown", "Unknown sp.")
        assert s.emoji == ""

    def test_frozen(self):
        s = Species("Bm", "Blue whale", "Balaenoptera musculus")
        with pytest.raises(Exception):
            s.code = "Bp"  # type: ignore[misc]


class TestRegistry:
    """Tests for the REGISTRY dict."""

    def test_all_whale_codes_present(self):
        for code in WHALE_CODES:
            assert code in REGISTRY, f"Missing species code: {code}"

    def test_all_call_type_codes_present(self):
        for code in CALL_CODES:
            assert code in REGISTRY, f"Missing call-type code: {code}"

    def test_total_size(self):
        assert len(REGISTRY) == len(WHALE_CODES) + len(CALL_CODES)

    def test_whale_species_have_scientific_names(self):
        for code in WHALE_CODES:
            assert REGISTRY[code].scientific, f"{code} missing scientific name"

    def test_call_types_have_empty_scientific_names(self):
        for code in CALL_CODES:
            assert REGISTRY[code].scientific == "", (
                f"{code} should have no scientific name"
            )

    def test_code_matches_key(self):
        for key, species in REGISTRY.items():
            assert species.code == key

    def test_minke_whale_code(self):
        assert REGISTRY["Ba"].common == "Minke whale"
        assert REGISTRY["Ba"].scientific == "Balaenoptera acutorostrata"

    def test_orca_code(self):
        assert REGISTRY["Oo"].common == "Orca"
        assert REGISTRY["Oo"].scientific == "Orcinus orca"


class TestDisplayName:
    """Tests for the display_name function."""

    def test_known_species_returns_common_name(self):
        result = display_name("Bm")
        assert "Blue whale" in result

    def test_known_species_with_emoji_includes_emoji(self):
        result = display_name("Bm")
        assert "🐋" in result

    def test_species_without_emoji_no_leading_space(self):
        result = display_name("Ba")
        assert not result.startswith(" ")
        assert "Minke whale" in result

    def test_unknown_code_returns_code_itself(self):
        assert display_name("ZZ") == "ZZ"

    def test_call_type_returns_code(self):
        # Call types have no scientific name so fall back to code
        assert display_name("Upcall") == "Upcall"

    def test_empty_code_returns_empty_string(self):
        assert display_name("") == ""


class TestScientificName:
    """Tests for the scientific_name function."""

    def test_known_species(self):
        assert scientific_name("Bm") == "Balaenoptera musculus"

    def test_humpback(self):
        assert scientific_name("Mn") == "Megaptera novaeangliae"

    def test_right_whale(self):
        assert scientific_name("Eg") == "Eubalaena glacialis"

    def test_unknown_code_returns_empty_string(self):
        assert scientific_name("ZZ") == ""

    def test_call_type_returns_empty_string(self):
        assert scientific_name("Upcall") == ""
