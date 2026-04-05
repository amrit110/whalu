"""Unit tests for whalu.sources."""

from whalu.sources import MBARI, ORCASOUND, REGISTRY, SourceInfo


class TestSourceInfoDataclass:
    """Tests for the SourceInfo dataclass."""

    def test_fields_accessible(self):
        assert MBARI.id == "mbari"
        assert MBARI.name == "MBARI Pacific Sound"

    def test_frozen(self):
        import pytest

        with pytest.raises(Exception):
            MBARI.id = "changed"  # type: ignore[misc]

    def test_species_is_list(self):
        assert isinstance(MBARI.species, list)
        assert len(MBARI.species) > 0

    def test_notes_is_list(self):
        assert isinstance(MBARI.notes, list)
        assert len(MBARI.notes) > 0


class TestMBARI:
    """Tests for the MBARI source definition."""

    def test_s3_bucket(self):
        assert MBARI.s3_bucket == "pacific-sound-16khz"

    def test_sample_rate(self):
        assert MBARI.sample_rate_hz == 16_000

    def test_bit_depth(self):
        assert MBARI.bit_depth == 24

    def test_depth_meters(self):
        assert MBARI.depth_m == 891

    def test_has_all_whale_species(self):
        species_text = " ".join(MBARI.species)
        for code in ["Bm", "Bp", "Mn", "Ba", "Be", "Bs", "Eg", "Oo"]:
            assert code in species_text, f"MBARI missing species code {code}"

    def test_description_non_empty(self):
        assert len(MBARI.description) > 50

    def test_s3_path_format_contains_bucket(self):
        assert "pacific-sound-16khz" in MBARI.s3_path_format


class TestOrcasound:
    """Tests for the Orcasound source definition."""

    def test_s3_bucket(self):
        assert ORCASOUND.s3_bucket == "acoustic-sandbox"

    def test_sample_rate(self):
        assert ORCASOUND.sample_rate_hz == 20_000

    def test_bit_depth(self):
        assert ORCASOUND.bit_depth == 16

    def test_primarily_orca(self):
        species_text = " ".join(ORCASOUND.species)
        assert "Oo" in species_text

    def test_shallow_deployment(self):
        assert ORCASOUND.depth_m < 100


class TestRegistry:
    """Tests for the sources REGISTRY."""

    def test_contains_mbari(self):
        assert "mbari" in REGISTRY
        assert REGISTRY["mbari"] is MBARI

    def test_contains_orcasound(self):
        assert "orcasound" in REGISTRY
        assert REGISTRY["orcasound"] is ORCASOUND

    def test_all_entries_are_source_info(self):
        for key, value in REGISTRY.items():
            assert isinstance(value, SourceInfo), f"{key} is not a SourceInfo"

    def test_id_matches_key(self):
        for key, value in REGISTRY.items():
            assert value.id == key
