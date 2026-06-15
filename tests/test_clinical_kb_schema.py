from ui.clinical_ui import load_clinical_kb


def test_clinical_kb_has_expected_top_level_sections():
    kb = load_clinical_kb()

    assert isinstance(kb, dict)
    assert isinstance(kb.get("meta"), dict)
    assert isinstance(kb.get("levels"), dict)
    assert isinstance(kb.get("sources"), list)

    meta = kb["meta"]
    assert meta.get("version")
    assert meta.get("review_date")
    assert meta.get("disclaimer")


def test_clinical_kb_levels_have_required_fields_and_known_refs():
    kb = load_clinical_kb()
    levels = kb["levels"]
    source_ids = {str(s.get("id")) for s in kb["sources"] if isinstance(s, dict)}

    for level in ["suspect", "probable", "critique"]:
        assert level in levels
        payload = levels[level]
        assert isinstance(payload, dict)
        assert str(payload.get("summary", "")).strip()
        assert str(payload.get("label", "")).strip()
        assert payload.get("priority") in {1, 2, 3}

        refs = payload.get("references")
        assert isinstance(refs, list)
        assert refs, f"{level} doit référencer au moins une source"
        assert all(ref in source_ids for ref in refs)


def test_clinical_kb_sources_have_unique_ids_and_minimum_metadata():
    kb = load_clinical_kb()
    sources = [s for s in kb["sources"] if isinstance(s, dict)]

    ids = [str(s.get("id", "")).strip() for s in sources]
    assert all(ids)
    assert len(ids) == len(set(ids))

    for src in sources:
        assert str(src.get("name", "")).strip()
        assert str(src.get("url", "")).strip()
        assert str(src.get("review_date", "")).strip()
