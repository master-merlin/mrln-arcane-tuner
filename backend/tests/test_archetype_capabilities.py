from app.engine.core.archetypes import ARCHETYPES, Archetype, build_field_visibility


def test_two_archetypes_exist():
    assert set(ARCHETYPES) == {"latent_diffusion", "unified_transformer"}


def test_registry_values_are_archetype_instances():
    assert all(isinstance(a, Archetype) for a in ARCHETYPES.values())


def test_field_visibility_accepts_plain_dict():
    # build_field_visibility must also work on a merged-capabilities dict, not
    # just an Archetype — the resolver (Task 3) passes a dict.
    fv = build_field_visibility({"has_vae": False, "te_cache": True})
    assert fv["cache_latents"]["supported"] is False        # has_vae False
    assert fv["cache_text_embeddings"]["supported"] is True  # te_cache True
    assert fv["block_swap_config"]["supported"] is True      # missing flag -> defaults True


def test_latent_diffusion_supports_caching():
    a = ARCHETYPES["latent_diffusion"]
    assert a.has_vae and a.has_external_te and a.latent_cache and a.te_cache


def test_unified_transformer_disables_caching():
    a = ARCHETYPES["unified_transformer"]
    assert not a.has_vae and not a.has_external_te
    assert not a.latent_cache and not a.te_cache and not a.supports_te_quantization


def test_field_visibility_hides_cache_for_unified():
    fv = build_field_visibility(ARCHETYPES["unified_transformer"])
    assert fv["cache_latents"]["supported"] is False
    assert fv["cache_text_embeddings"]["supported"] is False
    assert fv["cache_latents"].get("reason")


def test_field_visibility_shows_cache_for_latent():
    fv = build_field_visibility(ARCHETYPES["latent_diffusion"])
    assert fv["cache_latents"]["supported"] is True
    assert fv["cache_text_embeddings"]["supported"] is True
