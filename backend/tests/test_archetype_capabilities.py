from app.engine.core.archetypes import ARCHETYPES, Archetype, build_field_visibility


def test_three_archetypes_exist():
    assert set(ARCHETYPES) == {
        "latent_diffusion",
        "unified_transformer",
        "pixel_transformer",
    }


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


# ── pixel_transformer (prx_pixel): NO VAE but an EXTERNAL cacheable TE ──────


def test_pixel_transformer_caps():
    """Pixel-space DiT with a standalone text encoder: no VAE/latent cache,
    but the external TE exists and its embeddings ARE disk-cacheable."""
    a = ARCHETYPES["pixel_transformer"]
    assert not a.has_vae
    assert not a.latent_cache
    assert a.has_external_te
    assert a.te_cache
    assert not a.supports_train_te
    assert not a.supports_te_quantization
    assert not a.supports_block_swap
    # Image family: video flags stay off.
    assert not a.is_video and not a.has_audio and not a.dual_expert


def test_field_visibility_pixel_transformer_hides_latent_but_keeps_te_cache():
    """Field gating: cache_latents + low_vram hidden (no VAE), TE fields for
    the external encoder stay visible where meaningful."""
    fv = build_field_visibility(ARCHETYPES["pixel_transformer"])
    assert fv["cache_latents"]["supported"] is False
    assert fv["low_vram"]["supported"] is False
    assert fv["cache_latents"].get("reason")
    # The external TE keeps its cache + offload toggles...
    assert fv["cache_text_embeddings"]["supported"] is True
    assert fv["unload_text_encoder"]["supported"] is True
    # ...but no TE training / quantization / block swap.
    assert fv["train_text_encoder"]["supported"] is False
    assert fv["te_quantization"]["supported"] is False
    assert fv["block_swap_config"]["supported"] is False
