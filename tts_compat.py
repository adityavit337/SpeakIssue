"""
tts_compat.py — Compatibility shims for running Indic Parler-TTS (v0.2.2)
with Transformers v5.x.

Parler-TTS was built against transformers ≤ 4.46.1. Transformers v5
introduced several breaking changes:
  1. Removed `isin_mps_friendly` from `pytorch_utils`
  2. `ParlerTTSConfig.__init__` fails when diffing against a default config
  3. `PreTrainedModel` no longer inherits from `GenerationMixin`
  4. `tie_weights()` receives a new `recompute_mapping` kwarg
  5. `GenerationConfig.num_return_sequences` defaults to `None` instead of `1`
  6. `DynamicCache` replaced `key_cache`/`value_cache` lists with `layers`

Import this module BEFORE importing parler_tts to apply all patches.
"""

import torch


def apply_parler_tts_patches():
    """Apply all compatibility patches. Call once before loading the TTS model."""

    # ── BLOCK 1: PATCH 1 - isin_mps_friendly ──
    # Transformers v5 removed `isin_mps_friendly` from pytorch_utils.
    # This shim re-implements it for older code expecting it.
    import transformers.pytorch_utils as _pu
    if not hasattr(_pu, 'isin_mps_friendly'):
        def _isin_shim(elements, test_elements):
            if elements.dtype != test_elements.dtype:
                elements = elements.to(dtype=test_elements.dtype)
            return torch.isin(elements, test_elements)
        _pu.isin_mps_friendly = _isin_shim

    # ── BLOCK 2: PATCH 2 - config has_no_defaults_at_init ──
    # Prevents ParlerTTSConfig.__init__ from failing when diffing against a default config in v5.
    from parler_tts import configuration_parler_tts
    configuration_parler_tts.ParlerTTSConfig.has_no_defaults_at_init = True

    # ── BLOCK 3: PATCH 3 - inject GenerationMixin ──
    # In transformers v5, PreTrainedModel no longer inherits from GenerationMixin.
    # This manually injects it into ParlerTTSForConditionalGeneration's base classes.
    from transformers import GenerationMixin
    from parler_tts.modeling_parler_tts import ParlerTTSForConditionalGeneration

    if GenerationMixin not in ParlerTTSForConditionalGeneration.__bases__:
        ParlerTTSForConditionalGeneration.__bases__ = (
            GenerationMixin,
        ) + ParlerTTSForConditionalGeneration.__bases__

    # ── BLOCK 4: PATCH 4 - tie_weights ──
    # Transformers v5 passes a new `recompute_mapping` kwarg to tie_weights().
    # This monkey-patches the tie_weights method to handle it and manually tie embed_tokens.
    def _fixed_tie_weights(self, **kwargs):
        if not hasattr(self.config, 'tie_encoder_decoder'):
            self.config.tie_encoder_decoder = False
        # Manually tie text_encoder embed_tokens → shared weight
        if hasattr(self, 'text_encoder') and hasattr(self.text_encoder, 'encoder'):
            enc = self.text_encoder.encoder
            if hasattr(enc, 'embed_tokens') and hasattr(self.text_encoder, 'shared'):
                enc.embed_tokens.weight = self.text_encoder.shared.weight

    ParlerTTSForConditionalGeneration.tie_weights = _fixed_tie_weights

    # ── BLOCK 5: PATCH 5 - _expand_inputs_for_generation ──
    # GenerationConfig in v5 may default `expand_size` differently. This wraps
    # _expand_inputs_for_generation to safely handle `expand_size=None`.
    import transformers.generation.utils as _gen_utils
    _orig_expand = _gen_utils.GenerationMixin._expand_inputs_for_generation

    @staticmethod
    def _safe_expand(input_ids=None, expand_size=1, is_encoder_decoder=False, **model_kwargs):
        if expand_size is None:
            expand_size = 1
        if input_ids is None:
            kw = {}
            for k, v in model_kwargs.items():
                kw[k] = v.repeat_interleave(expand_size, dim=0) if isinstance(v, torch.Tensor) else v
            return None, kw
        return _orig_expand(
            input_ids=input_ids, expand_size=expand_size,
            is_encoder_decoder=is_encoder_decoder, **model_kwargs,
        )

    _gen_utils.GenerationMixin._expand_inputs_for_generation = _safe_expand

    # ── BLOCK 6: PATCH 6 - DynamicCache key_cache / value_cache ──
    # Transformers v5 replaced `key_cache` and `value_cache` lists with a `layers` structure.
    # This adds properties to DynamicCache to proxy access for older TTS code.
    from transformers import DynamicCache

    if not hasattr(DynamicCache, 'key_cache'):
        class _CacheProxy:
            """List-like proxy that maps cache.layers[i].keys/values back to
            the old cache.key_cache[i] / cache.value_cache[i] API."""
            def __init__(self, cache, attr):
                self._cache = cache
                self._attr = attr  # 'keys' or 'values'

            def __getitem__(self, idx):
                if idx < len(self._cache.layers):
                    return getattr(self._cache.layers[idx], self._attr)
                raise IndexError(f"Layer {idx} not found in cache")

            def __len__(self):
                return len(self._cache.layers)

            def append(self, val):
                pass  # no-op for compat

        DynamicCache.key_cache = property(lambda self: _CacheProxy(self, 'keys'))
        DynamicCache.value_cache = property(lambda self: _CacheProxy(self, 'values'))

    print("[tts_compat] All Parler-TTS v5 compatibility patches applied.")
