#!/usr/bin/env python3
"""Build a local NLLB-600M snapshot dir from the already-downloaded HF cache.

Creates models_cache/nllb-600m/ containing config/tokenizer files plus
model.safetensors (copied, since the cache snapshot only holds the safetensors
blob symlinked — we need a complete dir so load_nllb() takes the safetensors
path and never touches pytorch_model.bin).
No network access; no new downloads.
"""
from pathlib import Path
import shutil

CACHE = Path.home() / ".cache/huggingface/hub/models--facebook--nllb-200-distilled-600M"
DEST = Path("models_cache/nllb-600m")

snap_bin = next(CACHE.glob("snapshots/*/pytorch_model.bin"))
snap_st = next(CACHE.glob("snapshots/*/model.safetensors"))

DEST.mkdir(parents=True, exist_ok=True)
bin_dir = snap_bin.parent
for name in ["config.json", "generation_config.json", "sentencepiece.bpe.model",
             "special_tokens_map.json", "tokenizer.json", "tokenizer_config.json"]:
    src = bin_dir / name
    if src.exists():
        shutil.copy2(src, DEST / name)
    else:
        print("missing (optional):", name)
st_dest = DEST / "model.safetensors"
if not st_dest.exists():
    try:
        shutil.copy2(snap_st, st_dest)  # real copy, resolves symlink
    except OSError as e:  # e.g. disk full: fall back to a symlink into the cache
        print(f"copy failed ({e}); symlinking instead (no extra disk use)")
        st_dest.symlink_to(snap_st.resolve())
print("built", DEST, "->", sorted(p.name for p in DEST.iterdir()))
