# Fine-Tuning Feasibility Assessment — NLLB-200 distilled 600M on a 16 GB Intel MacBook Pro

Status: assessment only. **No model was downloaded, no heavy dependency installed,
no training started.** Everything below is an estimate from documented model
dimensions and this machine's specs; numbers marked "~" are approximations.

## 1. Target model

`facebook/nllb-200-distilled-600M`: ~615 M parameters total (~600 M after
embedding sharing; tied embeddings). At fp32 that is ~2.4 GB of weights.

## 2. Memory requirements

| Approach                                  | Weights             | Optimizer states + grads                                    | Peak RAM estimate        | Feasible on 16 GB?                                                                      |
| ----------------------------------------- | ------------------- | ----------------------------------------------------------- | ------------------------ | --------------------------------------------------------------------------------------- |
| Full fine-tune, fp32, AdamW               | ~2.4 GB             | ~2.4 (grads) + 4.8 (Adam moments) ≈ 7.2 GB                  | ~9.6–12 GB + activations | Marginal–No: with optimizer offloads barely; swapping thrash very likely with any batch |
| Full fine-tune, bf16 weights + 8-bit Adam | ~1.2 GB             | ~2.4 GB                                                     | ~4–6 GB + activations    | Possible but risky, slow, untested here                                                 |
| **LoRA / adapter (recommended)**          | frozen fp32 ~2.4 GB | LoRA params only (~0.1–1 %): ~50–200 MB + small optim state | ~3–5 GB + activations    | **Yes**, with seq len ≤ 128 and batch 1–2 + grad accumulation                           |

Activation memory dominates at inference-like batch sizes: encoder attention
over seq_len L is roughly O(L²) per layer; at L=128, batch 1, the 600M model
needs well under 1 GB of activations — comfortably inside budget. The macOS
unified-memory pressure plus the OS and the running FastAPI service must also
be counted; close the service during any training attempt.

Conclusion: **full fine-tuning is impractical; LoRA-style parameter-efficient
fine-tuning (PEFT) is the only realistic on-device option.**

## 3. Training time (CPU only — Intel MacBook Pro, no discrete GPU)

Assume MENYO-20k train = 10,070 pairs (measured), LoRA on q/v projections,
seq len 128, effective batch 16 via accumulation, 1 epoch ≈ 630 steps.

- Forward+backward on 600M params, CPU: ~3–8 s/step ⇒ **~40–90 min/epoch**
- 3 epochs: **2–5 hours**, assuming no thermal throttling (Intel MacBook Pros
  throttle heavily; add 30–50 % margin).
- torch threads: CPU training in PyTorch on macOS works but is inefficient;
  MPS backend does **not** support this generation of Intel Macs (MPS is
  Apple-silicon only), so there is **no usable accelerator** on this machine.

## 4. Disk usage

- Existing checkpoint (already present): ~2.5 GB.
- LoRA adapter save: ~30–100 MB.
- Full-checkpoint save (if attempted): +2.5 GB per checkpoint.
- NLLB SentencePiece/tokenizer + a HF `datasets` cache for tokenized data:
  ~200–500 MB.
- Total marginal disk: <1 GB for the LoRA path; ~3 GB for the full path.

## 5. CPU vs accelerator

- CPU: feasible only with PEFT and small batches (Section 3).
- CUDA GPU: any cloud GPU (T4/RTX-class) trains 3 epochs in **~15–30 min** for
  the LoRA setup and ~1–2 h full fine-tune with fp16/bf16.
- Recommended alternative: **remote GPU training** (Colab/RunPod/Lambda) with
  the dataset pushed privately; download only the LoRA adapter back.

## 6. Licensing restrictions (hard constraints)

- **Model**: `facebook/nllb-*` checkpoints are **CC-BY-NC 4.0** — non-commercial
  use only, attribution required. Any fine-tuned derivative inherits the
  constraint; YorubaBridge production use must remain non-commercial or switch
  checkpoints (this is already documented in `app/core/config.py`).
- **Dataset**: MENYO-20k is **CC BY-NC 4.0** (verified from the upstream
  `LICENSE`). Fine-tuning _on MENYO_ produces a non-commercial model; the
  resulting artifacts must not be used commercially or redistributed beyond the
  license terms. Attribution: Data and code by 'UD/NLP Semantics' group,
  University of Hamburg (menyo-20k_MT).
- Commercial deployment would require either an open dataset (e.g. FLORES-style
  data has different terms — check per-corpus) or explicit permission.

## 7. Recommendation

A **small controlled LoRA experiment is feasible** on this machine, but only
after the baseline is measured (Step 3 pending approval):

1. Scope: LoRA rank 8–16 on attention q/v, 1–3 epochs, 10,070 train pairs,
   dev split for early stopping, seq len ≤ 128.
2. Effort: ~3–6 h compute locally, or ~30 min on a rented GPU.
3. Gate: run only if the MENYO baseline shows a specific, addressable weakness
   (e.g. proverbs/news domain drop) — otherwise the baseline alone suffices.

Awaiting approval before any of: `pip install peft`, training runs, or any
further model downloads.
