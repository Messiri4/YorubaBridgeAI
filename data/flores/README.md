FLORES-200 evaluation subsets (official Meta release, CC-BY-SA 4.0).

Source archive: https://dl.fbaipublicfiles.com/nllb/flores200_dataset.tar.gz
dev/eng_Latn.dev -> dev.eng_Latn (first 60 lines)
dev/yor_Latn.dev -> dev.yor_Latn (first 60 lines)
devtest/eng_Latn.devtest -> devtest.eng_Latn (first 80 lines)
devtest/yor_Latn.devtest -> devtest.yor_Latn (first 80 lines)

The full FLORES-200 archive contains 997 dev + 1012 devtest aligned
sentences per language; we extracted prefixes sized for CPU evaluation.
Re-generate with the URLs above if you need larger splits.

License: CC-BY-SA 4.0 (dataset). Suitable for research/evaluation use.
Domain: Wikipedia-style formal text (news/wiki) — deliberately OUT of the
phrase-override table's chat/commerce domain, which is what makes it a
leakage-free held-out set for measuring the model itself.
