# Sprint 2 — Failure Analysis (numerical & placeholder restoration)

Artifact: `data/eval/integration_comparison_20261110T102741Z.csv` · pairs: 80 · split: devtest

## Artifact-evidence limitations (stated explicitly)

- The comparison CSV records per-sentence predictions and digit counts only;
  it does **NOT** record retry flags or placeholder traces, so per-row retry
  attribution below is marked NOT RECORDED rather than inferred.
- Run-log evidence (/tmp/integ_cmp_80.log): 22 restoration-failure retries logged (counts only; no per-row attribution).
- Digit-count mismatch is NOT treated as failure by itself: legitimate
  spelled-out renderings ("4" → "mẹ́rin") are classified as
  `rendered_as_words`. Only sources whose digit expressions are neither
  verbatim nor attested-rendered count as `dropped`.
- Meaning-change judgement is left for human linguistic review; this
  script does not claim semantic equivalence or difference.

## Automatic status summary (per direction, both arms)

| classification | baseline | integrated |
|---|---|---|
| no_numbers | 124 | 124 |
| preserved | 25 | 28 |
| rendered_as_words | 4 | 4 |
| dropped | 7 | 4 |

## Degeneration loops (separate failure class)

Sentences (either arm) showing a ≥6-token
consecutive repetition loop. These destroy sentence content wholesale,
often including numbers; they are decoding failures, not protection
failures, and are listed in the case table with their own status.


## Confirmed numerical-preservation failure cases

Cases where either arm dropped a numeric expression: **7**.

### eng_yor #2

- Source (eng): Like some other experts, he is skeptical about whether diabetes can be cured, noting that these findings have no relevance to people who already have Type 1 diabetes.
- Baseline prediction: Bíi ti àwọn ògbógi mìíràn, ó máa ń ṣiyèméjì bóyá a lè wo àrùn àtọ̀gbẹ sàn, ó sì sọ pé àwọn àbá yìí kò kan àwọn tó ti ní àrùn àtọ̀gbẹ Oríṣi Kìíní.
- Integrated prediction: Bíi ti àwọn ògbógi mìíràn, ó ń ṣiyèméjì nípa bóyá a lè wo àrùn àtọ̀gbẹ sàn, ó sọ pé àwọn ìwádìí yìí kò ní ìtumọ̀ kankan fún àwọn tó ti ní àrùn àtọ̀gbẹ Iru 1.
- Original numeric expressions: 1
- Observed (baseline): — — status: dropped
- Observed (integrated): 1 — status: preserved
- Dropped: baseline=['1']; integrated=none
- Retry occurred: NOT RECORDED in artifact
- Meaning changed: _pending human linguistic review_
- Probable mechanism: see §Mechanisms (analysis below); per-case attribution
  requires the placeholder traces the artifact does not store.

### eng_yor #50

- Source (eng): The Iraq Study Group presented its report at 12.00 GMT today.
- Baseline prediction: Ẹgbẹ́ Ìkẹ́kọ̀ọ́ Nílẹ̀ Iraq gbé ìròyìn rẹ̀ jáde ní aago méjìlá wákàtí GMT lónìí.
- Integrated prediction: Ẹgbẹ́ Ìwádìí Iraq gbé ìròyìn rẹ̀ jáde ní 12.00 GMT lónìí.
- Original numeric expressions: 12.00
- Observed (baseline): — — status: dropped
- Observed (integrated): 12.00 — status: preserved
- Dropped: baseline=['12.00']; integrated=none
- Retry occurred: NOT RECORDED in artifact
- Meaning changed: _pending human linguistic review_
- Probable mechanism: see §Mechanisms (analysis below); per-case attribution
  requires the placeholder traces the artifact does not store.

### eng_yor #55

- Source (eng): Current senator and Argentine First Lady Cristina Fernandez de Kirchner announced her presidential candidacy yesterday evening in La Plata, a city 50 kilometers (31 miles) away from Buenos Aires.
- Baseline prediction: Sànhẹ́dátì tó wà nísinsìnyí, tó sì tún jẹ́ obìnrin àkọ́kọ́ ní orílẹ̀-èdè Ajẹ́ntínà, Cristina Fernandez de Kirchner, kéde ìbò rẹ̀ gẹ́gẹ́ bí ààrẹ ní alẹ́ ọjọ́ òní ní ìlú La Plata, ìlú kan tó wà ní nǹkan bí àádọ́ta kìlómítà sí ìlú Buenos Aires.
- Integrated prediction: Sànhẹ́dátì tó wà nísinsìnyí, tó sì tún jẹ́ obìnrin àkọ́kọ́ ní orílẹ̀-èdè Ajẹ́ntínà, Cristina Fernandez de Kirchner, kéde ìbò rẹ̀ gẹ́gẹ́ bí ààrẹ ní alẹ́ ọjọ́ òní ní ìlú La Plata, ìlú kan tó wà ní nǹkan bí àádọ́ta kìlómítà sí ìlú Buenos Aires.
- Original numeric expressions: 50, 31
- Observed (baseline): — — status: dropped
- Observed (integrated): — — status: dropped
- Dropped: baseline=['31']; integrated=['31']
- Retry occurred: NOT RECORDED in artifact
- Meaning changed: _pending human linguistic review_
- Probable mechanism: see §Mechanisms (analysis below); per-case attribution
  requires the placeholder traces the artifact does not store.

### eng_yor #65

- Source (eng): Robin Uthappa made the innings highest score, 70 runs in just 41 balls by hitting 11 fours and 2 sixes.
- Baseline prediction: Robin Uthappa ló gba ìsọfúnni tó ga jù lọ nínú ìsọfúnni náà, ó ṣe ìsọfúnni àádọ́rin nínú ìsọfúnni mẹ́rìndínláàádọ́rin péré nípa fífi ìsọfúnni mẹ́rìndínláàádọ́rin àti ìsọfúnni mẹ́rìndínláàádọ́ta àti ìsọfúnni mẹ́rìndínláàádọ́ta.
- Integrated prediction: Robin Uthappa ṣe akọọlẹ ti o ga julọ, 70 run ni awọn bọọlu 41 nikan nipa titẹ 11 mẹrin ati 2 mẹfa.
- Original numeric expressions: 70, 41, 11, 2
- Observed (baseline): — — status: dropped
- Observed (integrated): 70, 41, 11, 2 — status: preserved
- Dropped: baseline=['41', '11', '2']; integrated=none
- Retry occurred: NOT RECORDED in artifact
- Meaning changed: _pending human linguistic review_
- Probable mechanism: see §Mechanisms (analysis below); per-case attribution
  requires the placeholder traces the artifact does not store.

### eng_yor #67

- Source (eng): But, after losing the captain's wicket India only made 36 runs loosing 7 wickets to end the innings.
- Baseline prediction: àmọ́, lẹ́yìn tí wọ́n pàdánù ìkọ́ ìkọ́ ìkọ́ ìkọ́ ìkọ́ ìkọ́ ìkọ́ ìkọ́ ìkọ́ ìkọ́ ìkọ́ ìkọ́ ìkọ́ ìkọ́ ìkọ́ ìkọ́ ìkọ́ ìkọ́ ìkọ́ ìkọ́ ìkọ́ ìkọ́ ìkọ́ ìkọ́ ìkọ́ ìkọ́ ìkọ́ ìkọ́ ìkọ́ ìkọ́ ìkọ́ ìkọ́ ìkọ́ ìkọ́ ìkọ́ ìkọ́ ìkọ́ ìkọ́ ìkọ́ ìkọ́ ìkọ́ ìkọ́ ìkọ́ ìkọ́ ìkọ́ ìkọ́ ìkọ́ ìkọ́ ìkọ́ ìkọ́ ìkọ́ ìkọ́ ìkọ́ ìkọ́ ìkọ́ ìkọ́ ìkọ́ ìkọ́ ìkọ́ ìkọ́ ìkọ́ ìkọ́ ìkọ́ ìkọ́ ìkọ́ ìkọ́ ìkọ́ ìkọ́ ìkọ́ ìkọ́ ìkọ́ ìkọ́ ìkọ́ ìkọ́ ìkọ́ ìkọ́ ìkọ́ ìkọ́ ìkọ́ ìkọ́ ìkọ
- Integrated prediction: àmọ́, lẹ́yìn tí wọ́n pàdánù ìkọ́ ìkọ́ ìkọ́ ìkọ́ ìkọ́ ìkọ́ ìkọ́ ìkọ́ ìkọ́ ìkọ́ ìkọ́ ìkọ́ ìkọ́ ìkọ́ ìkọ́ ìkọ́ ìkọ́ ìkọ́ ìkọ́ ìkọ́ ìkọ́ ìkọ́ ìkọ́ ìkọ́ ìkọ́ ìkọ́ ìkọ́ ìkọ́ ìkọ́ ìkọ́ ìkọ́ ìkọ́ ìkọ́ ìkọ́ ìkọ́ ìkọ́ ìkọ́ ìkọ́ ìkọ́ ìkọ́ ìkọ́ ìkọ́ ìkọ́ ìkọ́ ìkọ́ ìkọ́ ìkọ́ ìkọ́ ìkọ́ ìkọ́ ìkọ́ ìkọ́ ìkọ́ ìkọ́ ìkọ́ ìkọ́ ìkọ́ ìkọ́ ìkọ́ ìkọ́ ìkọ́ ìkọ́ ìkọ́ ìkọ́ ìkọ́ ìkọ́ ìkọ́ ìkọ́ ìkọ́ ìkọ́ ìkọ́ ìkọ́ ìkọ́ ìkọ́ ìkọ́ ìkọ́ ìkọ́ ìkọ́ ìkọ́ ìkọ́ ìkọ
- Original numeric expressions: 36, 7
- Observed (baseline): — — status: degeneration_loop
- Observed (integrated): — — status: degeneration_loop
- Dropped: baseline=['36', '7']; integrated=['36', '7']
- Retry occurred: NOT RECORDED in artifact
- Meaning changed: _pending human linguistic review_
- Probable mechanism: see §Mechanisms (analysis below); per-case attribution
  requires the placeholder traces the artifact does not store.

### eng_yor #71

- Source (eng): Prime Minister Stephen Harper has agreed to send the government's 'Clean Air Act' to an all-party committee for review, before its second reading, after Tuesday's 25 minute meeting with NDP leader Jack Layton at the PMO.
- Baseline prediction: Ààrẹ Stephen Harper ti gbà láti ránṣẹ́ 'Ìlànà Ẹ̀fúùfù Mímọ́' tí ìjọba gbé kalẹ̀ sí ìgbìmọ̀ gbogbo ẹgbẹ́ olóṣèlú fún àgbéyẹ̀wò, kí ó tó parí ìkẹ́kọ̀ọ́ kejì rẹ̀, lẹ́yìn ìpàdé ìṣẹ́jú mẹ́ẹ̀ẹ́dọ́gbọ̀n tó wáyé pẹ̀lú olùdarí NDP, Jack Layton ní PMO ní ọjọ́ Tuesday.
- Integrated prediction: Ààrẹ Stephen Harper ti gbà láti ránṣẹ́ 'Ìlànà Ẹ̀fúùfù Mímọ́' tí ìjọba gbé kalẹ̀ sí ìgbìmọ̀ gbogbo ẹgbẹ́ olóṣèlú fún àgbéyẹ̀wò, kí ó tó parí ìkẹ́kọ̀ọ́ kejì rẹ̀, lẹ́yìn ìpàdé ìṣẹ́jú mẹ́ẹ̀ẹ́dọ́gbọ̀n tó wáyé pẹ̀lú olùdarí NDP, Jack Layton ní PMO ní ọjọ́ Tuesday.
- Original numeric expressions: 25
- Observed (baseline): — — status: dropped
- Observed (integrated): — — status: dropped
- Dropped: baseline=['25']; integrated=['25']
- Retry occurred: NOT RECORDED in artifact
- Meaning changed: _pending human linguistic review_
- Probable mechanism: see §Mechanisms (analysis below); per-case attribution
  requires the placeholder traces the artifact does not store.

### eng_yor #74

- Source (eng): But Prime Minister John Howard has said the act was only to safeguard the facilities of the hospital from being downgraded by the Tasmanian government, in giving an extra AUD$45 million.
- Baseline prediction: ṣùgbọ́n, ààrẹ john howard sọ pé ohun tí wọ́n ṣe ni láti dáàbò bo ilé ìwòsàn náà kí ìjọba tasmanian má bàa sọ ọ́ di ẹni tí kò dára, ó sì fún un ní àfikún àádọ́rin ó lé ní mílíọ̀nù dọ́là ilẹ̀ australia.
- Integrated prediction: ṣùgbọ́n, ààrẹ john howard sọ pé ohun tí wọ́n ṣe ni láti dáàbò bo ilé ìwòsàn náà kí ìjọba tasmanian má bàa sọ ọ́ di ẹni tí kò dára, ó sì fún un ní àfikún àádọ́rin ó lé ní mílíọ̀nù dọ́là ilẹ̀ australia.
- Original numeric expressions: 45
- Observed (baseline): — — status: dropped
- Observed (integrated): — — status: dropped
- Dropped: baseline=['45']; integrated=['45']
- Retry occurred: NOT RECORDED in artifact
- Meaning changed: _pending human linguistic review_
- Probable mechanism: see §Mechanisms (analysis below); per-case attribution
  requires the placeholder traces the artifact does not store.

## Mechanism hypotheses (Phase 2 linkage; hypotheses, not proven causes)

1. Sentinel dropped by generation: NLLB omits `@@i@@` (or merges its `@`
   tokens with adjacent punctuation), so restoration verification fails
   (`missing`/`count_mismatch`) and the unprotected retry regenerates —
   the retry may itself drop the number, which surfaces as a `dropped`
   case in the integrated arm.
2. Spelled-out rendering pressure: even with sentinels, the decoder may
   translate the surrounding sentence and leave the sentinel untranslated
   or normalize digits to words for fluent output.
3. Reformatting: comma separators and date/time formats are rewritten by
   the model (e.g. `9,000` → `9000` or `12 noon` paraphrases), which
   counts as changed although semantically close; flagged for review.
4. Baseline weakness: without sentinels the model freely drops digits;
   the integrated arm reduces but does not eliminate this class.

## What would confirm mechanisms

- A small instrumented re-run recording per-row: protected input string,
  raw model output, restoration outcome kind, retry occurrence, final text.
- Tokenizer fragmentation probe of `@@i@@` (see
  `scripts/probe_placeholder_survival.py`).
- Per-case linguistic review to fill the meaning-changed column.

