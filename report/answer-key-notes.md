# Notes on the historical five-item sensitivity analysis

Reviewed on 15 November 2026. These notes document a new, source-based review of the five Phase 2 question IDs already listed in `analyze_all.py` as `FLAGGED`: 4800, 11671, 12048, 11204 and 6657. The released repository contains the flag list, questions, arguments and model responses, but no contemporaneous per-item explanation of why the author selected these five. The explanations below are therefore reconstructed assessments, not recovered historical author reasoning.

The main analysis retains the original benchmark keys and all five items. The existing five-item exclusion is an **exploratory sensitivity analysis using a historical flag list**, not a corrected benchmark, a preregistered exclusion rule, or evidence that all five keys are wrong. In particular, the concern about 6657 could not be substantiated. Model disagreement with a key is not itself evidence that the key is incorrect, and the argument-support audit did not validate answer keys or factual reasoning.

The source questions are in [`hard/questions.jsonl`](../hard/questions.jsonl); the original L3 texts are in [`hard/arguments.jsonl`](../hard/arguments.jsonl). Question IDs, rather than line positions, identify the records.

| Question ID | Category | Released key | Assessment from this review |
| --- | --- | --- | --- |
| 4800 | History | I | Strong directional contradiction; J fits the quoted passage better |
| 11671 | Engineering | E | Malformed or underspecified item; keyed expression is dimensionally inconsistent |
| 12048 | Engineering | J | Key conflicts with the standard ideal boost-converter calculation; D gives 0.27 |
| 11204 | Philosophy | C | Broad, ambiguous cultural claim; no unique replacement key established |
| 6657 | Health | C | Flag not substantiated; the keyed organismic perspective is defensible |

## 4800: Treaty of Brest-Litovsk

The question quotes Article III's removal of Russian sovereignty over territory west of the agreed line. The key, I, describes the eastern part of the Russian Empire; J describes its western part. The primary treaty text reproduced by Yale's Avalon Project confirms the westward wording and places the future status of these territories under arrangements involving Germany and Austria-Hungary. This supports a strong concern about the key's reversed direction. J is the better match among the supplied options, although its description as a surrender to Germany simplifies the treaty's wording. The passage also labels the treaty 14 March 1918, whereas the primary text is dated 3 March 1918. [Treaty text, Articles III–IV](https://avalon.law.yale.edu/20th_century/brest.asp).

The original key-target argument discusses Germany's eastern sphere and then calls the same territory the eastern part of Russia. That shift in geographical reference does not resolve the contradiction. This observation concerns the released argument, not a known historical reason for flagging the item.

## 11671: Steady inviscid flow

The prompt asks for velocity in terms of pressure, with `u = u(x)` and `v = w = 0`. The keyed expression is `u = sqrt((2gz - rho)/rho)`. Under the usual dimensional meanings of gravitational acceleration, height and density, `gz` has units of velocity squared while `rho` has units of mass per volume: these quantities cannot be subtracted. The key also contains no pressure variable.

Even with the extra assumptions of constant density and negligible height change, the standard Bernoulli relation gives `u = sqrt(2(p_total - p)/rho)` and requires a reference total pressure. Those assumptions and reference data are not supplied. NASA states the relation and its restrictions explicitly. [NASA, Bernoulli's equation](https://www1.grc.nasa.gov/beginners-guide-to-aeronautics/bernoullis-equation/).

This is strong evidence of a malformed or incomplete item, but not a basis for assigning another option as the corrected answer. The original argument's claim that pressure can be absorbed into a reference constant does not justify the dimensionally inconsistent subtraction.

## 12048: Step-up chopper duty cycle

For the standard ideal boost-converter interpretation, `D = 1 - Vin/Vout = 1 - 110/150 = 0.2667`, which rounds to option D, 0.27. The released key is J, 0.67. Texas Instruments gives the boost duty-cycle relation with an efficiency factor; setting efficiency to one recovers this ideal calculation. The question supplies no losses or other circuit conditions that justify the keyed value. [TI, *Basic Calculation of a Boost Converter's Power Stage*, equation 1](https://www.ti.com/lit/an/slva372d/slva372d.pdf).

The original key-target argument asserts a two-thirds duty cycle without a calculation that yields it. This supports treating the item as a likely key error under the conventional ideal interpretation. It does not alter the released key or rescore the original experiment.

## 11204: A fundamental theme of Japanese culture

The question asks for one of the most fundamental themes of Japanese culture and civilization, without naming a period, tradition, author or source. The keyed option C describes an eternal and merciful God. That broad wording does not map uniquely to the range of traditions invoked by the original argument. For example, Meiji Jingu describes many kami and emphasizes harmony with nature; Nichiren Shu explains Buddhist teaching in terms of karma. These are primary descriptions by religious institutions, not a ranking of themes for all Japanese culture. [Meiji Jingu, About Shinto](https://www.meijijingu.or.jp/en/about/aboutshinto/); [Nichiren Shu, What is Buddhism?](https://www.nichiren.or.jp/english/teachings/buddhism/).

The reconstructed concern is therefore ambiguity and insufficient context. The sources do not establish a unique replacement option, nor prove that the keyed statement is incompatible with every Japanese religious tradition. This flag has weaker support than the explicit contradictions in 4800 and 12048.

## 6657: Active control of development and aging

The prompt describes people as mostly in control of their development and their behavior as shaping how successfully they age. The released key C is the organismic perspective. The original key-target argument interprets that perspective in terms of active agency. That interpretation is compatible with Deci, Ryan and Williams's account of organismic models, in which individuals actively engage with their world and shape development. [Deci, Ryan and Williams (1996), *Need Satisfaction and the Self-Regulation of Learning*, p. 167](https://selfdeterminationtheory.org/SDT/documents/1996_DeciRyanWilliams_LID.pdf).

This source is not the provenance of the exam question and does not adjudicate its exact textbook taxonomy. It does, however, undermine a claim that the key is plainly erroneous. This review found no defensible basis to call 6657 an established answer-key error. It remains in the historical five-item sensitivity only to make that previously reported calculation traceable; any future exclusion policy should be independently specified and adjudicated.
