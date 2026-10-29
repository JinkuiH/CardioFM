# Zero-shot prompt bank

Generated from `cardiofm/prompts.py`. Every class uses exactly four templates; the
class embedding is the mean of their L2-normalised text embeddings, re-normalised.

Modality prefixes (the trailing colon and space are part of the prefix):

- `NCCT` -> `Non-contrast coronary calcium CT: `
- `CCTA` -> `Coronary CT angiography: `

## `burden` -- Coronary calcium burden, 5-class ordinal (NCCT or CCTA)

Shown with the `NCCT` prefix.

**none**

- `Non-contrast coronary calcium CT: coronary artery calcification burden is none`
- `Non-contrast coronary calcium CT: none coronary calcium burden`
- `Non-contrast coronary calcium CT: the calcium score indicates none burden`
- `Non-contrast coronary calcium CT: overall calcification is none`

**minimal**

- `Non-contrast coronary calcium CT: coronary artery calcification burden is minimal`
- `Non-contrast coronary calcium CT: minimal coronary calcium burden`
- `Non-contrast coronary calcium CT: the calcium score indicates minimal burden`
- `Non-contrast coronary calcium CT: overall calcification is minimal`

**mild**

- `Non-contrast coronary calcium CT: coronary artery calcification burden is mild`
- `Non-contrast coronary calcium CT: mild coronary calcium burden`
- `Non-contrast coronary calcium CT: the calcium score indicates mild burden`
- `Non-contrast coronary calcium CT: overall calcification is mild`

**moderate**

- `Non-contrast coronary calcium CT: coronary artery calcification burden is moderate`
- `Non-contrast coronary calcium CT: moderate coronary calcium burden`
- `Non-contrast coronary calcium CT: the calcium score indicates moderate burden`
- `Non-contrast coronary calcium CT: overall calcification is moderate`

**severe**

- `Non-contrast coronary calcium CT: coronary artery calcification burden is severe`
- `Non-contrast coronary calcium CT: severe coronary calcium burden`
- `Non-contrast coronary calcium CT: the calcium score indicates severe burden`
- `Non-contrast coronary calcium CT: overall calcification is severe`


## `stenosis` -- Maximal coronary stenosis, 5-class ordinal (CCTA)

Shown with the `CCTA` prefix.

**none**

- `Coronary CT angiography: maximal coronary stenosis is none`
- `Coronary CT angiography: none narrowing of a coronary artery`
- `Coronary CT angiography: the most severe stenosis is none`
- `Coronary CT angiography: worst stenosis graded as none`

**minimal**

- `Coronary CT angiography: maximal coronary stenosis is minimal`
- `Coronary CT angiography: minimal narrowing of a coronary artery`
- `Coronary CT angiography: the most severe stenosis is minimal`
- `Coronary CT angiography: worst stenosis graded as minimal`

**mild**

- `Coronary CT angiography: maximal coronary stenosis is mild`
- `Coronary CT angiography: mild narrowing of a coronary artery`
- `Coronary CT angiography: the most severe stenosis is mild`
- `Coronary CT angiography: worst stenosis graded as mild`

**moderate**

- `Coronary CT angiography: maximal coronary stenosis is moderate`
- `Coronary CT angiography: moderate narrowing of a coronary artery`
- `Coronary CT angiography: the most severe stenosis is moderate`
- `Coronary CT angiography: worst stenosis graded as moderate`

**severe**

- `Coronary CT angiography: maximal coronary stenosis is severe`
- `Coronary CT angiography: severe narrowing of a coronary artery`
- `Coronary CT angiography: the most severe stenosis is severe`
- `Coronary CT angiography: worst stenosis graded as severe`


## `obstructive` -- Obstructive CAD, binary (CCTA)

Shown with the `CCTA` prefix.

**positive**

- `Coronary CT angiography: obstructive coronary artery disease`
- `Coronary CT angiography: hemodynamically significant coronary stenosis`
- `Coronary CT angiography: flow-limiting coronary disease`
- `Coronary CT angiography: significant obstructive disease`

**negative**

- `Coronary CT angiography: no obstructive coronary artery disease`
- `Coronary CT angiography: no significant coronary stenosis`
- `Coronary CT angiography: non-obstructive coronary disease`
- `Coronary CT angiography: no flow-limiting disease`


## `vessel` -- Vessel-level calcium, binary per vessel (NCCT)

Shown with the `NCCT` prefix.

**LM:present**

- `Non-contrast coronary calcium CT: calcification is present in the LM`
- `Non-contrast coronary calcium CT: calcified plaque in the LM`
- `Non-contrast coronary calcium CT: the LM shows calcification`
- `Non-contrast coronary calcium CT: LM calcification`

**LM:absent**

- `Non-contrast coronary calcium CT: no calcification in the LM`
- `Non-contrast coronary calcium CT: the LM is free of calcification`
- `Non-contrast coronary calcium CT: no calcified plaque in the LM`
- `Non-contrast coronary calcium CT: LM without calcium`

**LAD:present**

- `Non-contrast coronary calcium CT: calcification is present in the LAD`
- `Non-contrast coronary calcium CT: calcified plaque in the LAD`
- `Non-contrast coronary calcium CT: the LAD shows calcification`
- `Non-contrast coronary calcium CT: LAD calcification`

**LAD:absent**

- `Non-contrast coronary calcium CT: no calcification in the LAD`
- `Non-contrast coronary calcium CT: the LAD is free of calcification`
- `Non-contrast coronary calcium CT: no calcified plaque in the LAD`
- `Non-contrast coronary calcium CT: LAD without calcium`

**LCx:present**

- `Non-contrast coronary calcium CT: calcification is present in the LCx`
- `Non-contrast coronary calcium CT: calcified plaque in the LCx`
- `Non-contrast coronary calcium CT: the LCx shows calcification`
- `Non-contrast coronary calcium CT: LCx calcification`

**LCx:absent**

- `Non-contrast coronary calcium CT: no calcification in the LCx`
- `Non-contrast coronary calcium CT: the LCx is free of calcification`
- `Non-contrast coronary calcium CT: no calcified plaque in the LCx`
- `Non-contrast coronary calcium CT: LCx without calcium`

**RCA:present**

- `Non-contrast coronary calcium CT: calcification is present in the RCA`
- `Non-contrast coronary calcium CT: calcified plaque in the RCA`
- `Non-contrast coronary calcium CT: the RCA shows calcification`
- `Non-contrast coronary calcium CT: RCA calcification`

**RCA:absent**

- `Non-contrast coronary calcium CT: no calcification in the RCA`
- `Non-contrast coronary calcium CT: the RCA is free of calcification`
- `Non-contrast coronary calcium CT: no calcified plaque in the RCA`
- `Non-contrast coronary calcium CT: RCA without calcium`

**PDA:present**

- `Non-contrast coronary calcium CT: calcification is present in the PDA`
- `Non-contrast coronary calcium CT: calcified plaque in the PDA`
- `Non-contrast coronary calcium CT: the PDA shows calcification`
- `Non-contrast coronary calcium CT: PDA calcification`

**PDA:absent**

- `Non-contrast coronary calcium CT: no calcification in the PDA`
- `Non-contrast coronary calcium CT: the PDA is free of calcification`
- `Non-contrast coronary calcium CT: no calcified plaque in the PDA`
- `Non-contrast coronary calcium CT: PDA without calcium`


## `plaque` -- Plaque composition, two binaries (CCTA)

Shown with the `CCTA` prefix.

**has_calcified:present**

- `Coronary CT angiography: calcified coronary plaque is present`
- `Coronary CT angiography: there is calcified plaque`
- `Coronary CT angiography: calcified atherosclerotic plaque`
- `Coronary CT angiography: presence of calcified plaque`

**has_calcified:absent**

- `Coronary CT angiography: no calcified coronary plaque`
- `Coronary CT angiography: absence of calcified plaque`
- `Coronary CT angiography: no calcified atherosclerotic plaque`
- `Coronary CT angiography: calcified plaque is absent`

**has_noncalcified:present**

- `Coronary CT angiography: noncalcified coronary plaque is present`
- `Coronary CT angiography: there is noncalcified plaque`
- `Coronary CT angiography: noncalcified atherosclerotic plaque`
- `Coronary CT angiography: presence of noncalcified plaque`

**has_noncalcified:absent**

- `Coronary CT angiography: no noncalcified coronary plaque`
- `Coronary CT angiography: absence of noncalcified plaque`
- `Coronary CT angiography: no noncalcified atherosclerotic plaque`
- `Coronary CT angiography: noncalcified plaque is absent`

## Scoring

| task | class score | prediction |
|---|---|---|
| `burden`, `stenosis` | cosine similarity to each of the 5 class embeddings | arg-max; AUROC from the softmax |
| `obstructive`, `vessel`, `plaque` | `sim(positive) - sim(negative)` | threshold 0; **not** a calibrated probability |
