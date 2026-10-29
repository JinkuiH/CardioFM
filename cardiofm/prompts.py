"""Zero-shot prompt bank.

Every class of every task uses exactly four templates. The final prompt is the
modality prefix concatenated directly with the template body -- the prefix ends
in a colon and a single space, and nothing further is inserted.

Scoring is not uniform across tasks:
  * ordinal tasks (burden, stenosis) -> cosine similarity to each of the five
    class embeddings; prediction is the arg-max, AUROC from the softmax.
  * binary tasks (obstructive, vessel, plaque) -> sim(positive) - sim(negative);
    the difference is the score and the threshold is 0. It is NOT a probability.
"""
from __future__ import annotations

from typing import Dict, List, Sequence

MODALITY_PREFIX: Dict[str, str] = {
    "NCCT": "Non-contrast coronary calcium CT: ",
    "CCTA": "Coronary CT angiography: ",
}

BURDEN_ORDER: Sequence[str] = ("none", "minimal", "mild", "moderate", "severe")
STENOSIS_ORDER: Sequence[str] = ("none", "minimal", "mild", "moderate", "severe")
VESSELS: Sequence[str] = ("LM", "LAD", "LCx", "RCA", "PDA")

#: Agatston cut-points behind ``BURDEN_ORDER``.
AGATSTON_BINS = ("0", "1-10", "11-100", "101-400", ">400")

TASKS = {
    "burden": "Coronary calcium burden, 5-class ordinal (NCCT or CCTA)",
    "stenosis": "Maximal coronary stenosis, 5-class ordinal (CCTA)",
    "obstructive": "Obstructive CAD, binary (CCTA)",
    "vessel": "Vessel-level calcium, binary per vessel (NCCT)",
    "plaque": "Plaque composition, two binaries (CCTA)",
}


def _bodies(task: str, class_name: str) -> List[str]:
    if task == "burden":
        return [f"coronary artery calcification burden is {class_name}",
                f"{class_name} coronary calcium burden",
                f"the calcium score indicates {class_name} burden",
                f"overall calcification is {class_name}"]
    if task == "stenosis":
        return [f"maximal coronary stenosis is {class_name}",
                f"{class_name} narrowing of a coronary artery",
                f"the most severe stenosis is {class_name}",
                f"worst stenosis graded as {class_name}"]
    if task == "obstructive":
        if class_name == "positive":
            return ["obstructive coronary artery disease",
                    "hemodynamically significant coronary stenosis",
                    "flow-limiting coronary disease",
                    "significant obstructive disease"]
        return ["no obstructive coronary artery disease",
                "no significant coronary stenosis",
                "non-obstructive coronary disease",
                "no flow-limiting disease"]
    if task == "vessel":
        vessel, present = class_name.split(":")
        if present == "present":
            return [f"calcification is present in the {vessel}",
                    f"calcified plaque in the {vessel}",
                    f"the {vessel} shows calcification",
                    f"{vessel} calcification"]
        return [f"no calcification in the {vessel}",
                f"the {vessel} is free of calcification",
                f"no calcified plaque in the {vessel}",
                f"{vessel} without calcium"]
    if task == "plaque":
        kind, present = class_name.split(":")
        word = "calcified" if kind == "has_calcified" else "noncalcified"
        if present == "present":
            return [f"{word} coronary plaque is present",
                    f"there is {word} plaque",
                    f"{word} atherosclerotic plaque",
                    f"presence of {word} plaque"]
        return [f"no {word} coronary plaque",
                f"absence of {word} plaque",
                f"no {word} atherosclerotic plaque",
                f"{word} plaque is absent"]
    raise ValueError(f"unknown task {task!r}; expected one of {sorted(TASKS)}")


def templates(task: str, class_name: str, modality: str) -> List[str]:
    """The four prompts for one class of one task, with the modality prefix applied."""
    if modality not in MODALITY_PREFIX:
        raise ValueError(f"unknown modality {modality!r}; expected NCCT or CCTA")
    return [MODALITY_PREFIX[modality] + b for b in _bodies(task, class_name)]


def class_names(task: str) -> List[str]:
    """The class names a task is scored over, in label order."""
    if task == "burden":
        return list(BURDEN_ORDER)
    if task == "stenosis":
        return list(STENOSIS_ORDER)
    if task == "obstructive":
        return ["positive", "negative"]
    if task == "vessel":
        return [f"{v}:{p}" for v in VESSELS for p in ("present", "absent")]
    if task == "plaque":
        return [f"{k}:{p}" for k in ("has_calcified", "has_noncalcified")
                for p in ("present", "absent")]
    raise ValueError(f"unknown task {task!r}")


def all_prompts(task: str, modality: str) -> Dict[str, List[str]]:
    """``{class_name: [4 prompts]}`` for a whole task -- handy for inspection."""
    return {c: templates(task, c, modality) for c in class_names(task)}
