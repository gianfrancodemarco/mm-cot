from enum import Enum

# These have been taken by a comment https://github.com/entitize/Fakeddit/issues/14
# Check if it is truthful


class LabelsTypes(Enum):
    TWO_WAY = "TWO_WAY"
    THREE_WAY = "THREE_WAY"
    SIX_WAY = "SIX_WAY"


class TwoWayLabels(Enum):
    TRUE: 1
    FALSE: 0


class ThreeWayLabels(Enum):
    TRUE: 0
    FAKE_WITH_TRUE_TEXT: 1
    FAKE_WITH_FALSE_TEXT: 0


class SixWayLabels(Enum):
    TRUE: 0
    SATIRE: 1
    FALSE_CONNECTION: 2
    IMPOSTER_CONTENT: 3
    MANIPULATED_CONTENT: 4
    MISLEADING_CONTENT: 5


def get_options_text(labels_type: LabelsTypes) -> str:
    if labels_type == LabelsTypes.TWO_WAY:
        return "(A) True (B) False"
    if labels_type == LabelsTypes.THREE_WAY:
        return "(A) True (B) Fake with true text (C) Fake with false text"
    if labels_type == LabelsTypes.SIX_WAY:
        return "(A) True (B) Satire (C) False connection (D) Imposter content (E) Manipulated content (F) Misleading content"


def get_label_column(labels_type: LabelsTypes) -> str:
    if labels_type == LabelsTypes.TWO_WAY:
        return "2_way_label"
    if labels_type == LabelsTypes.THREE_WAY:
        return "3_way_label"
    if labels_type == LabelsTypes.SIX_WAY:
        return "6_way_label"


def convert_int_to_label(label: str) -> int:
    
    _map = {
        0: "A",
        1: "B",
        2: "C",
        3: "D",
        4: "E",
        5: "F"
    }

    return _map.get(label)


def get_label_text(label: str) -> str:
    return f"The answer is ({label})"