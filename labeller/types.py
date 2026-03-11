"""
Data classes for speaker and modality metadata.
"""

from __future__ import annotations

import re
import warnings
from dataclasses import dataclass
from typing import Optional, Protocol, runtime_checkable

import numpy as np

from .config import (
    VALID_MODALITY,
    VALID_DIRECTION,
    VALID_VTL_CLASS,
    VALID_AGE,
    _AGE_CHILD_MAX,
    _AGE_TEEN_MAX,
)


@runtime_checkable
class FormantExtractor(Protocol):
    """
    Protocol for formant extractor backends.

    Any callable class that accepts (frame, sr, win_s, max_formant_hz, n_formants)
    and returns (freqs, bws, spectral_slope) satisfies this protocol.

    Both PraatFormantExtractor (labeller.formants) and TVLPFormantExtractor
    (tvlp) conform to this interface.
    """
    def __call__(
        self,
        frame: np.ndarray,
        sr: int,
        win_s: float,
        max_formant_hz: Optional[float] = None,
        n_formants: Optional[int] = None,
    ) -> tuple[np.ndarray, np.ndarray, float]: ...


@dataclass
class Modality:
    """
    Structured recording modality parsed from folder names.

    Folder name format:  Modality[_Register[_Direction[_Tag1[_Tag2...]]]]
    Examples:
        Spoken                   -> Modality("spoken", "M1", "exhale", frozenset())
        Sung_M2                  -> Modality("sung",   "M2", "exhale", frozenset())
        Sung_M2_Inhale           -> Modality("sung",   "M2", "inhale", frozenset())
        Sung_M3_Inhale_Pressed   -> Modality("sung",   "M3", "inhale", frozenset({"pressed"}))
        Sung_M1_Exhale_Pressed_Breathy
                                 -> Modality("sung",   "M1", "exhale", frozenset({"pressed","breathy"}))
    """
    modality:  str       # "spoken" | "sung"
    register:  str       # "M0", "M1", "M2", ... — voice register
    direction: str       # "exhale" | "inhale"
    tags:      frozenset # arbitrary lowercase annotation tags

    def __post_init__(self):
        if self.modality not in VALID_MODALITY:
            raise ValueError(f"modality={self.modality!r} not in {VALID_MODALITY}")
        if self.direction not in VALID_DIRECTION:
            raise ValueError(f"direction={self.direction!r} not in {VALID_DIRECTION}")
        object.__setattr__(self, "tags", frozenset(t.lower() for t in self.tags))

    @classmethod
    def from_path(cls, folder_name: str) -> "Modality":
        """
        Parse a modality folder name into a Modality instance with coercion.
        Unrecognised first token defaults to "spoken".
        Register defaults to "M1", direction to "exhale", tags to empty.
        """
        parts = folder_name.split("_")
        modality  = parts[0].lower()
        if modality not in VALID_MODALITY:
            warnings.warn(
                f"Unrecognised modality {parts[0]!r} in folder {folder_name!r}; "
                f"defaulting to 'spoken'.",
                stacklevel=2,
            )
            modality = "spoken"

        register  = "M1"
        direction = "exhale"
        tags      = []

        for part in parts[1:]:
            pl = part.lower()
            if re.match(r"^m\d+$", pl):
                register = part.upper()
            elif pl in VALID_DIRECTION:
                direction = pl
            else:
                tags.append(pl)

        return cls(modality=modality, register=register,
                   direction=direction, tags=frozenset(tags))

    def to_str(self) -> str:
        """Round-trip back to a canonical folder-name style string."""
        parts = [self.modality.capitalize(), self.register,
                 self.direction.capitalize()]
        parts += sorted(self.tags)
        return "_".join(parts)


@dataclass
class SpeakerMeta:
    """
    Per-speaker metadata used for VTL smoothing and demographic bookkeeping.

    vtl_class and vtl_prior_mm
    --------------------------
    vtl_class drives the group-level VTL prior (see VTL_PRIOR_MM).
    If vtl_prior_mm is set explicitly, the group-level blend is skipped
    entirely and the numeric value seeds the speaker-level smoother directly.
    Per-sample alpha still applies in both cases.

    age coercion
    ------------
    Numeric age is coerced to a label using thresholds defined in _AGE_CHILD_MAX
    and _AGE_TEEN_MAX:  <12 -> "child" | 12–15 -> "teen" | >=16 -> "adult".

    vtl_class coercion
    ------------------
    Numeric vtl_class (mm) is stored as vtl_prior_mm and vtl_class is set to
    the nearest named class; the numeric value takes precedence for smoothing.
    """
    vtl_class:    str         = "unknown"  # "small"|"medium"|"large"|"unknown"
    vtl_prior_mm: float|None  = None       # explicit mm override
    gender:       str         = "unknown"  # free-form: "woman","man","non-binary",...
    age:          str         = "unknown"  # "adult"|"teen"|"child"|"unknown"
    tags:         frozenset   = frozenset()# {"trans","voice_training","vfs",...}

    def __post_init__(self):
        # Coerce numeric age
        age = self.age
        if isinstance(age, (int, float)):
            if age < _AGE_CHILD_MAX:
                age = "child"
            elif age < _AGE_TEEN_MAX:
                age = "teen"
            else:
                age = "adult"
            object.__setattr__(self, "age", age)
        if self.age not in VALID_AGE:
            raise ValueError(f"age={self.age!r} not in {VALID_AGE}")

        # Coerce numeric vtl_class
        vtl_class = self.vtl_class
        if isinstance(vtl_class, (int, float)):
            mm = float(vtl_class)
            object.__setattr__(self, "vtl_prior_mm", mm)
            if mm < 135.0:
                vtl_class = "small"
            elif mm < 162.0:
                vtl_class = "medium"
            else:
                vtl_class = "large"
            object.__setattr__(self, "vtl_class", vtl_class)
        if self.vtl_class not in VALID_VTL_CLASS:
            raise ValueError(f"vtl_class={self.vtl_class!r} not in {VALID_VTL_CLASS}")

        object.__setattr__(self, "tags", frozenset(self.tags))
