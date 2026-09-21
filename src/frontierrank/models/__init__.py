"""Model adapters. `protocols` is the only thing the rest of the package imports."""
from .protocols import (RUBRIC_4LEVEL, ScorerProtocol, SlateRequest,
                        TeacherProtocol, TeacherVerdict, expected_grade)

__all__ = ["ScorerProtocol", "TeacherProtocol", "TeacherVerdict", "SlateRequest",
           "RUBRIC_4LEVEL", "expected_grade"]
