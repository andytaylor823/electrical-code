"""Pydantic model for structured NEC table data.

Used as the LLM structured-output response format, the manual correction
workflow (Phase 3), and the merge script (Phase 4).  See docs/table_cleaning.md
for the full process and docs/table_ambiguity.md for the ambiguity analysis
that motivated this design.
"""

from pydantic import BaseModel, model_validator


class TableStructure(BaseModel):
    """Structured representation of an NEC table extracted from OCR fragments.

    The LLM returns this schema via structured output.  The model_validator
    guarantees that every data_row has exactly len(column_headers) cells,
    eliminating the column-count and boundary-sensitivity ambiguities that
    plagued the earlier procedural approach.

    This same schema is used by the manual correction workflow (Phase 3) and
    the merge script (Phase 4).  See docs/table_cleaning.md for the full
    process and docs/table_ambiguity.md for the ambiguity analysis that
    motivated this design.
    """

    title: str
    column_headers: list[str]
    data_rows: list[list[str]]
    footnotes: list[str]

    @model_validator(mode="after")
    def validate_row_widths(self) -> "TableStructure":
        """Ensure every data row has exactly len(column_headers) cells."""
        n_cols = len(self.column_headers)
        for i, row in enumerate(self.data_rows):
            if len(row) != n_cols:
                raise ValueError(f"Row {i} has {len(row)} cells, expected {n_cols} (matching column_headers)")
        return self
