from __future__ import annotations

from hashlib import sha256
from typing import Annotated

from pydantic import BaseModel, ConfigDict, Field, model_validator

from samples import PASS1_RECORD_EXAMPLE_JSON
from samples import PASS2_RECORD_EXAMPLE_JSON

NonEmptyString = Annotated[str, Field(min_length=1)]
Score = Annotated[int, Field(ge=1, le=5)]


class Pass1Record(BaseModel):
    model_config = ConfigDict(extra="forbid")

    id: Annotated[str, Field(pattern=r"^[0-9a-f]{64}$")]
    question: NonEmptyString

    @model_validator(mode="after")
    def validate_id(self) -> "Pass1Record":
        if self.id != sha256(self.question.encode("utf-8")).hexdigest():
            raise ValueError("id must equal sha256(question)")
        return self


class FollowUpQuestion(BaseModel):
    model_config = ConfigDict(extra="forbid")

    question: Annotated[
        NonEmptyString,
        Field(description="A clarifying follow-up question that asks for missing context."),
    ]
    thinking: Annotated[
        NonEmptyString,
        Field(description="A short explanation of why this follow-up question is useful."),
    ]
    weight: Annotated[
        Score,
        Field(description="Expected information gain from this follow-up question on a 1 to 5 scale."),
    ]


class Pass2Record(BaseModel):
    model_config = ConfigDict(extra="forbid")

    id: Annotated[str, Field(pattern=r"^[0-9a-f]{64}$")]
    question: NonEmptyString
    difficulty_level: Annotated[
        Score,
        Field(description="Difficulty of answering the question well from the raw question alone."),
    ]
    follow_ups: Annotated[
        list[FollowUpQuestion],
        Field(min_length=3, max_length=7),
    ]

    @model_validator(mode="after")
    def validate_follow_ups(self) -> "Pass2Record":
        normalized = [" ".join(item.question.lower().split()) for item in self.follow_ups]
        if len(normalized) != len(set(normalized)):
            raise ValueError("follow_ups must contain distinct questions")
        return self


def main() -> None:
    pass1_record = Pass1Record.model_validate_json(PASS1_RECORD_EXAMPLE_JSON)
    pass2_record = Pass2Record.model_validate_json(PASS2_RECORD_EXAMPLE_JSON)
    print(pass1_record.model_dump_json())
    print(pass2_record.model_dump_json(indent=2))


if __name__ == "__main__":
    main()
