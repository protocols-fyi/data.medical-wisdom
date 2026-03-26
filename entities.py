from __future__ import annotations

from hashlib import sha256
from typing import Annotated

from pydantic import BaseModel, ConfigDict, Field, model_validator

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

    question: NonEmptyString
    thinking: NonEmptyString
    weight: Score


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
    pass1_record_example_json = """{"id":"9fc12ccbb5d32d43ea566549274bb05319ac3a14d87931bebf7f10cab1891896","question":"How serious is atrial fibrillation?"}"""
    pass2_record_example_json = """{
  "id": "9fc12ccbb5d32d43ea566549274bb05319ac3a14d87931bebf7f10cab1891896",
  "question": "How serious is atrial fibrillation?",
  "difficulty_level": 2,
  "follow_ups": [
    {
      "question": "Is this about you or someone else right now, and are there any red-flag symptoms such as chest pain, severe shortness of breath, fainting, confusion, weakness or numbness, or trouble speaking or seeing?",
      "thinking": "This separates a possible emergency from a routine information request. Urgency changes the next step entirely, so it has the highest information gain.",
      "weight": 5
    },
    {
      "question": "Has atrial fibrillation already been diagnosed, or are you asking because of a new irregular heartbeat or new symptoms?",
      "thinking": "This distinguishes a general education question from a new possible diagnosis. The answer changes what context is missing and what follow-up matters most.",
      "weight": 4
    },
    {
      "question": "What other health conditions, medications, or blood thinners are involved?",
      "thinking": "Comorbidities and current treatment strongly affect risk framing and what information would be most useful next.",
      "weight": 5
    }
  ]
}"""
    pass1_record = Pass1Record.model_validate_json(pass1_record_example_json)
    pass2_record = Pass2Record.model_validate_json(pass2_record_example_json)
    print(pass1_record.model_dump_json())
    print(pass2_record.model_dump_json(indent=2))


if __name__ == "__main__":
    main()
