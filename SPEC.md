# Consumer Health QA Dataset Generator Pipeline — SPEC

## 1. Purpose and Goal

This project creates the dataset that can be used to improve LLM model's capability to ask clarifying questions. 
The final deliverable is a data generation pipeline that will produce two outputs: 

1. a canonical question inventory in JSONL; and  
    `all_consumer_health_questions.jsonl` 
2. a second JSONL where each question has clarification questions produced with Anthropic Claude Haiku 4.5 on Amazon Bedrock.
   - a difficulty rating from 1 to 5,
   - 3–7 high-information-gain follow-up questions,
   - a short explanation for why each follow-up question is useful,
   - a weight from 1 to 5 for the expected information gain.
    `all_consumer_health_questions_enriched.jsonl`

## 2. Pydantic models

```python
from hashlib import sha256
from typing import Annotated

from pydantic import BaseModel, ConfigDict, Field, model_validator


NonEmptyString = Annotated[str, Field(min_length=1)]
Score = Annotated[int, Field(ge=1, le=5)]


class Pass1Record(BaseModel):
    model_config = ConfigDict(extra="forbid")

    # id is the lowercase hex sha256 of question
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

    # id is copied from Pass1Record and remains the lowercase hex sha256 of question
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
```

Example `Pass1Record` JSON:

```json
{"id":"f544f10c9ece2f2e3b65f16e86e70cf8476a1004ce5de21a3ad78d7c28d0c313","question":"Do you have information about Seniors' Health"}
```

Example `Pass2Record` JSON:

```json
{
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
}
```

File-level pass-1 check: ensure `id` values are unique across the JSONL.

## 3. Two-pass process

## 3.1 Pass 1 — Build the unified question JSONL

- Extract question text from the datasets in section 6.
- Ignore answers and other metadata.
- Generate `id = sha256(question)`.
- Write one unique `Pass1Record` per line to `all_consumer_health_questions.jsonl`.
- Pass 1 is complete when the validation checks in section 2 pass.

Ignore answers, labels, rankings, retrieval context, and other metadata in pass 1.
After extracting the final question string, generate `id` as the lowercase hex `sha256(question)`.

## 3.2 Pass 2 — Generate structured annotations with Anthropic Claude Haiku 4.5 on Amazon Bedrock

- Read each `Pass1Record` from pass 1.
- Call Claude Haiku 4.5 on Bedrock with the question and ask for JSON only.
- Parse the response as `Pass2Record`.
- Append valid records to `all_consumer_health_questions_enriched.jsonl`.

## 4. Canonical process summary

1. Extract question text from the datasets
2. Generate `id = sha256(question)` and write pass 1 as JSONL using `Pass1Record`.
3. Parse pass-1 lines with the `Pass1Record` Pydantic model and ensure `id` values are unique.
4. For each line in pass-1 JSON, pass 2 with Claude Haiku 4.5 on Bedrock.
6. Accept only pass-2 records that parse through `Pass2Record`.
7. Write the final enriched JSONL to the file.


### Reference links

#### Dataset sources

- MedQuAD (official): <https://github.com/abachaa/MedQuAD>
- LiveQA Medical / TREC 2017 (official): <https://github.com/abachaa/LiveQA_MedicalTask_TREC2017>
- LiveQA Medical / Hugging Face mirror: <https://huggingface.co/datasets/hyesunyun/liveqa_medical_trec2017>
- MEDIQA 2019 Task 3 QA (official): <https://github.com/abachaa/MEDIQA2019/tree/master/MEDIQA_Task3_QA>
- MEDIQA QA / BigBio mirror: <https://huggingface.co/datasets/bigbio/mediqa_qa>
- MedicationQA (official workbook repo): <https://github.com/abachaa/Medication_QA_MedInfo2019>
- MedicationQA / Hugging Face mirror: <https://huggingface.co/datasets/truehealth/medicationqa>
- MASH-QA (official repo): <https://github.com/mingzhu0527/MASHQA>
- MASH-QA / Hugging Face mirror: <https://huggingface.co/datasets/ZappY-AI/MASHQA-JSON>
- MedRedQA paper: <https://aclanthology.org/2023.ijcnlp-main.42/>
- MedRedQA official data collection: <https://data.csiro.au/collection/csiro%3A62454>
- MedRedQA / Hugging Face mirror: <https://huggingface.co/datasets/varun500/medredqa>
- HealthSearchQA paper: <https://www.nature.com/articles/s41586-023-06291-2>
- HealthSearchQA official supplemental XLSX: <https://static-content.springer.com/esm/art%3A10.1038%2Fs41586-023-06291-2/MediaObjects/41586_2023_6291_MOESM6_ESM.xlsx>
- HealthSearchQA / Hugging Face mirror: <https://huggingface.co/datasets/katielink/healthsearchqa>

#### Question text 

| Dataset | Where to get it | Question text |
| --- | --- | --- |
| `medquad` | Official MedQuAD XML repository | Read `<Question>` inside each `<QAPair>`. |
| `liveqa_med` | Official TREC 2017 LiveQA Medical XML. A flattened Hugging Face mirror is acceptable for loading. | Use `SUBJECT + "\n\n" + MESSAGE` when both exist. Otherwise use the non-empty one. Fall back to `NLM-Summary` only if needed. |
| `mediqa_qa` | Official MEDIQA 2019 Task 3 QA XML, or a BigBio / Hugging Face question-level mirror. | Use `QuestionText` in the XML, or `question` in the mirror. |
| `medication_qa` | Official MedicationQA workbook. A Hugging Face mirror is acceptable for loading. | Use the `Question` column. |
| `mash_qa` | Official MASH-QA archive, or a compatible Hugging Face mirror. | Use the `question` field. |
| `medredqa` | Official MedRedQA data collection when available. A Hugging Face mirror is acceptable for loading. | Use `Title + "\n\n" + Body` when both exist. Otherwise use the non-empty one. |
| `health_search_qa` | Official HealthSearchQA supplemental XLSX, or a Hugging Face mirror. | Use the `question` field. |

### Model and platform references

- Anthropic models overview: <https://platform.claude.com/docs/en/about-claude/models/overview>
- AWS Bedrock model parameters for Anthropic Claude: <https://docs.aws.amazon.com/bedrock/latest/userguide/model-parameters-anthropic-claude-messages.html>
