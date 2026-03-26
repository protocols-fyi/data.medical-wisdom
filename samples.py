from __future__ import annotations


PASS1_RECORD_EXAMPLE_JSON = """{"id":"9fc12ccbb5d32d43ea566549274bb05319ac3a14d87931bebf7f10cab1891896","question":"How serious is atrial fibrillation?"}"""

PASS2_RECORD_EXAMPLE_JSON = """{
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
