"""Run a fixed question set through the new agent for parity review.

Prints each question, the rewritten standalone question, the retrieved source
tags, and the grounded answer. Output is for manual side-by-side comparison
against the live Streamlit chat (pages/4_Chat.py), whose chain is Streamlit-bound
and not run headless here.

Run:  python api/scripts/parity_eval.py
Requires ingestion to have run first (python api/scripts/ingest.py).
"""
from __future__ import annotations

import sys

from portfolio_api.agent import answer

# ~15 questions spanning the interview_qa coverage (intro, WPP, TEV, Cloudserve,
# publications, education, behavioral, plus an out-of-scope refusal check).
QUESTIONS = [
    "Tell me about yourself.",
    "What are you working on currently?",
    "What kind of problems do you most enjoy working on?",
    "What did you build at WPP Media?",
    "Tell me about the SQL compiler work at WPP Media.",
    "How did you handle scale and guardrails at WPP Media?",
    "What did you do at Third Estate Ventures?",
    "Describe your computer vision and image annotation work.",
    "What did you work on at Cloudserve?",
    "Tell me about the BERT summarization and NER project.",
    "What are your strongest technical skills?",
    "Tell me about your publications.",
    "What is your educational background?",
    "Tell me about a time you reduced manual work for a team.",
    "What is the capital of France?",  # out-of-scope: should refuse / stay grounded
]


def main() -> int:
    print(f"Running parity eval over {len(QUESTIONS)} questions\n")
    for i, q in enumerate(QUESTIONS, 1):
        result = answer(q)
        print("=" * 80)
        print(f"Q{i}: {q}")
        if result.rewritten_question and result.rewritten_question != q:
            print(f"  rewritten: {result.rewritten_question}")
        print(f"  sources:   {', '.join(result.sources) if result.sources else '(none)'}")
        print(f"\n{result.answer}\n")
    print("=" * 80)
    return 0


if __name__ == "__main__":
    sys.exit(main())
