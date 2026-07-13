"""Standalone accuracy check for the health guardrail classifier.

Run with: python scripts/test_guardrail.py   (from backend/, with the venv active)
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from app.guardrail import check_health_related

# (prompt, expected_is_health_related)
CASES: list[tuple[str, bool]] = [
    # Obvious health
    ("I have a persistent headache and blurry vision for the past three days, what's wrong with me?", True),
    ("My child has a high fever of 39.5C and isn't eating, should I be worried?", True),
    ("I've been feeling extremely anxious and can't sleep for weeks, what should I do?", True),
    ("My knee hurts every time I go up stairs, especially in the morning.", True),
    ("I think I might have diabetes based on my symptoms, how do I treat it?", True),
    # Obvious non-health
    ("My laptop is broken and won't turn on.", False),
    ("My team keeps missing deadlines on our software project.", False),
    ("My car makes a weird grinding noise when I brake.", False),
    ("I can't figure out why my Python script keeps crashing with a segfault.", False),
    ("My small business isn't getting enough customers this quarter.", False),
    # Borderline (health mentioned as context, not the core problem)
    ("My employee has been calling in sick a lot lately and it's hurting our team's output.", False),
    ("My dog has been vomiting and won't eat, what should I do?", False),
    ("I'm exhausted all the time and can't focus at work — how do I fix my schedule?", False),
    ("Our nurse staffing software keeps double-booking shifts at the clinic.", False),
]


def main() -> int:
    failures = 0
    print(f"{'expected':<10} {'actual':<10} {'provider':<12} prompt")
    print("-" * 100)
    for prompt, expected in CASES:
        result = check_health_related(prompt)
        ok = result.is_health_related == expected
        if not ok:
            failures += 1
        status = "PASS" if ok else "FAIL"
        print(
            f"{str(expected):<10} {str(result.is_health_related):<10} {result.provider:<12} "
            f"[{status}] {prompt[:60]}"
        )
        if not ok:
            print(f"           -> reason given: {result.reason}")

    print("-" * 100)
    print(f"{len(CASES) - failures}/{len(CASES)} passed")
    return 1 if failures else 0


if __name__ == "__main__":
    raise SystemExit(main())
