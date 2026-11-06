"""Answer-scoring rules shared by the evaluator and the analysis.

`evaluate.py` scores rows as it writes them; `analyze.py` re-scores the saved
answers so older runs and newer runs are compared under the same rule. Both
import from here so the rule cannot drift between the two.
"""

import re

# Letter at the very start of a reply, allowing leading punctuation/markdown.
# The letter must end the reply or be followed by punctuation or a space and a
# capital/digit, so the article in "A survey question" is not read as option A.
LEADING_LETTER = re.compile(r"^\W*([A-Z])(?:$|[.):,*]|\s+(?=[A-Z0-9]))")


def leading_letter(answer_text, letters):
    """Option letter that a first-token answer starts with, or None.

    Lenient: text may follow the letter ("E. Strongly disagree"), and leading
    punctuation or markdown is skipped. A letter followed by lowercase text is
    not read as an option, so the article in "A survey question" does not count.
    ``letters`` is the set of valid option letters for the question, so a
    letter outside the options also returns None (the reply is flagged).
    """
    m = LEADING_LETTER.match(answer_text.strip())
    return m.group(1) if m and m.group(1) in letters else None
