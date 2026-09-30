"""
M5a -- Rule-based grammar mapper (Path A).

Runs LOCALLY.  Converts a sequence of recognised gloss tokens (e.g.
["I", "GO", "SCHOOL"]) into a fluent English sentence using
deterministic, hand-authored linguistic rules.

ISL (Indian Sign Language) grammar differs from English in several
systematic ways that these rules attempt to handle:

1. Word order: ISL is primarily SOV (Subject-Object-Verb), while
   English is SVO.  Example: ISL "I SCHOOL GO" -> English "I go to school."

2. Articles: ISL omits articles (a, an, the). The mapper inserts them
   based on heuristics.

3. Prepositions: ISL often omits or repositions prepositions. Common
   verb-preposition pairings are restored.

4. Tense markers: ISL uses separate signs for tense (PAST, FUTURE,
   FINISH). These are converted to English verb inflections.

5. Pronouns: ISL uses spatial pointing; the classifier outputs tokens
   like "I", "YOU", "HE", "SHE" etc.

6. Spelling mode: consecutive single-letter tokens are joined into a
   spelled word.

This is deliberately a *limited* rule-based system -- it won't produce
perfect English for all inputs, and that's the whole point of comparing
it with the LLM path (Path B) in the evaluation.

Usage:
    from src.grammar_mapper.rule_mapper import RuleBasedMapper
    mapper = RuleBasedMapper()
    sentence = mapper.map("I SCHOOL GO")
"""

from __future__ import annotations


# ── Linguistic knowledge bases ──────────────────────────────────────

# Pronouns recognised by the system
PRONOUNS = {"I", "YOU", "HE", "SHE", "IT", "WE", "THEY", "ME", "MY"}

# Nouns that typically take "to" as a preposition with movement verbs
LOCATION_NOUNS = {
    "SCHOOL", "HOME", "HOSPITAL", "MARKET", "OFFICE", "PARK",
    "COLLEGE", "TEMPLE", "CHURCH", "MOSQUE", "SHOP", "STATION",
    "BANK", "LIBRARY", "HOTEL", "RESTAURANT",
}

# Common verbs and their base/3rd-person/past forms
# Format: GLOSS -> (base, 3rd_person_singular, past)
VERB_FORMS: dict[str, tuple[str, str, str]] = {
    "GO":       ("go",      "goes",     "went"),
    "COME":     ("come",    "comes",    "came"),
    "EAT":      ("eat",     "eats",     "ate"),
    "DRINK":    ("drink",   "drinks",   "drank"),
    "WANT":     ("want",    "wants",    "wanted"),
    "NEED":     ("need",    "needs",    "needed"),
    "LIKE":     ("like",    "likes",    "liked"),
    "LOVE":     ("love",    "loves",    "loved"),
    "HELP":     ("help",    "helps",    "helped"),
    "WORK":     ("work",    "works",    "worked"),
    "PLAY":     ("play",    "plays",    "played"),
    "LEARN":    ("learn",   "learns",   "learned"),
    "TEACH":    ("teach",   "teaches",  "taught"),
    "MAKE":     ("make",    "makes",    "made"),
    "GIVE":     ("give",    "gives",    "gave"),
    "TAKE":     ("take",    "takes",    "took"),
    "SEE":      ("see",     "sees",     "saw"),
    "HEAR":     ("hear",    "hears",    "heard"),
    "SPEAK":    ("speak",   "speaks",   "spoke"),
    "TELL":     ("tell",    "tells",    "told"),
    "ASK":      ("ask",     "asks",     "asked"),
    "KNOW":     ("know",    "knows",    "knew"),
    "THINK":    ("think",   "thinks",   "thought"),
    "FEEL":     ("feel",    "feels",    "felt"),
    "LIVE":     ("live",    "lives",    "lived"),
    "WALK":     ("walk",    "walks",    "walked"),
    "RUN":      ("run",     "runs",     "ran"),
    "SIT":      ("sit",     "sits",     "sat"),
    "STAND":    ("stand",   "stands",   "stood"),
    "SLEEP":    ("sleep",   "sleeps",   "slept"),
    "CALL":     ("call",    "calls",    "called"),
    "BUY":      ("buy",     "buys",     "bought"),
    "READ":     ("read",    "reads",    "read"),
    "WRITE":    ("write",   "writes",   "wrote"),
    "STOP":     ("stop",    "stops",    "stopped"),
    "WAIT":     ("wait",    "waits",    "waited"),
    "THANK":    ("thank",   "thanks",   "thanked"),
}

# Verbs that typically pair with "to" + location
MOVEMENT_VERBS = {"GO", "COME", "WALK", "RUN"}

# Tense marker glosses
TENSE_MARKERS = {
    "PAST": "past",
    "FINISH": "past",       # ISL uses FINISH as a past-tense marker
    "FUTURE": "future",
    "WILL": "future",
    "NOW": "present",
    "TODAY": "present",
    "YESTERDAY": "past",
    "TOMORROW": "future",
}

# Subjects that take 3rd-person singular verb forms
THIRD_PERSON_SINGULAR = {"HE", "SHE", "IT"}

# Greetings and fixed phrases -- output as-is
FIXED_PHRASES: dict[str, str] = {
    "HELLO": "Hello.",
    "THANK_YOU": "Thank you.",
    "SORRY": "Sorry.",
    "PLEASE": "Please.",
    "YES": "Yes.",
    "NO": "No.",
    "GOOD_MORNING": "Good morning.",
    "GOOD_NIGHT": "Good night.",
    "GOODBYE": "Goodbye.",
    "WELCOME": "Welcome.",
    "OK": "OK.",
}

# Words that should get an article inserted before them
# (simple heuristic -- nouns that aren't proper nouns or pronouns)
COMMON_NOUNS = LOCATION_NOUNS | {
    "BOOK", "WATER", "FOOD", "PHONE", "NAME", "FAMILY",
    "FRIEND", "DOCTOR", "TEACHER", "STUDENT", "CHILD",
    "CAR", "BUS", "TRAIN", "HOUSE", "ROOM",
}

# Vowel-starting words get "an" instead of "a"
VOWELS = set("AEIOU")


class RuleBasedMapper:
    """Deterministic gloss-to-English sentence mapper.

    This is Path A of the SignScribe comparison.  It applies
    hand-authored rules to convert ISL gloss sequences into
    English sentences.
    """

    def map(self, gloss_input: str | list[str]) -> str:
        """Convert gloss tokens into an English sentence.

        Parameters
        ----------
        gloss_input : str or list[str]
            Either a space-separated string of glosses or a list.
            Example: "I SCHOOL GO" or ["I", "SCHOOL", "GO"]

        Returns
        -------
        str
            The generated English sentence.
        """
        if isinstance(gloss_input, str):
            tokens = gloss_input.strip().upper().split()
        else:
            tokens = [t.upper() for t in gloss_input]

        if not tokens:
            return ""

        # Step 0: Check for single fixed phrases
        if len(tokens) == 1 and tokens[0] in FIXED_PHRASES:
            return FIXED_PHRASES[tokens[0]]

        # Step 1: Merge consecutive single-letter tokens into spelled words
        tokens = self._merge_spelled_words(tokens)

        # Step 2: Extract tense markers
        tense, tokens = self._extract_tense(tokens)

        # Step 3: Identify sentence components (subject, verb, objects)
        subject, verb_gloss, objects = self._parse_sov(tokens)

        # Step 4: Build the English sentence in SVO order
        sentence = self._build_sentence(subject, verb_gloss, objects, tense)

        return sentence

    def map_sequence(self, gloss_sequence: list[str]) -> str:
        """Map a sequence of gloss tokens accumulated over time.

        Filters out repeated consecutive tokens (common in real-time
        recognition where the same gesture is classified multiple
        times in a row).
        """
        # De-duplicate consecutive repetitions
        deduped: list[str] = []
        for token in gloss_sequence:
            if not deduped or token != deduped[-1]:
                deduped.append(token)

        return self.map(deduped)

    # ── Internal processing steps ───────────────────────────────

    @staticmethod
    def _merge_spelled_words(tokens: list[str]) -> list[str]:
        """Merge consecutive single-character tokens into one word.

        Example: ["H", "E", "L", "L", "O"] -> ["HELLO"]
        But: ["H", "E", "GO", "L", "O"] -> ["HE", "GO", "LO"]
        """
        merged: list[str] = []
        i = 0
        while i < len(tokens):
            if len(tokens[i]) == 1 and tokens[i].isalpha():
                # Start collecting letters
                word = tokens[i]
                j = i + 1
                while j < len(tokens) and len(tokens[j]) == 1 and tokens[j].isalpha():
                    word += tokens[j]
                    j += 1
                merged.append(word)
                i = j
            else:
                merged.append(tokens[i])
                i += 1
        return merged

    @staticmethod
    def _extract_tense(tokens: list[str]) -> tuple[str, list[str]]:
        """Remove tense markers from the token list and return the tense.

        Returns
        -------
        tense : str
            One of "present", "past", "future".
        remaining : list[str]
            Tokens without the tense markers.
        """
        tense = "present"
        remaining: list[str] = []

        for token in tokens:
            if token in TENSE_MARKERS:
                tense = TENSE_MARKERS[token]
            else:
                remaining.append(token)

        return tense, remaining

    @staticmethod
    def _parse_sov(
        tokens: list[str],
    ) -> tuple[str | None, str | None, list[str]]:
        """Parse ISL SOV order into (subject, verb, objects/complements).

        ISL typically follows Subject-Object-Verb order.  This parser
        identifies the components using part-of-speech heuristics.

        Returns
        -------
        subject : str or None
        verb_gloss : str or None  (the original GLOSS, not inflected)
        objects : list[str]  (remaining tokens in original order)
        """
        subject = None
        verb_gloss = None
        objects: list[str] = []

        # Find subject (first pronoun)
        remaining = list(tokens)
        for i, tok in enumerate(remaining):
            if tok in PRONOUNS:
                subject = tok
                remaining.pop(i)
                break

        # In SOV, the verb is typically last.
        # Scan from the end to find a known verb.
        for i in range(len(remaining) - 1, -1, -1):
            if remaining[i] in VERB_FORMS:
                verb_gloss = remaining[i]
                remaining.pop(i)
                break

        # Everything else is objects/complements
        objects = remaining

        return subject, verb_gloss, objects

    def _build_sentence(
        self,
        subject: str | None,
        verb_gloss: str | None,
        objects: list[str],
        tense: str,
    ) -> str:
        """Assemble the English sentence from parsed components.

        Follows SVO order: Subject + Verb + Objects/Complements
        """
        parts: list[str] = []

        # Subject
        if subject:
            parts.append(subject.capitalize() if subject == "I" else subject.lower())
            # Special: "I" stays capitalised
            if subject == "I":
                parts[-1] = "I"

        # Verb (conjugated)
        if verb_gloss:
            verb_form = self._conjugate_verb(verb_gloss, subject, tense)
            parts.append(verb_form)

        # Objects / complements
        for obj in objects:
            word = self._format_object(obj, verb_gloss)
            parts.append(word)

        # If no components found, just join the original tokens
        if not parts:
            parts = [t.lower() for t in objects]

        sentence = " ".join(parts)

        # Capitalise first letter, add period
        if sentence:
            sentence = sentence[0].upper() + sentence[1:]
            if not sentence.endswith((".", "!", "?")):
                sentence += "."

        return sentence

    @staticmethod
    def _conjugate_verb(
        verb_gloss: str, subject: str | None, tense: str
    ) -> str:
        """Return the correctly inflected English verb form."""
        if verb_gloss not in VERB_FORMS:
            return verb_gloss.lower()

        base, third, past = VERB_FORMS[verb_gloss]

        if tense == "past":
            return past
        elif tense == "future":
            return f"will {base}"
        else:
            # Present tense
            if subject and subject in THIRD_PERSON_SINGULAR:
                return third
            return base

    @staticmethod
    def _format_object(obj_gloss: str, verb_gloss: str | None) -> str:
        """Format an object/complement token for English output.

        - Inserts prepositions (e.g., "to school" for movement verbs)
        - Inserts articles for common nouns
        - Lowercases non-proper-noun tokens
        """
        word = obj_gloss

        # Fixed phrases embedded in a sentence
        if word in FIXED_PHRASES:
            return FIXED_PHRASES[word].rstrip(".")

        # Insert "to" before location nouns when preceded by movement verbs
        prefix = ""
        if word in LOCATION_NOUNS and verb_gloss in MOVEMENT_VERBS:
            prefix = "to "
        elif word in COMMON_NOUNS:
            # Insert article
            article = "an" if word[0] in VOWELS else "a"
            prefix = f"{article} "

        return prefix + word.lower()


# ── Convenience function for quick testing ──────────────────────────

def demo():
    """Run a few example translations to verify the mapper."""
    mapper = RuleBasedMapper()

    examples = [
        "I SCHOOL GO",
        "I FOOD EAT",
        "HE HOSPITAL GO PAST",
        "SHE BOOK READ",
        "I WATER DRINK FUTURE",
        "HELLO",
        "THANK_YOU",
        "H E L L O",
        "YOU HELP WANT",
        "WE PARK GO TOMORROW",
        "I FRIEND HELP",
        "HE TEACHER ASK",
    ]

    print("=" * 60)
    print("  SignScribe - Rule-Based Grammar Mapper Demo (Phase 4)")
    print("=" * 60)
    print()

    for gloss in examples:
        sentence = mapper.map(gloss)
        print(f"  ISL gloss : {gloss}")
        print(f"  English   : {sentence}")
        print()

    print("=" * 60)


if __name__ == "__main__":
    demo()
