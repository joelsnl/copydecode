from __future__ import annotations

import unittest

from copydecode.glossary import Glossary, glossary_from_data
from copydecode.prompts import SOURCE_STYLE, job_style


class GlossaryTests(unittest.TestCase):
    def test_empty_glossary_does_not_rewrite_any_genre(self) -> None:
        glossary = Glossary()
        sample = "The shifu 师傅 took a breath of 气 and called the 公子."
        self.assertEqual(glossary.apply_to_text(sample), sample)

    def test_user_glossary_still_applies(self) -> None:
        glossary = glossary_from_data({"terms": [{"source": "Jindan", "target": "Golden Core"}]})
        self.assertIn("Golden Core", glossary.apply_to_text("His Jindan broke."))

    def test_substitution_is_single_pass(self) -> None:
        # One term's output must never be re-matched by another term.
        glossary = glossary_from_data(
            {"terms": [{"source": "Wang", "target": "Wang Lin"}, {"source": "Lin", "target": "Forest"}]}
        )
        self.assertEqual(glossary.apply_to_text("Wang nodded at Lin."), "Wang Lin nodded at Forest.")

    def test_longest_source_wins_at_same_position(self) -> None:
        glossary = glossary_from_data(
            {"terms": [{"source": "Wang Lin", "target": "Elder Wang Lin"}, {"source": "Wang", "target": "King"}]}
        )
        self.assertEqual(glossary.apply_to_text("Wang Lin met Wang."), "Elder Wang Lin met King.")

    def test_target_backslashes_are_literal(self) -> None:
        glossary = glossary_from_data({"terms": [{"source": "the path", "target": "C:\\new\\g1"}]})
        self.assertEqual(glossary.apply_to_text("Follow the path now."), "Follow C:\\new\\g1 now.")

    def test_foreign_sources_match_without_word_boundaries(self) -> None:
        glossary = glossary_from_data({"terms": [{"source": "师兄", "target": "Senior Brother"}]})
        self.assertEqual(glossary.apply_to_text("他叫了师兄一声。"), "他叫了Senior Brother一声。")

    def test_terms_added_after_first_apply_take_effect(self) -> None:
        from copydecode.glossary import Term

        glossary = glossary_from_data({"terms": [{"source": "Jindan", "target": "Golden Core"}]})
        self.assertEqual(glossary.apply_to_text("Jindan"), "Golden Core")
        glossary.add(Term("shifu", "master"))
        self.assertEqual(glossary.apply_to_text("the shifu spoke"), "the master spoke")

    def test_hit_counts_label_source_and_target(self) -> None:
        glossary = glossary_from_data({"terms": [{"source": "Jindan", "target": "Golden Core"}]})
        counts = glossary.hit_counts("Jindan cracked. The Jindan glowed.")
        self.assertEqual(counts, {"Jindan → Golden Core": 2})

    def test_source_style_is_genre_neutral(self) -> None:
        self.assertNotIn("xianxia", SOURCE_STYLE.lower())
        self.assertNotIn("cultivation", SOURCE_STYLE.lower())
        self.assertIn("register", SOURCE_STYLE.lower())
        extra = job_style("Keep it terse.")
        self.assertTrue(extra.startswith(SOURCE_STYLE))
        self.assertIn("Keep it terse.", extra)


if __name__ == "__main__":
    unittest.main()
