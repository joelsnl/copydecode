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

    def test_source_style_is_genre_neutral(self) -> None:
        self.assertNotIn("xianxia", SOURCE_STYLE.lower())
        self.assertNotIn("cultivation", SOURCE_STYLE.lower())
        self.assertIn("register", SOURCE_STYLE.lower())
        extra = job_style("Keep it terse.")
        self.assertTrue(extra.startswith(SOURCE_STYLE))
        self.assertIn("Keep it terse.", extra)


if __name__ == "__main__":
    unittest.main()
