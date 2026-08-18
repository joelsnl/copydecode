from __future__ import annotations

import unittest

from copydecode.glossary import Glossary, Term
from copydecode.spans import (
    EditProgram,
    Span,
    pack_span_jobs,
    replacement_ok,
    span_jobs_for,
    span_length_ok,
    split_units,
    tag_text,
    trim_echo,
)


class SplitUnitsTests(unittest.TestCase):
    def test_roundtrip(self) -> None:
        samples = [
            "The sky was clear. He walked on.",
            '"You dare!" the youth said. Then he left.',
            "No period here",
            "Wait... what happened next?",
            "He paused… then continued.",
            "Nascent Soul. Golden Core. Qi.",
            "",
        ]
        for text in samples:
            self.assertEqual("".join(split_units(text)), text)

    def test_ellipsis_stays_together(self) -> None:
        units = split_units("Wait... what happened next?")
        self.assertTrue(any("Wait..." in unit or "Wait…" in unit for unit in units) or units[0].startswith("Wait"))
        self.assertEqual("".join(units), "Wait... what happened next?")


class TagTests(unittest.TestCase):
    def test_short_trailing_keep_is_copied(self) -> None:
        text = "He could not help but smile. The valley was quiet."
        program = tag_text(text, "polish", "auto")
        self.assertEqual(program.stitched({}), text)
        kinds = [span.kind for span in program.spans]
        self.assertEqual(kinds, ["REPLACE", "KEEP"])
        self.assertGreater(program.keep_ratio, 0.2)

    def test_keep_replace_keep(self) -> None:
        text = (
            "The mountain wind was cold. "
            "He could not help but smile. "
            "Then he walked toward the sect gate."
        )
        program = tag_text(text, "polish", "auto")
        kinds = [span.kind for span in program.spans]
        self.assertIn("REPLACE", kinds)
        self.assertIn("KEEP", kinds)
        self.assertEqual(program.stitched({}), text)
        replace_text = "".join(s.text for s in program.spans if s.kind == "REPLACE")
        self.assertIn("could not help but", replace_text)

    def test_absorb_short_keep_bridge(self) -> None:
        text = (
            "He could not help but smile. The old path forked ahead. "
            "He sucked in a cold air. "
            "The sun rose over the valley."
        )
        program = tag_text(text, "polish", "auto")
        self.assertEqual(program.stitched({}), text)
        replace_text = "".join(s.text for s in program.spans if s.kind == "REPLACE")
        self.assertIn("The old path forked ahead", replace_text)
        kinds = [span.kind for span in program.spans]
        self.assertEqual(kinds.count("REPLACE"), 1)

    def test_force_dirty_when_clean(self) -> None:
        text = "The mountain wind was cold. Then he walked toward the sect gate."
        program = tag_text(text, "polish", "auto", force_dirty=True)
        self.assertTrue(any(span.kind == "REPLACE" for span in program.spans))
        self.assertEqual(program.stitched({}), text)

    def test_skip_off_rewrites_clean_text(self) -> None:
        text = "The mountain wind was cold."
        program = tag_text(text, "polish", "off")
        self.assertEqual(program.spans[0].kind, "REPLACE")

    def test_low_keep_ratio_collapses(self) -> None:
        text = (
            "He could not help but smile. "
            "He sucked in a cold air. "
            "Incontinently the youth said in a deep voice that this daddy will "
            "already was very much so in the next moment."
        )
        program = tag_text(text, "polish", "auto")
        self.assertEqual(len(program.spans), 1)
        self.assertEqual(program.spans[0].kind, "REPLACE")

    def test_glossary_does_not_break_join(self) -> None:
        glossary = Glossary(terms=[Term("Jindan", "Golden Core")])
        text = "His Jindan broke. The mountain was quiet."
        program = tag_text(text, "polish", "auto", glossary)
        self.assertEqual(program.stitched({}), text)


class StitchTests(unittest.TestCase):
    def test_replace_only_dirty_span(self) -> None:
        program = EditProgram(
            (
                Span("KEEP", "The mountain wind was cold. "),
                Span("REPLACE", "He could not help but smile. "),
                Span("KEEP", "Then he walked on."),
            )
        )
        out = program.stitched({1: "He smiled despite himself. "})
        self.assertTrue(out.startswith("The mountain wind was cold. "))
        self.assertTrue(out.endswith("Then he walked on."))
        self.assertIn("smiled despite himself", out)
        self.assertNotIn("could not help but", out)

    def test_jobs_point_at_replace_indexes(self) -> None:
        program = tag_text(
            "The mountain wind was cold. He could not help but smile. Then he walked on.",
            "polish",
            "auto",
        )
        jobs = span_jobs_for(3, program)
        self.assertTrue(jobs)
        self.assertTrue(all(job.seg_index == 3 for job in jobs))
        for job in jobs:
            self.assertEqual(program.spans[job.span_index].kind, "REPLACE")
            self.assertEqual(program.spans[job.span_index].text, job.text)

    def test_pack_respects_max_chars(self) -> None:
        program = tag_text(
            "He could not help but smile. The path was quiet. He sucked in a cold air.",
            "polish",
            "auto",
        )
        jobs = span_jobs_for(0, program)
        packed = pack_span_jobs(jobs, max_chars=40)
        self.assertGreaterEqual(len(packed), 1)
        self.assertEqual(sum(len(p) for p in packed), len(jobs))

    def test_trim_echo_strips_keep_context(self) -> None:
        source = "He could not help but smile."
        before = "The mountain wind was cold."
        after = "Then he walked on."
        echoed = f"{before} He smiled. {after}"
        self.assertEqual(trim_echo(echoed, source, before, after), "He smiled.")

    def test_trim_echo_handles_clipped_context_markers(self) -> None:
        # clip_context prepends/appends an ellipsis on the clipped side; the
        # model echoes the real text without it.
        source = "He could not help but smile."
        before = "…wind was cold."
        after = "Then he walked on…"
        echoed = "wind was cold. He smiled. Then he walked on"
        self.assertEqual(trim_echo(echoed, source, before, after), "He smiled.")

    def test_trim_echo_rejects_whole_passage_echo(self) -> None:
        source = "He could not help but smile."
        before = "AAA "
        after = " ZZZ"
        echoed = before + source + " extra extra extra extra extra extra extra extra" + after
        self.assertEqual(trim_echo(echoed, source, before, after), source)

    def test_span_length_ok(self) -> None:
        self.assertTrue(span_length_ok("could not help but smile", "couldn't help smiling"))
        self.assertFalse(span_length_ok("could not help but smile", ""))
        self.assertFalse(span_length_ok("Here we come.", "x" * 200))

    def test_replacement_ok_keeps_faithful_polish(self) -> None:
        source = (
            "Su Qing walked to the counter, knocked on the table, and said in a cold voice: "
            '"Put away your melon seed peels, we have customers."'
        )
        after = (
            "Su Qing walked to the counter, knocked on the table, and said in a cold voice: "
            '"Clear away your melon seeds, we have customers."'
        )
        self.assertTrue(replacement_ok(source, after))
        self.assertTrue(
            replacement_ok(
                'Jiang Kai subconsciously took the phone away, but couldn\'t help but raise the corners of his mouth: "How was the surgery?"',
                '"How did the operation go?" Jiang Kai asked with a slight smile as he unconsciously moved the phone away.',
            )
        )

    def test_replacement_ok_rejects_hallucination_and_leaks(self) -> None:
        self.assertFalse(
            replacement_ok(
                'Jiang Kai\'s eyes narrowed: "Here we come."',
                "The news of Jiang Kai's arrival spread like wildfire, and within a short time, the entire city was filled with the news of Jiang Kai's arrival.",
            )
        )
        self.assertFalse(
            replacement_ok(
                "His red eyes shone with the light of gossip.",
                "The two of them were like a pair of birds, flying together in the sky.",
            )
        )
        self.assertFalse(
            replacement_ok(
                'The word "开" is written everywhere in bright red, like wounds that have not yet healed.',
                "The house is about to be demolished, and the car stops in front of it.",
            )
        )
        self.assertFalse(
            replacement_ok(
                "The corners of her mouth raised slightly, revealing a playful smile.",
                'She had a mischievous glint in her eyes as she replied, "I\'ll show',
            )
        )
        self.assertFalse(
            replacement_ok(
                "Jiang Kai couldn't help but raise the corners of his mouth.",
                "Do not add or remove [n] labels. Do not add extra text.",
            )
        )
        source = (
            'The car stopped in front of a low-cost rental house that was about to be demolished. '
            'The word "开" is written everywhere in bright red, like wounds that have not yet healed.'
        )
        wrapped = (
            "The house is in a dilapidated state, with the walls cracked and the roof leaking. "
            "The windows are broken, and the door is hanging off its hinges.\n"
            + source
            + " The house"
        )
        self.assertFalse(replacement_ok(source, wrapped))
        gossip = (
            "His red eyes shone with the light of gossip. He came over to wink and made the famous "
            'gesture with extremely exaggerated mouth gestures: "Check the post?"'
        )
        self.assertFalse(
            replacement_ok(
                gossip,
                '"Have you seen the latest news?" he asked, winking and making an exaggerated gesture with his mouth.',
            )
        )


if __name__ == "__main__":
    unittest.main()
