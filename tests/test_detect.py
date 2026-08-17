from __future__ import annotations

import unittest

from copydecode.detect import detect_mode, foreign_script_ratio, guess_nllb_src_lang


class DetectTests(unittest.TestCase):
    def test_english_is_polish(self) -> None:
        self.assertEqual(
            detect_mode("She walked into the room and sat by the window."),
            "polish",
        )

    def test_chinese_is_translate(self) -> None:
        self.assertEqual(detect_mode("他走进房间，在窗边坐下。这是一段中文。"), "translate")
        self.assertEqual(guess_nllb_src_lang("他走进房间，在窗边坐下。"), "zho_Hans")

    def test_japanese_and_korean(self) -> None:
        self.assertEqual(detect_mode("彼は部屋に入って窓のそばに座った。"), "translate")
        self.assertEqual(guess_nllb_src_lang("彼は部屋に入って窓のそばに座った。"), "jpn_Jpan")
        self.assertEqual(detect_mode("그는 방에 들어가 창가에 앉았다."), "translate")
        self.assertEqual(guess_nllb_src_lang("그는 방에 들어가 창가에 앉았다."), "kor_Hang")

    def test_arabic_and_cyrillic(self) -> None:
        self.assertEqual(detect_mode("دخل الغرفة وجلس بجانب النافذة."), "translate")
        self.assertEqual(guess_nllb_src_lang("دخل الغرفة وجلس بجانب النافذة."), "arb_Arab")
        self.assertEqual(detect_mode("Он вошёл в комнату и сел у окна."), "translate")
        self.assertEqual(guess_nllb_src_lang("Он вошёл в комнату и сел у окна."), "rus_Cyrl")

    def test_foreign_ratio_ignores_short_latin(self) -> None:
        self.assertLess(foreign_script_ratio("Hello world"), 0.1)


if __name__ == "__main__":
    unittest.main()
