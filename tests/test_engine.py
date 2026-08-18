from __future__ import annotations

import unittest
from unittest.mock import MagicMock, patch

from copydecode.engine import EngineError, EngineInfo, LLMEngine, classify_host, pick_model


def _response(status: int, text: str = "", payload=None) -> MagicMock:
    response = MagicMock()
    response.status_code = status
    response.text = text
    response.json.return_value = payload if payload is not None else {}
    return response


def _server(routes: dict[str, MagicMock]):
    def fake_ok(url: str, timeout: float = 1.2):
        for suffix, response in routes.items():
            if url.endswith(suffix):
                return response
        return None

    return fake_ok


class ClassifyHostTests(unittest.TestCase):
    def test_ollama_by_api_tags(self) -> None:
        routes = {"/api/tags": _response(200, "{}")}
        with patch("copydecode.engine._ok", side_effect=_server(routes)):
            self.assertEqual(classify_host("http://127.0.0.1:11434").kind, "ollama")

    def test_llamacpp_by_props(self) -> None:
        routes = {
            "/props": _response(200, "{}"),
            "/v1/models": _response(200, '{"data":[{"id":"model.gguf"}]}'),
        }
        with patch("copydecode.engine._ok", side_effect=_server(routes)):
            self.assertEqual(classify_host("http://127.0.0.1:8080").kind, "llamacpp")

    def test_vllm_is_not_mistaken_for_llamacpp(self) -> None:
        # vLLM serves /health and /v1/models but not llama.cpp's /props.
        routes = {
            "/health": _response(200, ""),
            "/v1/models": _response(200, '{"data":[{"id":"qwen","owned_by":"vllm"}]}'),
        }
        with patch("copydecode.engine._ok", side_effect=_server(routes)):
            info = classify_host("http://127.0.0.1:8000")
        self.assertEqual(info.kind, "vllm")

    def test_plain_openai_server(self) -> None:
        routes = {"/v1/models": _response(200, '{"data":[{"id":"gpt-x"}]}')}
        with patch("copydecode.engine._ok", side_effect=_server(routes)):
            self.assertEqual(classify_host("http://127.0.0.1:8000").kind, "openai")

    def test_no_server_raises(self) -> None:
        with patch("copydecode.engine._ok", return_value=None):
            with self.assertRaises(EngineError):
                classify_host("http://127.0.0.1:9")


class PickModelTests(unittest.TestCase):
    def test_prefers_largest_sized_model_that_fits(self) -> None:
        names = ["qwen2.5:3b", "qwen2.5:14b", "qwen2.5:32b"]
        self.assertEqual(pick_model(names, 14.0), "qwen2.5:14b")

    def test_unknown_size_never_beats_a_sized_fit(self) -> None:
        names = ["mystery-model", "qwen2.5:7b"]
        self.assertEqual(pick_model(names, 14.0), "qwen2.5:7b")

    def test_unknown_size_is_last_resort(self) -> None:
        self.assertEqual(pick_model(["mystery-model"], 14.0), "mystery-model")

    def test_empty_list(self) -> None:
        self.assertIsNone(pick_model([], 14.0))


class CountPromptTokensTests(unittest.TestCase):
    def test_llamacpp_uses_server_tokenizer(self) -> None:
        engine = LLMEngine(EngineInfo("llamacpp", "http://127.0.0.1:8080", "llama.cpp"), model="q")
        try:
            fake = MagicMock()
            fake.status_code = 200
            fake.json.return_value = {"tokens": [1, 2, 3]}
            with patch.object(engine._client, "post", return_value=fake) as post:
                self.assertEqual(engine.count_prompt_tokens("hello world"), 3)
            post.assert_called_once()
        finally:
            engine.close()

    def test_other_engines_return_none(self) -> None:
        engine = LLMEngine(EngineInfo("vllm", "http://127.0.0.1:8000", "vLLM"), model="q")
        try:
            self.assertIsNone(engine.count_prompt_tokens("hello"))
        finally:
            engine.close()


if __name__ == "__main__":
    unittest.main()
