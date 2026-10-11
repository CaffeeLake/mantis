"""Tests for how get_llm_kwargs shapes a request per provider route.

The pipeline is provider-agnostic underneath (litellm); these pin the few
places the harness adds route-specific behaviour, so a change to one route
cannot silently diverge from another.
"""

from __future__ import annotations

import os
import sys
import unittest
from pathlib import Path
from unittest import mock

_REF_ROOT = str(Path(__file__).resolve().parent.parent)
if _REF_ROOT not in sys.path:
    sys.path.insert(0, _REF_ROOT)

from core.config import DEFAULT_MODEL, get_llm_kwargs, normalize_model_id  # noqa: E402

_CATEGORIES = {
    "HARM_CATEGORY_DANGEROUS_CONTENT",
    "HARM_CATEGORY_HATE_SPEECH",
    "HARM_CATEGORY_HARASSMENT",
    "HARM_CATEGORY_SEXUALLY_EXPLICIT",
}

_CRED_VARS = ("GEMINI_API_KEY", "GOOGLE_API_KEY", "VERTEXAI_PROJECT",
              "GOOGLE_CLOUD_PROJECT", "GOOGLE_APPLICATION_CREDENTIALS",
              "ANTHROPIC_API_KEY", "OPENAI_API_KEY", "MANTIS_MODEL", "MODEL_ID",
              "LLM_API_BASE")


def _clean_env(**extra):
    env = {k: v for k, v in os.environ.items() if k not in _CRED_VARS}
    env.update(extra)
    return mock.patch.dict(os.environ, env, clear=True)


def _relaxed(kwargs) -> bool:
    settings = kwargs.get("safety_settings") or []
    return ({s["category"] for s in settings} == _CATEGORIES
            and all(s["threshold"] == "BLOCK_NONE" for s in settings))


class DefaultModelTest(unittest.TestCase):
    def test_default_is_a_supported_gemini_flash(self):
        self.assertEqual(DEFAULT_MODEL, "vertex_ai/gemini-3.8-flash")

    def test_bare_gemini_routes_by_credentials(self):
        with _clean_env(GEMINI_API_KEY="k"):
            self.assertEqual(normalize_model_id("gemini-3.8-flash"), "gemini/gemini-3.8-flash")
        with _clean_env(GOOGLE_CLOUD_PROJECT="p"):
            self.assertEqual(normalize_model_id("gemini-3.8-flash"), "vertex_ai/gemini-3.8-flash")


class SafetySettingsPerRouteTest(unittest.TestCase):
    """The category-filter relaxation follows the model, not the route.

    Before this, only the vertex_ai/ branch set safety_settings: the same
    defensive-analysis prompt could be filtered through an API key and pass
    through a project.
    """

    def test_vertex_route_relaxes_filters(self):
        with _clean_env(GOOGLE_CLOUD_PROJECT="p"):
            model, kwargs = get_llm_kwargs(model_id="vertex_ai/gemini-3.8-flash")
        self.assertEqual(model, "vertex_ai/gemini-3.8-flash")
        self.assertTrue(_relaxed(kwargs), kwargs.get("safety_settings"))
        self.assertEqual(kwargs.get("vertex_project"), "p")

    def test_gemini_api_route_relaxes_filters_too(self):
        with _clean_env(GEMINI_API_KEY="k"):
            model, kwargs = get_llm_kwargs(model_id="gemini-3.8-flash")
        self.assertEqual(model, "gemini/gemini-3.8-flash")
        self.assertTrue(_relaxed(kwargs), kwargs.get("safety_settings"))
        self.assertNotIn("vertex_project", kwargs)

    def test_non_gemini_routes_are_untouched(self):
        with _clean_env(ANTHROPIC_API_KEY="k"):
            _, anthropic = get_llm_kwargs(model_id="anthropic/claude-opus-5")
        with _clean_env(OPENAI_API_KEY="k", LLM_API_BASE="http://x/v1"):
            _, openai_compat = get_llm_kwargs(model_id="openai/glm-5.3")
            # A Gemini model served by an OpenAI-compatible gateway takes
            # OpenAI parameters; the Gemini-only one would be rejected there.
            _, gateway_gemini = get_llm_kwargs(model_id="openai/gemini-3.8-flash")
        self.assertNotIn("safety_settings", anthropic)
        self.assertNotIn("safety_settings", openai_compat)
        self.assertNotIn("safety_settings", gateway_gemini)
        self.assertEqual(openai_compat.get("api_base"), "http://x/v1")

    def test_vertex_global_is_only_set_on_vertex(self):
        import litellm

        with mock.patch.object(litellm, "vertex_ai_safety_settings", None, create=True):
            with _clean_env(GEMINI_API_KEY="k"):
                get_llm_kwargs(model_id="gemini/gemini-3.8-flash")
            self.assertIsNone(litellm.vertex_ai_safety_settings)
            with _clean_env(GOOGLE_CLOUD_PROJECT="p"):
                get_llm_kwargs(model_id="vertex_ai/gemini-3.8-flash")
            self.assertIsNotNone(litellm.vertex_ai_safety_settings)


if __name__ == "__main__":
    unittest.main()
