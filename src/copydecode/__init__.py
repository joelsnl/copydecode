"""Local KEEP/REPLACE copy-edit and any-language → English translation.

Public API:

- :func:`polish_paragraphs` / :func:`wants_polish` — polish plain paragraph lists.
- :class:`JobConfig` / :func:`run_job` — full document jobs (what the CLI runs).
- :func:`load_document` / :func:`write_document` and the ``READERS``/``WRITERS``
  registries — file formats.
- :class:`LLMEngine` / :func:`discover_engine` — talk to a local LLM server.
- :class:`CopydecodeError` — base class of every deliberate error.

Copyright (c) 2026 Joel Sunil
SPDX-License-Identifier: AGPL-3.0-or-later
"""

from copydecode.api import polish_paragraphs, wants_polish
from copydecode.document import Chapter, Document, Segment
from copydecode.engine import EngineError, EngineInfo, LLMEngine, discover_engine
from copydecode.errors import CopydecodeError, DocumentError
from copydecode.glossary import Glossary, Term, load_glossary_file
from copydecode.io import READERS, WRITERS, load_document, write_document
from copydecode.pipeline import JobConfig, run_job

__version__ = "0.3.0"

__all__ = [
    "Chapter",
    "CopydecodeError",
    "Document",
    "DocumentError",
    "EngineError",
    "EngineInfo",
    "Glossary",
    "JobConfig",
    "LLMEngine",
    "READERS",
    "Segment",
    "Term",
    "WRITERS",
    "__version__",
    "discover_engine",
    "load_document",
    "load_glossary_file",
    "polish_paragraphs",
    "run_job",
    "wants_polish",
    "write_document",
]
