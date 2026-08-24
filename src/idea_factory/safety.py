"""Shared conservative content firewalls for model-authored payloads."""

from __future__ import annotations

from collections.abc import Iterator
import re
import unicodedata


_ABSOLUTE_PRIORITY_PATTERNS = tuple(
    re.compile(pattern)
    for pattern in (
        r"\bworld first\b",
        r"\bfirst ever\b",
        r"\bfirst in (?:the )?world\b",
        r"\bfirst of its kind\b",
        r"\b(?:first system|system first) ever\b",
        r"\bnever (?:(?:previously|ever|been) )*(?:attempted|done|studied|built|reported|explored)\b",
        r"\b(?:has|have) not (?:(?:previously|ever|been) )*(?:attempted|done|studied|built|reported|explored)\b",
        r"\bnot (?:(?:previously|ever|been) )*(?:attempted|done|studied|built|reported|explored)\b",
        r"\bno (?:(?:existing|prior|previous|published) ){1,3}work\b"
        r"(?! (?:queue|queues|item|items|directory|directories|file|files|tree|trees|flow|flows|"
        r"stream|streams|load|loads|package|packages|unit|units|worker|workers|buffer|buffers|log|logs)\b)",
        r"\bno one\b",
        r"\bnobody\b",
        r"\b(?:highest|top) priority\b",
        r"\bmost important\b",
        r"\bunprecedented\b",
        r"\bnovel(?:ty)?\b",
    )
)
_SENSITIVE_KEYS = frozenset(
    {
        "access_token", "access_secret", "client_secret", "api_key", "authorization", "password",
        "private_key", "secret",
    }
)
_SECRET_VALUE_PATTERNS = tuple(
    re.compile(pattern, re.I)
    for pattern in (
        r"\bBearer\s+[A-Za-z0-9._~+/=-]{8,}", r"\bghp_[A-Za-z0-9]{20,}",
        r"\bgithub_pat_[A-Za-z0-9_]{20,}", r"\bxox[baprs]-[A-Za-z0-9-]{20,}",
        r"\b(?:AKIA|ASIA)[0-9A-Z]{16}\b", r"\bAIza[0-9A-Za-z_-]{30,}\b",
        r"\bya29\.[0-9A-Za-z_-]{20,}\b", r"-----BEGIN (?:RSA |EC |OPENSSH )?PRIVATE KEY-----",
        r"\bsk-[A-Za-z0-9_-]{8,}",
        r"\b[a-z][a-z0-9+.-]*://[^/\s:@]+:[^@\s/]+@",
        r"\b(?:access[\s_-]?(?:token|secret)|client[\s_-]?secret|api[\s_-]?key|authorization|password|private[\s_-]?key)\s*(?::|=|\bis\b)\s*\S+",
    )
)


def _strings(value: object) -> Iterator[str]:
    if type(value) is str:
        yield value
    elif type(value) is dict:
        for item in value.values():
            yield from _strings(item)
    elif type(value) in {list, tuple}:
        for item in value:
            yield from _strings(item)


def _normalize_security_text(text: str) -> str:
    normalized = unicodedata.normalize("NFKC", text)
    return "".join(character for character in normalized if unicodedata.category(character) != "Cf")


def _canonical_phrase_text(text: str) -> str:
    normalized = _normalize_security_text(text).casefold()
    normalized = re.sub(r"(?<=[a-z0-9])(?:['\u2019]s)\b", "", normalized)
    separated = "".join(character if character.isalnum() else " " for character in normalized)
    return " ".join(separated.split())


def _tokens(text: str) -> tuple[str, ...]:
    return tuple(re.findall(r"[a-z0-9]+", _normalize_security_text(text).casefold()))


def normalize_free_text(text: str) -> str:
    return " ".join(_tokens(text))


def contains_global_priority_claim(value: object) -> bool:
    for text in _strings(value):
        canonical = _canonical_phrase_text(text)
        if any(pattern.search(canonical) for pattern in _ABSOLUTE_PRIORITY_PATTERNS):
            return True
    return False


def _sensitive_key(key: object) -> bool:
    if type(key) is not str:
        return False
    normalized = re.sub(r"[^a-z0-9]+", "_", _normalize_security_text(key).casefold()).strip("_")
    return normalized in _SENSITIVE_KEYS or any(normalized.endswith("_" + item) for item in _SENSITIVE_KEYS)


def contains_secret(value: object) -> bool:
    if type(value) is dict:
        return any(_sensitive_key(key) or contains_secret(item) for key, item in value.items())
    if type(value) in {list, tuple}:
        return any(contains_secret(item) for item in value)
    return type(value) is str and any(pattern.search(_normalize_security_text(value)) for pattern in _SECRET_VALUE_PATTERNS)
