"""
Foundation unit tests for credential providers.

Covers StaticProvider and BearerTokenProvider:
- Provider metadata (provider_id, supported_types)
- Credential validation (empty keys, valid keys, token expiration)
- Token expiration and should_refresh behavior
- Refresh behavior (static unchanged, bearer raises)
- can_handle for supported credential types
"""

from __future__ import annotations

from datetime import UTC, datetime, timedelta

import pytest
from pydantic import SecretStr

from framework.credentials import (
    BearerTokenProvider,
    CredentialKey,
    CredentialObject,
    CredentialRefreshError,
    CredentialType,
    StaticProvider,
)


class TestStaticProvider:
    """Tests for StaticProvider metadata, validation, and refresh behavior."""

    def test_provider_metadata(self) -> None:
        """Provider ID and supported types are correct."""
        provider = StaticProvider()
        assert provider.provider_id == "static"
        assert provider.supported_types == [
            CredentialType.API_KEY,
            CredentialType.BASIC_AUTH,
            CredentialType.CUSTOM,
        ]

    def test_validate_with_valid_key(self) -> None:
        """Validation passes when at least one key has a non-empty value."""
        provider = StaticProvider()
        cred = CredentialObject(
            id="test",
            credential_type=CredentialType.API_KEY,
            keys={"api_key": CredentialKey(name="api_key", value=SecretStr("valid"))},
        )
        assert provider.validate(cred) is True

    def test_validate_empty_keys_fails(self) -> None:
        """Validation fails when credential has no keys."""
        provider = StaticProvider()
        cred = CredentialObject(id="test", keys={})
        assert provider.validate(cred) is False

    def test_validate_empty_or_whitespace_key_fails(self) -> None:
        """Validation fails when only key has empty or whitespace-only value."""
        provider = StaticProvider()
        for value in ("", "   ", "\t\n"):
            cred = CredentialObject(
                id="test",
                keys={"k": CredentialKey(name="k", value=SecretStr(value))},
            )
            assert provider.validate(cred) is False

    def test_refresh_returns_unchanged(self) -> None:
        """Static credentials are returned unchanged by refresh."""
        provider = StaticProvider()
        cred = CredentialObject(
            id="test",
            keys={"k": CredentialKey(name="k", value=SecretStr("v"))},
        )
        refreshed = provider.refresh(cred)
        assert refreshed is cred
        assert refreshed.get_key("k") == "v"

    def test_should_refresh_always_false(self) -> None:
        """Static provider never requires refresh."""
        provider = StaticProvider()
        cred = CredentialObject(
            id="test",
            keys={"k": CredentialKey(name="k", value=SecretStr("v"))},
        )
        assert provider.should_refresh(cred) is False

    def test_can_handle_supported_types(self) -> None:
        """can_handle returns True for API_KEY, BASIC_AUTH, CUSTOM."""
        provider = StaticProvider()
        for cred_type in (CredentialType.API_KEY, CredentialType.BASIC_AUTH, CredentialType.CUSTOM):
            cred = CredentialObject(id="x", credential_type=cred_type, keys={})
            assert provider.can_handle(cred) is True
        cred_oauth = CredentialObject(id="x", credential_type=CredentialType.OAUTH2, keys={})
        assert provider.can_handle(cred_oauth) is False


class TestBearerTokenProvider:
    """Tests for BearerTokenProvider metadata, validation, expiration, and refresh."""

    def test_provider_metadata(self) -> None:
        """Provider ID and supported types are correct."""
        provider = BearerTokenProvider()
        assert provider.provider_id == "bearer_token"
        assert provider.supported_types == [CredentialType.BEARER_TOKEN]

    def test_validate_missing_token_fails(self) -> None:
        """Validation fails when access_token and token are missing."""
        provider = BearerTokenProvider()
        cred = CredentialObject(
            id="test",
            credential_type=CredentialType.BEARER_TOKEN,
            keys={"other": CredentialKey(name="other", value=SecretStr("x"))},
        )
        assert provider.validate(cred) is False

    def test_validate_expired_token_fails(self) -> None:
        """Validation fails when token key is expired."""
        provider = BearerTokenProvider()
        past = datetime.now(UTC) - timedelta(minutes=1)
        cred = CredentialObject(
            id="test",
            credential_type=CredentialType.BEARER_TOKEN,
            keys={
                "access_token": CredentialKey(
                    name="access_token", value=SecretStr("jwt"), expires_at=past
                )
            },
        )
        assert provider.validate(cred) is False

    def test_validate_valid_token_passes(self) -> None:
        """Validation passes when token exists and is not expired."""
        provider = BearerTokenProvider()
        future = datetime.now(UTC) + timedelta(hours=1)
        cred = CredentialObject(
            id="test",
            credential_type=CredentialType.BEARER_TOKEN,
            keys={
                "access_token": CredentialKey(
                    name="access_token", value=SecretStr("jwt"), expires_at=future
                )
            },
        )
        assert provider.validate(cred) is True

    def test_validate_accepts_token_key(self) -> None:
        """Validation accepts key named 'token' as well as 'access_token'."""
        provider = BearerTokenProvider()
        future = datetime.now(UTC) + timedelta(hours=1)
        cred = CredentialObject(
            id="test",
            credential_type=CredentialType.BEARER_TOKEN,
            keys={"token": CredentialKey(name="token", value=SecretStr("jwt"), expires_at=future)},
        )
        assert provider.validate(cred) is True

    def test_should_refresh_when_near_expiry(self) -> None:
        """should_refresh returns True when token is within 5 minutes of expiry."""
        provider = BearerTokenProvider()
        # Exactly 4 minutes from now: within 5-min buffer
        near = datetime.now(UTC) + timedelta(minutes=4)
        cred = CredentialObject(
            id="test",
            credential_type=CredentialType.BEARER_TOKEN,
            keys={
                "access_token": CredentialKey(
                    name="access_token", value=SecretStr("jwt"), expires_at=near
                )
            },
        )
        assert provider.should_refresh(cred) is True

    def test_should_refresh_false_when_far_future(self) -> None:
        """should_refresh returns False when token expiry is far in future."""
        provider = BearerTokenProvider()
        future = datetime.now(UTC) + timedelta(hours=1)
        cred = CredentialObject(
            id="test",
            credential_type=CredentialType.BEARER_TOKEN,
            keys={
                "access_token": CredentialKey(
                    name="access_token", value=SecretStr("jwt"), expires_at=future
                )
            },
        )
        assert provider.should_refresh(cred) is False

    def test_refresh_raises_credential_refresh_error(self) -> None:
        """refresh raises CredentialRefreshError with message referencing credential id."""
        provider = BearerTokenProvider()
        cred = CredentialObject(
            id="my_bearer",
            credential_type=CredentialType.BEARER_TOKEN,
            keys={
                "access_token": CredentialKey(
                    name="access_token", value=SecretStr("jwt"), expires_at=None
                )
            },
        )
        with pytest.raises(CredentialRefreshError) as exc_info:
            provider.refresh(cred)
        assert "my_bearer" in str(exc_info.value)
        assert "cannot be refreshed" in str(exc_info.value).lower()

    def test_can_handle_bearer_token_only(self) -> None:
        """can_handle returns True for BEARER_TOKEN, False for other types."""
        provider = BearerTokenProvider()
        cred_bearer = CredentialObject(id="x", credential_type=CredentialType.BEARER_TOKEN, keys={})
        assert provider.can_handle(cred_bearer) is True
        cred_api = CredentialObject(id="x", credential_type=CredentialType.API_KEY, keys={})
        assert provider.can_handle(cred_api) is False
