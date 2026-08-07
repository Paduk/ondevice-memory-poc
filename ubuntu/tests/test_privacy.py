from palmclaw_ubuntu.privacy import (
    REDACTED,
    detect_sensitive_spans,
    redact_data,
    redact_data_for_cloud,
    redact_for_cloud,
    redact_secrets,
)


def test_redacts_common_secret_shapes():
    value = (
        "Authorization: Bearer abcdefghijklmnopqrstuvwxyz "
        "api_key=sk-abcdefghijklmnop1234 password=hunter2"
    )
    redacted = redact_secrets(value)
    assert "abcdefghijklmnopqrstuvwxyz" not in redacted
    assert "sk-" not in redacted
    assert "hunter2" not in redacted
    assert redacted.count(REDACTED) == 3


def test_redacts_nested_data():
    redacted = redact_data(
        {
            "items": ["token=abcdefghijklmnop", {"safe": "ok"}],
            "api_key": "plain-secret-value",
            "Authorization": "plain-bearer-value",
        }
    )
    assert redacted["items"][0] == f"token={REDACTED}"
    assert redacted["items"][1]["safe"] == "ok"
    assert redacted["api_key"] == REDACTED
    assert redacted["Authorization"] == REDACTED


def test_detects_and_redacts_pii_for_cloud():
    value = "Email alice@example.com, phone +82 10-1234-5678, address 123 Main Street"
    redacted, report = redact_for_cloud(value)

    assert "alice@example.com" not in redacted
    assert "10-1234-5678" not in redacted
    assert "123 Main Street" not in redacted
    assert "[REDACTED_EMAIL]" in redacted
    assert report.category_counts == {
        "address": 1,
        "email": 1,
        "phone": 1,
    }
    assert report.sensitive_chars_after == 0


def test_cloud_data_redaction_covers_nested_values_and_sensitive_keys():
    value = {
        "messages": [{"content": "Reach me at alice@example.com"}],
        "api_key": "plain-secret",
        "encrypted_content": "opaque-provider-state",
    }
    redacted, report = redact_data_for_cloud(value)

    assert "alice@example.com" not in str(redacted)
    assert redacted["api_key"] == REDACTED
    assert redacted["encrypted_content"] == "opaque-provider-state"
    assert report.redacted_count == 2
    assert report.sensitive_chars_after == 0


def test_generic_id_field_is_not_treated_as_an_opaque_provider_id():
    redacted, report = redact_data_for_cloud({"id": "123456-1234567"})

    assert redacted["id"] == "[REDACTED_GOVERNMENT_ID]"
    assert report.category_counts == {"government_id": 1}


def test_phone_detector_does_not_classify_iso_date():
    spans = detect_sensitive_spans("Run date: 2026-07-24")
    assert all(span.category != "phone" for span in spans)


def test_uuid_is_not_classified_as_pii():
    spans = detect_sensitive_spans(
        "Evaluation ID: bb42421c-6450-445d-a79a-e289f12e615e"
    )
    assert spans == ()
