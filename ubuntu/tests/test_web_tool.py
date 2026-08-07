from __future__ import annotations

import threading

from palmclaw_ubuntu.models import ToolCall, ToolDefinition, ToolResult
from palmclaw_ubuntu.tools import (
    ToolExecutionContext,
    ToolRegistry,
    WebFetchTool,
    WebResponse,
)


def _execute(tool: WebFetchTool, url: str):
    return ToolRegistry([tool]).execute(
        ToolCall(
            id="web-call",
            name="web_fetch",
            arguments={"url": url},
        ),
        ToolExecutionContext(session_id="session-1"),
    )[0]


def test_web_fetch_uses_validated_pinned_public_ip():
    observed: list[tuple[str, str]] = []

    def requester(url, pinned_ip, timeout_seconds, max_bytes):
        del timeout_seconds, max_bytes
        observed.append((url, pinned_ip))
        return WebResponse(
            status_code=200,
            headers={"content-type": "text/plain; charset=utf-8"},
            content="안전한 응답".encode(),
        )

    tool = WebFetchTool(
        timeout_seconds=1,
        max_bytes=1_000,
        max_redirects=2,
        resolver=lambda host, port: ["93.184.216.34"],
        requester=requester,
    )
    result = _execute(tool, "https://example.com/page")

    assert not result.is_error
    assert observed == [("https://example.com/page", "93.184.216.34")]
    assert result.metadata["resolved_ip"] == "93.184.216.34"


def test_private_and_mixed_dns_answers_are_blocked_before_request():
    requested = threading.Event()

    def requester(url, pinned_ip, timeout_seconds, max_bytes):
        del url, pinned_ip, timeout_seconds, max_bytes
        requested.set()
        raise AssertionError("request must not run")

    for addresses in (
        ["127.0.0.1"],
        ["169.254.169.254"],
        ["10.0.0.1"],
        ["93.184.216.34", "192.168.1.1"],
    ):
        tool = WebFetchTool(
            timeout_seconds=1,
            max_bytes=1_000,
            max_redirects=2,
            resolver=lambda host, port, values=addresses: values,
            requester=requester,
        )
        result = _execute(tool, "https://example.com/")
        assert result.is_error
        assert result.metadata["error_code"] == "permission_denied"
    assert not requested.is_set()


def test_redirect_target_is_revalidated_and_private_target_is_blocked():
    requested: list[str] = []

    def resolver(host, port):
        del port
        return ["93.184.216.34"] if host == "public.example" else ["127.0.0.1"]

    def requester(url, pinned_ip, timeout_seconds, max_bytes):
        del pinned_ip, timeout_seconds, max_bytes
        requested.append(url)
        return WebResponse(
            status_code=302,
            headers={"location": "https://internal.example/secret"},
            content=b"",
        )

    tool = WebFetchTool(
        timeout_seconds=1,
        max_bytes=1_000,
        max_redirects=2,
        resolver=resolver,
        requester=requester,
    )
    result = _execute(tool, "https://public.example/start")

    assert result.is_error
    assert result.metadata["error_code"] == "permission_denied"
    assert requested == ["https://public.example/start"]


def test_non_https_credentials_and_oversized_responses_are_blocked():
    def success(url, pinned_ip, timeout_seconds, max_bytes):
        del url, pinned_ip, timeout_seconds, max_bytes
        return WebResponse(
            status_code=200,
            headers={"content-type": "text/plain"},
            content=b"too large",
        )

    tool = WebFetchTool(
        timeout_seconds=1,
        max_bytes=3,
        max_redirects=0,
        resolver=lambda host, port: ["93.184.216.34"],
        requester=success,
    )

    assert (
        _execute(tool, "http://example.com").metadata["error_code"]
        == "permission_denied"
    )
    assert (
        _execute(tool, "https://user:pass@example.com").metadata["error_code"]
        == "permission_denied"
    )
    oversized = _execute(tool, "https://example.com")
    assert oversized.is_error
    assert oversized.metadata["error_code"] == "execution_error"


class _SlowTool:
    def __init__(self):
        self.started = threading.Event()
        self.release = threading.Event()
        self.definition = ToolDefinition(
            name="slow",
            description="Wait until released",
            parameters={
                "type": "object",
                "properties": {},
                "additionalProperties": False,
            },
            timeout_seconds=0.05,
        )

    def run(self, arguments, context):
        del arguments, context
        self.started.set()
        self.release.wait(timeout=5)
        return ToolResult(tool_call_id="", content="late")


def test_tool_timeout_is_structured():
    tool = _SlowTool()
    registry = ToolRegistry([tool])
    result, _ = registry.execute(
        ToolCall(id="slow-call", name="slow", arguments={}),
        ToolExecutionContext(session_id="session-1"),
    )
    tool.release.set()

    assert result.is_error
    assert result.metadata["error_code"] == "timeout"


class _LargeResultTool:
    definition = ToolDefinition(
        name="large",
        description="Return a large result",
        parameters={
            "type": "object",
            "properties": {},
            "additionalProperties": False,
        },
        timeout_seconds=1,
    )

    def run(self, arguments, context):
        del arguments, context
        return ToolResult(tool_call_id="", content="x" * 100)


def test_tool_result_is_bounded():
    registry = ToolRegistry([_LargeResultTool()], max_result_chars=10)
    result, _ = registry.execute(
        ToolCall(id="large-call", name="large", arguments={}),
        ToolExecutionContext(session_id="session-1"),
    )

    assert result.content == ("x" * 10) + "\n[TRUNCATED]"
