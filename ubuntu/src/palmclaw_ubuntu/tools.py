from __future__ import annotations

import hashlib
import http.client
import ipaddress
import json
import socket
import ssl
import time
from collections.abc import Callable, Mapping, Sequence
from concurrent.futures import ThreadPoolExecutor
from concurrent.futures import TimeoutError as FutureTimeout
from dataclasses import dataclass
from typing import Any, Protocol
from urllib.parse import urljoin, urlsplit

from jsonschema import Draft202012Validator
from jsonschema.exceptions import ValidationError

from palmclaw_ubuntu.models import (
    ToolCall,
    ToolDefinition,
    ToolResult,
)
from palmclaw_ubuntu.tool_memory_schema import (
    ToolMemoryOntology,
    build_tool_memory_ontology,
)
from palmclaw_ubuntu.workspace import WorkspaceResolver


@dataclass(frozen=True)
class ToolExecutionContext:
    session_id: str


class Tool(Protocol):
    definition: ToolDefinition

    def run(
        self,
        arguments: Mapping[str, Any],
        context: ToolExecutionContext,
    ) -> ToolResult: ...


class ToolRegistry:
    def __init__(
        self,
        tools: Sequence[Tool] = (),
        *,
        max_result_chars: int = 20_000,
    ):
        self._tools: dict[str, Tool] = {}
        self.max_result_chars = max_result_chars
        for tool in tools:
            self.register(tool)

    def register(self, tool: Tool) -> None:
        name = tool.definition.name
        if name in self._tools:
            raise ValueError(f"Duplicate tool: {name}")
        Draft202012Validator.check_schema(tool.definition.parameters)
        self._tools[name] = tool

    def definitions(self) -> list[ToolDefinition]:
        return [tool.definition for tool in self._tools.values()]

    def definition(self, name: str) -> ToolDefinition | None:
        tool = self._tools.get(name)
        return tool.definition if tool else None

    def names(self) -> list[str]:
        return sorted(self._tools)

    def memory_ontology(self) -> ToolMemoryOntology:
        return build_tool_memory_ontology(self.definitions())

    @staticmethod
    def fingerprint(call: ToolCall) -> str:
        canonical = json.dumps(
            {
                "name": call.name,
                "arguments": dict(call.arguments),
            },
            ensure_ascii=False,
            separators=(",", ":"),
            sort_keys=True,
        )
        return hashlib.sha256(canonical.encode("utf-8")).hexdigest()

    def validate(self, call: ToolCall) -> ToolResult | None:
        tool = self._tools.get(call.name)
        if tool is None:
            return self._error(
                call.id,
                "unknown_tool",
                f"Unknown tool: {call.name}",
            )
        try:
            Draft202012Validator(tool.definition.parameters).validate(
                dict(call.arguments)
            )
        except ValidationError as exc:
            location = ".".join(str(part) for part in exc.absolute_path)
            suffix = f" at {location}" if location else ""
            return self._error(
                call.id,
                "invalid_arguments",
                f"Invalid arguments{suffix}: {exc.message}",
            )
        return None

    def execute(
        self,
        call: ToolCall,
        context: ToolExecutionContext,
    ) -> tuple[ToolResult, int]:
        started = time.monotonic()
        validation_error = self.validate(call)
        if validation_error is not None:
            return validation_error, self._elapsed_ms(started)

        tool = self._tools.get(call.name)
        assert tool is not None

        executor = ThreadPoolExecutor(
            max_workers=1,
            thread_name_prefix=f"tool-{call.name}",
        )
        future = executor.submit(tool.run, call.arguments, context)
        try:
            result = future.result(timeout=tool.definition.timeout_seconds)
        except FutureTimeout:
            future.cancel()
            result = self._error(
                call.id,
                "timeout",
                f"Tool timed out after {tool.definition.timeout_seconds:g}s",
            )
        except PermissionError as exc:
            result = self._error(
                call.id,
                "permission_denied",
                str(exc),
            )
        except Exception as exc:  # Tool failures become structured results.
            result = self._error(
                call.id,
                "execution_error",
                f"{type(exc).__name__}: {exc}",
            )
        finally:
            executor.shutdown(wait=False, cancel_futures=True)

        bounded_content = result.content[: self.max_result_chars]
        if len(result.content) > self.max_result_chars:
            bounded_content += "\n[TRUNCATED]"
        return (
            ToolResult(
                tool_call_id=call.id,
                content=bounded_content,
                is_error=result.is_error,
                metadata=result.metadata,
            ),
            self._elapsed_ms(started),
        )

    @staticmethod
    def _elapsed_ms(started: float) -> int:
        return max(0, round((time.monotonic() - started) * 1000))

    @staticmethod
    def _error(call_id: str, code: str, message: str) -> ToolResult:
        return ToolResult(
            tool_call_id=call_id,
            content=json.dumps(
                {"error": {"code": code, "message": message}},
                ensure_ascii=False,
            ),
            is_error=True,
            metadata={"error_code": code},
        )


class FileReadTool:
    def __init__(
        self,
        resolver: WorkspaceResolver,
        *,
        timeout_seconds: float,
        max_file_bytes: int,
    ):
        self.resolver = resolver
        self.max_file_bytes = max_file_bytes
        self.definition = ToolDefinition(
            name="file_read",
            description=(
                "Read a UTF-8 text file inside the current session or shared "
                "workspace. Returns path, byte count, and content."
            ),
            parameters={
                "type": "object",
                "properties": {
                    "path": {
                        "type": "string",
                        "description": (
                            "Relative, session://, or shared:// workspace path"
                        ),
                    }
                },
                "required": ["path"],
                "additionalProperties": False,
            },
            timeout_seconds=timeout_seconds,
            side_effect="none",
            retry_safety="safe",
        )

    def run(
        self,
        arguments: Mapping[str, Any],
        context: ToolExecutionContext,
    ) -> ToolResult:
        path = self.resolver.resolve(
            context.session_id,
            str(arguments["path"]),
            must_exist=True,
        )
        if not path.is_file():
            raise IsADirectoryError(f"Not a file: {arguments['path']}")
        stat = path.stat()
        if stat.st_nlink > 1:
            raise PermissionError("Hard-linked files are not allowed")
        size = stat.st_size
        if size > self.max_file_bytes:
            raise ValueError(f"File exceeds {self.max_file_bytes} byte read limit")
        content = path.read_text(encoding="utf-8")
        payload = {
            "path": self.resolver.display_path(context.session_id, path),
            "bytes": size,
            "content": content,
        }
        return ToolResult(
            tool_call_id="",
            content=json.dumps(payload, ensure_ascii=False),
            metadata={"path": payload["path"], "bytes": size},
        )


class FileWriteTool:
    def __init__(
        self,
        resolver: WorkspaceResolver,
        *,
        timeout_seconds: float,
        max_file_bytes: int,
    ):
        self.resolver = resolver
        self.max_file_bytes = max_file_bytes
        self.definition = ToolDefinition(
            name="file_write",
            description=(
                "Write UTF-8 text inside the current session or shared "
                "workspace. Existing files require overwrite=true."
            ),
            parameters={
                "type": "object",
                "properties": {
                    "path": {
                        "type": "string",
                        "description": (
                            "Relative, session://, or shared:// workspace path"
                        ),
                    },
                    "content": {"type": "string"},
                    "overwrite": {
                        "type": "boolean",
                        "default": False,
                    },
                },
                "required": ["path", "content", "overwrite"],
                "additionalProperties": False,
            },
            timeout_seconds=timeout_seconds,
            side_effect="workspace_write",
            retry_safety="unsafe",
        )

    def run(
        self,
        arguments: Mapping[str, Any],
        context: ToolExecutionContext,
    ) -> ToolResult:
        content = str(arguments["content"])
        encoded = content.encode("utf-8")
        if len(encoded) > self.max_file_bytes:
            raise ValueError(f"Content exceeds {self.max_file_bytes} byte write limit")
        path = self.resolver.resolve(
            context.session_id,
            str(arguments["path"]),
        )
        if path.exists() and path.stat().st_nlink > 1:
            raise PermissionError("Hard-linked files are not allowed")
        if path.exists() and not bool(arguments.get("overwrite", False)):
            raise FileExistsError("Target exists; set overwrite=true to replace it")
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(content, encoding="utf-8")
        display_path = self.resolver.display_path(context.session_id, path)
        payload = {
            "path": display_path,
            "bytes": len(encoded),
            "written": True,
        }
        return ToolResult(
            tool_call_id="",
            content=json.dumps(payload, ensure_ascii=False),
            metadata={"path": display_path, "bytes": len(encoded)},
        )


@dataclass(frozen=True)
class WebResponse:
    status_code: int
    headers: Mapping[str, str]
    content: bytes


class _PinnedHTTPSConnection(http.client.HTTPSConnection):
    def __init__(
        self,
        host: str,
        pinned_ip: str,
        *,
        port: int,
        timeout: float,
    ):
        super().__init__(
            host,
            port=port,
            timeout=timeout,
            context=ssl.create_default_context(),
        )
        self._pinned_ip = pinned_ip

    def connect(self) -> None:
        self.sock = socket.create_connection(
            (self._pinned_ip, self.port),
            self.timeout,
            self.source_address,
        )
        self.sock = self._context.wrap_socket(
            self.sock,
            server_hostname=self.host,
        )


class WebFetchTool:
    def __init__(
        self,
        *,
        timeout_seconds: float,
        max_bytes: int,
        max_redirects: int,
        resolver: Callable[[str, int], Sequence[str]] | None = None,
        requester: (Callable[[str, str, float, int], WebResponse] | None) = None,
    ):
        self.max_bytes = max_bytes
        self.max_redirects = max_redirects
        self._resolver = resolver or self._resolve_host
        self._requester = requester or self._fetch_once
        self.definition = ToolDefinition(
            name="web_fetch",
            description=(
                "Fetch public HTTPS text content. Private, loopback, local, "
                "credential-bearing, and non-HTTPS URLs are blocked."
            ),
            parameters={
                "type": "object",
                "properties": {
                    "url": {
                        "type": "string",
                        "description": "Public HTTPS URL",
                    }
                },
                "required": ["url"],
                "additionalProperties": False,
            },
            timeout_seconds=timeout_seconds,
            side_effect="network_read",
            retry_safety="safe",
        )

    def run(
        self,
        arguments: Mapping[str, Any],
        context: ToolExecutionContext,
    ) -> ToolResult:
        del context
        current_url = str(arguments["url"])
        redirects = 0
        while True:
            host, port, pinned_ip = self._validate_target(current_url)
            response = self._requester(
                current_url,
                pinned_ip,
                self.definition.timeout_seconds,
                self.max_bytes,
            )
            if response.status_code in {301, 302, 303, 307, 308}:
                location = response.headers.get("location")
                if not location:
                    raise ValueError("Redirect response has no Location header")
                if redirects >= self.max_redirects:
                    raise ValueError("Web redirect limit exceeded")
                current_url = urljoin(current_url, location)
                redirects += 1
                continue
            if response.status_code < 200 or response.status_code >= 300:
                raise ValueError(f"HTTP status {response.status_code}")
            if len(response.content) > self.max_bytes:
                raise ValueError(f"Response exceeds {self.max_bytes} byte limit")
            content_type = response.headers.get("content-type", "")
            charset = self._charset(content_type)
            text = response.content.decode(charset, errors="replace")
            payload = {
                "url": current_url,
                "status": response.status_code,
                "content_type": content_type,
                "bytes": len(response.content),
                "content": text,
            }
            return ToolResult(
                tool_call_id="",
                content=json.dumps(payload, ensure_ascii=False),
                metadata={
                    "url": current_url,
                    "host": host,
                    "port": port,
                    "resolved_ip": pinned_ip,
                    "redirects": redirects,
                    "bytes": len(response.content),
                },
            )

    def _validate_target(self, url: str) -> tuple[str, int, str]:
        parsed = urlsplit(url)
        if parsed.scheme.lower() != "https":
            raise PermissionError("Only HTTPS URLs are allowed")
        if parsed.username is not None or parsed.password is not None:
            raise PermissionError("URL credentials are not allowed")
        host = parsed.hostname
        if not host:
            raise ValueError("URL host is required")
        try:
            port = parsed.port or 443
        except ValueError as exc:
            raise ValueError("Invalid URL port") from exc
        addresses = tuple(dict.fromkeys(self._resolver(host, port)))
        if not addresses:
            raise OSError(f"Could not resolve host: {host}")
        parsed_addresses = []
        for address in addresses:
            try:
                parsed_address = ipaddress.ip_address(address)
            except ValueError as exc:
                raise OSError(f"Resolver returned an invalid IP: {address}") from exc
            if not parsed_address.is_global:
                raise PermissionError(
                    f"Non-public address is blocked: {parsed_address}"
                )
            parsed_addresses.append(str(parsed_address))
        return host, port, parsed_addresses[0]

    @staticmethod
    def _resolve_host(host: str, port: int) -> Sequence[str]:
        return [
            str(item[4][0])
            for item in socket.getaddrinfo(
                host,
                port,
                type=socket.SOCK_STREAM,
            )
        ]

    @staticmethod
    def _charset(content_type: str) -> str:
        for part in content_type.split(";")[1:]:
            key, separator, value = part.strip().partition("=")
            if separator and key.lower() == "charset":
                return value.strip("\"'") or "utf-8"
        return "utf-8"

    @staticmethod
    def _fetch_once(
        url: str,
        pinned_ip: str,
        timeout_seconds: float,
        max_bytes: int,
    ) -> WebResponse:
        parsed = urlsplit(url)
        host = parsed.hostname
        if host is None:
            raise ValueError("URL host is required")
        port = parsed.port or 443
        target = parsed.path or "/"
        if parsed.query:
            target += f"?{parsed.query}"
        connection = _PinnedHTTPSConnection(
            host,
            pinned_ip,
            port=port,
            timeout=timeout_seconds,
        )
        display_host = f"[{host}]" if ":" in host else host
        host_header = display_host if port == 443 else f"{display_host}:{port}"
        try:
            connection.request(
                "GET",
                target,
                headers={
                    "Host": host_header,
                    "User-Agent": "PalmClaw-Ubuntu/0.1",
                    "Accept": "text/*,application/json,application/xml;q=0.9",
                },
            )
            response = connection.getresponse()
            content = response.read(max_bytes + 1)
            headers = {key.lower(): value for key, value in response.getheaders()}
            return WebResponse(
                status_code=response.status,
                headers=headers,
                content=content,
            )
        finally:
            connection.close()
