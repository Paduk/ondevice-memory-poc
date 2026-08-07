from __future__ import annotations


class TokenCounter:
    def __init__(self, encoding_name: str = "o200k_base"):
        import tiktoken

        self._encoding = tiktoken.get_encoding(encoding_name)

    def count(self, text: str) -> int:
        if not text:
            return 0
        return len(self._encoding.encode(text))

    def truncate(self, text: str, max_tokens: int) -> str:
        if max_tokens <= 0 or not text:
            return ""
        encoded = self._encoding.encode(text)
        if len(encoded) <= max_tokens:
            return text
        return self._encoding.decode(encoded[:max_tokens]).rstrip()
