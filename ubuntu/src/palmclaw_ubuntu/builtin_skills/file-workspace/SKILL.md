---
name: file-workspace
description: Read and write text files in the isolated session workspace. 파일 조회와 작성.
always: false
---

# File workspace

Use `file_read` and `file_write` only for files needed by the user's request.

- Prefer relative or `session://` paths.
- Use `shared://` only when the user explicitly requests shared workspace data.
- Do not claim a file was read or written until the corresponding Tool succeeds.
- Existing files require an explicit `overwrite=true` Tool argument.
