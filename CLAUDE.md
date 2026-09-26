# book_transcriber

Fork of RastislavKish/book_transcriber (remote `upstream`). Code, comments, commit messages and the README are in English, in the style of the surrounding code.

## Compatibility

Existing `config.toml` files and command-line options keep their meaning and defaults. New config fields are optional, and their defaults reproduce the previous behavior. The README's Cerebras example is a test (`config::tests`).

## Building and checking

Build and test with default features only; the `mupdf` feature compiles MuPDF's C code, which is unwanted here, so `src/pdf.rs` stays unverified by builds. A step is done when these pass, as in CI:

```
cargo fmt
cargo build
cargo clippy --all-targets -- -D warnings
cargo test
```

## Output for screen readers

Everything btr prints is plain text for screen readers: one `label: value` per line, words instead of symbolic notation (`pages 1 to 2`, not `1..=2`), no colors, no column padding. Install hints name the software to install (Poppler, DjVuLibre, Claude Code) and Linux package names, never one particular Windows package manager.

## claude-cli backend

The flags in `src/claude_cli.rs` were verified against Claude Code 2.1.280; check `claude --help` before changing them. Facts about its stream-json output that the docs don't make obvious:

- `-p` with `--output-format stream-json` requires `--verbose`.
- API failures arrive as a `result` event with `subtype: "success"` and `is_error: true`; the preceding assistant event carries the category in a top-level `error` field.
- `rate_limit_event` appears on every run; only `status: "rejected"` means a usage limit was hit.

`tests/claude_cli.rs` (`harness = false`) is both the integration test and the fake `claude` it runs. When the backend's arguments or stdin message change, update the fake's checks with them.

## Test fixtures

`tests/fixtures/*.pdf` and `*.djvu` are hand-made and marked binary in `.gitattributes`, because line-ending conversion breaks the PDF's xref offsets. Tests that need Poppler or DjVuLibre skip with a message when the tools are missing.
