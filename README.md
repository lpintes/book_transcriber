# book_transcriber

This is a simple program which lets me transcribe a PDF, a DjVu or a directory of PNG / JPG images using an arbitrary large language model. LLMs, including relatively small models often runnable on consumer hardware, have gotten pretty good at creating highly accurate transcriptions of even complex documents, where they can wastly outperform traditional solutions. The user can request the output to be processed in a specific way, meaning the LLM can handle page structure - headings, tables, transcribe math notation, describe diagrams, images and plots, and even understand spatially aligned structures, like the Pascal triangle.

With this program, I'm trying to find out the best workflow for processing books with a screenreader, as well as determine the accuracy, strenghts and limitations of LLMs used for this purpose.

## Disclaimer

This project is 100% coded by Claude. I'm just lightly skimming through the output, but I'm not actively writing code nor steering the architectural decisions, because for the size of the project it's not worth it. The program is doing what I need it to do, and it's doing it really well, that's the important part for me. Anyone else is free to decide their priorities for themselves. As always, the project, by its license, does not come with any warranty, see the license text for more details.

## Usage

### An example config file

Since this is a project using large language models, you first need to configure the providers and models to be used. Create a config.toml in ~/.config/book_transcriber (on Windows, `%APPDATA%\book_transcriber\config.toml`, e.g. `C:\Users\<name>\AppData\Roaming\book_transcriber\config.toml`) and give it the following content, replacing the services, models and instructions according to your needs. A config file elsewhere can be selected with `--config`.

```
default_model="qwen-3.8-27b"
default_prompt="Hello! Please transcribe these pages to Markdown. Use LaTeX for math expressions and replace any diagrams or images with placeholder alt descriptions, containing the relevant information for the particular image or diagram."

[providers]

[providers.Cerebras]

base_url="https://api.cerebras.ai/v1"
api_key="..."

[models]

[models."qwen-3.8-27b"]

provider="Cerebras"
model_id="qwen-3.8-27b"
```

Any OpenAI-compatible `/chat/completions` API works this way. Optional model settings are `reasoning_effort`, `max_completion_tokens` (default 25000), `dpi` (see below) and `input_price_per_mtok` / `output_price_per_mtok` for a cost estimate.

### Using a Claude subscription through Claude Code

Instead of an API key, transcription can run through a locally installed and logged-in [Claude Code](https://code.claude.com/docs/en/setup) CLI. Requests are then covered by your Claude subscription (Pro, Max, ...) rather than billed per token to an API account. Add a provider of kind `claude-cli`:

```
[providers.ClaudeCLI]
kind="claude-cli"
# command="claude"      # optional, the program to run; default "claude" from PATH
# extra_args=[]         # optional extra arguments for the CLI

[models.claude-sonnet]
provider="ClaudeCLI"
model_id="sonnet"       # an alias (sonnet, opus, ...) or a full model ID
dpi=150
```

Then `btr -m claude-sonnet book.pdf` (or set `default_model="claude-sonnet"`). Run `claude` once beforehand and log in.

Things worth knowing about this backend:

- A subscription has usage limits instead of a price per token. Requests therefore run one at a time unless you pass `--jobs`. When a limit is reached and the CLI reports when it resets, btr stops and tells you how long to wait; run the same command again later and finished pages are skipped.
- The token usage is reported as usual. The CLI also reports a cost in USD, which btr prints, but with a subscription that figure is only indicative.
- Every batch runs as a fresh `claude -p` process in an empty temporary directory, with all tools, MCP servers, user and project settings and session history disabled. The model only sees the prompt and the page images.
- `reasoning_effort` and `max_completion_tokens` don't apply and are ignored with a warning.

### Page resolution

PDF and DjVu pages are rendered to images at 200 DPI by default. `--dpi` overrides this for one run, and a model can set its own default with `dpi` in its config section. Claude models scale images down anyway once their longer side exceeds about 1568 pixels (2576 for the newest models), while an A4 page at 200 DPI is about 1650 by 2340 pixels, so around 130 to 150 DPI sends less data without losing detail.

### Transcription

I like to alias book_transcriber as btr:

```sh
btr book.pdf
```

Transcribes all pages in book.pdf, and puts them into a book.md file.

```sh
btr book.pdf output_directory
```

Takes book.pdf and saves individual transcription pages into directory output_directory. Any already transcribed pages are skipped.

```sh
btr book.djvu output_directory
```

DjVu files work the same way as PDFs.

```sh
btr book_pages output_directory
```

Reads images from directory book_pages and saves transcriptions into output_directory. If output_directory contains a plain-text file called prompt, this prompt is used for the transcription.

```sh
btr -s 120 -n 10 book.pdf output_directory
```

Transcribes 10 pages starting with page 120 and saves the result into output_directory.

The program also offers other configurable parameters, for example, how many images are given to the model at once, or how many API requests are performed simultaneously. See ```btr --help``` for more information.

## Installation

### Required programs

PDF and DjVu pages are rendered by external programs, which must be in PATH:

- PDF: `pdfinfo` and `pdftoppm` from Poppler.
- DjVu: `djvused` and `ddjvu` from DjVuLibre.
- The `claude-cli` backend: `claude` from Claude Code.

btr checks for them before it starts and names whatever is missing. Only the ones needed for the given input and model are required.

On Linux, install the `poppler-utils` and `djvulibre-bin` packages, e.g. `sudo apt install poppler-utils djvulibre-bin` on Debian or Ubuntu.

On Windows, any installation of Poppler and DjVuLibre works as long as their programs are in PATH. With [Scoop](https://scoop.sh), for example:

```
scoop install poppler djvulibre
```

### Build

You need [Rust](https://rustup.rs). On Linux nothing else is needed. On Windows, the Visual Studio C++ build tools that rustup asks for are enough; Clang, CMake or building C libraries are not required.

```sh
cargo build --release -q
```

The result will be placed in the target/release directory (`book_transcriber` on Linux, `book_transcriber.exe` on Windows). Copy it to a directory in PATH, or run `cargo install --path .`.

### Rendering PDFs with MuPDF

Instead of Poppler, PDFs can be rendered by a statically linked MuPDF, so that no external programs are needed for PDF input. This requires Clang and a C toolchain at build time:

```sh
cargo build --release -q --features mupdf
```

With the feature enabled, MuPDF is always used for PDFs.

## License

Copyright (C) 2026 Rastislav Kish

This program is free software: you can redistribute it and/or modify
it under the terms of the GNU Affero General Public License as published by
the Free Software Foundation, version 3.

This program is distributed in the hope that it will be useful,
but WITHOUT ANY WARRANTY; without even the implied warranty of
MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the
GNU Affero General Public License for more details.

You should have received a copy of the GNU Affero General Public License
along with this program. If not, see <https://www.gnu.org/licenses/>.

