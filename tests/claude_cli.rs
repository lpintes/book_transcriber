//! End-to-end test of the claude-cli backend against a fake `claude`.
//!
//! This test binary plays both roles. Run by cargo, it writes a config whose
//! `command` points at itself and runs book_transcriber. Run by
//! book_transcriber with the CLI's arguments, it acts as the fake `claude`:
//! it checks the arguments and the stdin message and prints canned
//! stream-json output.

use std::io::Read;
use std::path::{Path, PathBuf};
use std::process::{Command, Output};

use base64::Engine;
use base64::engine::general_purpose::STANDARD as BASE64;
use serde_json::{Value, json};

/// A 16x16 red PNG.
const RED_PNG: &str = "iVBORw0KGgoAAAANSUhEUgAAABAAAAAQCAYAAAAf8/9hAAAAAXNSR0IArs4c6QAAAARnQU1BAACxjwv8YQUAAAAJcEhZcwAADsMAAA7DAcdvqGQAAAAdSURBVDhPY/jPwPCfEsyALkAqHjVg1IBRAwaLAQAwxP4Q7zYsrwAAAABJRU5ErkJggg==";

const NOT_LOGGED_IN: &str = include_str!("fixtures/claude_cli/not_logged_in.jsonl");
const BLANK_PDF: &[u8] = include_bytes!("fixtures/blank-2-pages.pdf");
const RED_DJVU: &[u8] = include_bytes!("fixtures/blank-2-pages.djvu");

/// Selects the fake's behavior; unset means success.
const MODE_VAR: &str = "FAKE_CLAUDE_MODE";
/// The book's directory, which the CLI must not run in.
const BOOK_DIR_VAR: &str = "FAKE_CLAUDE_BOOK_DIR";

fn main() {
    let args: Vec<String> = std::env::args().skip(1).collect();
    if args.iter().any(|a| a == "--input-format") {
        fake_claude(&args);
        return;
    }

    transcribes_pages_through_the_cli();
    println!("test transcribes_pages_through_the_cli ... ok");
    not_logged_in_stops_the_run();
    println!("test not_logged_in_stops_the_run ... ok");
    missing_cli_is_reported_up_front();
    println!("test missing_cli_is_reported_up_front ... ok");
    if Command::new("pdftoppm").arg("-v").output().is_ok() {
        transcribes_a_pdf_with_diacritics_in_its_path();
        println!("test transcribes_a_pdf_with_diacritics_in_its_path ... ok");
    } else {
        println!("test transcribes_a_pdf_with_diacritics_in_its_path ... skipped (Poppler is not installed)");
    }
    if Command::new("ddjvu").arg("--help").output().is_ok() {
        transcribes_remaining_djvu_pages();
        println!("test transcribes_remaining_djvu_pages ... ok");
    } else {
        println!("test transcribes_remaining_djvu_pages ... skipped (DjVuLibre is not installed)");
    }
}

// ---- the fake `claude` ----

fn fake_claude(args: &[String]) {
    if std::env::var(MODE_VAR).as_deref() == Ok("not_logged_in") {
        print!("{NOT_LOGGED_IN}");
        std::process::exit(1);
    }
    match check_request(args) {
        Ok(images) => {
            let text = format!("Fake transcription of {images} image(s).");
            let lines = [
                json!({"type": "system", "subtype": "init", "tools": [], "mcp_servers": []}),
                json!({"type": "assistant", "message": {"role": "assistant",
                    "content": [{"type": "text", "text": text}]}}),
                json!({"type": "result", "subtype": "success", "is_error": false,
                    "result": text, "total_cost_usd": 0.01,
                    "usage": {"input_tokens": 100, "cache_creation_input_tokens": 20,
                        "cache_read_input_tokens": 3, "output_tokens": 7}}),
            ];
            for line in lines {
                println!("{line}");
            }
        }
        Err(problem) => {
            let result = json!({"type": "result", "subtype": "success", "is_error": true,
                "result": format!("fake claude: {problem}")});
            println!("{result}");
            std::process::exit(1);
        }
    }
}

/// Validate the command line and the stdin message; return the image count.
fn check_request(args: &[String]) -> Result<usize, String> {
    let has_flag = |flag: &str| args.iter().any(|a| a == flag);
    let value_of = |flag: &str| {
        args.iter()
            .position(|a| a == flag)
            .and_then(|i| args.get(i + 1))
            .map(String::as_str)
    };
    for flag in ["-p", "--verbose", "--strict-mcp-config", "--no-session-persistence"] {
        if !has_flag(flag) {
            return Err(format!("missing {flag} in {args:?}"));
        }
    }
    if value_of("--tools") != Some("") {
        return Err(format!("tools are not disabled: {args:?}"));
    }
    if value_of("--model") != Some("test-model") {
        return Err(format!("unexpected --model: {args:?}"));
    }
    if value_of("--extra") != Some("yes") {
        return Err(format!("extra_args were not passed: {args:?}"));
    }

    if let Ok(book_dir) = std::env::var(BOOK_DIR_VAR) {
        let cwd = std::env::current_dir().map_err(|e| e.to_string())?;
        if same_dir(&cwd, Path::new(&book_dir)) {
            return Err("running in the book's directory".into());
        }
    }

    let mut input = String::new();
    std::io::stdin()
        .read_to_string(&mut input)
        .map_err(|e| e.to_string())?;
    let lines: Vec<&str> = input.lines().filter(|l| !l.trim().is_empty()).collect();
    if lines.len() != 1 {
        return Err(format!("expected one stdin line, got {}", lines.len()));
    }
    let message: Value = serde_json::from_str(lines[0]).map_err(|e| e.to_string())?;
    if message["type"] != "user" || message["message"]["role"] != "user" {
        return Err("stdin is not a user message".into());
    }
    let content = message["message"]["content"]
        .as_array()
        .ok_or("message has no content array")?;
    if content.first().map(|b| &b["type"]) != Some(&json!("text")) {
        return Err("the first block is not the prompt text".into());
    }
    let images: Vec<&Value> = content.iter().filter(|b| b["type"] == "image").collect();
    if images.is_empty() {
        return Err("no image block".into());
    }
    for image in &images {
        let source = &image["source"];
        if source["type"] != "base64" || source["media_type"] != "image/png" {
            return Err(format!("unexpected image source: {source}"));
        }
        let data = BASE64
            .decode(source["data"].as_str().unwrap_or_default())
            .map_err(|e| format!("image data is not base64: {e}"))?;
        if !data.starts_with(b"\x89PNG") {
            return Err("image data is not a PNG".into());
        }
    }
    Ok(images.len())
}

fn same_dir(a: &Path, b: &Path) -> bool {
    match (a.canonicalize(), b.canonicalize()) {
        (Ok(a), Ok(b)) => a == b,
        _ => false,
    }
}

// ---- the tests ----

struct Fixture {
    root: PathBuf,
    book: PathBuf,
    out: PathBuf,
    config: PathBuf,
}

impl Fixture {
    fn new(name: &str) -> Self {
        // Diacritics and spaces in every path, as is common on Windows.
        let root = std::env::temp_dir().join(format!(
            "book_transcriber test časť {}-{name}",
            std::process::id()
        ));
        let _ = std::fs::remove_dir_all(&root);
        let book = root.join("book");
        std::fs::create_dir_all(&book).unwrap();
        let png = BASE64.decode(RED_PNG).unwrap();
        for page in ["1.png", "2.png"] {
            std::fs::write(book.join(page), &png).unwrap();
        }

        let exe = std::env::current_exe().unwrap();
        let config = root.join("config.toml");
        std::fs::write(
            &config,
            format!(
                "default_model = \"test\"\n\
                 [providers.Fake]\n\
                 kind = \"claude-cli\"\n\
                 command = '{}'\n\
                 extra_args = [\"--extra\", \"yes\"]\n\
                 [models.test]\n\
                 provider = \"Fake\"\n\
                 model_id = \"test-model\"\n",
                exe.display()
            ),
        )
        .unwrap();

        Self {
            out: root.join("out"),
            root,
            book,
            config,
        }
    }

    /// Transcribe the image directory into the per-page output directory.
    fn run(&self, mode: Option<&str>) -> Output {
        self.run_with(&self.book, Some(&self.out), mode)
    }

    fn run_with(&self, input: &Path, output: Option<&Path>, mode: Option<&str>) -> Output {
        let book_dir = if input.is_dir() {
            input
        } else {
            input.parent().unwrap()
        };
        let mut cmd = Command::new(env!("CARGO_BIN_EXE_book_transcriber"));
        cmd.arg(input);
        if let Some(output) = output {
            cmd.arg(output);
        }
        cmd.arg("--config")
            .arg(&self.config)
            .env(BOOK_DIR_VAR, book_dir)
            .env_remove(MODE_VAR);
        if let Some(mode) = mode {
            cmd.env(MODE_VAR, mode);
        }
        cmd.output().unwrap()
    }
}

impl Drop for Fixture {
    fn drop(&mut self) {
        let _ = std::fs::remove_dir_all(&self.root);
    }
}

fn transcribes_pages_through_the_cli() {
    let fixture = Fixture::new("success");
    let output = fixture.run(None);
    let stdout = String::from_utf8_lossy(&output.stdout);
    let stderr = String::from_utf8_lossy(&output.stderr);
    assert!(
        output.status.success(),
        "btr failed\nstdout:\n{stdout}\nstderr:\n{stderr}"
    );

    for page in ["1.md", "2.md"] {
        let text = std::fs::read_to_string(fixture.out.join(page)).unwrap();
        assert_eq!(text, "Fake transcription of 1 image(s).");
    }
    // claude-cli defaults to one request at a time.
    assert!(!stdout.contains("in parallel"), "{stdout}");
    // Two requests, each 100 + 20 + 3 input and 7 output tokens.
    assert!(stdout.contains("input tokens: 246"), "{stdout}");
    assert!(stdout.contains("output tokens: 14"), "{stdout}");
    assert!(stdout.contains("reported cost: $0.0200"), "{stdout}");
}

fn not_logged_in_stops_the_run() {
    let fixture = Fixture::new("not-logged-in");
    let output = fixture.run(Some("not_logged_in"));
    let stderr = String::from_utf8_lossy(&output.stderr);
    assert!(!output.status.success(), "btr should fail\n{stderr}");
    assert!(stderr.contains("log in"), "{stderr}");
    assert!(stderr.contains("1 batch(es) were not attempted"), "{stderr}");
    assert!(!fixture.out.join("1.md").exists());
}

fn missing_cli_is_reported_up_front() {
    let fixture = Fixture::new("missing-cli");
    let exe = std::env::current_exe().unwrap();
    let config = std::fs::read_to_string(&fixture.config).unwrap().replace(
        &exe.display().to_string(),
        "book-transcriber-no-such-claude",
    );
    std::fs::write(&fixture.config, config).unwrap();

    let output = fixture.run(None);
    let stderr = String::from_utf8_lossy(&output.stderr);
    assert!(!output.status.success(), "btr should fail\n{stderr}");
    assert!(
        stderr.contains("Not found: book-transcriber-no-such-claude"),
        "{stderr}"
    );
    assert!(stderr.contains("install Claude Code"), "{stderr}");
    // Nothing was created or attempted.
    assert!(!fixture.out.exists());
    assert!(!stderr.contains("failed:"), "{stderr}");
}

fn transcribes_a_pdf_with_diacritics_in_its_path() {
    let fixture = Fixture::new("pdf");
    let pdf = fixture.root.join("Kniha – časť 1.pdf");
    std::fs::write(&pdf, BLANK_PDF).unwrap();

    // No output argument: both pages go into one file next to the PDF.
    let output = fixture.run_with(&pdf, None, None);
    let stdout = String::from_utf8_lossy(&output.stdout);
    let stderr = String::from_utf8_lossy(&output.stderr);
    assert!(
        output.status.success(),
        "btr failed\nstdout:\n{stdout}\nstderr:\n{stderr}"
    );
    assert!(stdout.contains("PDF: 2 pages"), "{stdout}");
    let text = std::fs::read_to_string(fixture.root.join("Kniha – časť 1.md")).unwrap();
    assert_eq!(
        text,
        "Fake transcription of 1 image(s).\n\nFake transcription of 1 image(s)."
    );
}

fn transcribes_remaining_djvu_pages() {
    let fixture = Fixture::new("djvu");
    // The extension is matched case-insensitively.
    let djvu = fixture.root.join("Kniha – časť 1.DJVU");
    std::fs::write(&djvu, RED_DJVU).unwrap();
    // Page 1 is already done and must be skipped.
    std::fs::create_dir_all(&fixture.out).unwrap();
    std::fs::write(fixture.out.join("1.md"), "done earlier").unwrap();

    let output = fixture.run_with(&djvu, Some(&fixture.out), None);
    let stdout = String::from_utf8_lossy(&output.stdout);
    let stderr = String::from_utf8_lossy(&output.stderr);
    assert!(
        output.status.success(),
        "btr failed\nstdout:\n{stdout}\nstderr:\n{stderr}"
    );
    assert!(stdout.contains("DjVu: 2 pages"), "{stdout}");
    assert!(stdout.contains("1 to transcribe, 1 already done"), "{stdout}");
    let page = |name: &str| std::fs::read_to_string(fixture.out.join(name)).unwrap();
    assert_eq!(page("1.md"), "done earlier");
    assert_eq!(page("2.md"), "Fake transcription of 1 image(s).");
}
