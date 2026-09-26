//! Backend that runs the locally installed Claude Code CLI, so transcription
//! is billed to the user's Claude subscription instead of an API key.

use std::io::{Read, Write};
use std::path::{Path, PathBuf};
use std::process::{Child, Command, ExitStatus, Stdio};
use std::time::{Duration, Instant, SystemTime, UNIX_EPOCH};

use anyhow::{Context, Result, bail};
use serde_json::{Value, json};

use crate::transcriber::{
    Backend, Fatal, RetryConfig, Usage, backoff, clean_output, read_image_base64, truncate,
};

/// A CLI process running longer than this is killed (same as the HTTP timeout).
const TIMEOUT: Duration = Duration::from_secs(600);

/// Minimum wait before retrying after hitting a usage limit or an overload.
const BUSY_MIN_DELAY: Duration = Duration::from_secs(30);

/// Replaces Claude Code's own system prompt, which describes a coding agent,
/// costs thousands of input tokens per request and is irrelevant here. The
/// actual transcription instructions travel in the user message.
const SYSTEM_PROMPT: &str = "You transcribe images of book pages. Follow the user's \
instructions exactly and output only the requested text.";

const INSTALL_HINT: &str = "install Claude Code (see https://code.claude.com/docs/en/setup), \
or set `command` in the provider config to the full path of the `claude` program";

pub struct ClaudeCliBackend {
    command: String,
    args: Vec<String>,
    /// An empty working directory, so the CLI sees neither the book nor any
    /// project settings of the directory btr was started from.
    work_dir: PathBuf,
}

impl ClaudeCliBackend {
    pub fn new(command: &str, extra_args: &[String], model_id: &str) -> Result<Self> {
        let work_dir =
            std::env::temp_dir().join(format!("book_transcriber-cli-{}", std::process::id()));
        std::fs::create_dir_all(&work_dir)
            .with_context(|| format!("creating temp directory {}", work_dir.display()))?;

        let mut args: Vec<String> = Vec::new();
        args.extend(
            [
                // Non-interactive: read one request from stdin, answer, exit.
                "-p",
                "--input-format",
                "stream-json",
                "--output-format",
                "stream-json",
                // With -p, stream-json output is rejected unless --verbose is set.
                "--verbose",
                // No tools at all: the model must not read or write files or
                // run commands; it only turns images into text.
                "--tools",
                "",
                // Load no MCP servers (none are passed via --mcp-config).
                "--strict-mcp-config",
                // Ignore user, project and local settings, including hooks and
                // CLAUDE.md files.
                "--setting-sources",
                "",
                // These one-shot runs don't belong in the session history.
                "--no-session-persistence",
                "--disable-slash-commands",
                "--system-prompt",
                SYSTEM_PROMPT,
                // An alias ("sonnet") or a full model ID.
                "--model",
                model_id,
            ]
            .map(String::from),
        );
        args.extend(extra_args.iter().cloned());

        Ok(Self {
            command: command.to_string(),
            args,
            work_dir,
        })
    }

    /// Run the CLI once for the given stdin line and classify the outcome.
    fn try_once(&self, line: &str) -> Result<(String, Usage), Failure> {
        let mut child = Command::new(&self.command)
            .args(&self.args)
            .current_dir(&self.work_dir)
            .stdin(Stdio::piped())
            .stdout(Stdio::piped())
            .stderr(Stdio::piped())
            .spawn()
            .map_err(|e| {
                if e.kind() == std::io::ErrorKind::NotFound {
                    Failure::Fatal(format!(
                        "could not find the Claude Code CLI `{}`; {INSTALL_HINT}",
                        self.command
                    ))
                } else {
                    Failure::Failed(format!("starting `{}`: {e}", self.command))
                }
            })?;
        let mut stdin = child.stdin.take().expect("stdin is piped");
        let mut stdout = child.stdout.take().expect("stdout is piped");
        let mut stderr = child.stderr.take().expect("stderr is piped");

        // Base64 page images easily reach megabytes, far more than a pipe
        // buffers. Writing stdin while stdout and stderr go unread would
        // deadlock both processes, so each stream gets its own thread.
        let (status, out, err) = std::thread::scope(|scope| {
            scope.spawn(move || {
                // A write error means the process already exited; its output
                // explains why.
                let _ = stdin.write_all(line.as_bytes());
                // Dropping stdin closes it, telling the CLI the input is complete.
            });
            let out = scope.spawn(move || read_all(&mut stdout));
            let err = scope.spawn(move || read_all(&mut stderr));
            let status = wait_with_timeout(&mut child, TIMEOUT);
            (
                status,
                out.join().unwrap_or_default(),
                err.join().unwrap_or_default(),
            )
        });

        match status {
            Err(e) => Err(Failure::Failed(format!(
                "waiting for `{}`: {e}",
                self.command
            ))),
            Ok(None) => Err(Failure::Transient(format!(
                "`{}` did not finish within {} s and was stopped",
                self.command,
                TIMEOUT.as_secs()
            ))),
            Ok(Some(status)) => classify(&out, status.success(), &err, unix_now()),
        }
    }
}

impl Drop for ClaudeCliBackend {
    fn drop(&mut self) {
        let _ = std::fs::remove_dir_all(&self.work_dir);
    }
}

impl Backend for ClaudeCliBackend {
    /// Overload and usage-limit failures are retried with longer waits; a usage
    /// limit with a known reset time, a missing login or a missing CLI stop the
    /// whole run instead.
    fn transcribe(
        &self,
        prompt: &str,
        images: &[&Path],
        retry: RetryConfig,
    ) -> Result<(String, Usage)> {
        let line = user_message(prompt, images)?;
        let busy_retry = RetryConfig {
            base_delay: BUSY_MIN_DELAY,
            max_delay: Duration::from_secs(600),
            ..retry
        };

        let mut attempt: u32 = 0;
        loop {
            let (message, delay) = match self.try_once(&line) {
                Ok(result) => return Ok(result),
                Err(Failure::Fatal(message)) => return Err(Fatal(message).into()),
                Err(Failure::Failed(message)) => bail!("{message}"),
                Err(Failure::Busy(message)) => {
                    let delay = backoff(attempt, &busy_retry).max(BUSY_MIN_DELAY);
                    (message, delay)
                }
                Err(Failure::Transient(message)) => (message, backoff(attempt, &retry)),
            };
            if attempt >= retry.max_retries {
                bail!("{message}");
            }
            eprintln!(
                "  attempt {} failed: {}; retrying in {:.1}s ({} left)",
                attempt + 1,
                message,
                delay.as_secs_f64(),
                retry.max_retries - attempt,
            );
            std::thread::sleep(delay);
            attempt += 1;
        }
    }

    /// Subscriptions have usage limits, so requests run one at a time unless
    /// the user asks for more.
    fn default_jobs(&self) -> usize {
        1
    }
}

/// Why a CLI run failed, grouped by what the caller should do about it.
#[derive(Debug)]
enum Failure {
    /// Retrying cannot help and neither can other batches: stop the run.
    Fatal(String),
    /// Usage limit or overload: retry after a long wait.
    Busy(String),
    /// Timeout or server error: retry after the normal backoff.
    Transient(String),
    /// Give up on this batch.
    Failed(String),
}

/// The single stream-json input line: a user message with the prompt followed
/// by the page images as base64 image blocks.
fn user_message(prompt: &str, images: &[&Path]) -> Result<String> {
    let mut content: Vec<Value> = Vec::with_capacity(images.len() + 1);
    content.push(json!({ "type": "text", "text": prompt }));
    for path in images {
        let (media_type, data) = read_image_base64(path)?;
        content.push(json!({
            "type": "image",
            "source": { "type": "base64", "media_type": media_type, "data": data },
        }));
    }
    let message = json!({
        "type": "user",
        "message": { "role": "user", "content": content },
    });
    Ok(format!("{message}\n"))
}

fn read_all(reader: &mut impl Read) -> String {
    let mut buf = Vec::new();
    let _ = reader.read_to_end(&mut buf);
    String::from_utf8_lossy(&buf).into_owned()
}

/// Wait for the process to exit; kill it once `timeout` elapses, returning
/// `None` in that case.
fn wait_with_timeout(child: &mut Child, timeout: Duration) -> std::io::Result<Option<ExitStatus>> {
    let deadline = Instant::now() + timeout;
    loop {
        match child.try_wait() {
            Ok(Some(status)) => return Ok(Some(status)),
            Ok(None) if Instant::now() < deadline => {
                std::thread::sleep(Duration::from_millis(100));
            }
            other => {
                let _ = child.kill();
                let _ = child.wait();
                return other.map(|_| None);
            }
        }
    }
}

fn unix_now() -> u64 {
    SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .map(|d| d.as_secs())
        .unwrap_or(0)
}

/// Everything of interest in the stream-json output of one CLI run.
#[derive(Debug, Default)]
struct Stream {
    /// The final `result` event, if the CLI got that far.
    result: Option<ResultEvent>,
    /// Text blocks of the assistant's messages, concatenated in order.
    assistant_text: String,
    /// Error category of the last synthetic assistant message, e.g.
    /// `authentication_failed` or `rate_limit`.
    api_error: Option<String>,
    /// A `rate_limit_event` reported that the usage limit rejects requests.
    limit_rejected: bool,
    /// When that limit resets, in Unix seconds.
    resets_at: Option<u64>,
}

#[derive(Debug)]
struct ResultEvent {
    is_error: bool,
    /// The final answer, or the error description on failure.
    text: String,
    api_error_status: Option<u64>,
    usage: Usage,
}

/// Extract the result, assistant text, error category and usage-limit state
/// from the CLI's stream-json output. Lines that aren't JSON are ignored.
fn parse_stream(stdout: &str) -> Stream {
    let mut stream = Stream::default();
    for line in stdout.lines() {
        let Ok(event) = serde_json::from_str::<Value>(line.trim()) else {
            continue;
        };
        match event.get("type").and_then(Value::as_str) {
            Some("assistant") => {
                if let Some(category) = event.get("error").and_then(Value::as_str) {
                    // A synthetic message describing an API error, not an answer.
                    stream.api_error = Some(category.to_string());
                } else if let Some(blocks) =
                    event.pointer("/message/content").and_then(Value::as_array)
                {
                    for block in blocks {
                        if block.get("type").and_then(Value::as_str) == Some("text")
                            && let Some(text) = block.get("text").and_then(Value::as_str)
                        {
                            stream.assistant_text.push_str(text);
                        }
                    }
                }
            }
            Some("rate_limit_event") => {
                let info = &event["rate_limit_info"];
                if info.get("status").and_then(Value::as_str) == Some("rejected") {
                    stream.limit_rejected = true;
                    stream.resets_at = info.get("resetsAt").and_then(Value::as_u64);
                }
            }
            Some("result") => stream.result = Some(parse_result(&event)),
            _ => {}
        }
    }
    stream
}

fn parse_result(event: &Value) -> ResultEvent {
    let subtype = event.get("subtype").and_then(Value::as_str).unwrap_or("");
    // Error subtypes carry `errors` instead of `result`.
    let text = match event.get("result").and_then(Value::as_str) {
        Some(text) => text.to_string(),
        None => event
            .get("errors")
            .and_then(Value::as_array)
            .map(|errors| {
                errors
                    .iter()
                    .filter_map(Value::as_str)
                    .collect::<Vec<_>>()
                    .join("; ")
            })
            .filter(|joined| !joined.is_empty())
            .unwrap_or_else(|| subtype.to_string()),
    };
    let tokens = |key: &str| {
        event
            .get("usage")
            .and_then(|u| u.get(key))
            .and_then(Value::as_u64)
            .unwrap_or(0)
    };
    ResultEvent {
        is_error: event
            .get("is_error")
            .and_then(Value::as_bool)
            .unwrap_or(false)
            || subtype != "success",
        text,
        api_error_status: event.get("api_error_status").and_then(Value::as_u64),
        usage: Usage {
            // Cached prompt tokens are still input tokens of this request.
            prompt_tokens: tokens("input_tokens")
                + tokens("cache_creation_input_tokens")
                + tokens("cache_read_input_tokens"),
            completion_tokens: tokens("output_tokens"),
            cost_usd: event.get("total_cost_usd").and_then(Value::as_f64),
        },
    }
}

/// Decide what a finished CLI run amounts to. `now` (Unix seconds) is used to
/// report how long until a usage limit resets.
fn classify(
    stdout: &str,
    exited_ok: bool,
    stderr: &str,
    now: u64,
) -> Result<(String, Usage), Failure> {
    let stream = parse_stream(stdout);

    match &stream.result {
        Some(result) if exited_ok && !result.is_error => {
            return Ok((clean_output(&result.text), result.usage));
        }
        None if exited_ok
            && stream.api_error.is_none()
            && !stream.assistant_text.trim().is_empty() =>
        {
            // No result event, but the answer itself arrived.
            return Ok((clean_output(&stream.assistant_text), Usage::default()));
        }
        _ => {}
    }

    let message = stream
        .result
        .as_ref()
        .map(|r| r.text.as_str())
        .filter(|text| !text.trim().is_empty())
        .or_else(|| Some(stderr).filter(|text| !text.trim().is_empty()))
        .map(|text| truncate(text, 500))
        .unwrap_or_else(|| String::from("the CLI exited without a result"));
    let category = stream.api_error.as_deref();
    let status = stream.result.as_ref().and_then(|r| r.api_error_status);

    if category == Some("authentication_failed")
        || message.contains("Not logged in")
        || message.contains("/login")
    {
        return Err(Failure::Fatal(format!(
            "Claude Code is not logged in ({message}); run `claude`, log in, then run btr again"
        )));
    }
    if category == Some("model_not_found") {
        return Err(Failure::Fatal(message));
    }

    let usage_limit =
        stream.limit_rejected || (message.contains("hit your") && message.contains("limit"));
    if usage_limit {
        return Err(match stream.resets_at.filter(|&at| at > now) {
            Some(at) => Failure::Fatal(format!(
                "usage limit reached ({message}); it resets in {}. Run btr again after \
that; finished pages are skipped",
                format_wait(at - now)
            )),
            None => Failure::Busy(format!("usage limit reached: {message}")),
        });
    }
    if matches!(category, Some("rate_limit" | "overloaded")) || matches!(status, Some(429 | 529)) {
        return Err(Failure::Busy(message));
    }
    if category == Some("server_error") || matches!(status, Some(500..=599)) {
        return Err(Failure::Transient(message));
    }
    Err(Failure::Failed(message))
}

/// A wait in whole minutes, e.g. `2 h 15 min`.
fn format_wait(secs: u64) -> String {
    let minutes = secs.div_ceil(60);
    match (minutes / 60, minutes % 60) {
        (0, m) => format!("{m} min"),
        (h, 0) => format!("{h} h"),
        (h, m) => format!("{h} h {m} min"),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    const SUCCESS: &str = include_str!("../tests/fixtures/claude_cli/success.jsonl");
    const NOT_LOGGED_IN: &str = include_str!("../tests/fixtures/claude_cli/not_logged_in.jsonl");
    const RATE_LIMIT: &str = include_str!("../tests/fixtures/claude_cli/rate_limit.jsonl");
    const NO_RESULT: &str = include_str!("../tests/fixtures/claude_cli/no_result.jsonl");

    /// `resetsAt` in the rate-limit fixture.
    const RESETS_AT: u64 = 1_790_460_600;

    #[test]
    fn success_returns_text_and_usage() {
        let (text, usage) = classify(SUCCESS, true, "", 0).unwrap();
        assert_eq!(text, "# Chapter One\n\nIt was a bright cold day.");
        assert_eq!(usage.prompt_tokens, 10 + 6600 + 200);
        assert_eq!(usage.completion_tokens, 67);
        assert_eq!(usage.cost_usd, Some(0.013545));
    }

    #[test]
    fn allowed_rate_limit_event_is_not_an_error() {
        let stream = parse_stream(SUCCESS);
        assert!(!stream.limit_rejected);
        assert!(stream.resets_at.is_none());
    }

    #[test]
    fn not_logged_in_is_fatal_with_login_hint() {
        match classify(NOT_LOGGED_IN, false, "", 0) {
            Err(Failure::Fatal(message)) => {
                assert!(message.contains("Not logged in"), "{message}");
                assert!(message.contains("run `claude`"), "{message}");
            }
            other => panic!("expected a fatal failure, got {other:?}"),
        }
    }

    #[test]
    fn usage_limit_with_reset_time_stops_the_run() {
        let now = RESETS_AT - 2 * 3600 - 15 * 60;
        match classify(RATE_LIMIT, false, "", now) {
            Err(Failure::Fatal(message)) => {
                assert!(message.contains("session limit"), "{message}");
                assert!(message.contains("2 h 15 min"), "{message}");
            }
            other => panic!("expected a fatal failure, got {other:?}"),
        }
    }

    #[test]
    fn usage_limit_past_its_reset_time_is_retried() {
        let result = classify(RATE_LIMIT, false, "", RESETS_AT + 10);
        assert!(matches!(result, Err(Failure::Busy(_))), "{result:?}");
    }

    #[test]
    fn missing_result_falls_back_to_assistant_text() {
        let (text, usage) = classify(NO_RESULT, true, "", 0).unwrap();
        assert_eq!(text, "First paragraph.\n\nSecond paragraph.");
        assert_eq!(usage.prompt_tokens, 0);
    }

    #[test]
    fn missing_result_after_a_crash_is_a_failure() {
        let result = classify(NO_RESULT, false, "boom", 0);
        assert!(matches!(result, Err(Failure::Failed(_))), "{result:?}");
    }

    #[test]
    fn no_output_reports_stderr() {
        let stderr = "Error: When using --print, --output-format=stream-json requires --verbose";
        match classify("", false, stderr, 0) {
            Err(Failure::Failed(message)) => assert_eq!(message, stderr),
            other => panic!("expected a failure, got {other:?}"),
        }
    }

    #[test]
    fn overload_is_retried_after_a_long_wait() {
        let stdout = r#"{"type":"assistant","message":{"content":[{"type":"text","text":"API Error: Repeated 529 Overloaded errors."}]},"error":"overloaded","is_api_error_message":true}
{"type":"result","subtype":"success","is_error":true,"api_error_status":529,"result":"API Error: Repeated 529 Overloaded errors."}"#;
        let result = classify(stdout, false, "", 0);
        assert!(matches!(result, Err(Failure::Busy(_))), "{result:?}");
    }

    #[test]
    fn wait_is_formatted_in_minutes() {
        assert_eq!(format_wait(30), "1 min");
        assert_eq!(format_wait(3600), "1 h");
        assert_eq!(format_wait(3 * 3600 + 5 * 60), "3 h 5 min");
    }
}
