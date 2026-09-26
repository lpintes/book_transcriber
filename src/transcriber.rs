use std::path::Path;
use std::time::Duration;

use anyhow::{Context, Result, bail};
use base64::Engine;
use base64::engine::general_purpose::STANDARD as BASE64;
use serde_json::{Value, json};

use crate::claude_cli::ClaudeCliBackend;
use crate::config::{DEFAULT_MAX_COMPLETION_TOKENS, Endpoint, ModelConfig, ResolvedModel};

/// Token usage returned by the provider for one request.
#[derive(Debug, Default, Clone, Copy)]
pub struct Usage {
    pub prompt_tokens: u64,
    pub completion_tokens: u64,
    /// Cost estimate reported by the backend itself, if it reports one.
    pub cost_usd: Option<f64>,
}

impl Usage {
    pub fn add(&mut self, other: &Usage) {
        self.prompt_tokens += other.prompt_tokens;
        self.completion_tokens += other.completion_tokens;
        if let Some(cost) = other.cost_usd {
            *self.cost_usd.get_or_insert(0.0) += cost;
        }
    }
}

/// An error that makes every remaining batch pointless (not logged in, usage
/// limit reached, ...). The run stops instead of attempting further batches.
#[derive(Debug)]
pub struct Fatal(pub String);

impl std::fmt::Display for Fatal {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.write_str(&self.0)
    }
}

impl std::error::Error for Fatal {}

/// How to retry requests that fail with a transient error (network problems,
/// rate limits, provider overload, 5xx).
#[derive(Debug, Clone, Copy)]
pub struct RetryConfig {
    /// Number of retries after the initial attempt.
    pub max_retries: u32,
    /// Base delay for exponential backoff.
    pub base_delay: Duration,
    /// Upper bound on any single backoff wait.
    pub max_delay: Duration,
}

impl Default for RetryConfig {
    fn default() -> Self {
        Self {
            max_retries: 5,
            base_delay: Duration::from_secs(2),
            max_delay: Duration::from_secs(60),
        }
    }
}

/// A failed HTTP attempt, tagged with whether it is worth retrying and any
/// server-provided `Retry-After` hint.
struct Attempt {
    retryable: bool,
    message: String,
    retry_after: Option<Duration>,
}

/// A way of sending a prompt plus page images to a model.
pub trait Backend {
    /// Send `prompt` plus every image to the model and return the cleaned text
    /// response together with the reported token usage. Transient failures are
    /// retried per `retry`.
    fn transcribe(
        &self,
        prompt: &str,
        images: &[&Path],
        retry: RetryConfig,
    ) -> Result<(String, Usage)>;

    /// How many requests to run in parallel when `--jobs` is not given.
    fn default_jobs(&self) -> usize;
}

/// Create the backend matching the model's provider kind.
pub fn backend_for(model: &ResolvedModel<'_>) -> Result<Box<dyn Backend + Sync>> {
    match model.endpoint {
        Endpoint::Openai { base_url, api_key } => Ok(Box::new(OpenAiBackend::new(
            model.model,
            base_url,
            api_key,
        )?)),
        Endpoint::ClaudeCli {
            command,
            extra_args,
        } => {
            if model.model.reasoning_effort.is_some() || model.model.max_completion_tokens.is_some()
            {
                eprintln!(
                    "warning: model '{}' sets reasoning_effort or max_completion_tokens; \
the claude-cli backend ignores them",
                    model.name
                );
            }
            Ok(Box::new(ClaudeCliBackend::new(
                command,
                extra_args,
                &model.model.model_id,
            )?))
        }
    }
}

/// An OpenAI-compatible `/chat/completions` HTTP API.
pub struct OpenAiBackend {
    client: reqwest::blocking::Client,
    url: String,
    api_key: String,
    model_id: String,
    max_completion_tokens: u32,
    reasoning_effort: Option<String>,
}

impl OpenAiBackend {
    pub fn new(model: &ModelConfig, base_url: &str, api_key: &str) -> Result<Self> {
        let client = reqwest::blocking::Client::builder()
            .timeout(Duration::from_secs(600))
            .build()
            .context("building HTTP client")?;
        Ok(Self {
            client,
            url: format!("{}/chat/completions", base_url.trim_end_matches('/')),
            api_key: api_key.to_string(),
            model_id: model.model_id.clone(),
            max_completion_tokens: model
                .max_completion_tokens
                .unwrap_or(DEFAULT_MAX_COMPLETION_TOKENS),
            reasoning_effort: model.reasoning_effort.clone(),
        })
    }
}

impl Backend for OpenAiBackend {
    /// Transient failures are retried with exponential backoff per `retry`.
    fn transcribe(
        &self,
        prompt: &str,
        images: &[&Path],
        retry: RetryConfig,
    ) -> Result<(String, Usage)> {
        // Build the request body once; only the network call is retried.
        let mut content: Vec<Value> = Vec::with_capacity(images.len() + 1);
        content.push(json!({ "type": "text", "text": prompt }));
        for path in images {
            let data_url = encode_image(path)?;
            content.push(json!({
                "type": "image_url",
                "image_url": { "url": data_url },
            }));
        }

        let mut body = json!({
            "model": self.model_id,
            "messages": [ { "role": "user", "content": content } ],
            "max_completion_tokens": self.max_completion_tokens,
        });
        if let Some(effort) = &self.reasoning_effort {
            body["reasoning_effort"] = json!(effort);
        }

        let mut attempt: u32 = 0;
        loop {
            match self.try_once(&body) {
                Ok(result) => return Ok(result),
                Err(err) => {
                    if !err.retryable || attempt >= retry.max_retries {
                        bail!("{}", err.message);
                    }
                    let delay = err
                        .retry_after
                        .map(|d| d.min(Duration::from_secs(300)))
                        .unwrap_or_else(|| backoff(attempt, &retry));
                    eprintln!(
                        "  attempt {} failed: {}; retrying in {:.1}s ({} left)",
                        attempt + 1,
                        err.message,
                        delay.as_secs_f64(),
                        retry.max_retries - attempt,
                    );
                    std::thread::sleep(delay);
                    attempt += 1;
                }
            }
        }
    }

    fn default_jobs(&self) -> usize {
        4
    }
}

impl OpenAiBackend {
    /// One HTTP attempt. Errors carry a retryable flag so the caller can decide.
    fn try_once(&self, body: &Value) -> Result<(String, Usage), Attempt> {
        let url = &self.url;
        let response = match self
            .client
            .post(url)
            .bearer_auth(&self.api_key)
            .json(body)
            .send()
        {
            Ok(r) => r,
            // Connection/timeout/DNS failures are transient; retry them.
            Err(e) => {
                return Err(Attempt {
                    retryable: true,
                    message: format!("sending request to {url}: {e}"),
                    retry_after: None,
                });
            }
        };

        let status = response.status();
        let retry_after = parse_retry_after(&response);
        let text = response.text().unwrap_or_default();

        if !status.is_success() {
            // Retry on rate limiting (429), request timeout (408), the
            // Anthropic-style overload (529), and any 5xx. Other 4xx (bad
            // request, auth, not found) are terminal.
            let code = status.as_u16();
            let retryable = matches!(code, 408 | 429 | 529) || status.is_server_error();
            return Err(Attempt {
                retryable,
                message: format!("provider returned {status}: {}", truncate(&text, 500)),
                retry_after,
            });
        }

        let value: Value = match serde_json::from_str(&text) {
            Ok(v) => v,
            Err(e) => {
                return Err(Attempt {
                    retryable: false,
                    message: format!("parsing response body as JSON: {e}"),
                    retry_after: None,
                });
            }
        };

        let message = match value
            .get("choices")
            .and_then(|c| c.get(0))
            .and_then(|c| c.get("message"))
            .and_then(|m| m.get("content"))
            .and_then(|c| c.as_str())
        {
            Some(m) => m.to_string(),
            None => {
                return Err(Attempt {
                    retryable: false,
                    message: format!(
                        "response did not contain choices[0].message.content: {}",
                        truncate(&text, 500)
                    ),
                    retry_after: None,
                });
            }
        };

        let usage = Usage {
            prompt_tokens: value
                .pointer("/usage/prompt_tokens")
                .and_then(Value::as_u64)
                .unwrap_or(0),
            completion_tokens: value
                .pointer("/usage/completion_tokens")
                .and_then(Value::as_u64)
                .unwrap_or(0),
            cost_usd: None,
        };

        Ok((clean_output(&message), usage))
    }
}

/// Exponential backoff with full jitter: a random wait in
/// `[0, min(max_delay, base * 2^attempt)]`.
pub(crate) fn backoff(attempt: u32, retry: &RetryConfig) -> Duration {
    let cap = retry.max_delay.as_secs_f64();
    let exp = retry.base_delay.as_secs_f64() * 2f64.powi(attempt as i32);
    let ceiling = exp.min(cap);
    Duration::from_secs_f64(ceiling * jitter_fraction())
}

/// A pseudo-random fraction in `[0, 1)`, derived from the clock to avoid a
/// dependency on a random-number crate.
fn jitter_fraction() -> f64 {
    use std::time::{SystemTime, UNIX_EPOCH};
    let nanos = SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .map(|d| d.subsec_nanos())
        .unwrap_or(0);
    (nanos % 1_000_000) as f64 / 1_000_000.0
}

/// Parse a `Retry-After` header expressed in seconds.
fn parse_retry_after(response: &reqwest::blocking::Response) -> Option<Duration> {
    response
        .headers()
        .get(reqwest::header::RETRY_AFTER)?
        .to_str()
        .ok()?
        .trim()
        .parse::<u64>()
        .ok()
        .map(Duration::from_secs)
}

pub(crate) fn truncate(s: &str, max: usize) -> String {
    let s = s.trim();
    if s.chars().count() <= max {
        s.to_string()
    } else {
        let cut: String = s.chars().take(max).collect();
        format!("{cut}…")
    }
}

/// Read an image file and return a `data:` URL with base64 payload.
fn encode_image(path: &Path) -> Result<String> {
    let (mime, data) = read_image_base64(path)?;
    Ok(format!("data:{mime};base64,{data}"))
}

/// Read an image file and return its MIME type and base64-encoded contents.
pub(crate) fn read_image_base64(path: &Path) -> Result<(&'static str, String)> {
    let mime = match path
        .extension()
        .and_then(|e| e.to_str())
        .map(str::to_ascii_lowercase)
        .as_deref()
    {
        Some("png") => "image/png",
        Some("jpg") | Some("jpeg") => "image/jpeg",
        other => bail!("unsupported image extension: {other:?}"),
    };
    let bytes = std::fs::read(path).with_context(|| format!("reading image {}", path.display()))?;
    Ok((mime, BASE64.encode(bytes)))
}

/// Strip reasoning-model artifacts: `<think>` blocks and a single wrapping
/// markdown code fence, so what we write is the transcription itself.
pub(crate) fn clean_output(raw: &str) -> String {
    let mut text = raw.to_string();

    // Remove <think>...</think> blocks (some reasoning models emit them).
    while let Some(start) = text.find("<think>") {
        if let Some(end) = text[start..].find("</think>") {
            let end = start + end + "</think>".len();
            text.replace_range(start..end, "");
        } else {
            break;
        }
    }

    let trimmed = text.trim();

    // Unwrap a single fenced block that spans the whole response, e.g.
    // ```markdown\n...\n``` .
    if trimmed.starts_with("```")
        && let Some(first_newline) = trimmed.find('\n')
    {
        let after_fence = &trimmed[first_newline + 1..];
        if let Some(close) = after_fence.rfind("```") {
            return after_fence[..close].trim_end().to_string();
        }
    }

    trimmed.to_string()
}
