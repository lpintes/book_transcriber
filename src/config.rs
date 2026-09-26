use std::collections::HashMap;
use std::path::{Path, PathBuf};

use anyhow::{Context, Result, anyhow};
use serde::Deserialize;

/// Top-level configuration, loaded from
/// `~/.config/book_transcriber/config.toml`.
#[derive(Debug, Deserialize)]
pub struct Config {
    /// Name of the model (a key in `models`) used when `--model` is not given.
    pub default_model: String,
    /// Prompt used when the output directory has no `prompt` file. Falls back
    /// to a built-in default when unset.
    #[serde(default)]
    pub default_prompt: Option<String>,
    /// Providers keyed by name, e.g. `[providers.Cerebras]`.
    #[serde(default)]
    pub providers: HashMap<String, Provider>,
    /// Models keyed by name, e.g. `[models."qwen-3.8-27b"]`.
    #[serde(default)]
    pub models: HashMap<String, ModelConfig>,
}

#[derive(Debug, Deserialize)]
pub struct Provider {
    /// How requests are sent. Defaults to an OpenAI-compatible HTTP API.
    #[serde(default)]
    pub kind: ProviderKind,
    /// OpenAI-compatible base URL, e.g. `https://api.cerebras.ai/v1`.
    /// Required for `kind = "openai"`.
    #[serde(default)]
    pub base_url: Option<String>,
    /// Required for `kind = "openai"`.
    #[serde(default)]
    pub api_key: Option<String>,
    /// Program to run for `kind = "claude-cli"`; defaults to `claude` from PATH.
    #[serde(default)]
    pub command: Option<String>,
    /// Extra arguments passed to the CLI for `kind = "claude-cli"`.
    #[serde(default)]
    pub extra_args: Vec<String>,
}

#[derive(Debug, Default, Clone, Copy, PartialEq, Eq, Deserialize)]
#[serde(rename_all = "kebab-case")]
pub enum ProviderKind {
    /// An OpenAI-compatible `/chat/completions` HTTP API.
    #[default]
    Openai,
    /// The locally installed Claude Code CLI.
    ClaudeCli,
}

/// Connection details of a provider, validated for its kind.
pub enum Endpoint<'a> {
    Openai {
        base_url: &'a str,
        api_key: &'a str,
    },
    ClaudeCli {
        command: &'a str,
        extra_args: &'a [String],
    },
}

/// Used when a model does not set `max_completion_tokens`.
pub const DEFAULT_MAX_COMPLETION_TOKENS: u32 = 25_000;

#[derive(Debug, Deserialize)]
pub struct ModelConfig {
    /// Name of the provider (a key in `providers`).
    pub provider: String,
    /// The provider-side model identifier sent in the request body.
    pub model_id: String,
    /// Optional reasoning effort ("low"/"medium"/"high"). Omit for none.
    #[serde(default)]
    pub reasoning_effort: Option<String>,
    /// Upper bound on tokens the model may generate per request
    /// (default: `DEFAULT_MAX_COMPLETION_TOKENS`).
    #[serde(default)]
    pub max_completion_tokens: Option<u32>,
    /// Resolution to render document pages at for this model; `--dpi`
    /// overrides it. Lets models that downscale large images anyway get
    /// smaller renders.
    #[serde(default)]
    pub dpi: Option<f32>,
    /// Optional pricing, USD per 1M tokens, used only for cost reporting.
    #[serde(default)]
    pub input_price_per_mtok: Option<f64>,
    #[serde(default)]
    pub output_price_per_mtok: Option<f64>,
}

/// A model together with the provider it resolves to.
pub struct ResolvedModel<'a> {
    /// The model's key in `models`.
    pub name: &'a str,
    pub model: &'a ModelConfig,
    pub endpoint: Endpoint<'a>,
}

impl Config {
    /// `~/.config/book_transcriber/config.toml`
    pub fn default_path() -> Result<PathBuf> {
        let dir = dirs::config_dir()
            .context("could not determine the user config directory (~/.config)")?;
        Ok(dir.join("book_transcriber").join("config.toml"))
    }

    pub fn load(path: &Path) -> Result<Config> {
        let text = std::fs::read_to_string(path)
            .with_context(|| format!("reading config file {}", path.display()))?;
        let config: Config = toml::from_str(&text)
            .with_context(|| format!("parsing config file {}", path.display()))?;
        Ok(config)
    }

    /// Resolve a model by name (or the default) together with its provider.
    pub fn resolve<'a>(&'a self, name: Option<&str>) -> Result<ResolvedModel<'a>> {
        let name = name.unwrap_or(&self.default_model);
        let (name, model) = self
            .models
            .get_key_value(name)
            .ok_or_else(|| anyhow!("model '{name}' is not defined in [models]"))?;
        let provider = self.providers.get(&model.provider).ok_or_else(|| {
            anyhow!(
                "model '{name}' references provider '{}', which is not defined in [providers]",
                model.provider
            )
        })?;
        let endpoint = provider.endpoint(&model.provider)?;
        Ok(ResolvedModel {
            name,
            model,
            endpoint,
        })
    }
}

impl Provider {
    /// Check that the fields required by this provider's kind are present.
    fn endpoint<'a>(&'a self, name: &str) -> Result<Endpoint<'a>> {
        match self.kind {
            ProviderKind::Openai => {
                let base_url = self.base_url.as_deref().ok_or_else(|| {
                    anyhow!("provider '{name}' is of kind \"openai\" but has no base_url")
                })?;
                let api_key = self.api_key.as_deref().ok_or_else(|| {
                    anyhow!("provider '{name}' is of kind \"openai\" but has no api_key")
                })?;
                Ok(Endpoint::Openai { base_url, api_key })
            }
            ProviderKind::ClaudeCli => Ok(Endpoint::ClaudeCli {
                command: self.command.as_deref().unwrap_or("claude"),
                extra_args: &self.extra_args,
            }),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// The example config from the README must keep working unchanged.
    const README_EXAMPLE: &str = r#"
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
"#;

    #[test]
    fn readme_example_is_an_openai_provider() {
        let config: Config = toml::from_str(README_EXAMPLE).unwrap();
        assert_eq!(config.providers["Cerebras"].kind, ProviderKind::Openai);
        let resolved = config.resolve(None).unwrap();
        assert_eq!(resolved.model.model_id, "qwen-3.8-27b");
        assert_eq!(resolved.name, "qwen-3.8-27b");
        assert!(resolved.model.max_completion_tokens.is_none());
        assert!(resolved.model.reasoning_effort.is_none());
        match resolved.endpoint {
            Endpoint::Openai { base_url, api_key } => {
                assert_eq!(base_url, "https://api.cerebras.ai/v1");
                assert_eq!(api_key, "...");
            }
            Endpoint::ClaudeCli { .. } => panic!("expected an openai endpoint"),
        }
    }

    #[test]
    fn openai_provider_without_api_key_is_rejected() {
        let config: Config = toml::from_str(
            r#"
default_model = "m"
[providers.P]
base_url = "https://example.com/v1"
[models.m]
provider = "P"
model_id = "x"
"#,
        )
        .unwrap();
        let err = config.resolve(None).err().unwrap().to_string();
        assert!(err.contains("'P'") && err.contains("api_key"), "{err}");
    }

    #[test]
    fn claude_cli_provider_needs_no_url_or_key() {
        let config: Config = toml::from_str(
            r#"
default_model = "claude-sonnet"
[providers.ClaudeCLI]
kind = "claude-cli"
[models.claude-sonnet]
provider = "ClaudeCLI"
model_id = "sonnet"
"#,
        )
        .unwrap();
        let resolved = config.resolve(None).unwrap();
        match resolved.endpoint {
            Endpoint::ClaudeCli {
                command,
                extra_args,
            } => {
                assert_eq!(command, "claude");
                assert!(extra_args.is_empty());
            }
            Endpoint::Openai { .. } => panic!("expected a claude-cli endpoint"),
        }
    }
}
