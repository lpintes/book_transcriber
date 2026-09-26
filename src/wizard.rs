//! A plain-text setup wizard that writes a first config file when none exists.
//!
//! Questions go one per line with numbered choices and the default in
//! brackets, so the dialog works well with a screen reader.

use std::io::{BufRead, Write};
use std::path::Path;

use anyhow::{Context, Result};

use crate::tools;

const OPENAI_BASE_URL: &str = "https://api.openai.com/v1";

/// Ask the user for a minimal configuration and write it to `path`. Returns
/// `false` when the user skips, declines or ends the input.
pub fn run(input: &mut impl BufRead, out: &mut impl Write, path: &Path) -> Result<bool> {
    let mut prompt = Prompt { input, out };
    match prompt.wizard(path) {
        Ok(created) => Ok(created),
        // End of input means the user left the wizard.
        Err(Answer::Eof) => Ok(false),
        Err(Answer::Io(e)) => Err(e).context("reading the setup answers"),
    }
}

enum Answer {
    Eof,
    Io(std::io::Error),
}

impl From<std::io::Error> for Answer {
    fn from(e: std::io::Error) -> Self {
        Answer::Io(e)
    }
}

struct Prompt<'a, R, W> {
    input: &'a mut R,
    out: &'a mut W,
}

impl<R: BufRead, W: Write> Prompt<'_, R, W> {
    fn wizard(&mut self, path: &Path) -> Result<bool, Answer> {
        writeln!(self.out, "No config file found at {}.", path.display())?;
        if !self.yes_no("Create one now? Type y to create it, or s to skip.")? {
            return Ok(false);
        }

        let claude_found = tools::is_installed("claude");
        writeln!(self.out, "How should btr reach the model?")?;
        writeln!(
            self.out,
            "1. Claude Code with your Claude subscription (claude-cli)"
        )?;
        writeln!(self.out, "2. An OpenAI-compatible API with an API key")?;
        let choice = self.choice(2, if claude_found { 1 } else { 2 })?;

        let (provider_name, provider_body, model_id, dpi) = if choice == 1 {
            if !claude_found {
                writeln!(
                    self.out,
                    "Note: the claude program was not found in PATH. Install Claude Code \
(see https://code.claude.com/docs/en/setup) and log in before transcribing."
                )?;
            }
            let model_id = self.text(
                "Model (sonnet, opus, haiku or a full model ID)",
                Some("sonnet"),
            )?;
            let body = String::from("kind = \"claude-cli\"\n");
            (String::from("ClaudeCLI"), body, model_id, Some(150))
        } else {
            let name = self.text("Provider name", Some("OpenAI"))?;
            let base_url = self.text("Base URL", Some(OPENAI_BASE_URL))?;
            writeln!(
                self.out,
                "The API key is shown as you type and stored as plain text in the config file."
            )?;
            let api_key = self.text("API key", None)?;
            let model_id = self.text("Model ID", None)?;
            let body = format!(
                "base_url = {}\napi_key = {}\n",
                quote(&base_url),
                quote(&api_key)
            );
            (name, body, model_id, None)
        };
        let model_name = self.text("Name for this model in the config", Some(&model_id))?;

        let mut config = format!(
            "default_model = {}\n\n[providers.{}]\n{provider_body}\n[models.{}]\nprovider = {}\nmodel_id = {}\n",
            quote(&model_name),
            quote(&provider_name),
            quote(&model_name),
            quote(&provider_name),
            quote(&model_id),
        );
        if let Some(dpi) = dpi {
            config.push_str(&format!("dpi = {dpi}\n"));
        }

        writeln!(self.out, "The config file will contain:")?;
        writeln!(self.out)?;
        write!(self.out, "{config}")?;
        writeln!(self.out)?;
        if !self.yes_no("Write this file?")? {
            return Ok(false);
        }

        if let Some(dir) = path.parent() {
            std::fs::create_dir_all(dir)?;
        }
        std::fs::write(path, &config)?;
        writeln!(
            self.out,
            "Wrote {}. You can edit it later, for example to set default_prompt.",
            path.display()
        )?;
        writeln!(self.out)?;
        Ok(true)
    }

    /// One line of input, trimmed; `Eof` when the input has ended.
    fn line(&mut self, question: &str, default: Option<&str>) -> Result<String, Answer> {
        match default {
            Some(default) => write!(self.out, "{question} [{default}]: ")?,
            None => write!(self.out, "{question}: ")?,
        }
        self.out.flush()?;
        let mut line = String::new();
        if self.input.read_line(&mut line)? == 0 {
            writeln!(self.out)?;
            return Err(Answer::Eof);
        }
        Ok(line.trim().to_string())
    }

    /// A free-text answer; an empty answer takes the default, or is asked
    /// again when there is none.
    fn text(&mut self, question: &str, default: Option<&str>) -> Result<String, Answer> {
        loop {
            let answer = self.line(question, default)?;
            match (answer.is_empty(), default) {
                (false, _) => return Ok(answer),
                (true, Some(default)) => return Ok(default.to_string()),
                (true, None) => writeln!(self.out, "A value is required.")?,
            }
        }
    }

    /// y or n (s and skip count as n); Enter means y.
    fn yes_no(&mut self, question: &str) -> Result<bool, Answer> {
        loop {
            let answer = self.line(question, Some("y"))?.to_ascii_lowercase();
            match answer.as_str() {
                "" | "y" | "yes" => return Ok(true),
                "n" | "no" | "s" | "skip" => return Ok(false),
                _ => writeln!(self.out, "Please type y or s.")?,
            }
        }
    }

    /// A number from 1 to `count`.
    fn choice(&mut self, count: usize, default: usize) -> Result<usize, Answer> {
        let default_text = default.to_string();
        loop {
            let answer = self.text("Choice", Some(&default_text))?;
            match answer.parse::<usize>() {
                Ok(n) if (1..=count).contains(&n) => return Ok(n),
                _ => writeln!(self.out, "Please type a number from 1 to {count}.")?,
            }
        }
    }
}

/// A TOML basic string with any needed escapes.
fn quote(s: &str) -> String {
    toml::Value::String(s.to_string()).to_string()
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::config::{Config, Endpoint};

    /// Run the wizard on scripted answers; returns (created, transcript, path).
    fn script(name: &str, answers: &str) -> (bool, String, std::path::PathBuf) {
        let dir = std::env::temp_dir().join(format!(
            "book_transcriber-wizard-{}-{name}",
            std::process::id()
        ));
        let _ = std::fs::remove_dir_all(&dir);
        let path = dir.join("book_transcriber").join("config.toml");
        let mut out = Vec::new();
        let created = run(&mut answers.as_bytes(), &mut out, &path).unwrap();
        (created, String::from_utf8(out).unwrap(), path)
    }

    fn cleanup(path: &Path) {
        let _ = std::fs::remove_dir_all(path.parent().unwrap().parent().unwrap());
    }

    #[test]
    fn claude_cli_with_defaults() {
        let (created, _, path) = script("claude", "\n1\n\n\n\n");
        assert!(created);
        let config = Config::load(&path).unwrap();
        let model = config.resolve(None).unwrap();
        assert_eq!(model.name, "sonnet");
        assert_eq!(model.model.model_id, "sonnet");
        assert_eq!(model.model.dpi, Some(150.0));
        match model.endpoint {
            Endpoint::ClaudeCli { command, .. } => assert_eq!(command, "claude"),
            Endpoint::Openai { .. } => panic!("expected a claude-cli endpoint"),
        }
        cleanup(&path);
    }

    #[test]
    fn openai_asks_again_for_required_values() {
        let answers = "y\n2\nMy \"Provider\"\n\n\nsk-123\ngpt-x\nmine\ny\n";
        let (created, transcript, path) = script("openai", answers);
        assert!(created);
        assert!(transcript.contains("A value is required."), "{transcript}");
        let config = Config::load(&path).unwrap();
        let model = config.resolve(None).unwrap();
        assert_eq!(model.name, "mine");
        assert_eq!(model.model.provider, "My \"Provider\"");
        assert_eq!(model.model.model_id, "gpt-x");
        assert!(model.model.dpi.is_none());
        match model.endpoint {
            Endpoint::Openai { base_url, api_key } => {
                assert_eq!(base_url, OPENAI_BASE_URL);
                assert_eq!(api_key, "sk-123");
            }
            Endpoint::ClaudeCli { .. } => panic!("expected an openai endpoint"),
        }
        cleanup(&path);
    }

    #[test]
    fn invalid_choice_is_asked_again() {
        let (created, transcript, path) = script("choice", "y\n7\n1\nopus\n\ny\n");
        assert!(created);
        assert!(
            transcript.contains("Please type a number from 1 to 2."),
            "{transcript}"
        );
        cleanup(&path);
    }

    #[test]
    fn skip_writes_nothing() {
        let (created, _, path) = script("skip", "s\n");
        assert!(!created);
        assert!(!path.exists());
    }

    #[test]
    fn end_of_input_writes_nothing() {
        let (created, _, path) = script("eof", "y\n1\n");
        assert!(!created);
        assert!(!path.exists());
    }

    #[test]
    fn declining_the_summary_writes_nothing() {
        let (created, transcript, path) = script("decline", "y\n1\n\n\nn\n");
        assert!(!created);
        assert!(transcript.contains("model_id = \"sonnet\""), "{transcript}");
        assert!(!path.exists());
    }
}
