//! Checks up front that the external programs a run needs are installed, so a
//! missing tool is reported once instead of failing page after page.

use std::path::Path;

use anyhow::{Result, bail};

/// External programs that come from one package, plus how to get them.
pub struct Requirement {
    /// Program names looked up in PATH, or paths to the programs.
    programs: Vec<String>,
    /// What to install to get the programs.
    advice: String,
}

impl Requirement {
    pub fn poppler() -> Self {
        Self {
            programs: vec!["pdfinfo".into(), "pdftoppm".into()],
            advice: package_advice("Poppler", "poppler-utils"),
        }
    }

    pub fn djvulibre() -> Self {
        Self {
            programs: vec!["djvused".into(), "ddjvu".into()],
            advice: package_advice("DjVuLibre", "djvulibre-bin"),
        }
    }

    pub fn claude_cli(command: &str) -> Self {
        Self {
            programs: vec![command.into()],
            advice: String::from(
                "install Claude Code (see https://code.claude.com/docs/en/setup) and run \
`claude` once to log in, or set `command` in the provider config to the full path \
of the program",
            ),
        }
    }
}

fn package_advice(name: &str, linux_package: &str) -> String {
    if cfg!(windows) {
        format!("install {name} and make sure the directory with its programs is in PATH")
    } else {
        format!("install {name} (on Linux, the {linux_package} package)")
    }
}

/// Fail with a single message that lists every missing program and how to
/// install it.
pub fn check(requirements: &[Requirement]) -> Result<()> {
    let problems: Vec<String> = requirements
        .iter()
        .filter_map(|req| {
            let missing: Vec<&str> = req
                .programs
                .iter()
                .map(String::as_str)
                .filter(|program| !is_installed(program))
                .collect();
            (!missing.is_empty()).then(|| {
                format!(
                    "Not found: {}. To fix this, {}.",
                    missing.join(", "),
                    req.advice
                )
            })
        })
        .collect();
    if problems.is_empty() {
        Ok(())
    } else {
        bail!("required programs are missing.\n{}", problems.join("\n"))
    }
}

/// Whether `program` can be started: a path to an existing file, or a bare
/// name found in one of the PATH directories.
pub fn is_installed(program: &str) -> bool {
    let path = Path::new(program);
    if path.components().count() > 1 {
        return is_program_file(path);
    }
    std::env::var_os("PATH").is_some_and(|dirs| {
        std::env::split_paths(&dirs).any(|dir| is_program_file(&dir.join(program)))
    })
}

fn is_program_file(path: &Path) -> bool {
    // On Windows a bare name only starts `<name>.exe`.
    if cfg!(windows) && path.extension().is_none() {
        return path.with_extension("exe").is_file();
    }
    path.is_file()
}

#[cfg(test)]
mod tests {
    use super::*;

    const MISSING: &str = "book-transcriber-no-such-program";

    #[test]
    fn existing_path_is_installed() {
        let exe = std::env::current_exe().unwrap();
        assert!(is_installed(exe.to_str().unwrap()));
    }

    #[test]
    fn unknown_program_is_not_installed() {
        assert!(!is_installed(MISSING));
        assert!(!is_installed("/no/such/dir/claude"));
    }

    #[test]
    fn missing_programs_are_reported_together() {
        let requirements = [
            Requirement {
                programs: vec![MISSING.into(), "other-missing-program".into()],
                advice: String::from("install Something"),
            },
            Requirement::claude_cli(MISSING),
        ];
        let message = check(&requirements).unwrap_err().to_string();
        assert!(
            message.contains(&format!(
                "Not found: {MISSING}, other-missing-program. To fix this, install Something."
            )),
            "{message}"
        );
        assert!(message.contains("install Claude Code"), "{message}");
        assert_eq!(message.lines().count(), 3, "{message}");
    }

    #[test]
    fn nothing_missing_passes() {
        let exe = std::env::current_exe().unwrap();
        let requirement = Requirement {
            programs: vec![exe.to_str().unwrap().into()],
            advice: String::new(),
        };
        assert!(check(&[requirement]).is_ok());
    }
}
