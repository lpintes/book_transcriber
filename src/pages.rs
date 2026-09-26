//! Documents whose pages are rendered to images before transcription.

use std::path::{Path, PathBuf};
use std::process::{Command, Output};

use anyhow::{Context, Result, bail};

/// A paged document that can render single pages to PNG files.
pub trait PageSource {
    fn page_count(&self) -> Result<usize>;

    /// Render 1-based `page` at `dpi` into the PNG file `dest`.
    fn render_png(&self, page: usize, dpi: f32, dest: &Path) -> Result<()>;
}

/// Open a PDF with MuPDF when built with the `mupdf` feature, otherwise with
/// the external Poppler tools.
pub fn open_pdf(path: &Path) -> Result<Box<dyn PageSource>> {
    #[cfg(feature = "mupdf")]
    {
        Ok(Box::new(crate::pdf::Pdf::open(path)?))
    }
    #[cfg(not(feature = "mupdf"))]
    {
        Ok(Box::new(Poppler::new(path)))
    }
}

/// A PDF rendered by Poppler's `pdfinfo` and `pdftoppm`.
#[cfg_attr(feature = "mupdf", allow(dead_code))]
pub struct Poppler {
    path: PathBuf,
}

#[cfg_attr(feature = "mupdf", allow(dead_code))]
impl Poppler {
    pub fn new(path: &Path) -> Self {
        Self {
            path: path.to_path_buf(),
        }
    }
}

impl PageSource for Poppler {
    fn page_count(&self) -> Result<usize> {
        let output = run_tool(Command::new("pdfinfo").arg(&self.path), "pdfinfo")?;
        let stdout = String::from_utf8_lossy(&output.stdout);
        parse_pdfinfo_pages(&stdout).with_context(|| {
            format!(
                "pdfinfo reported no page count for {}",
                self.path.display()
            )
        })
    }

    fn render_png(&self, page: usize, dpi: f32, dest: &Path) -> Result<()> {
        // pdftoppm takes an output prefix and appends ".png" itself.
        if dest.extension().and_then(|e| e.to_str()) != Some("png") {
            bail!("render target {} must end in .png", dest.display());
        }
        let prefix = dest.with_extension("");
        let page = page.to_string();
        run_tool(
            Command::new("pdftoppm")
                .arg("-r")
                .arg(dpi.to_string())
                .args(["-f", &page, "-l", &page, "-png", "-singlefile"])
                .arg(&self.path)
                .arg(&prefix),
            "pdftoppm",
        )?;
        if !dest.is_file() {
            bail!("pdftoppm did not write {}", dest.display());
        }
        Ok(())
    }
}

/// The value of the `Pages:` line in `pdfinfo` output.
fn parse_pdfinfo_pages(output: &str) -> Option<usize> {
    output.lines().find_map(|line| {
        line.strip_prefix("Pages:")
            .and_then(|rest| rest.trim().parse().ok())
    })
}

/// Run an external tool, failing with its stderr when it exits unsuccessfully.
pub fn run_tool(cmd: &mut Command, name: &str) -> Result<Output> {
    let output = match cmd.output() {
        Ok(output) => output,
        Err(e) if e.kind() == std::io::ErrorKind::NotFound => {
            bail!("`{name}` was not found in PATH")
        }
        Err(e) => return Err(e).with_context(|| format!("running `{name}`")),
    };
    if !output.status.success() {
        bail!(
            "`{name}` failed ({}): {}",
            output.status,
            String::from_utf8_lossy(&output.stderr).trim()
        );
    }
    Ok(output)
}

#[cfg(test)]
mod tests {
    use super::*;

    const BLANK_PDF: &[u8] = include_bytes!("../tests/fixtures/blank-2-pages.pdf");

    /// A fresh temp directory whose name has diacritics and spaces.
    fn diacritics_dir(name: &str) -> PathBuf {
        let dir = std::env::temp_dir().join(format!(
            "book_transcriber {name} čšž {}",
            std::process::id()
        ));
        let _ = std::fs::remove_dir_all(&dir);
        std::fs::create_dir_all(&dir).unwrap();
        dir
    }

    /// Whether `tool` can be started; tests needing it are skipped otherwise.
    fn have_tool(tool: &str, version_arg: &str) -> bool {
        let found = Command::new(tool).arg(version_arg).output().is_ok();
        if !found {
            eprintln!("skipping: `{tool}` is not installed");
        }
        found
    }

    #[test]
    fn pdfinfo_pages_line_is_parsed() {
        let output = "Title:           x\nPages:           312\nEncrypted:       no\n";
        assert_eq!(parse_pdfinfo_pages(output), Some(312));
        assert_eq!(parse_pdfinfo_pages("Title: x\n"), None);
    }

    #[test]
    fn poppler_handles_paths_with_diacritics() {
        if !have_tool("pdfinfo", "-v") || !have_tool("pdftoppm", "-v") {
            return;
        }
        let dir = diacritics_dir("poppler");
        let pdf = dir.join("Kniha – časť 1.pdf");
        std::fs::write(&pdf, BLANK_PDF).unwrap();

        let source = Poppler::new(&pdf);
        assert_eq!(source.page_count().unwrap(), 2);
        let dest = dir.join("strana 2.png");
        source.render_png(2, 72.0, &dest).unwrap();
        let png = std::fs::read(&dest).unwrap();
        assert!(png.starts_with(b"\x89PNG"));

        std::fs::remove_dir_all(&dir).unwrap();
    }
}
