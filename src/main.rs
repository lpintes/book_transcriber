mod claude_cli;
mod config;
mod pages;
#[cfg(feature = "mupdf")]
mod pdf;
mod tools;
mod transcriber;
mod wizard;

use std::cmp::Ordering;
use std::collections::HashMap;
use std::io::IsTerminal;
use std::path::{Path, PathBuf};
use std::sync::Mutex;
use std::sync::atomic::{AtomicBool, AtomicUsize, Ordering as AtomicOrdering};
use std::thread;

use anyhow::{Context, Result, bail};
use clap::error::ErrorKind;
use clap::{CommandFactory, Parser};

use std::time::Duration;

use config::{Config, Endpoint};
use pages::{DocumentKind, PageSource};
use tools::Requirement;
use transcriber::{Fatal, RetryConfig, Usage};

/// Marker the model is asked to place between pages in a multi-image batch.
const PAGE_BREAK: &str = "<<<--- PAGE BREAK --->>>";

/// Resolution for rendering document pages when neither `--dpi` nor the
/// model's `dpi` is set.
const DEFAULT_DPI: f32 = 200.0;

const DEFAULT_PROMPT: &str = "\
You are transcribing scanned pages of a book into clean Markdown plain text. \
Reproduce the text faithfully, preserving reading order, paragraphs, headings, \
lists, and emphasis using Markdown. Do not add commentary, do not summarize, \
and do not wrap your answer in a code fence. Output only the transcription.";

/// Transcribe scanned book pages (a PDF, a DjVu or a directory of images) into
/// Markdown using a vision-capable LLM.
#[derive(Parser, Debug)]
// Plain text only: colored output reads poorly with a screen reader.
#[command(name = "btr", version, about, color = clap::ColorChoice::Never)]
struct Args {
    /// Input source: a directory of images (.png / .jpg / .jpeg), a .pdf file
    /// or a .djvu file. A prompt file named after it (e.g. `book.prompt` next to
    /// `book.pdf`), if present, is used as the user prompt.
    // Optional for clap only, so that a first run without arguments can still
    // reach the setup wizard; `run` requires it right after that.
    input: Option<PathBuf>,

    /// Output location. A directory writes one Markdown file per page (and a
    /// `prompt` file there, if present, is used as the user prompt). If omitted,
    /// the pages go into a work directory named after the input (e.g.
    /// `document.pdf` -> `document.btr`), so an interrupted run can be resumed,
    /// and all transcribed pages are combined into a single file next to the
    /// input (`document.md`).
    output: Option<PathBuf>,

    /// How many images to send to the model in a single request.
    #[arg(short, long, default_value_t = 1)]
    batch_size: usize,

    /// 1-indexed page (position in the sorted list) to start from.
    #[arg(short, long, default_value_t = 1)]
    start: usize,

    /// Number of pages to transcribe (default: all remaining from --start).
    #[arg(short = 'n', long)]
    count: Option<usize>,

    /// Model name to use (a key in [models]); overrides default_model.
    #[arg(short, long)]
    model: Option<String>,

    /// Path to the config file (default: ~/.config/book_transcriber/config.toml,
    /// on Windows %APPDATA%\book_transcriber\config.toml).
    #[arg(long)]
    config: Option<PathBuf>,

    /// Re-transcribe pages even if their output file already exists
    /// (resume/skip is on by default).
    #[arg(long)]
    overwrite: bool,

    /// Retries per request on transient errors (rate limits, overload, 5xx,
    /// network failures) before giving up. Uses exponential backoff.
    #[arg(long, default_value_t = 5)]
    max_retries: u32,

    /// Number of requests to run in parallel (default: 4 for HTTP providers,
    /// 1 for claude-cli).
    #[arg(short, long)]
    jobs: Option<usize>,

    /// Resolution to render PDF and DjVu pages at. Lower values hurt
    /// OCR quality; ~200-300 is a good range. Default: the model's `dpi` from
    /// the config, else 200.
    #[arg(long)]
    dpi: Option<f32>,
}

impl Args {
    /// The input path, which `run` checks for before anything uses it.
    fn input(&self) -> &Path {
        self.input
            .as_deref()
            .expect("input is checked at the start of run")
    }
}

/// A temporary directory removed when this guard is dropped.
struct TempDir {
    path: PathBuf,
}

impl TempDir {
    fn new() -> Result<Self> {
        let path = std::env::temp_dir().join(format!("book_transcriber-{}", std::process::id()));
        std::fs::create_dir_all(&path)
            .with_context(|| format!("creating temp directory {}", path.display()))?;
        Ok(Self { path })
    }
}

impl Drop for TempDir {
    fn drop(&mut self) {
        let _ = std::fs::remove_dir_all(&self.path);
    }
}

fn main() {
    if let Err(err) = run() {
        eprintln!("error: {err:#}");
        std::process::exit(1);
    }
}

fn run() -> Result<()> {
    let args = Args::parse();

    if args.batch_size == 0 {
        bail!("--batch-size must be at least 1");
    }
    if args.start == 0 {
        bail!("--start is 1-indexed and must be at least 1");
    }

    let config_path = match &args.config {
        Some(p) => p.clone(),
        None => Config::default_path()?,
    };
    // Offer to create a missing config interactively; if the user skips it,
    // loading below reports the missing file as before.
    let mut created_config = false;
    if !config_path.exists() && std::io::stdin().is_terminal() && std::io::stdout().is_terminal() {
        created_config = wizard::run(
            &mut std::io::stdin().lock(),
            &mut std::io::stdout(),
            &config_path,
        )?;
    }
    if args.input.is_none() {
        if created_config {
            println!(
                "To transcribe a book, run btr again with its file or directory, e.g. btr book.pdf"
            );
            return Ok(());
        }
        Args::command()
            .error(
                ErrorKind::MissingRequiredArgument,
                "the following required argument was not provided: <INPUT>",
            )
            .exit();
    }
    let config = Config::load(&config_path)?;
    let model = config.resolve(args.model.as_deref())?;
    let dpi = args.dpi.or(model.model.dpi).unwrap_or(DEFAULT_DPI);
    if dpi.is_nan() || dpi <= 0.0 {
        bail!("DPI must be positive, got {dpi}");
    }

    let document = DocumentKind::of(args.input());
    if document.is_none() && !args.input().is_dir() {
        bail!(
            "input {} is neither a .pdf or .djvu file nor a directory",
            args.input().display()
        );
    }

    // Report missing external programs before creating or rendering anything.
    let mut requirements: Vec<Requirement> = document
        .and_then(DocumentKind::requirement)
        .into_iter()
        .collect();
    if let Endpoint::ClaudeCli { command, .. } = model.endpoint {
        requirements.push(Requirement::claude_cli(command));
    }
    tools::check(&requirements)?;

    // Directory output => one file per page; no output => one file per page in a
    // work directory, combined into a single file named after the input.
    let output = match &args.output {
        Some(dir) => Output {
            dir: dir.clone(),
            combined: None,
        },
        None => {
            let combined = combined_output_path(args.input())?;
            let dir = work_dir_path(args.input())?;
            // Without a work directory the file is not btr's to regenerate.
            if combined.exists() && !dir.exists() && !args.overwrite {
                bail!(
                    "output file {} already exists; pass --overwrite to replace it, \
or give an output directory to write one file per page",
                    combined.display()
                );
            }
            Output {
                dir,
                combined: Some(combined),
            }
        }
    };
    std::fs::create_dir_all(&output.dir)
        .with_context(|| format!("creating output directory {}", output.dir.display()))?;

    let (prompt, prompt_source) = load_prompt(
        args.output.as_deref(),
        args.input(),
        config.default_prompt.as_deref(),
    )?;

    println!(
        "Model: {} ({} via {})",
        model.name, model.model.model_id, model.model.provider
    );
    println!("Prompt: {prompt_source}");
    if output.combined.is_some() {
        println!("Work directory: {}", output.dir.display());
    }

    // Build the list of pages to transcribe from a PDF, a DjVu or an image dir.
    // For a document, rendered page images live in `_tmp`, kept alive until the run
    // finishes.
    let mut _tmp: Option<TempDir> = None;
    let (pending, all) = match document {
        Some(kind) => {
            let doc = kind.open(args.input())?;
            let (pages, all, tmp) =
                document_pages(&args, &output.dir, doc.as_ref(), kind.name(), dpi)?;
            _tmp = Some(tmp);
            (pages, all)
        }
        None => image_dir_pages(&args, &output.dir)?,
    };

    if pending.is_empty() {
        println!("Nothing to transcribe.");
        if let Some(path) = &output.combined {
            write_combined(path, &output.dir, &all)?;
        }
        return Ok(());
    }

    let backend = transcriber::backend_for(&model)?;
    let retry = RetryConfig {
        max_retries: args.max_retries,
        base_delay: Duration::from_secs(2),
        max_delay: Duration::from_secs(60),
    };

    let batches: Vec<&[Page]> = pending.chunks(args.batch_size).collect();
    let total_batches = batches.len();
    let jobs = args.jobs.unwrap_or_else(|| backend.default_jobs());
    let workers = jobs.max(1).min(total_batches);
    if workers > 1 {
        println!("Running {workers} requests in parallel.");
    }

    // Shared state for the worker pool.
    let next = AtomicUsize::new(0); // index of the next batch to claim
    let done = AtomicUsize::new(0); // completed batches, for progress display
    let stop = AtomicBool::new(false); // set after a fatal error
    let total = Mutex::new(Usage::default());
    let print_lock = Mutex::new(()); // serializes multi-line console output
    let failures: Mutex<Vec<String>> = Mutex::new(Vec::new());

    thread::scope(|scope| {
        for _ in 0..workers {
            scope.spawn(|| {
                loop {
                    if stop.load(AtomicOrdering::Relaxed) {
                        break;
                    }
                    let idx = next.fetch_add(1, AtomicOrdering::Relaxed);
                    if idx >= total_batches {
                        break;
                    }
                    let batch = batches[idx];
                    let paths: Vec<&Path> = batch.iter().map(|p| p.image.as_path()).collect();
                    let batch_prompt = build_prompt(&prompt, batch.len());
                    let label = batch_label(batch);

                    match backend.transcribe(&batch_prompt, &paths, retry) {
                        Ok((text, usage)) => {
                            total.lock().unwrap().add(&usage);
                            let split = split_batch(batch, &text);
                            let summary = deliver(&output.dir, batch, split);
                            let n = done.fetch_add(1, AtomicOrdering::Relaxed) + 1;
                            let _lock = print_lock.lock().unwrap();
                            match summary {
                                Ok(s) => println!("[{n}/{total_batches}] {label}: {s}"),
                                Err(e) => {
                                    eprintln!("[{n}/{total_batches}] {label}: write failed: {e:#}");
                                    failures.lock().unwrap().push(format!("{label}: {e:#}"));
                                }
                            }
                        }
                        Err(e) => {
                            if e.downcast_ref::<Fatal>().is_some() {
                                stop.store(true, AtomicOrdering::Relaxed);
                            }
                            let _lock = print_lock.lock().unwrap();
                            eprintln!("{label}: failed: {e:#}");
                            failures.lock().unwrap().push(format!("{label}: {e:#}"));
                        }
                    }
                }
            });
        }
    });

    report_usage(&total.into_inner().unwrap(), &model);

    // In single-file mode, combine everything transcribed so far, including
    // pages from earlier runs.
    if let Some(path) = &output.combined {
        write_combined(path, &output.dir, &all)?;
    }

    let failures = failures.into_inner().unwrap();
    if !failures.is_empty() {
        eprintln!("\n{} batch(es) failed:", failures.len());
        for f in &failures {
            eprintln!("  {f}");
        }
        let started = next.into_inner().min(total_batches);
        if started < total_batches {
            eprintln!(
                "Stopped after a fatal error; {} batch(es) were not attempted.",
                total_batches - started
            );
        }
        eprintln!("Re-run to retry the failed pages (already-done pages are skipped).");
        bail!("{} batch(es) failed", failures.len());
    }
    Ok(())
}

/// Where transcriptions go.
struct Output {
    /// One Markdown file per page is written here: the output directory, or in
    /// single-file mode the work directory next to the input.
    dir: PathBuf,
    /// In single-file mode, the file all transcribed pages are combined into.
    combined: Option<PathBuf>,
}

/// One page of the input, whether selected for this run or not.
struct Entry {
    /// Name of its transcription file without `.md` (e.g. `12`).
    name: String,
    /// Human-readable name for messages (e.g. `12.png` or `page 3`).
    label: String,
}

/// One page to transcribe.
struct Page {
    /// Source image (an input file, or a rendered document page in a temp dir).
    image: PathBuf,
    /// Human-readable name for progress output (e.g. `12.png` or `page 3`).
    label: String,
    /// Destination file.
    output: PathBuf,
}

fn file_name(path: &Path) -> String {
    path.file_name()
        .and_then(|n| n.to_str())
        .unwrap_or_default()
        .to_string()
}

/// The destination file for one page, and whether to skip the page because
/// the file already exists and resume is active.
fn page_output(dir: &Path, name: &str, overwrite: bool) -> (PathBuf, bool) {
    let path = dir.join(format!("{name}.md"));
    let skip = !overwrite && path.exists();
    (path, skip)
}

/// Selected, not-yet-done pages from a directory of images (natural order),
/// and all the images as entries.
fn image_dir_pages(args: &Args, dir: &Path) -> Result<(Vec<Page>, Vec<Entry>)> {
    let mut images = list_images(args.input())?;
    images.sort_by(|a, b| natural_cmp(&file_name(a), &file_name(b)));
    if images.is_empty() {
        bail!(
            "no .png/.jpg/.jpeg images found in {}",
            args.input().display()
        );
    }

    let start_idx = args.start - 1;
    if start_idx >= images.len() {
        bail!(
            "--start {} is past the last page ({} images available)",
            args.start,
            images.len()
        );
    }
    let end_idx = match args.count {
        Some(n) => (start_idx + n).min(images.len()),
        None => images.len(),
    };

    let mut pending = Vec::new();
    let mut skipped = 0usize;
    for img in &images[start_idx..end_idx] {
        let (out, skip) = page_output(dir, &file_stem(img), args.overwrite);
        if skip {
            skipped += 1;
            continue;
        }
        pending.push(Page {
            image: img.clone(),
            label: file_name(img),
            output: out,
        });
    }
    let all = images
        .iter()
        .map(|img| Entry {
            name: file_stem(img),
            label: file_name(img),
        })
        .collect();

    println!(
        "Pages {} to {} of {} selected; {} to transcribe, {} already done.",
        args.start,
        end_idx,
        images.len(),
        pending.len(),
        skipped
    );
    Ok((pending, all))
}

/// Selected, not-yet-done pages from a paged document such as a PDF (`kind`
/// names it in messages), and all its pages as entries. Each pending page is
/// rendered to a PNG in a temp directory (returned so it outlives
/// transcription); output files are named by page number (e.g. `3.md`).
fn document_pages(
    args: &Args,
    dir: &Path,
    doc: &dyn PageSource,
    kind: &str,
    dpi: f32,
) -> Result<(Vec<Page>, Vec<Entry>, TempDir)> {
    let total = doc.page_count()?;
    if total == 0 {
        bail!("{kind} {} has no pages", args.input().display());
    }

    let start_idx = args.start - 1;
    if start_idx >= total {
        bail!(
            "--start {} is past the last page ({total} pages in the {kind})",
            args.start
        );
    }
    let end = match args.count {
        Some(n) => (start_idx + n).min(total),
        None => total,
    };
    println!(
        "{kind}: {total} pages; rendering pages {} to {end} at {dpi:.0} DPI.",
        args.start
    );

    let tmp = TempDir::new()?;
    let mut pending = Vec::new();
    let mut skipped = 0usize;
    for page in args.start..=end {
        // `page` is the 1-based page number the user sees.
        let name = page.to_string();
        let (out, skip) = page_output(dir, &name, args.overwrite);
        if skip {
            skipped += 1;
            continue;
        }
        let image = tmp.path.join(format!("{name}.png"));
        doc.render_png(page, dpi, &image)
            .with_context(|| format!("rendering {kind} page {page}"))?;
        pending.push(Page {
            image,
            label: format!("page {page}"),
            output: out,
        });
    }
    let all = (1..=total)
        .map(|page| Entry {
            name: page.to_string(),
            label: format!("page {page}"),
        })
        .collect();

    println!(
        "Pages {} to {end} selected; {} to transcribe, {skipped} already done.",
        args.start,
        pending.len()
    );
    Ok((pending, all, tmp))
}

/// A file next to the input, named after it with extension `ext`:
/// `document.pdf` -> `document.<ext>`, directory `mybook/` -> `mybook.<ext>`.
fn sibling_path(input: &Path, ext: &str) -> Option<PathBuf> {
    let stem = input
        .file_stem()
        .and_then(|s| s.to_str())
        .filter(|s| !s.is_empty())?;
    let parent = input.parent().unwrap_or_else(|| Path::new(""));
    Some(parent.join(format!("{stem}.{ext}")))
}

/// Derive the single-file output path from the input: `document.pdf` ->
/// `document.md`, directory `mybook/` -> `mybook.md`, placed next to the input.
fn combined_output_path(input: &Path) -> Result<PathBuf> {
    sibling_path(input, "md")
        .with_context(|| format!("cannot derive an output file name from {}", input.display()))
}

/// Derive the single-file work directory from the input: `document.pdf` ->
/// `document.btr`, placed next to it.
fn work_dir_path(input: &Path) -> Result<PathBuf> {
    sibling_path(input, "btr").with_context(|| {
        format!(
            "cannot derive a work directory name from {}",
            input.display()
        )
    })
}

fn list_images(dir: &Path) -> Result<Vec<PathBuf>> {
    let mut out = Vec::new();
    let entries = std::fs::read_dir(dir)
        .with_context(|| format!("reading input directory {}", dir.display()))?;
    for entry in entries {
        let path = entry?.path();
        let is_image = path
            .extension()
            .and_then(|e| e.to_str())
            .map(|e| matches!(e.to_ascii_lowercase().as_str(), "png" | "jpg" | "jpeg"))
            .unwrap_or(false);
        if path.is_file() && is_image {
            out.push(path);
        }
    }
    Ok(out)
}

/// Resolve the prompt, returning it with a description of where it came from.
/// The first that exists wins: a `prompt` file in the per-page output directory
/// (`prompt_dir`, `None` in single-file mode), a `<input name>.prompt` file next
/// to the input (e.g. `book.prompt` beside `book.pdf`), the config's
/// `default_prompt`, and the built-in default.
fn load_prompt(
    prompt_dir: Option<&Path>,
    input: &Path,
    config_default: Option<&str>,
) -> Result<(String, String)> {
    let files = prompt_dir
        .map(|dir| dir.join("prompt"))
        .into_iter()
        .chain(sibling_path(input, "prompt"));
    for file in files {
        if file.is_file() {
            let text = std::fs::read_to_string(&file)
                .with_context(|| format!("reading prompt file {}", file.display()))?;
            return Ok((text.trim().to_string(), format!("file {}", file.display())));
        }
    }
    Ok(match config_default {
        Some(text) => (
            text.to_string(),
            String::from("default_prompt from the config"),
        ),
        None => (DEFAULT_PROMPT.to_string(), String::from("built-in default")),
    })
}

/// Augment the base prompt with page-delimiter instructions for batches > 1.
fn build_prompt(base: &str, image_count: usize) -> String {
    if image_count <= 1 {
        base.to_string()
    } else {
        format!(
            "{base}\n\nYou are given {image_count} page images in order. \
Transcribe each one, and separate consecutive pages with a line containing \
exactly:\n{PAGE_BREAK}\nDo not add this marker before the first page or after \
the last one."
        )
    }
}

fn batch_label(batch: &[Page]) -> String {
    match (batch.first(), batch.last()) {
        (Some(first), Some(last)) => range_label(&first.label, &last.label),
        _ => String::from("(empty)"),
    }
}

/// `page 3`, or `page 3 to page 5` for a range of pages.
fn range_label(first: &str, last: &str) -> String {
    if first == last {
        first.to_string()
    } else {
        format!("{first} to {last}")
    }
}

/// The result of splitting a batch response into per-page text.
enum BatchSplit<'a> {
    /// One text section per page, aligned to the batch.
    Pages(Vec<(&'a Page, String)>),
    /// The model didn't delimit as asked; the whole response, unsplit.
    Unsplit(String),
}

/// Split a batch response on the page-break marker. A single-image batch needs
/// no marker; a multi-image batch whose section count doesn't match is returned
/// unsplit rather than risking a misaligned guess.
fn split_batch<'a>(batch: &'a [Page], text: &str) -> BatchSplit<'a> {
    if batch.len() == 1 {
        return BatchSplit::Pages(vec![(&batch[0], text.trim().to_string())]);
    }
    let parts: Vec<&str> = text.split(PAGE_BREAK).map(str::trim).collect();
    if parts.len() == batch.len() {
        BatchSplit::Pages(
            batch
                .iter()
                .zip(parts)
                .map(|(page, part)| (page, part.to_string()))
                .collect(),
        )
    } else {
        BatchSplit::Unsplit(text.to_string())
    }
}

/// Write the pages of one batch into `dir`, returning a short human-readable
/// summary.
fn deliver(dir: &Path, batch: &[Page], split: BatchSplit) -> Result<String> {
    match split {
        BatchSplit::Pages(pairs) => {
            let mut names = Vec::with_capacity(pairs.len());
            for (page, text) in &pairs {
                std::fs::write(&page.output, text)
                    .with_context(|| format!("writing {}", page.output.display()))?;
                names.push(file_name(&page.output));
            }
            Ok(format!("wrote {}", names.join(", ")))
        }
        BatchSplit::Unsplit(raw) => {
            // Don't guess a split; dump the raw response so nothing is lost.
            let name = format!(
                "{}-{}.raw.md",
                file_stem(&batch[0].image),
                file_stem(&batch[batch.len() - 1].image)
            );
            let path = dir.join(&name);
            std::fs::write(&path, &raw).with_context(|| format!("writing {}", path.display()))?;
            Ok(format!(
                "could not split {} pages; wrote raw response to {name} \
(re-run these pages with --batch-size 1)",
                batch.len()
            ))
        }
    }
}

/// Combine the page files in `dir` into one document at `path`, in the order
/// of `all`. A raw response that could not be split stands in for its pages
/// while none of them has a file of its own. Each other run of pages without a
/// file (not transcribed yet or failed) becomes one placeholder line rather
/// than a silent gap.
fn write_combined(path: &Path, dir: &Path, all: &[Entry]) -> Result<()> {
    let mut texts = Vec::with_capacity(all.len());
    for entry in all {
        let file = dir.join(format!("{}.md", entry.name));
        texts.push(read_if_exists(&file)?);
    }
    let raws = raw_responses(dir, all, &texts)?;

    let mut sections = Vec::new();
    let mut gaps = Vec::new();
    let mut unsplit = Vec::new();
    let mut gap: Option<(usize, usize)> = None;
    let mut i = 0;
    while i < all.len() {
        if let Some(text) = &texts[i] {
            close_gap(gap.take(), all, &mut gaps, &mut sections);
            sections.push(text.clone());
            i += 1;
        } else if let Some(raw) = raws.iter().find(|raw| raw.first == i) {
            close_gap(gap.take(), all, &mut gaps, &mut sections);
            let label = range_label(&all[raw.first].label, &all[raw.last].label);
            sections.push(format!(
                "<!-- {label}: not split into pages -->\n\n{}",
                raw.text
            ));
            unsplit.push(label);
            i = raw.last + 1;
        } else {
            gap = Some((gap.map_or(i, |(first, _)| first), i));
            i += 1;
        }
    }
    close_gap(gap, all, &mut gaps, &mut sections);

    let doc = sections.join("\n\n");
    std::fs::write(path, &doc).with_context(|| format!("writing {}", path.display()))?;
    let done = texts.iter().filter(|text| text.is_some()).count();
    println!(
        "\nWrote {}: {done} of {} pages transcribed.",
        path.display(),
        all.len()
    );
    if !unsplit.is_empty() {
        println!("Not split into pages: {}", unsplit.join(", "));
    }
    if !gaps.is_empty() {
        println!("Not transcribed: {}", gaps.join(", "));
    }
    Ok(())
}

/// Record a run of pages without a transcription, given as indices into
/// `all`, and mark it in the document.
fn close_gap(
    gap: Option<(usize, usize)>,
    all: &[Entry],
    gaps: &mut Vec<String>,
    sections: &mut Vec<String>,
) {
    if let Some((first, last)) = gap {
        let label = range_label(&all[first].label, &all[last].label);
        sections.push(format!("<!-- {label}: not transcribed -->"));
        gaps.push(label);
    }
}

/// A raw response in the work directory, for the pages `first..=last` of `all`.
struct RawResponse {
    first: usize,
    last: usize,
    text: String,
}

/// The raw responses in `dir` (named like `12-16.raw.md` by `deliver`) whose
/// pages all still lack a file of their own. Raw responses whose pages all have
/// one now are obsolete and deleted; partly covered ones are left alone.
fn raw_responses(dir: &Path, all: &[Entry], texts: &[Option<String>]) -> Result<Vec<RawResponse>> {
    let index: HashMap<&str, usize> = all
        .iter()
        .enumerate()
        .map(|(i, entry)| (entry.name.as_str(), i))
        .collect();
    let mut raws = Vec::new();
    let entries =
        std::fs::read_dir(dir).with_context(|| format!("reading directory {}", dir.display()))?;
    for entry in entries {
        let path = entry?.path();
        let Some(range) = file_name(&path)
            .strip_suffix(".raw.md")
            .and_then(|stem| raw_range(stem, &index))
        else {
            continue;
        };
        let (first, last) = range;
        let pages = &texts[first..=last];
        if pages.iter().all(Option::is_some) {
            std::fs::remove_file(&path).with_context(|| format!("removing {}", path.display()))?;
        } else if pages.iter().all(Option::is_none) {
            let text = std::fs::read_to_string(&path)
                .with_context(|| format!("reading {}", path.display()))?;
            raws.push(RawResponse { first, last, text });
        }
    }
    raws.sort_by_key(|raw| raw.first);
    Ok(raws)
}

/// The indices of the first and last page in a raw response's name such as
/// `12-16`. Page names may contain `-` themselves, so every split is tried.
fn raw_range(stem: &str, index: &HashMap<&str, usize>) -> Option<(usize, usize)> {
    stem.match_indices('-').find_map(|(at, _)| {
        let first = *index.get(&stem[..at])?;
        let last = *index.get(&stem[at + 1..])?;
        (first <= last).then_some((first, last))
    })
}

/// The contents of `path`, or `None` if it doesn't exist.
fn read_if_exists(path: &Path) -> Result<Option<String>> {
    match std::fs::read_to_string(path) {
        Ok(text) => Ok(Some(text)),
        Err(e) if e.kind() == std::io::ErrorKind::NotFound => Ok(None),
        Err(e) => Err(e).with_context(|| format!("reading {}", path.display())),
    }
}

fn file_stem(path: &Path) -> String {
    path.file_stem()
        .and_then(|s| s.to_str())
        .unwrap_or("page")
        .to_string()
}

fn report_usage(total: &Usage, model: &config::ResolvedModel<'_>) {
    // One "label: value" per line, without column padding, so screen readers
    // don't stumble over runs of spaces.
    println!("\nToken usage:");
    println!("  input tokens: {}", total.prompt_tokens);
    println!("  output tokens: {}", total.completion_tokens);
    println!(
        "  total tokens: {}",
        total.prompt_tokens + total.completion_tokens
    );

    let in_price = model.model.input_price_per_mtok;
    let out_price = model.model.output_price_per_mtok;
    if in_price.is_some() || out_price.is_some() {
        let cost = total.prompt_tokens as f64 / 1e6 * in_price.unwrap_or(0.0)
            + total.completion_tokens as f64 / 1e6 * out_price.unwrap_or(0.0);
        println!("  estimated cost: ${cost:.4}");
    }
    if let Some(cost) = total.cost_usd {
        println!(
            "  reported cost: ${cost:.4} (the backend's own estimate; \
with a subscription this is only indicative)"
        );
    }
}

/// Compare two strings so that embedded numbers order numerically:
/// `2.png` sorts before `10.png`.
fn natural_cmp(a: &str, b: &str) -> Ordering {
    let a: Vec<char> = a.chars().collect();
    let b: Vec<char> = b.chars().collect();
    let (mut i, mut j) = (0usize, 0usize);

    while i < a.len() && j < b.len() {
        if a[i].is_ascii_digit() && b[j].is_ascii_digit() {
            let si = i;
            while i < a.len() && a[i].is_ascii_digit() {
                i += 1;
            }
            let sj = j;
            while j < b.len() && b[j].is_ascii_digit() {
                j += 1;
            }
            // Compare the digit runs, ignoring leading zeros.
            let na: String = a[si..i].iter().collect();
            let nb: String = b[sj..j].iter().collect();
            let ta = na.trim_start_matches('0');
            let tb = nb.trim_start_matches('0');
            let ord = ta.len().cmp(&tb.len()).then_with(|| ta.cmp(tb));
            if ord != Ordering::Equal {
                return ord;
            }
            // Equal value: longer run (more leading zeros) sorts first.
            let ord = na.len().cmp(&nb.len());
            if ord != Ordering::Equal {
                return ord;
            }
        } else {
            let ord = a[i].cmp(&b[j]);
            if ord != Ordering::Equal {
                return ord;
            }
            i += 1;
            j += 1;
        }
    }
    a.len().cmp(&b.len())
}

#[cfg(test)]
mod tests {
    use super::*;

    /// A temp dir holding `Kniha – časť 1.pdf` and an output directory `out`.
    fn book_dir(name: &str) -> (PathBuf, PathBuf, PathBuf) {
        let dir = std::env::temp_dir().join(format!(
            "book_transcriber-prompt-{}-{name}",
            std::process::id()
        ));
        let _ = std::fs::remove_dir_all(&dir);
        let out = dir.join("out");
        std::fs::create_dir_all(&out).unwrap();
        let input = dir.join("Kniha – časť 1.pdf");
        std::fs::write(&input, b"%PDF").unwrap();
        (dir, input, out)
    }

    #[test]
    fn sibling_path_follows_the_input_name() {
        assert_eq!(
            sibling_path(Path::new("dir/book.pdf"), "prompt"),
            Some(PathBuf::from("dir/book.prompt"))
        );
        assert_eq!(
            sibling_path(Path::new("dir/pages/"), "md"),
            Some(PathBuf::from("dir/pages.md"))
        );
        assert_eq!(sibling_path(Path::new(".."), "md"), None);
    }

    #[test]
    fn prompt_sources_in_order() {
        let (dir, input, out) = book_dir("order");
        let load = |prompt_dir: Option<&Path>, config: Option<&str>| {
            load_prompt(prompt_dir, &input, config).unwrap()
        };

        let (text, source) = load(Some(&out), None);
        assert_eq!(text, DEFAULT_PROMPT);
        assert_eq!(source, "built-in default");

        let (text, source) = load(Some(&out), Some("from config"));
        assert_eq!(text, "from config");
        assert_eq!(source, "default_prompt from the config");

        let book_prompt = dir.join("Kniha – časť 1.prompt");
        std::fs::write(&book_prompt, "from the book\n").unwrap();
        let (text, source) = load(Some(&out), Some("from config"));
        assert_eq!(text, "from the book");
        assert_eq!(source, format!("file {}", book_prompt.display()));
        // Single-file mode has no output directory but still finds it.
        assert_eq!(load(None, Some("from config")).0, "from the book");

        std::fs::write(out.join("prompt"), "from the output directory").unwrap();
        assert_eq!(
            load(Some(&out), Some("from config")).0,
            "from the output directory"
        );

        std::fs::remove_dir_all(&dir).unwrap();
    }

    #[test]
    fn combined_file_marks_each_gap_once() {
        let (dir, input, _) = book_dir("combined");
        let work = work_dir_path(&input).unwrap();
        assert_eq!(work, dir.join("Kniha – časť 1.btr"));
        std::fs::create_dir_all(&work).unwrap();
        std::fs::write(work.join("2.md"), "two").unwrap();
        std::fs::write(work.join("5.md"), "five").unwrap();
        std::fs::write(work.join("6.md"), "six").unwrap();
        // Stands in for pages 3 and 4.
        std::fs::write(work.join("3-4.raw.md"), "three and four").unwrap();
        // Page 2 has its own file: ignored, but kept for page 1.
        std::fs::write(work.join("1-2.raw.md"), "one and two").unwrap();
        // Both pages have their own files: obsolete.
        std::fs::write(work.join("5-6.raw.md"), "five and six").unwrap();
        let all: Vec<Entry> = (1..=7)
            .map(|page| Entry {
                name: page.to_string(),
                label: format!("page {page}"),
            })
            .collect();

        let combined = combined_output_path(&input).unwrap();
        write_combined(&combined, &work, &all).unwrap();
        assert_eq!(
            std::fs::read_to_string(&combined).unwrap(),
            "<!-- page 1: not transcribed -->\n\ntwo\n\n\
<!-- page 3 to page 4: not split into pages -->\n\nthree and four\n\n\
five\n\nsix\n\n<!-- page 7: not transcribed -->"
        );
        assert!(work.join("1-2.raw.md").exists());
        assert!(work.join("3-4.raw.md").exists());
        assert!(!work.join("5-6.raw.md").exists());

        std::fs::remove_dir_all(&dir).unwrap();
    }

    #[test]
    fn raw_range_allows_dashes_in_page_names() {
        let index: HashMap<&str, usize> = [("scan-1", 0), ("scan-2", 1)].into();
        assert_eq!(raw_range("scan-1-scan-2", &index), Some((0, 1)));
        assert_eq!(raw_range("scan-2-scan-1", &index), None);
        assert_eq!(raw_range("scan-1", &index), None);
    }
}
