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
use clap::Parser;

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
#[command(version, about, color = clap::ColorChoice::Never)]
struct Args {
    /// Input source: a directory of images (.png / .jpg / .jpeg), a .pdf file
    /// or a .djvu file.
    input: PathBuf,

    /// Output location. A directory writes one Markdown file per page (and a
    /// `prompt` file there, if present, is used as the user prompt). If omitted,
    /// all pages are combined into a single file named after the input
    /// (e.g. `document.pdf` -> `document.md`), written next to it.
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
    if !config_path.exists() && std::io::stdin().is_terminal() && std::io::stdout().is_terminal() {
        wizard::run(
            &mut std::io::stdin().lock(),
            &mut std::io::stdout(),
            &config_path,
        )?;
    }
    let config = Config::load(&config_path)?;
    let model = config.resolve(args.model.as_deref())?;
    let dpi = args.dpi.or(model.model.dpi).unwrap_or(DEFAULT_DPI);
    if dpi.is_nan() || dpi <= 0.0 {
        bail!("DPI must be positive, got {dpi}");
    }

    let document = DocumentKind::of(&args.input);
    if document.is_none() && !args.input.is_dir() {
        bail!(
            "input {} is neither a .pdf or .djvu file nor a directory",
            args.input.display()
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

    // Directory output => one file per page; no output => a single combined file
    // named after the input.
    let output = match &args.output {
        Some(dir) => Output::PerPage(dir.clone()),
        None => Output::Single(combined_output_path(&args.input)?),
    };
    match &output {
        Output::PerPage(dir) => {
            std::fs::create_dir_all(dir)
                .with_context(|| format!("creating output directory {}", dir.display()))?;
        }
        Output::Single(path) => {
            if path.exists() && !args.overwrite {
                bail!(
                    "output file {} already exists; pass --overwrite to replace it, \
or give an output directory to write one file per page",
                    path.display()
                );
            }
        }
    }

    let prompt_dir = match &output {
        Output::PerPage(dir) => Some(dir.as_path()),
        Output::Single(_) => None,
    };
    let prompt = load_prompt(prompt_dir, config.default_prompt.as_deref())?;

    println!(
        "Model: {} ({} via {})",
        model.name, model.model.model_id, model.model.provider
    );

    // Build the list of pages to transcribe from a PDF, a DjVu or an image dir.
    // For a document, rendered page images live in `_tmp`, kept alive until the run
    // finishes.
    let mut _tmp: Option<TempDir> = None;
    let pending = match document {
        Some(kind) => {
            let doc = kind.open(&args.input)?;
            let (pages, tmp) = document_pages(&args, &output, doc.as_ref(), kind.name(), dpi)?;
            _tmp = Some(tmp);
            pages
        }
        None => image_dir_pages(&args, &output)?,
    };

    if pending.is_empty() {
        println!("Nothing to do.");
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
    // Single-file mode buffers (order, text) here for ordered assembly.
    let collected: Mutex<Vec<(usize, String)>> = Mutex::new(Vec::new());

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
                            let summary = deliver(&output, batch, split, &collected);
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

    // In single-file mode, assemble the collected pages in order and write once.
    if let Output::Single(path) = &output {
        write_combined(path, &pending, collected.into_inner().unwrap())?;
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
enum Output {
    /// One Markdown file per page, written into this directory.
    PerPage(PathBuf),
    /// All pages combined into this single Markdown file.
    Single(PathBuf),
}

/// One page to transcribe.
struct Page {
    /// Source image (an input file, or a rendered document page in a temp dir).
    image: PathBuf,
    /// Human-readable name for progress output (e.g. `12.png` or `page 3`).
    label: String,
    /// Position in the selected sequence; used to reassemble single-file output.
    order: usize,
    /// Destination file in per-page mode; `None` in single-file mode.
    output: Option<PathBuf>,
}

fn file_name(path: &Path) -> String {
    path.file_name()
        .and_then(|n| n.to_str())
        .unwrap_or_default()
        .to_string()
}

/// The destination file for one page in per-page mode, or `None` in single-file
/// mode. Returns `None` via the `skip` flag when the file already exists and
/// resume is active (per-page mode only).
fn page_output(output: &Output, name: &str, overwrite: bool) -> (Option<PathBuf>, bool) {
    match output {
        Output::PerPage(dir) => {
            let path = dir.join(format!("{name}.md"));
            let skip = !overwrite && path.exists();
            (Some(path), skip)
        }
        Output::Single(_) => (None, false),
    }
}

/// Selected, not-yet-done pages from a directory of images (natural order).
fn image_dir_pages(args: &Args, output: &Output) -> Result<Vec<Page>> {
    let mut images = list_images(&args.input)?;
    images.sort_by(|a, b| natural_cmp(&file_name(a), &file_name(b)));
    if images.is_empty() {
        bail!(
            "no .png/.jpg/.jpeg images found in {}",
            args.input.display()
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
    for (order, img) in images[start_idx..end_idx].iter().enumerate() {
        let stem = img.file_stem().and_then(|s| s.to_str()).unwrap_or("page");
        let (out, skip) = page_output(output, stem, args.overwrite);
        if skip {
            skipped += 1;
            continue;
        }
        pending.push(Page {
            image: img.clone(),
            label: file_name(img),
            order,
            output: out,
        });
    }

    println!(
        "Pages {} to {} of {} selected; {} to transcribe, {} already done.",
        args.start,
        end_idx,
        images.len(),
        pending.len(),
        skipped
    );
    Ok(pending)
}

/// Selected, not-yet-done pages from a paged document such as a PDF (`kind`
/// names it in messages). Each pending page is rendered to a PNG in a temp
/// directory (returned so it outlives transcription); in per-page mode output
/// files are named by page number (e.g. `3.md`).
fn document_pages(
    args: &Args,
    output: &Output,
    doc: &dyn PageSource,
    kind: &str,
    dpi: f32,
) -> Result<(Vec<Page>, TempDir)> {
    let total = doc.page_count()?;
    if total == 0 {
        bail!("{kind} {} has no pages", args.input.display());
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
    for (order, page) in (args.start..=end).enumerate() {
        // `page` is the 1-based page number the user sees.
        let name = page.to_string();
        let (out, skip) = page_output(output, &name, args.overwrite);
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
            order,
            output: out,
        });
    }

    println!(
        "Pages {} to {end} selected; {} to transcribe, {skipped} already done.",
        args.start,
        pending.len()
    );
    Ok((pending, tmp))
}

/// Derive the single-file output path from the input: `document.pdf` ->
/// `document.md`, directory `mybook/` -> `mybook.md`, placed next to the input.
fn combined_output_path(input: &Path) -> Result<PathBuf> {
    let stem = match input.file_stem().and_then(|s| s.to_str()) {
        Some(s) if !s.is_empty() => s,
        _ => bail!("cannot derive an output file name from {}", input.display()),
    };
    let parent = input.parent().unwrap_or_else(|| Path::new(""));
    Ok(parent.join(format!("{stem}.md")))
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

/// Resolve the prompt: a `prompt` file in the per-page output directory
/// overrides everything, then the config's `default_prompt`, then the built-in
/// default. `prompt_dir` is `None` in single-file mode (no directory to hold a
/// prompt file), so only the config/built-in defaults apply there.
fn load_prompt(prompt_dir: Option<&Path>, config_default: Option<&str>) -> Result<String> {
    if let Some(dir) = prompt_dir {
        let prompt_file = dir.join("prompt");
        if prompt_file.exists() {
            let text = std::fs::read_to_string(&prompt_file)
                .with_context(|| format!("reading prompt file {}", prompt_file.display()))?;
            return Ok(text.trim().to_string());
        }
    }
    Ok(config_default.unwrap_or(DEFAULT_PROMPT).to_string())
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
        (Some(first), _) if batch.len() == 1 => first.label.clone(),
        (Some(first), Some(last)) => format!("{} to {}", first.label, last.label),
        _ => String::from("(empty)"),
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

/// Write (per-page mode) or buffer (single-file mode) the pages of one batch,
/// returning a short human-readable summary.
fn deliver(
    output: &Output,
    batch: &[Page],
    split: BatchSplit,
    collected: &Mutex<Vec<(usize, String)>>,
) -> Result<String> {
    match output {
        Output::PerPage(dir) => match split {
            BatchSplit::Pages(pairs) => {
                let mut names = Vec::with_capacity(pairs.len());
                for (page, text) in &pairs {
                    let path = page.output.as_ref().expect("per-page mode has a path");
                    std::fs::write(path, text)
                        .with_context(|| format!("writing {}", path.display()))?;
                    names.push(file_name(path));
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
                std::fs::write(&path, &raw)
                    .with_context(|| format!("writing {}", path.display()))?;
                Ok(format!(
                    "could not split {} pages; wrote raw response to {name} \
(re-run these pages with --batch-size 1)",
                    batch.len()
                ))
            }
        },
        Output::Single(_) => {
            let mut buf = collected.lock().unwrap();
            match split {
                BatchSplit::Pages(pairs) => {
                    let labels: Vec<&str> = pairs.iter().map(|(p, _)| p.label.as_str()).collect();
                    let summary = format!("transcribed {}", labels.join(", "));
                    for (page, text) in pairs {
                        buf.push((page.order, text));
                    }
                    Ok(summary)
                }
                BatchSplit::Unsplit(raw) => {
                    // Keep the whole response as one block at the batch's start.
                    buf.push((batch[0].order, raw));
                    Ok(format!(
                        "could not split {} pages; kept as one block \
(re-run with --batch-size 1)",
                        batch.len()
                    ))
                }
            }
        }
    }
}

/// Assemble the buffered single-file pages in page order and write the combined
/// document. Pages that failed leave a visible placeholder rather than a silent
/// gap.
fn write_combined(path: &Path, pending: &[Page], collected: Vec<(usize, String)>) -> Result<()> {
    let texts: HashMap<usize, String> = collected.into_iter().collect();
    let mut sections = Vec::with_capacity(pending.len());
    let mut missing = 0usize;
    for page in pending {
        match texts.get(&page.order) {
            Some(text) => sections.push(text.clone()),
            None => {
                missing += 1;
                sections.push(format!("<!-- {}: transcription failed -->", page.label));
            }
        }
    }
    let doc = sections.join("\n\n");
    std::fs::write(path, &doc).with_context(|| format!("writing {}", path.display()))?;
    println!("\nWrote {} ({} pages).", path.display(), pending.len());
    if missing > 0 {
        eprintln!(
            "Note: {missing} page(s) failed and are marked with placeholders in the document."
        );
    }
    Ok(())
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
