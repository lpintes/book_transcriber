use std::path::Path;

use anyhow::{Context, Result};
use mupdf::pixmap::ImageFormat;
use mupdf::{Colorspace, Document, Matrix};

use crate::pages::PageSource;

/// A PDF opened for rendering pages to raster images.
pub struct Pdf {
    doc: Document,
}

impl Pdf {
    pub fn open(path: &Path) -> Result<Self> {
        let doc =
            Document::open(path).with_context(|| format!("opening PDF {}", path.display()))?;
        Ok(Self { doc })
    }

    /// Render 0-based `index` to PNG bytes at the given DPI (PDF user space is
    /// 72 units/inch, so the scale factor is `dpi / 72`).
    fn render_page_png(&self, index: i32, dpi: f32) -> Result<Vec<u8>> {
        let page = self
            .doc
            .load_page(index)
            .with_context(|| format!("loading PDF page {}", index + 1))?;
        let scale = dpi / 72.0;
        let matrix = Matrix::new_scale(scale, scale);
        let pixmap = page
            .to_pixmap(&matrix, &Colorspace::device_rgb(), false, true)
            .with_context(|| format!("rendering PDF page {}", index + 1))?;

        let mut png = Vec::new();
        pixmap
            .write_to(&mut png, ImageFormat::PNG)
            .with_context(|| format!("encoding PDF page {} as PNG", index + 1))?;
        Ok(png)
    }
}

impl PageSource for Pdf {
    fn page_count(&self) -> Result<usize> {
        let count = self.doc.page_count().context("reading PDF page count")?;
        Ok(count.max(0) as usize)
    }

    fn render_png(&self, page: usize, dpi: f32, dest: &Path) -> Result<()> {
        let png = self.render_page_png((page - 1) as i32, dpi)?;
        std::fs::write(dest, png).with_context(|| format!("writing {}", dest.display()))
    }
}
