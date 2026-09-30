/*
 * Copyright (c) 2023.
 *
 * This software is free software;
 *
 * You can redistribute it or modify it under terms of the MIT, Apache License or Zlib license
 */

//! A baseline JPEG decoded one MCU row at a time.
//!
//! [`JpegDecoder::decode_into`] fills the whole output buffer before it returns,
//! which for a large image is a long time to hold no pixels at all. The decoder
//! underneath has never worked that way: its loop reads one MCU row, colour
//! converts it, and moves on, so the pixels at the top of the image exist long
//! before the bottom is read. [`McuRowReader`] is that loop with its turns
//! separated by a return, so a caller can act on a band of rows while the rest
//! of the image is still being read.
//!
//! Three things it will not do, because a band that is wrong is worse than a
//! band that is late:
//!
//! - **It will not hand out a band of a progressive image.** A progressive
//!   image's blocks are only final once every scan has landed, so the pixels a
//!   scan has produced so far are not early pixels, they are wrong pixels.
//!   [`NotBandable::Progressive`] is how such an image is routed away, and it is
//!   the reason [`JpegDecoder::is_progressive`] is public.
//! - **It will not hand out a band out of order, twice, or skip one.** The rows
//!   a band covers are not an argument to anything: [`McuRowReader::next_row`]
//!   decodes the next MCU row and advances its own cursor, and there is no way
//!   to name a row, ask for one again, or see one twice. The band borrows the
//!   reader, so the next one cannot be asked for until the caller has let go of
//!   this one.
//! - **It will not silently stop early.** A stream that ends before the image
//!   does, or takes a shape this reader does not follow, is an error and no
//!   further bands: a caller that catches one abandons the bands it has and
//!   decodes the image whole, which is the only way a truncated picture can
//!   avoid being shown as a complete one.
//!
//! Rows are handed over in the output colour space the decoder was configured
//! with, exactly as [`JpegDecoder::decode_into`] writes them, and one setup
//! feeds both — so the bands of an image and the whole-image decode of it are
//! the same picture.
//!
//! ```no_run
//! use zune_core::bytestream::ZCursor;
//! use zune_jpeg::{JpegDecoder, McuRowReader};
//!
//! let data = std::fs::read("large.jpg").unwrap();
//! let mut decoder = JpegDecoder::new(ZCursor::new(&data));
//! decoder.decode_headers().unwrap();
//! // Progressive images have no usable band; decode those whole.
//! assert!(!decoder.is_progressive());
//! let mut rows = McuRowReader::new(decoder).unwrap();
//! let stride = rows.width() * rows.components();
//! while let Some(band) = rows.next_row().unwrap() {
//!     for row in band.pixels().chunks_exact(stride) {
//!         // ... one row of the image, band.start() + n from the top
//!         let _ = row;
//!     }
//! }
//! ```

use alloc::vec::Vec;
use core::num::NonZeroUsize;

use zune_core::bytestream::ZByteReaderTrait;
use zune_core::colorspace::ColorSpace;

use crate::decoder::JpegDecoder;
use crate::errors::DecodeErrors;
use crate::mcu::{BaselineRun, McuContinuation};

/// Why a decoder cannot hand back its image one MCU row at a time.
///
/// Every one of these is a routing answer rather than a failure: the image may
/// be fine, it simply has to be decoded whole.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
#[non_exhaustive]
pub enum NotBandable {
    /// The headers have not been read, so nothing is known about the image yet.
    /// Call [`JpegDecoder::decode_headers`] first.
    HeadersNotRead,
    /// The image is progressive. Its scans each carry part of every block, so
    /// no prefix of the image is final until the last scan has landed and any
    /// band taken before then would be a wrong picture rather than an early one.
    Progressive,
    /// The image's components are not all in its first scan, so no MCU row can
    /// be colour converted until a later scan has supplied the rest of them.
    /// This is the shape a baseline image takes when its scans are interleaved
    /// with its MCUs.
    SplitScan
}

/// One MCU row of a baseline image, decoded.
///
/// Borrowed from the reader that produced it, so the rows are only valid until
/// the next [`McuRowReader::next_row`] — which is also why a band cannot be read
/// twice: the borrow checker will not let a caller ask for the next one while
/// this one is still in hand.
pub struct McuRowBand<'a> {
    /// First output row of the image this band covers; `0` for the first band,
    /// and the end of the previous band for every later one.
    start:     usize,
    /// How many output rows it covers. Never zero, so a band that carries no
    /// rows cannot be built.
    rows:      NonZeroUsize,
    /// Bytes per row: the image width times the output colour space's component
    /// count, which is the stride [`Self::pixels`] is laid out in.
    row_bytes: usize,
    /// The pixels, `rows * row_bytes` bytes, row-major from `start`.
    pixels:    &'a [u8]
}

impl<'a> McuRowBand<'a> {
    /// First output row of the image this band covers.
    #[must_use]
    pub const fn start(&self) -> usize {
        self.start
    }

    /// One past the last output row it covers.
    #[must_use]
    pub const fn end(&self) -> usize {
        self.start + self.rows.get()
    }

    /// How many output rows it covers; never zero.
    #[must_use]
    pub const fn rows(&self) -> NonZeroUsize {
        self.rows
    }

    /// Bytes per row, the stride [`Self::pixels`] is laid out in.
    #[must_use]
    pub const fn row_bytes(&self) -> usize {
        self.row_bytes
    }

    /// The pixels of those rows, row-major, `rows * row_bytes` bytes.
    #[must_use]
    pub const fn pixels(&self) -> &'a [u8] {
        self.pixels
    }
}

/// A baseline JPEG read one MCU row at a time.
///
/// The rows come from the decoder's own baseline loop, run one turn per
/// [`Self::next_row`], so a band is not an approximation of a decode happening
/// somewhere else: it is the part of the image the decoder has actually reached.
pub struct McuRowReader<T> {
    /// The decoder doing the reading, held for its components, tables and
    /// options.
    decoder: JpegDecoder<T>,
    /// The baseline loop's state, carried across turns. Built on the first turn
    /// rather than in [`Self::new`], so that opening a reader cannot fail for a
    /// reason other than the image not being one it can follow.
    run:     Option<BaselineRun>,
    /// One MCU row's output, reused. Sized by [`BaselineRun::row_capacity`], so
    /// the loop can never write past it.
    scratch: Vec<u8>,
    /// MCU rows decoded so far; the cursor every band is derived from, and the
    /// only thing that decides which rows come next.
    cursor:  usize,
    /// Output rows handed out so far.
    emitted: usize,
    /// Rows of the image still to be handed out as the grey a whole-image
    /// decode fills them with when the stream ends early. Non-zero only after
    /// a truncated stream, and then every row left in the image is this many.
    padding: usize,
    /// Whether the scratch still holds the rows of the turn that found the end
    /// of the stream, and so has to be overwritten with the grey the padding
    /// bands hand out before the first of them is taken.
    padding_needs_fill: bool,
    /// Set once a turn has failed, so a caller that keeps asking cannot be told
    /// a decode is still in progress when it is not.
    failed:  bool
}

impl<T: ZByteReaderTrait> McuRowReader<T> {
    /// Open `decoder` for row-wise reading, or say why it cannot be.
    ///
    /// The decoder's headers must already have been read — that is what tells
    /// the reader whether the image is one it can follow, and it is what
    /// [`JpegDecoder::decode_headers`] is for.
    ///
    /// # Errors
    /// See [`NotBandable`], every one of which means the image has to be decoded
    /// whole. The decoder is not handed back: none of these says the image is
    /// bad, and a caller that acts on one of them starts a whole-image decode,
    /// which is what the refusal is telling it to do.
    pub fn new(decoder: JpegDecoder<T>) -> Result<Self, NotBandable> {
        decoder.mcu_rows_available()?;
        Ok(Self {
            decoder,
            run: None,
            scratch: Vec::new(),
            cursor: 0,
            emitted: 0,
            padding: 0,
            padding_needs_fill: false,
            failed: false
        })
    }

    /// Image width in pixels, as the headers gave it.
    #[must_use]
    pub fn width(&self) -> usize {
        self.decoder.dimensions().map_or(0, |(width, _)| width)
    }

    /// Image height in pixels, as the headers gave it.
    #[must_use]
    pub fn height(&self) -> usize {
        self.decoder.dimensions().map_or(0, |(_, height)| height)
    }

    /// Samples per pixel of the decoded output.
    ///
    /// This is the configured output colour space's component count — one for
    /// [`ColorSpace::Luma`], three for [`ColorSpace::RGB`] — so one output pixel
    /// of [`McuRowBand::pixels`] is this many bytes.
    #[must_use]
    pub fn components(&self) -> usize {
        self.color_space().num_components()
    }

    /// The colour space the bands are in, which is the one the decoder was
    /// configured to output.
    #[must_use]
    pub fn color_space(&self) -> ColorSpace {
        self.decoder.options().jpeg_get_out_colorspace()
    }

    /// Decode the next MCU row, or `None` once every row of the image has been
    /// handed out.
    ///
    /// The band covers the rows immediately after the last one's, and there is
    /// no argument to make it anything else, so bands cannot overlap, arrive out
    /// of order, or skip rows.
    ///
    /// A truncated stream is not a failure: a whole-image decode fills the rest
    /// of the image with mid-grey and returns, and the bands left in it are
    /// that same grey, so the two paths still agree on what a truncated image is.
    ///
    /// # Errors
    /// - Whatever the decode failed with, and then every later call fails the
    ///   same way: once a turn has failed the reader is spent, because a reader
    ///   that resumed would be handing out rows of a stream it no longer
    ///   understands. The decode is set up on the first turn, so a decoder whose
    ///   components or tables cannot be prepared fails here rather than in
    ///   [`Self::new`].
    /// - If the stream takes a shape this reader does not follow — a second
    ///   start of scan, an inter-scan marker, or an explicit terminate — even
    ///   though the headers promised a single sequential baseline scan. The rows
    ///   already handed out are still what they said they were, but this reader
    ///   will not finish the image, and the caller should decode it whole.
    pub fn next_row(&mut self) -> Result<Option<McuRowBand<'_>>, DecodeErrors> {
        if self.failed {
            return Err(DecodeErrors::FormatStatic(
                "The row-wise decode of this image already failed"
            ));
        }
        if self.run.is_none() {
            let run = match self.decoder.baseline_run_setup() {
                Ok(run) => run,
                Err(error) => {
                    self.failed = true;
                    return Err(error);
                }
            };
            self.scratch = vec![0; run.row_capacity * run.width * self.components()];
            self.run = Some(run);
        }
        let (height, mcu_height) = {
            let run = self.run.as_ref().expect("the setup ran above");
            (self.height(), run.mcu_height)
        };
        if self.emitted >= height {
            return Ok(None);
        }
        let row_bytes = self.width() * self.components();
        // The tail of a truncated image. A whole-image decode fills the rest of
        // its buffer with mid-grey at the point the stream runs out and then
        // returns successfully; these bands are that same grey, handed out a
        // band's worth at a time, so that the two paths still agree about what a
        // truncated image is rather than one of them calling it a failed decode.
        if self.padding > 0 {
            if core::mem::take(&mut self.padding_needs_fill) {
                self.scratch.fill(128);
            }
            let rows = core::cmp::min(self.padding, self.scratch.len() / row_bytes);
            let rows = NonZeroUsize::new(rows).expect("a padding band holds rows");
            let start = self.emitted;
            self.padding -= rows.get();
            self.emitted += rows.get();
            return Ok(Some(McuRowBand {
                start,
                rows,
                row_bytes,
                pixels: &self.scratch[..rows.get() * row_bytes]
            }));
        }
        // Every MCU row has been read and the image's rows have still not all
        // been handed out, which means the decode produced fewer of them than
        // the header named. There is nothing left to read, so this is the end of
        // the image as far as this reader is concerned, and a caller that needs
        // the whole extent will refuse the short raster.
        if self.cursor >= mcu_height {
            return Ok(None);
        }
        // Every band starts where the last one ended and comes from the next MCU
        // row; the only state a band is derived from is this cursor.
        let index = self.cursor;
        let (terminate, written) = {
            let run = self.run.as_mut().expect("the setup ran above");
            // The loop writes at `pixels_written`, so zeroing it is what makes a
            // turn land at the start of this band's own buffer rather than at an
            // offset into the image. It is the one thing a whole-image decode and
            // a row-wise one do differently, and it is one line on purpose.
            run.pixels_written = 0;
            let terminate = self
                .decoder
                .baseline_mcu_row(run, index, &mut self.scratch);
            (terminate, run.pixels_written)
        };
        let terminate = match terminate {
            Ok(terminate) => terminate,
            Err(error) => {
                self.failed = true;
                return Err(error);
            }
        };
        // The two ways a stream can end before its image does, both of which a
        // whole-image decode answers by filling the rest of the image with
        // mid-grey and returning successfully. This reader answers the same way,
        // so a truncated image is the same picture on both paths — grey at the
        // bottom — rather than a whole decode on one and a failed one on the
        // other. The rows this turn did produce are real and are handed out
        // below; everything after them is the grey.
        let truncated = match terminate {
            McuContinuation::Ok => false,
            McuContinuation::PrematureEnd | McuContinuation::Terminate => true,
            // Anything else means the stream is not the single sequential scan
            // the headers described — a second start of scan, or an inter-scan
            // marker the whole-image loop would follow by re-reading its MCU
            // rows from the top, which is not something a cursor can express.
            // Rather than guess at what the rest of it means, stop: the caller
            // has bands it can throw away and a whole decode that will read this
            // image the way the rest of the library does.
            _ => {
                self.failed = true;
                return Err(DecodeErrors::FormatStatic(
                    "The image's scans are not one sequential baseline scan"
                ));
            }
        };
        self.cursor += 1;
        // The image's last MCU row is padded out to a whole MCU, and a whole
        // decode stops at the height the header named because the buffer it
        // writes into does. Clamping here is the same thing said once, and it is
        // also what a truncated image's last real band is clamped by.
        let rows = core::cmp::min(written / row_bytes, height - self.emitted);
        if truncated {
            self.cursor = mcu_height;
            self.padding = height - self.emitted - rows;
            self.padding_needs_fill = true;
        }
        let Some(rows) = NonZeroUsize::new(rows) else {
            if truncated {
                // This turn produced no rows of its own, so the grey starts
                // where the last real band ended; the padding branch above takes
                // it from here.
                return self.next_row();
            }
            self.failed = true;
            return Err(DecodeErrors::FormatStatic(
                "An MCU row produced no output rows"
            ));
        };
        let start = self.emitted;
        self.emitted += rows.get();
        Ok(Some(McuRowBand {
            start,
            rows,
            row_bytes,
            pixels: &self.scratch[..rows.get() * row_bytes]
        }))
    }
}
