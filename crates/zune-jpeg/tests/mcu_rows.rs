//! What a row-wise decode promises its caller.
//!
//! Three things, in the order a caller meets them: the bands tile the image and
//! cover every row of it exactly once, the pixels they carry are the pixels a
//! whole-image decode of the same file produces, and an image that has no usable
//! bands — anything progressive, and the split-scan shape a few baseline images
//! take — says so before a single row is handed out rather than producing the
//! pixels of a picture that is not finished.

use std::fs;
use std::path::{Path, PathBuf};

use zune_core::bytestream::ZCursor;
use zune_core::colorspace::ColorSpace;
use zune_core::options::DecoderOptions;
use zune_jpeg::{JpegDecoder, McuRowReader, NotBandable};

/// The largest fixture [`a_row_wise_decode_is_the_whole_image_decode`] takes;
/// above it a comparison costs minutes rather than milliseconds in a test build.
const SMALL_FIXTURE: usize = 5_000_000;

/// Below this a fixture is banded in every output colour space rather than the
/// two that are cheap enough to run over a megapixel.
const SHAVING_FIXTURE: usize = 1_000_000;

/// Every fixture the repository ships, which between them cover the sampling
/// shapes, the component counts and the four input colour spaces the decoder
/// handles.
fn fixtures() -> Vec<PathBuf> {
    let dir = Path::new(env!("CARGO_MANIFEST_DIR")).join("../../test-images/jpeg");
    let mut paths: Vec<PathBuf> = fs::read_dir(&dir)
        .expect("the repository ships jpeg fixtures")
        .filter_map(Result::ok)
        .map(|entry| entry.path())
        .filter(|path| {
            path.is_file()
                && matches!(
                    path.extension().and_then(|e| e.to_str()),
                    Some("jpg" | "jpeg")
                )
        })
        .collect();
    paths.sort();
    assert!(paths.len() > 20, "the corpus went missing");
    paths
}

/// A decoder configured exactly as the whole-image path configures one, with
/// the headers read.
fn opened(data: &[u8], out: ColorSpace) -> JpegDecoder<ZCursor<&[u8]>> {
    let options = DecoderOptions::default()
        .set_strict_mode(false)
        .set_max_width(usize::MAX)
        .set_max_height(usize::MAX)
        .jpeg_set_out_colorspace(out);
    let mut decoder = JpegDecoder::new_with_options(ZCursor::new(data), options);
    decoder.decode_headers().expect("headers decode");
    decoder
}

/// Everything a whole-image decode of `data` produces, through the same
/// options the banded one is opened with.
///
/// `None` for a colour space the decoder does not implement for this image —
/// CMYK to Luma, say — which is a property of the pair and not of the path, so
/// there is nothing to compare and nothing to say about the banded one.
fn whole(data: &[u8], out: ColorSpace) -> Option<Vec<u8>> {
    let mut decoder = opened(data, out);
    let mut buffer = vec![0; decoder.output_buffer_size().expect("headers read")];
    decoder.decode_into(&mut buffer).ok()?;
    Some(buffer)
}

/// Drive a reader to the end, checking as it goes that the bands tile the
/// image: the first starts at row zero, each starts where the last ended, and
/// the last ends at the height.
fn bands(reader: &mut McuRowReader<ZCursor<&[u8]>>) -> Vec<u8> {
    let height = reader.height();
    let stride = reader.width() * reader.components();
    let mut collected = Vec::new();
    let mut next_row = 0;
    while let Some(band) = reader.next_row().expect("a valid image does not fail") {
        assert_eq!(band.start(), next_row, "bands are contiguous and in order");
        assert_eq!(band.row_bytes(), stride, "a band row is one image row");
        assert_eq!(band.pixels().len(), band.rows().get() * stride);
        next_row = band.end();
        collected.extend_from_slice(band.pixels());
    }
    assert_eq!(next_row, height, "the bands cover every row of the image");
    assert_eq!(collected.len(), height * stride);
    collected
}

/// The small half of the corpus, banded in every output colour space and
/// compared against the whole-image decode of the same file.
///
/// The large fixtures are compared by [`a_large_image_is_banded_the_same_way`]
/// instead: a 7680x4320 decode in an unoptimized test build costs minutes, and
/// the shapes worth covering — every sampling factor, every component count,
/// every input colour space — are all in the small ones.
#[test]
fn a_row_wise_decode_is_the_whole_image_decode() {
    let mut exercised = 0;
    for path in fixtures() {
        let data = fs::read(&path).expect("fixture reads");
        let decoder = opened(&data, ColorSpace::RGB);
        let (width, height) = decoder.dimensions().expect("headers were read");
        if decoder.mcu_rows_available().is_err() || width * height > SMALL_FIXTURE {
            continue;
        }
        // Every colour space on the small ones, where it costs nothing, and
        // the two the whole-image path implements for everything on the rest.
        let colours: &[ColorSpace] = if width * height <= SHAVING_FIXTURE {
            &[ColorSpace::RGB, ColorSpace::RGBA, ColorSpace::Luma]
        } else {
            &[ColorSpace::RGB, ColorSpace::RGBA]
        };
        compare(&data, &path, colours);
        exercised += 1;
    }
    assert!(
        exercised >= 12,
        "only {exercised} fixtures were bandable; the corpus should have more"
    );
}

/// The large fixtures, in the one output colour space the whole-image path is
/// guaranteed to implement for all of them. The truncated one is here rather
/// than with the others because its last band is the grey it shares with a
/// whole-image decode.
#[test]
fn a_large_image_is_banded_the_same_way() {
    for name in [
        "large_vertical_samp_7680_4320.jpg",
        "incomplete_image.jpg",
    ] {
        let path = Path::new(env!("CARGO_MANIFEST_DIR"))
            .join("../../test-images/jpeg")
            .join(name);
        let data = fs::read(&path).expect("fixture reads");
        assert_eq!(
            opened(&data, ColorSpace::RGB).mcu_rows_available(),
            Ok(()),
            "{name} is baseline and bandable"
        );
        compare(&data, &path, &[ColorSpace::RGB]);
    }
}

/// Band `data` in each of `out` and check it against the whole-image decode.
fn compare(data: &[u8], path: &Path, out: &[ColorSpace]) {
    for out in out {
        let Some(reference) = whole(data, *out) else {
            continue;
        };
        let mut reader = McuRowReader::new(opened(data, *out)).expect("the probe said it can");
        assert_eq!(reader.color_space(), *out);
        assert_eq!(reader.components(), out.num_components());
        let banded = bands(&mut reader);
        assert_eq!(
            banded,
            reference,
            "the bands of {} are not the image, in {out:?}",
            path.display()
        );
    }
}

#[test]
fn a_progressive_image_is_refused_before_any_row_is_handed_out() {
    // Progressive, and between them: a four-component YCCK image, a
    // down-sampled grayscale one, and an RGB one.
    for name in [
        "Kiara_limited_progressive_four_components.jpg",
        "down_sampled_grayscale_prog.jpg",
        "weird_sampling_2.jpeg",
        "huffman_third_index.jpg",
    ] {
        let path = Path::new(env!("CARGO_MANIFEST_DIR"))
            .join("../../test-images/jpeg")
            .join(name);
        let data = fs::read(&path).expect("fixture reads");
        let decoder = opened(&data, ColorSpace::RGB);
        assert!(decoder.is_progressive(), "{name} is progressive");
        assert_eq!(decoder.mcu_rows_available(), Err(NotBandable::Progressive));
        // The refusal is the constructor's, so a caller cannot hold a reader
        // for an image it would then have to be told not to read.
        assert!(McuRowReader::new(decoder).is_err(), "{name} must not open");
    }
}

#[test]
fn a_baseline_image_whose_scans_are_split_is_refused() {
    // Baseline, but each component arrives in its own scan, so no MCU row can
    // be colour converted until the last one has been read. `is_progressive`
    // alone does not catch these, which is why the refusal is its own answer.
    for name in [
        "non_interleaved_420_64x64.jpg",
        "non_interleaved_444_64x64.jpg",
        "tiny_non_interleaved_444.jpg",
        "sos_news.jpeg",
    ] {
        let path = Path::new(env!("CARGO_MANIFEST_DIR"))
            .join("../../test-images/jpeg")
            .join(name);
        let data = fs::read(&path).expect("fixture reads");
        let decoder = opened(&data, ColorSpace::RGB);
        assert!(!decoder.is_progressive(), "{name} is baseline");
        assert_eq!(decoder.mcu_rows_available(), Err(NotBandable::SplitScan));
        assert!(McuRowReader::new(decoder).is_err(), "{name} must not open");
    }
}

#[test]
fn the_headers_have_to_be_read_before_the_question_can_be_asked() {
    let data = fs::read(fixtures()[0].clone()).expect("fixture reads");
    let decoder = JpegDecoder::new(ZCursor::new(data.as_slice()));
    assert!(!decoder.is_progressive(), "nothing is known before the headers");
    assert_eq!(decoder.mcu_rows_available(), Err(NotBandable::HeadersNotRead));
    assert!(McuRowReader::new(decoder).is_err());
}

#[test]
fn a_truncated_stream_fails_and_stays_failed() {
    let path = Path::new(env!("CARGO_MANIFEST_DIR"))
        .join("../../test-images/jpeg/incomplete_image.jpg");
    let data = fs::read(&path).expect("fixture reads");
    let mut reader = McuRowReader::new(opened(&data, ColorSpace::RGB)).expect("baseline");
    let mut rows = 0;
    loop {
        match reader.next_row() {
            Ok(Some(band)) => rows += band.rows().get(),
            Ok(None) => break,
            Err(_) => {
                assert!(rows > 0, "the failure is mid-stream, not at the start");
                // A reader that resumed would be handing out rows of a stream
                // it no longer understands, so it refuses instead.
                assert!(reader.next_row().is_err());
                return;
            }
        }
    }
    // A truncated image that this decoder fills in is not a failure; either
    // ending is fine so long as the rows never went backwards or repeated,
    // which `bands()` above checks. This test only insists that a failure is
    // terminal.
    assert!(rows > 0);
}

#[test]
fn a_reader_hands_out_the_rows_of_the_image_at_most_once() {
    let data = fs::read(
        Path::new(env!("CARGO_MANIFEST_DIR")).join("../../test-images/jpeg/medium_horiz_samp_2500x1786.jpg"),
    )
    .expect("fixture reads");
    let mut reader = McuRowReader::new(opened(&data, ColorSpace::RGB)).expect("baseline");
    let height = reader.height();
    let mut seen = vec![false; height];
    let mut count = 0;
    while let Some(band) = reader.next_row().expect("decodes") {
        for (row, seen) in seen
            .iter_mut()
            .enumerate()
            .take(band.end())
            .skip(band.start())
        {
            assert!(!*seen, "row {row} was handed out twice");
            *seen = true;
            count += 1;
        }
    }
    assert_eq!(count, height);
    assert!(seen.iter().all(|row| *row), "every row was handed out");
    // Past the end there are no more bands, however often it is asked.
    assert!(reader.next_row().expect("still no failure").is_none());
    assert!(reader.next_row().expect("still no failure").is_none());
}
