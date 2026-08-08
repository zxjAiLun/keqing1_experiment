use std::collections::VecDeque;
use std::fs;
use std::io::{Read, Seek, Write};
use std::path::{Path, PathBuf};
use std::process;
use std::time::Instant;
use std::time::{SystemTime, UNIX_EPOCH};

use serde::Serialize;
use zip::write::FileOptions;
use zip::CompressionMethod;
use zip::ZipWriter;

pub fn collect_mjson_files(data_dirs: &[String], smoke: bool) -> Result<Vec<String>, String> {
    let mut out: Vec<String> = Vec::new();
    for data_dir in data_dirs {
        let path = Path::new(data_dir);
        if !path.exists() {
            return Err(format!("data dir does not exist: {data_dir}"));
        }
        if !path.is_dir() {
            return Err(format!("data dir is not a directory: {data_dir}"));
        }
        let mut entries: Vec<PathBuf> = fs::read_dir(path)
            .map_err(|err| format!("failed to read dir {data_dir}: {err}"))?
            .filter_map(|entry| entry.ok().map(|e| e.path()))
            .filter(|p| p.extension().and_then(|s| s.to_str()) == Some("mjson"))
            .collect();
        entries.sort();
        if smoke {
            entries.truncate(1);
        }
        out.extend(entries.into_iter().map(|p| p.to_string_lossy().to_string()));
    }
    Ok(out)
}

pub fn output_npz_path(output_root: &Path, input_file: &str) -> PathBuf {
    let input_path = Path::new(input_file);
    let ds_name = input_path
        .parent()
        .and_then(|p| p.file_name())
        .and_then(|s| s.to_str())
        .unwrap_or("dataset");
    output_root.join(ds_name).join(
        input_path
            .file_stem()
            .and_then(|s| s.to_str())
            .map(|s| format!("{s}.npz"))
            .unwrap_or_else(|| "sample.npz".to_string()),
    )
}

pub fn format_duration_s(seconds: f64) -> String {
    if seconds >= 3600.0 {
        format!("{:.1}h", seconds / 3600.0)
    } else if seconds >= 60.0 {
        format!("{:.1}m", seconds / 60.0)
    } else {
        format!("{:.0}s", seconds)
    }
}

pub fn print_export_progress(
    label: &str,
    start: &Instant,
    done: usize,
    total: usize,
    processed_file_count: usize,
    skipped_existing_file_count: usize,
    exported_sample_count: usize,
    recent_file_seconds: &VecDeque<f64>,
    jobs: usize,
) {
    let elapsed_s = start.elapsed().as_secs_f64();
    let recent_avg_s = if recent_file_seconds.is_empty() {
        0.0
    } else {
        recent_file_seconds.iter().sum::<f64>() / recent_file_seconds.len() as f64
    };
    let remaining = total.saturating_sub(done);
    let eta_s = recent_avg_s * remaining as f64 / jobs.max(1) as f64;
    let pct = if total == 0 {
        100.0
    } else {
        done as f64 * 100.0 / total as f64
    };
    println!(
        "[{label} preprocess] {done}/{total} ({pct:.1}%) | workers={} new={processed_file_count} skip={skipped_existing_file_count} samples={exported_sample_count} | 已运行={} 预计剩余={}",
        jobs.max(1),
        format_duration_s(elapsed_s),
        format_duration_s(eta_s),
    );
}

pub fn temp_npz_path(path: &Path) -> PathBuf {
    let nonce = SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .map(|duration| duration.as_nanos())
        .unwrap_or(0);
    let file_name = path
        .file_name()
        .and_then(|name| name.to_str())
        .unwrap_or("export.npz");
    path.with_file_name(format!(".{file_name}.tmp-{}-{nonce}", process::id(),))
}

pub fn npy_header(descr: &str, shape: &[usize]) -> Vec<u8> {
    let shape_str = if shape.len() == 1 {
        format!("({},)", shape[0])
    } else {
        let parts: Vec<String> = shape.iter().map(|v| v.to_string()).collect();
        format!("({})", parts.join(", "))
    };
    let mut header = format!(
        "{{'descr': '{}', 'fortran_order': False, 'shape': {}, }}",
        descr, shape_str
    )
    .into_bytes();
    let preamble_len = 10usize;
    while (preamble_len + header.len() + 1) % 16 != 0 {
        header.push(b' ');
    }
    header.push(b'\n');
    let mut out = Vec::with_capacity(preamble_len + header.len());
    out.extend_from_slice(b"\x93NUMPY");
    out.push(1);
    out.push(0);
    out.extend_from_slice(&(header.len() as u16).to_le_bytes());
    out.extend_from_slice(&header);
    out
}

pub fn write_npy_f16<W: Write + Seek>(
    zip: &mut ZipWriter<W>,
    name: &str,
    shape: &[usize],
    data: &[u16],
) -> Result<(), String> {
    let options = FileOptions::default().compression_method(CompressionMethod::Stored);
    zip.start_file(name, options)
        .map_err(|err| format!("failed to start {name}: {err}"))?;
    zip.write_all(&npy_header("<f2", shape))
        .map_err(|err| format!("failed to write header {name}: {err}"))?;
    for value in data {
        zip.write_all(&value.to_le_bytes())
            .map_err(|err| format!("failed to write data {name}: {err}"))?;
    }
    Ok(())
}

pub fn write_npy_f32<W: Write + Seek>(
    zip: &mut ZipWriter<W>,
    name: &str,
    shape: &[usize],
    data: &[f32],
) -> Result<(), String> {
    let options = FileOptions::default().compression_method(CompressionMethod::Stored);
    zip.start_file(name, options)
        .map_err(|err| format!("failed to start {name}: {err}"))?;
    zip.write_all(&npy_header("<f4", shape))
        .map_err(|err| format!("failed to write header {name}: {err}"))?;
    for value in data {
        zip.write_all(&value.to_le_bytes())
            .map_err(|err| format!("failed to write data {name}: {err}"))?;
    }
    Ok(())
}

pub fn write_npy_i16<W: Write + Seek>(
    zip: &mut ZipWriter<W>,
    name: &str,
    shape: &[usize],
    data: &[i16],
) -> Result<(), String> {
    let options = FileOptions::default().compression_method(CompressionMethod::Stored);
    zip.start_file(name, options)
        .map_err(|err| format!("failed to start {name}: {err}"))?;
    zip.write_all(&npy_header("<i2", shape))
        .map_err(|err| format!("failed to write header {name}: {err}"))?;
    for value in data {
        zip.write_all(&value.to_le_bytes())
            .map_err(|err| format!("failed to write data {name}: {err}"))?;
    }
    Ok(())
}

pub fn write_npy_i32<W: Write + Seek>(
    zip: &mut ZipWriter<W>,
    name: &str,
    shape: &[usize],
    data: &[i32],
) -> Result<(), String> {
    let options = FileOptions::default().compression_method(CompressionMethod::Stored);
    zip.start_file(name, options)
        .map_err(|err| format!("failed to start {name}: {err}"))?;
    zip.write_all(&npy_header("<i4", shape))
        .map_err(|err| format!("failed to write header {name}: {err}"))?;
    for value in data {
        zip.write_all(&value.to_le_bytes())
            .map_err(|err| format!("failed to write data {name}: {err}"))?;
    }
    Ok(())
}

pub fn write_npy_i8<W: Write + Seek>(
    zip: &mut ZipWriter<W>,
    name: &str,
    shape: &[usize],
    data: &[i8],
) -> Result<(), String> {
    let options = FileOptions::default().compression_method(CompressionMethod::Stored);
    zip.start_file(name, options)
        .map_err(|err| format!("failed to start {name}: {err}"))?;
    zip.write_all(&npy_header("|i1", shape))
        .map_err(|err| format!("failed to write header {name}: {err}"))?;
    let bytes: Vec<u8> = data.iter().map(|v| *v as u8).collect();
    zip.write_all(&bytes)
        .map_err(|err| format!("failed to write data {name}: {err}"))?;
    Ok(())
}

pub fn write_npy_u8<W: Write + Seek>(
    zip: &mut ZipWriter<W>,
    name: &str,
    shape: &[usize],
    data: &[u8],
) -> Result<(), String> {
    let options = FileOptions::default().compression_method(CompressionMethod::Stored);
    zip.start_file(name, options)
        .map_err(|err| format!("failed to start {name}: {err}"))?;
    zip.write_all(&npy_header("|u1", shape))
        .map_err(|err| format!("failed to write header {name}: {err}"))?;
    zip.write_all(data)
        .map_err(|err| format!("failed to write data {name}: {err}"))?;
    Ok(())
}

pub fn write_npy_unicode_scalar<W: Write + Seek>(
    zip: &mut ZipWriter<W>,
    name: &str,
    value: &str,
) -> Result<(), String> {
    let options = FileOptions::default().compression_method(CompressionMethod::Stored);
    zip.start_file(name, options)
        .map_err(|err| format!("failed to start {name}: {err}"))?;
    let char_len = value.chars().count();
    zip.write_all(&npy_header(&format!("<U{char_len}"), &[]))
        .map_err(|err| format!("failed to write header {name}: {err}"))?;
    for ch in value.chars() {
        zip.write_all(&(ch as u32).to_le_bytes())
            .map_err(|err| format!("failed to write data {name}: {err}"))?;
    }
    Ok(())
}

pub fn read_npy_first_dim_from_zip(
    path: &Path,
    member_name: &str,
) -> Result<Option<usize>, String> {
    if !path.exists() {
        return Ok(None);
    }
    let file = fs::File::open(path)
        .map_err(|err| format!("failed to open npz {}: {err}", path.display()))?;
    let mut zip = zip::ZipArchive::new(file)
        .map_err(|err| format!("failed to open npz {}: {err}", path.display()))?;
    let mut member = match zip.by_name(member_name) {
        Ok(item) => item,
        Err(_) => return Ok(None),
    };
    let mut magic = [0u8; 6];
    if member.read_exact(&mut magic).is_err() || &magic != b"\x93NUMPY" {
        return Ok(None);
    }
    let mut version = [0u8; 2];
    member
        .read_exact(&mut version)
        .map_err(|err| format!("failed to read npy version {}: {err}", path.display()))?;
    let header_len = match version {
        [1, 0] | [2, 0] => {
            let mut raw = [0u8; 2];
            member
                .read_exact(&mut raw)
                .map_err(|err| format!("failed to read npy header {}: {err}", path.display()))?;
            u16::from_le_bytes(raw) as usize
        }
        [3, 0] => {
            let mut raw = [0u8; 4];
            member
                .read_exact(&mut raw)
                .map_err(|err| format!("failed to read npy header {}: {err}", path.display()))?;
            u32::from_le_bytes(raw) as usize
        }
        _ => return Ok(None),
    };
    let mut header = vec![0u8; header_len];
    member
        .read_exact(&mut header)
        .map_err(|err| format!("failed to read npy header {}: {err}", path.display()))?;
    let header_text = String::from_utf8_lossy(&header);
    let Some(shape_start) = header_text.find("'shape': (") else {
        return Ok(None);
    };
    let shape_text = &header_text[shape_start + "'shape': (".len()..];
    let dim_text: String = shape_text
        .chars()
        .take_while(|ch| ch.is_ascii_digit())
        .collect();
    if dim_text.is_empty() {
        return Ok(None);
    }
    Ok(dim_text.parse::<usize>().ok())
}

pub fn write_json_manifest<T: Serialize>(
    output_dir: &Path,
    file_name: &str,
    manifest: &T,
) -> Result<String, String> {
    fs::create_dir_all(output_dir).map_err(|err| {
        format!(
            "failed to create output dir {}: {err}",
            output_dir.display()
        )
    })?;
    let manifest_path = output_dir.join(file_name);
    let text = serde_json::to_string_pretty(manifest)
        .map_err(|err| format!("failed to serialize export manifest: {err}"))?;
    fs::write(&manifest_path, text).map_err(|err| {
        format!(
            "failed to write manifest {}: {err}",
            manifest_path.display()
        )
    })?;
    Ok(manifest_path.to_string_lossy().to_string())
}

pub fn finalize_temp_npz(
    mut zip: ZipWriter<fs::File>,
    temp_path: &Path,
    final_path: &Path,
    sync_file: bool,
) -> Result<(), String> {
    let file = zip
        .finish()
        .map_err(|err| format!("failed to finish npz {}: {err}", temp_path.display()))?;
    if sync_file {
        file.sync_all()
            .map_err(|err| format!("failed to sync temp npz {}: {err}", temp_path.display()))?;
    }
    drop(file);
    if let Some(parent) = final_path.parent() {
        fs::create_dir_all(parent)
            .map_err(|err| format!("failed to create output dir {}: {err}", parent.display()))?;
    }
    fs::rename(temp_path, final_path).map_err(|err| {
        format!(
            "failed to move temp npz {} -> {}: {err}",
            temp_path.display(),
            final_path.display()
        )
    })?;
    Ok(())
}
