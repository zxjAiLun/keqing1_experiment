extern crate _native;

use _native::keqingv4_export::build_keqingv4_cached_records_with_options;
use _native::xmodel1_export::ExportRunOptions;
use std::env;
use std::path::PathBuf;

fn usage() -> &'static str {
    "Usage: keqingv4_export --output-dir <dir> [--data-dir <dir> ...] [--data-dirs <dir> ...] [--smoke] [--progress-every <n>] [--jobs <n>] [--limit-files <n>] [--force]"
}

fn parse_args() -> Result<(Vec<String>, String, ExportRunOptions), String> {
    let mut data_dirs: Vec<String> = Vec::new();
    let mut output_dir: Option<String> = None;
    let mut smoke = false;
    let mut resume = true;
    let mut progress_every = 20usize;
    let mut jobs = 0usize;
    let mut limit_files = 0usize;

    let mut args = env::args().skip(1).peekable();
    while let Some(arg) = args.next() {
        match arg.as_str() {
            "--data-dir" | "--data_dir" => {
                let value = args
                    .next()
                    .ok_or_else(|| format!("missing value for {arg}\n{}", usage()))?;
                data_dirs.push(value);
            }
            "--data-dirs" | "--data_dirs" => {
                let mut consumed_any = false;
                while let Some(next) = args.peek() {
                    if next.starts_with('-') {
                        break;
                    }
                    consumed_any = true;
                    data_dirs.push(args.next().expect("peeked arg must exist"));
                }
                if !consumed_any {
                    return Err(format!("missing value for {arg}\n{}", usage()));
                }
            }
            "--output-dir" | "--output_dir" => {
                let value = args
                    .next()
                    .ok_or_else(|| format!("missing value for {arg}\n{}", usage()))?;
                output_dir = Some(value);
            }
            "--progress-every" | "--progress_every" => {
                let value = args
                    .next()
                    .ok_or_else(|| format!("missing value for {arg}\n{}", usage()))?;
                progress_every = value
                    .parse::<usize>()
                    .map_err(|_| format!("invalid integer for {arg}: {value}"))?;
            }
            "--jobs" => {
                let value = args
                    .next()
                    .ok_or_else(|| format!("missing value for {arg}\n{}", usage()))?;
                jobs = value
                    .parse::<usize>()
                    .map_err(|_| format!("invalid integer for {arg}: {value}"))?;
            }
            "--limit-files" | "--limit_files" => {
                let value = args
                    .next()
                    .ok_or_else(|| format!("missing value for {arg}\n{}", usage()))?;
                limit_files = value
                    .parse::<usize>()
                    .map_err(|_| format!("invalid integer for {arg}: {value}"))?;
            }
            "--smoke" => smoke = true,
            "--force" => resume = false,
            "-h" | "--help" => {
                println!("{}", usage());
                std::process::exit(0);
            }
            other => return Err(format!("unknown argument: {other}\n{}", usage())),
        }
    }

    let output_dir = output_dir.ok_or_else(|| format!("--output-dir is required\n{}", usage()))?;
    Ok((
        data_dirs,
        output_dir,
        ExportRunOptions {
            smoke,
            resume,
            progress_every,
            jobs,
            limit_files,
        },
    ))
}

fn main() {
    if let Err(err) = run() {
        eprintln!("{err}");
        std::process::exit(1);
    }
}

fn run() -> Result<(), String> {
    let (data_dirs, output_dir, options) = parse_args()?;
    let output_dir = PathBuf::from(output_dir);
    let output_dir_str = output_dir.to_string_lossy().to_string();
    let (file_count, manifest_path, produced_npz) =
        build_keqingv4_cached_records_with_options(&data_dirs, &output_dir_str, options)?;
    println!(
        "Rust keqingv4 export completed: file_count={} manifest={} produced_npz={}",
        file_count, manifest_path, produced_npz
    );
    if produced_npz || options.smoke {
        return Ok(());
    }
    Err("Rust keqingv4 preprocess completed without npz output; full export is required before training.".to_string())
}
