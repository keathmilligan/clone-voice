"""Command-line interface for Clone Voice."""

from pathlib import Path
from typing import Annotated, Optional, TYPE_CHECKING

import typer
from rich.console import Console

# Only import for type checking, not at runtime
if TYPE_CHECKING:
    from rich.table import Table

app = typer.Typer(
    name="clone-voice",
    help="Local voice cloning using XTTS v2",
    add_completion=False,
)
console = Console()


def _status(msg: str) -> None:
    """Print a status message."""
    console.print(f"[dim]{msg}[/dim]", highlight=False)


@app.command()
def record(
    name: Annotated[
        Optional[str],
        typer.Option("--name", "-n", help="Name for the recording"),
    ] = None,
    duration: Annotated[
        int,
        typer.Option("--duration", "-d", help="Recording duration in seconds"),
    ] = 10,
    device: Annotated[
        Optional[int],
        typer.Option(
            "--device", "-D", help="Audio input device ID (use 'devices' command to list)"
        ),
    ] = None,
    interactive: Annotated[
        bool,
        typer.Option(
            "--interactive/--no-interactive", "-i/-I", help="Interactive mode with prompts"
        ),
    ] = True,
    output: Annotated[
        Optional[Path],
        typer.Option("--output", "-o", help="Output file path"),
    ] = None,
) -> None:
    """Record a voice sample from microphone."""
    _status("Loading audio modules...")
    from clone_voice.config import reload_settings
    from clone_voice.recorder import Recorder

    reload_settings(recording_duration=duration)
    recorder = Recorder()

    if interactive:
        recorder.record_interactive(output_path=output, name=name, device=device)
    else:
        recorder.record(duration=duration, output_path=output, name=name, device=device)


@app.command()
def process(
    input_path: Annotated[
        Path,
        typer.Argument(help="Input audio file to process"),
    ],
    output: Annotated[
        Optional[Path],
        typer.Option("--output", "-o", help="Output file path"),
    ] = None,
    no_denoise: Annotated[
        bool,
        typer.Option("--no-denoise", help="Skip noise reduction"),
    ] = False,
    no_normalize: Annotated[
        bool,
        typer.Option("--no-normalize", help="Skip normalization"),
    ] = False,
    no_trim: Annotated[
        bool,
        typer.Option("--no-trim", help="Skip silence trimming"),
    ] = False,
) -> None:
    """Process an audio file (denoise, normalize, trim)."""
    _status("Loading audio processing modules...")
    from clone_voice.preprocessor import Preprocessor

    preprocessor = Preprocessor()

    output_path = preprocessor.process(
        audio_path=input_path,
        output_path=output,
        denoise=not no_denoise,
        normalize=not no_normalize,
        trim=not no_trim,
    )

    console.print(f"\n[bold green]Processing complete![/bold green]")
    console.print(f"Output: {output_path}")


@app.command()
def generate(
    text: Annotated[
        str,
        typer.Argument(help="Text to synthesize"),
    ],
    voice: Annotated[
        Path,
        typer.Option("--voice", "-v", help="Path to voice sample WAV file"),
    ],
    output: Annotated[
        Optional[Path],
        typer.Option("--output", "-o", help="Output file path"),
    ] = None,
    language: Annotated[
        str,
        typer.Option("--language", "-l", help="Language code"),
    ] = "en",
    speed: Annotated[
        float,
        typer.Option("--speed", "-s", help="Speech speed multiplier (0.5-2.0)"),
    ] = 1.0,
    no_fp16: Annotated[
        bool,
        typer.Option("--no-fp16", help="Disable half-precision (slower but more compatible)"),
    ] = False,
) -> None:
    """Generate speech from text using a voice sample."""
    _status("Loading TTS engine (this may take a moment)...")
    from clone_voice.synthesizer import Synthesizer

    synthesizer = Synthesizer(use_fp16=not no_fp16)

    output_path = synthesizer.generate(
        text=text,
        speaker_wav=voice,
        output_path=output,
        language=language,
        speed=speed,
    )

    console.print(f"\n[bold green]Generation complete![/bold green]")
    console.print(f"Output: {output_path}")


@app.command()
def batch(
    input_file: Annotated[
        Path,
        typer.Argument(help="Text file to synthesize"),
    ],
    voice: Annotated[
        Path,
        typer.Option("--voice", "-v", help="Path to voice sample WAV file"),
    ],
    output_dir: Annotated[
        Optional[Path],
        typer.Option("--output-dir", "-o", help="Output directory"),
    ] = None,
    language: Annotated[
        str,
        typer.Option("--language", "-l", help="Language code"),
    ] = "en",
    speed: Annotated[
        float,
        typer.Option("--speed", "-s", help="Speech speed multiplier (0.5-2.0)"),
    ] = 1.0,
    no_split: Annotated[
        bool,
        typer.Option("--no-split", help="Don't split into sentences"),
    ] = False,
    no_fp16: Annotated[
        bool,
        typer.Option("--no-fp16", help="Disable half-precision (slower but more compatible)"),
    ] = False,
) -> None:
    """Generate speech from a text file."""
    _status("Loading TTS engine (this may take a moment)...")
    from clone_voice.synthesizer import Synthesizer

    synthesizer = Synthesizer(use_fp16=not no_fp16)

    output_paths = synthesizer.generate_from_file(
        input_file=input_file,
        speaker_wav=voice,
        output_dir=output_dir,
        language=language,
        split_sentences=not no_split,
        speed=speed,
    )

    console.print(f"\n[bold green]Batch generation complete![/bold green]")
    console.print(f"Generated {len(output_paths)} files")


@app.command(name="list")
def list_samples() -> None:
    """List recorded voice samples."""
    from rich.table import Table
    from clone_voice.config import get_settings

    settings = get_settings()
    samples_dir = settings.get_samples_path()

    if not samples_dir.exists():
        recordings = []
    else:
        recordings = sorted(samples_dir.glob("*.wav"))

    if not recordings:
        console.print("[yellow]No recordings found.[/yellow]")
        console.print("Use 'clone-voice record' to create a voice sample.")
        return

    table = Table(title="Voice Samples")
    table.add_column("Name", style="cyan")
    table.add_column("Path", style="dim")
    table.add_column("Size", justify="right")

    for path in recordings:
        size = path.stat().st_size
        size_str = f"{size / 1024:.1f} KB" if size < 1024 * 1024 else f"{size / 1024 / 1024:.1f} MB"
        table.add_row(path.stem, str(path.parent), size_str)

    console.print(table)


@app.command()
def devices(
    all_apis: Annotated[
        bool,
        typer.Option("--all", "-a", help="Show devices from all audio APIs (includes duplicates)"),
    ] = False,
) -> None:
    """List available audio input devices."""
    from rich.table import Table
    from rich.panel import Panel

    _status("Loading audio modules...")
    import sounddevice as sd

    all_devices = sd.query_devices()
    host_apis = sd.query_hostapis()

    # Get default input device for each API
    api_defaults = {}
    for api_idx, api in enumerate(host_apis):
        default_dev = api.get("default_input_device", -1)
        if default_dev is not None and default_dev >= 0:
            api_defaults[api_idx] = default_dev

    # Build list of input devices with full info
    input_devices = []
    for i, device in enumerate(all_devices):
        if device["max_input_channels"] > 0:
            api_idx = device["hostapi"]
            api_name = host_apis[api_idx]["name"] if api_idx < len(host_apis) else "Unknown"
            is_api_default = api_defaults.get(api_idx) == i
            input_devices.append(
                {
                    "id": i,
                    "name": device["name"],
                    "channels": device["max_input_channels"],
                    "sample_rate": device["default_samplerate"],
                    "latency_low": device["default_low_input_latency"],
                    "latency_high": device["default_high_input_latency"],
                    "api": api_name,
                    "api_idx": api_idx,
                    "is_default": is_api_default,
                }
            )

    has_default = any(d["is_default"] for d in input_devices)

    if not input_devices:
        console.print("[yellow]No input devices found.[/yellow]")
        return

    # Filter to recommended API (WASAPI on Windows) unless --all specified
    if not all_apis:
        # Prefer WASAPI, then WDM-KS, then others
        preferred_apis = ["Windows WASAPI", "Windows WDM-KS", "MME", "Windows DirectSound"]
        for preferred in preferred_apis:
            filtered = [d for d in input_devices if d["api"] == preferred]
            if filtered:
                input_devices = filtered
                console.print(f"[dim]Showing {preferred} devices (use --all for all APIs)[/dim]\n")
                break

    # Create table
    table = Table(title="Audio Input Devices", show_lines=True)
    table.add_column("ID", style="cyan", justify="right", no_wrap=True)
    table.add_column("Name", style="bold")
    table.add_column("API", style="dim")
    table.add_column("Ch", justify="right")
    table.add_column("Sample Rate", justify="right")
    table.add_column("Latency (ms)", justify="right")
    table.add_column("", justify="center")  # Default indicator

    for device in input_devices:
        latency_ms = f"{device['latency_low'] * 1000:.1f} - {device['latency_high'] * 1000:.1f}"
        default_marker = "[green]*[/green]" if device["is_default"] else ""

        table.add_row(
            str(device["id"]),
            device["name"],
            device["api"],
            str(device["channels"]),
            f"{device['sample_rate']:.0f} Hz",
            latency_ms,
            default_marker,
        )

    console.print(table)

    if has_default:
        console.print("\n[dim]* = Default device for this API[/dim]")

    console.print("[dim]Tip: Use -D <ID> with record command to select device[/dim]")


@app.command()
def info(
    audio_path: Annotated[
        Path,
        typer.Argument(help="Audio file to inspect"),
    ],
) -> None:
    """Show information about an audio file."""
    from rich.table import Table

    _status("Loading audio modules...")
    from clone_voice.preprocessor import Preprocessor

    preprocessor = Preprocessor()
    audio_info = preprocessor.get_audio_info(audio_path)

    table = Table(title=f"Audio Info: {audio_path.name}")
    table.add_column("Property", style="cyan")
    table.add_column("Value")

    table.add_row("Path", str(audio_info["path"]))
    table.add_row("Duration", f"{audio_info['duration']:.2f} seconds")
    table.add_row("Sample Rate", f"{audio_info['sample_rate']} Hz")
    table.add_row("Samples", f"{audio_info['samples']:,}")
    table.add_row("RMS Level", f"{audio_info['rms_db']:.1f} dB")

    console.print(table)


@app.command()
def models() -> None:
    """List available TTS models."""
    _status("Loading TTS engine (this may take a moment)...")
    from TTS.api import TTS

    console.print("[blue]Available TTS models:[/blue]\n")

    all_models = TTS().list_models()

    # Filter for voice cloning capable models
    cloning_models = [m for m in all_models if "xtts" in m.lower() or "your_tts" in m.lower()]

    console.print("[bold]Recommended for voice cloning:[/bold]")
    for model in cloning_models:
        console.print(f"  • {model}")

    console.print(f"\n[dim]Total available: {len(all_models)} models[/dim]")
    console.print("[dim]Use --help with any command for more options[/dim]")


if __name__ == "__main__":
    app()
