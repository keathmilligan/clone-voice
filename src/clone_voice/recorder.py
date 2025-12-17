"""Audio recording module for capturing voice samples."""

from pathlib import Path
from datetime import datetime

import numpy as np
import sounddevice as sd
from scipy.io import wavfile
from rich.console import Console
from rich.progress import Progress, SpinnerColumn, TextColumn, BarColumn, TimeElapsedColumn

from clone_voice.config import get_settings

console = Console()


class Recorder:
    """Cross-platform audio recorder for voice samples."""

    def __init__(self, sample_rate: int | None = None, channels: int | None = None):
        """Initialize recorder with settings."""
        self.settings = get_settings()
        self.sample_rate = sample_rate or self.settings.sample_rate
        self.channels = channels or self.settings.recording_channels

    def list_devices(self) -> list[dict]:
        """List available audio input devices."""
        devices = sd.query_devices()
        input_devices = []
        for i, device in enumerate(devices):
            if device["max_input_channels"] > 0:
                input_devices.append(
                    {
                        "id": i,
                        "name": device["name"],
                        "channels": device["max_input_channels"],
                        "sample_rate": device["default_samplerate"],
                    }
                )
        return input_devices

    def _get_device_sample_rate(self, device: int | None) -> int:
        """Get the native sample rate for a device."""
        if device is not None:
            device_info = sd.query_devices(device)
            return int(device_info["default_samplerate"])
        else:
            # Use default input device's sample rate
            default_device = sd.query_devices(kind="input")
            return int(default_device["default_samplerate"])

    def record(
        self,
        duration: int | None = None,
        device: int | None = None,
        output_path: Path | None = None,
        name: str | None = None,
    ) -> Path:
        """
        Record audio from microphone.

        Args:
            duration: Recording duration in seconds
            device: Audio device ID (None for default)
            output_path: Custom output path for the recording
            name: Name for the recording file (without extension)

        Returns:
            Path to the saved recording
        """
        duration = duration or self.settings.recording_duration

        # Use device's native sample rate to avoid compatibility issues
        sample_rate = self._get_device_sample_rate(device)

        # Determine output path
        if output_path is None:
            samples_dir = self.settings.get_samples_path()
            samples_dir.mkdir(parents=True, exist_ok=True)

            if name is None:
                timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                name = f"recording_{timestamp}"

            output_path = samples_dir / f"{name}.wav"

        # Get device name for display
        if device is not None:
            device_info = sd.query_devices(device)
            device_name = device_info["name"]
        else:
            device_info = sd.query_devices(kind="input")
            device_name = f"{device_info['name']} (default)"

        console.print(f"\n[bold blue]Recording[/bold blue]")
        console.print(f"  Device: {device_name}")
        console.print(f"  Duration: {duration}s")
        console.print(f"  Sample rate: {sample_rate} Hz")
        console.print(f"  Output: {output_path}")
        console.print("\n[dim]Press Ctrl+C to stop early[/dim]\n")

        try:
            with Progress(
                SpinnerColumn(),
                TextColumn("[progress.description]{task.description}"),
                BarColumn(),
                TimeElapsedColumn(),
                console=console,
            ) as progress:
                task = progress.add_task("Recording", total=duration)

                # Record audio using device's native sample rate
                recording = sd.rec(
                    int(duration * sample_rate),
                    samplerate=sample_rate,
                    channels=self.channels,
                    device=device,
                    dtype=np.float32,
                )

                # Update progress while recording
                for i in range(duration):
                    sd.sleep(1000)  # Sleep 1 second
                    progress.update(task, advance=1)

                sd.wait()  # Wait for recording to complete

        except KeyboardInterrupt:
            console.print("\n[yellow]Recording stopped early[/yellow]")
            sd.stop()
            # Get partial recording
            recording = recording[: int(sd.get_stream().time * sample_rate)]

        # Normalize and convert to int16
        recording = recording.flatten()
        max_val = np.max(np.abs(recording))
        if max_val > 0:
            recording = recording / max_val  # Normalize to [-1, 1]
        recording_int16 = (recording * 32767).astype(np.int16)

        # Save recording with the device's sample rate
        wavfile.write(output_path, sample_rate, recording_int16)

        console.print(f"\n[green]Recording saved![/green]")
        console.print(f"  Path: {output_path}")
        console.print(f"  Duration: {len(recording_int16) / sample_rate:.1f}s")

        return output_path

    def record_interactive(
        self,
        output_path: Path | None = None,
        name: str | None = None,
        device: int | None = None,
    ) -> Path:
        """
        Interactive recording with countdown and prompts.

        Args:
            output_path: Custom output path
            name: Name for the recording
            device: Audio device ID (None for default)

        Returns:
            Path to the saved recording
        """
        console.print("\n[bold]Voice Sample Recording[/bold]")
        console.print("─" * 40)

        # Show selected device
        if device is not None:
            devices = self.list_devices()
            device_info = next((d for d in devices if d["id"] == device), None)
            if device_info:
                console.print(f"\n[cyan]Using device:[/cyan] {device_info['name']}")
        else:
            default_device = sd.query_devices(kind="input")
            console.print(f"\n[cyan]Using device:[/cyan] {default_device['name']} (default)")

        console.print(
            "\nTips for best results:\n"
            "  • Use a quiet environment\n"
            "  • Keep microphone 6-12 inches from mouth\n"
            "  • Speak naturally at normal volume\n"
            "  • Read a passage or speak freely for 10-30 seconds\n"
        )

        # Countdown
        console.print("[bold]Starting in:[/bold]")
        for i in range(3, 0, -1):
            console.print(f"  {i}...")
            sd.sleep(1000)

        return self.record(output_path=output_path, name=name, device=device)


def list_recordings() -> list[Path]:
    """List all recordings in the samples directory."""
    settings = get_settings()
    samples_dir = settings.get_samples_path()

    if not samples_dir.exists():
        return []

    return sorted(samples_dir.glob("*.wav"))
