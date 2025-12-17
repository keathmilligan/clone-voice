"""Audio preprocessing module for voice samples."""

from pathlib import Path
import time

import numpy as np
import librosa
import noisereduce as nr
from scipy.io import wavfile
from rich.console import Console
from rich.progress import Progress, SpinnerColumn, TextColumn, TimeElapsedColumn

from clone_voice.config import get_settings

console = Console()


class Preprocessor:
    """Audio preprocessor for cleaning and normalizing voice samples."""

    def __init__(self, target_sample_rate: int | None = None):
        """Initialize preprocessor."""
        self.settings = get_settings()
        self.target_sample_rate = target_sample_rate or self.settings.sample_rate

    def load_audio(self, audio_path: Path) -> tuple[np.ndarray, int]:
        """
        Load audio file.

        Args:
            audio_path: Path to audio file

        Returns:
            Tuple of (audio_data, sample_rate)
        """
        audio, sr = librosa.load(audio_path, sr=None)
        return audio, sr

    def resample(self, audio: np.ndarray, orig_sr: int, target_sr: int | None = None) -> np.ndarray:
        """
        Resample audio to target sample rate.

        Args:
            audio: Audio data
            orig_sr: Original sample rate
            target_sr: Target sample rate (uses settings if None)

        Returns:
            Resampled audio
        """
        target_sr = target_sr or self.target_sample_rate
        if orig_sr != target_sr:
            audio = librosa.resample(audio, orig_sr=orig_sr, target_sr=target_sr)
        return audio

    def normalize(self, audio: np.ndarray, target_db: float = -20.0) -> np.ndarray:
        """
        Normalize audio to target dB level.

        Args:
            audio: Audio data
            target_db: Target loudness in dB

        Returns:
            Normalized audio
        """
        # Calculate current RMS
        rms = np.sqrt(np.mean(audio**2))
        if rms > 0:
            # Calculate target RMS from dB
            target_rms = 10 ** (target_db / 20)
            # Scale audio
            audio = audio * (target_rms / rms)

        # Clip to prevent clipping
        audio = np.clip(audio, -1.0, 1.0)

        return audio

    def reduce_noise(
        self,
        audio: np.ndarray,
        sample_rate: int,
        prop_decrease: float = 0.8,
    ) -> np.ndarray:
        """
        Apply noise reduction to audio.

        Args:
            audio: Audio data
            sample_rate: Sample rate
            prop_decrease: Proportion to reduce noise by (0.0-1.0)

        Returns:
            Denoised audio
        """
        return nr.reduce_noise(
            y=audio,
            sr=sample_rate,
            prop_decrease=prop_decrease,
            stationary=False,
        )

    def trim_silence(
        self,
        audio: np.ndarray,
        top_db: int = 30,
        frame_length: int = 2048,
        hop_length: int = 512,
    ) -> np.ndarray:
        """
        Trim silence from beginning and end of audio.

        Args:
            audio: Audio data
            top_db: Threshold in dB below reference to consider silence
            frame_length: Frame length for analysis
            hop_length: Hop length for analysis

        Returns:
            Trimmed audio
        """
        trimmed, _ = librosa.effects.trim(
            audio,
            top_db=top_db,
            frame_length=frame_length,
            hop_length=hop_length,
        )
        return trimmed

    def process(
        self,
        audio_path: Path,
        output_path: Path | None = None,
        denoise: bool = True,
        normalize: bool = True,
        trim: bool = True,
        target_db: float = -20.0,
    ) -> Path:
        """
        Process audio file with full pipeline.

        Args:
            audio_path: Input audio path
            output_path: Output path (defaults to input with _processed suffix)
            denoise: Apply noise reduction
            normalize: Apply normalization
            trim: Trim silence
            target_db: Target loudness for normalization

        Returns:
            Path to processed audio
        """
        audio_path = Path(audio_path)

        if output_path is None:
            output_path = audio_path.parent / f"{audio_path.stem}_processed.wav"

        console.print(f"\n[bold blue]Processing Audio[/bold blue]")
        console.print(f"  Input: {audio_path.name}")

        start_time = time.time()

        # Load audio
        with Progress(
            SpinnerColumn(),
            TextColumn("[progress.description]{task.description}"),
            TimeElapsedColumn(),
            console=console,
            transient=True,
        ) as progress:
            progress.add_task("Loading audio...", total=None)
            audio, sr = self.load_audio(audio_path)

        console.print(f"  Duration: {len(audio) / sr:.2f}s @ {sr}Hz")

        # Resample if needed
        if sr != self.target_sample_rate:
            with Progress(
                SpinnerColumn(),
                TextColumn("[progress.description]{task.description}"),
                console=console,
                transient=True,
            ) as progress:
                progress.add_task(f"Resampling to {self.target_sample_rate}Hz...", total=None)
                audio = self.resample(audio, sr, self.target_sample_rate)
                sr = self.target_sample_rate
            console.print(f"  Resampled to {sr}Hz")

        # Apply processing steps
        if trim:
            original_len = len(audio)
            audio = self.trim_silence(audio)
            trimmed_amount = (original_len - len(audio)) / sr
            console.print(f"  Trimmed {trimmed_amount:.2f}s of silence")

        if denoise:
            with Progress(
                SpinnerColumn(),
                TextColumn("[progress.description]{task.description}"),
                TimeElapsedColumn(),
                console=console,
                transient=True,
            ) as progress:
                progress.add_task("Reducing noise...", total=None)
                audio = self.reduce_noise(audio, sr)
            console.print("  Applied noise reduction")

        if normalize:
            audio = self.normalize(audio, target_db)
            console.print(f"  Normalized to {target_db}dB")

        # Save processed audio
        audio_int16 = (audio * 32767).astype(np.int16)
        wavfile.write(output_path, sr, audio_int16)

        elapsed = time.time() - start_time
        console.print(f"\n[green]Processed in {elapsed:.1f}s:[/green] {output_path}")

        return output_path

    def get_audio_info(self, audio_path: Path) -> dict:
        """
        Get information about an audio file.

        Args:
            audio_path: Path to audio file

        Returns:
            Dictionary with audio information
        """
        audio, sr = self.load_audio(audio_path)
        duration = len(audio) / sr

        return {
            "path": audio_path,
            "duration": duration,
            "sample_rate": sr,
            "samples": len(audio),
            "rms_db": 20 * np.log10(np.sqrt(np.mean(audio**2)) + 1e-10),
        }
