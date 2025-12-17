"""Text-to-speech synthesizer using XTTS v2 for voice cloning."""

from pathlib import Path
import hashlib
import json
import os
import time
from typing import Iterator

# Auto-accept Coqui TTS license agreement
os.environ["COQUI_TOS_AGREED"] = "1"

import torch
import torchaudio
from TTS.tts.configs.xtts_config import XttsConfig
from TTS.tts.models.xtts import Xtts
from rich.console import Console
from rich.progress import (
    Progress,
    TextColumn,
    BarColumn,
    TaskProgressColumn,
    TimeElapsedColumn,
    TimeRemainingColumn,
)

from clone_voice.config import get_settings

# Use force_terminal and legacy_windows to avoid Unicode rendering issues on Windows
console = Console(force_terminal=True, legacy_windows=False)


class Synthesizer:
    """Voice cloning synthesizer using XTTS v2 with optimizations."""

    def __init__(
        self,
        model_name: str | None = None,
        device: str | None = None,
        use_fp16: bool = True,
    ):
        """
        Initialize synthesizer.

        Args:
            model_name: TTS model name (defaults to settings)
            device: Compute device (cuda/cpu/auto)
            use_fp16: Use half-precision for faster inference (GPU only)
        """
        self.settings = get_settings()
        self.model_name = model_name or self.settings.model_name
        self.device = device or self.settings.get_device()
        self.use_fp16 = use_fp16 and self.device == "cuda"
        self._model: Xtts | None = None
        self._config: XttsConfig | None = None

        # Speaker embedding cache: {audio_path_hash: (gpt_cond_latent, speaker_embedding)}
        self._speaker_cache: dict[str, tuple] = {}
        self._cache_dir = self.settings.base_dir / "cache" / "speakers"

    @property
    def model(self) -> Xtts:
        """Lazy-load the XTTS model."""
        if self._model is None:
            self._load_model()
        return self._model

    def _get_model_path(self) -> Path:
        """Get path to downloaded model."""
        # TTS stores models in ~/.local/share/tts on Linux, AppData on Windows
        from TTS.utils.manage import ModelManager

        manager = ModelManager()
        model_path, _, _ = manager.download_model(self.model_name)
        return Path(model_path)

    def _load_model(self) -> None:
        """Load the XTTS model directly for better control."""
        console.print(f"\n[bold blue]Loading TTS Model[/bold blue]")
        console.print(f"  Model: {self.model_name}")
        console.print(f"  Device: {self.device}")
        console.print(f"  Half-precision: {self.use_fp16}")
        console.print()

        start_time = time.time()

        console.print("[dim]Downloading/locating model...[/dim]")
        model_path = self._get_model_path()

        console.print("[dim]Loading model configuration...[/dim]")
        config_path = model_path / "config.json"
        self._config = XttsConfig()
        self._config.load_json(str(config_path))

        console.print("[dim]Initializing model...[/dim]")
        self._model = Xtts.init_from_config(self._config)

        console.print("[dim]Loading model weights...[/dim]")
        self._model.load_checkpoint(
            self._config,
            checkpoint_dir=str(model_path),
            eval=True,
        )

        console.print(f"[dim]Moving model to {self.device}...[/dim]")
        self._model = self._model.to(self.device)

        # Enable inference mode optimizations
        torch.set_grad_enabled(False)

        elapsed = time.time() - start_time
        console.print(f"[green]Model loaded in {elapsed:.1f}s[/green]\n")

        # Load cached speaker embeddings
        self._load_speaker_cache()

    def _get_speaker_hash(self, audio_path: Path) -> str:
        """Get hash for speaker audio file (based on path and modification time)."""
        stat = audio_path.stat()
        key = f"{audio_path.absolute()}:{stat.st_mtime}:{stat.st_size}"
        return hashlib.md5(key.encode()).hexdigest()

    def _load_speaker_cache(self) -> None:
        """Load speaker embeddings from disk cache."""
        if not self._cache_dir.exists():
            return

        cache_index = self._cache_dir / "index.json"
        if not cache_index.exists():
            return

        try:
            with open(cache_index) as f:
                index = json.load(f)
            console.print(f"[dim]Found {len(index)} cached speaker embeddings[/dim]")
        except Exception:
            pass

    def _get_speaker_embedding(
        self, audio_path: Path, gpt_cond_len: int = 6
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Get speaker conditioning latents, using cache if available.

        Args:
            audio_path: Path to reference audio
            gpt_cond_len: Length of GPT conditioning (6=faster, 30=better quality)

        Returns:
            Tuple of (gpt_cond_latent, speaker_embedding)
        """
        audio_hash = self._get_speaker_hash(audio_path)

        # Check memory cache
        if audio_hash in self._speaker_cache:
            console.print("[dim]Using cached speaker embedding[/dim]")
            return self._speaker_cache[audio_hash]

        # Check disk cache
        cache_file = self._cache_dir / f"{audio_hash}.pt"
        if cache_file.exists():
            try:
                data = torch.load(cache_file, map_location=self.device, weights_only=True)
                gpt_cond_latent = data["gpt_cond_latent"].float()
                speaker_embedding = data["speaker_embedding"].float()

                self._speaker_cache[audio_hash] = (gpt_cond_latent, speaker_embedding)
                console.print("[dim]Loaded speaker embedding from cache[/dim]")
                return gpt_cond_latent, speaker_embedding
            except Exception as e:
                console.print(f"[yellow]Cache load failed: {e}[/yellow]")

        # Compute speaker embedding
        console.print("[dim]Computing speaker embedding...[/dim]")
        start = time.time()

        # Get conditioning latents - returns as float32
        gpt_cond_latent, speaker_embedding = self.model.get_conditioning_latents(
            audio_path=str(audio_path),
            gpt_cond_len=gpt_cond_len,
            gpt_cond_chunk_len=6,
            max_ref_length=30,
        )

        elapsed = time.time() - start
        console.print(f"[dim]Speaker embedding computed in {elapsed:.1f}s[/dim]")

        # Ensure tensors are on correct device
        gpt_cond_latent = gpt_cond_latent.to(self.device)
        speaker_embedding = speaker_embedding.to(self.device)

        # Save to disk cache (save as FP32 for compatibility)
        self._cache_dir.mkdir(parents=True, exist_ok=True)
        try:
            torch.save(
                {
                    "gpt_cond_latent": gpt_cond_latent.float().cpu(),
                    "speaker_embedding": speaker_embedding.float().cpu(),
                },
                cache_file,
            )
        except Exception as e:
            console.print(f"[yellow]Cache save failed: {e}[/yellow]")

        # Store in memory cache
        self._speaker_cache[audio_hash] = (gpt_cond_latent, speaker_embedding)

        return gpt_cond_latent, speaker_embedding

    def generate(
        self,
        text: str,
        speaker_wav: Path | str,
        output_path: Path | None = None,
        language: str | None = None,
        speed: float = 1.0,
        temperature: float = 0.65,
        top_p: float = 0.85,
        top_k: int = 50,
        repetition_penalty: float = 10.0,
    ) -> Path:
        """
        Generate speech from text using voice cloning.

        Args:
            text: Text to synthesize
            speaker_wav: Path to reference audio for voice cloning
            output_path: Output path for generated audio
            language: Language code (defaults to settings)
            speed: Speech speed multiplier (1.0 = normal)
            temperature: Sampling temperature (lower = more deterministic)
            top_p: Nucleus sampling threshold
            top_k: Top-k sampling
            repetition_penalty: Penalty for repeating tokens

        Returns:
            Path to generated audio file
        """
        speaker_wav = Path(speaker_wav)
        language = language or self.settings.language

        # Generate output path if not provided
        if output_path is None:
            outputs_dir = self.settings.get_outputs_path()
            outputs_dir.mkdir(parents=True, exist_ok=True)
            text_hash = hashlib.md5(text.encode()).hexdigest()[:8]
            output_path = outputs_dir / f"output_{text_hash}.wav"

        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)

        text_preview = text[:60] + "..." if len(text) > 60 else text
        console.print(f"[blue]Generating speech...[/blue]")
        console.print(f"  Text: {text_preview}")
        console.print(f"  Voice: {speaker_wav.name}")
        console.print(f"  Length: {len(text)} chars")
        if speed != 1.0:
            console.print(f"  Speed: {speed}x")

        start_time = time.time()

        # Get speaker embedding (cached if possible)
        gpt_cond_latent, speaker_embedding = self._get_speaker_embedding(speaker_wav)

        # Generate speech
        console.print("[dim]Synthesizing audio...[/dim]")
        synth_start = time.time()

        out = self.model.inference(
            text=text,
            language=language,
            gpt_cond_latent=gpt_cond_latent,
            speaker_embedding=speaker_embedding,
            temperature=temperature,
            top_p=top_p,
            top_k=top_k,
            repetition_penalty=repetition_penalty,
            speed=speed,
            enable_text_splitting=True,
        )

        synth_elapsed = time.time() - synth_start

        # Save audio
        wav = torch.tensor(out["wav"]).unsqueeze(0)
        torchaudio.save(str(output_path), wav, 24000)

        elapsed = time.time() - start_time
        audio_duration = len(out["wav"]) / 24000
        rtf = synth_elapsed / audio_duration if audio_duration > 0 else 0

        console.print(
            f"[green]Generated in {elapsed:.1f}s[/green] (synthesis: {synth_elapsed:.1f}s, RTF: {rtf:.2f})"
        )
        console.print(f"  Output: {output_path}\n")

        return output_path

    def generate_stream(
        self,
        text: str,
        speaker_wav: Path | str,
        language: str | None = None,
        speed: float = 1.0,
        chunk_size: int = 20,
    ) -> Iterator[torch.Tensor]:
        """
        Generate speech in streaming mode for faster first audio.

        Args:
            text: Text to synthesize
            speaker_wav: Path to reference audio
            language: Language code
            speed: Speech speed multiplier
            chunk_size: Size of audio chunks to yield

        Yields:
            Audio tensor chunks
        """
        speaker_wav = Path(speaker_wav)
        language = language or self.settings.language

        gpt_cond_latent, speaker_embedding = self._get_speaker_embedding(speaker_wav)

        console.print("[dim]Streaming synthesis...[/dim]")

        for chunk in self.model.inference_stream(
            text=text,
            language=language,
            gpt_cond_latent=gpt_cond_latent,
            speaker_embedding=speaker_embedding,
            stream_chunk_size=chunk_size,
            speed=speed,
            enable_text_splitting=True,
        ):
            yield chunk

    def generate_batch(
        self,
        texts: list[str],
        speaker_wav: Path | str,
        output_dir: Path | None = None,
        language: str | None = None,
        prefix: str = "output",
        speed: float = 1.0,
    ) -> list[Path]:
        """
        Generate speech for multiple texts efficiently.

        Args:
            texts: List of texts to synthesize
            speaker_wav: Path to reference audio
            output_dir: Output directory
            language: Language code
            prefix: Prefix for output filenames
            speed: Speech speed multiplier

        Returns:
            List of paths to generated audio files
        """
        speaker_wav = Path(speaker_wav)
        language = language or self.settings.language

        if output_dir is None:
            output_dir = self.settings.get_outputs_path()
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

        total_chars = sum(len(t) for t in texts)
        console.print(f"\n[bold blue]Batch Generation[/bold blue]")
        console.print(f"  Items: {len(texts)}")
        console.print(f"  Total chars: {total_chars:,}")
        console.print(f"  Voice: {speaker_wav.name}")
        console.print(f"  Output: {output_dir}\n")

        # Pre-compute speaker embedding once for entire batch
        gpt_cond_latent, speaker_embedding = self._get_speaker_embedding(speaker_wav)

        output_paths = []
        start_time = time.time()

        with Progress(
            TextColumn("[progress.description]{task.description}"),
            BarColumn(),
            TaskProgressColumn(),
            TimeElapsedColumn(),
            TimeRemainingColumn(),
            console=console,
        ) as progress:
            task = progress.add_task("Generating...", total=len(texts))

            for i, text in enumerate(texts):
                output_path = output_dir / f"{prefix}_{i:04d}.wav"

                out = self.model.inference(
                    text=text,
                    language=language,
                    gpt_cond_latent=gpt_cond_latent,
                    speaker_embedding=speaker_embedding,
                    speed=speed,
                    enable_text_splitting=True,
                )

                wav = torch.tensor(out["wav"]).unsqueeze(0)
                torchaudio.save(str(output_path), wav, 24000)

                output_paths.append(output_path)
                progress.update(task, advance=1, description=f"[{i + 1}/{len(texts)}]")

        elapsed = time.time() - start_time
        avg_time = elapsed / len(texts) if texts else 0
        console.print(f"\n[green]Batch complete![/green]")
        console.print(f"  Generated: {len(output_paths)} files")
        console.print(f"  Total time: {elapsed:.1f}s")
        console.print(f"  Avg per item: {avg_time:.2f}s")

        return output_paths

    def generate_from_file(
        self,
        input_file: Path,
        speaker_wav: Path | str,
        output_dir: Path | None = None,
        language: str | None = None,
        split_sentences: bool = True,
        speed: float = 1.0,
    ) -> list[Path]:
        """
        Generate speech from a text file.

        Args:
            input_file: Path to text file
            speaker_wav: Path to reference audio
            output_dir: Output directory
            language: Language code
            split_sentences: Split text into sentences for batch processing
            speed: Speech speed multiplier

        Returns:
            List of paths to generated audio files
        """
        input_file = Path(input_file)
        text = input_file.read_text(encoding="utf-8").strip()

        if not text:
            raise ValueError(f"Empty text file: {input_file}")

        if split_sentences:
            import re

            sentences = re.split(r"(?<=[.!?])\s+", text)
            sentences = [s.strip() for s in sentences if s.strip()]

            console.print(f"[blue]Split into {len(sentences)} sentences[/blue]")

            return self.generate_batch(
                texts=sentences,
                speaker_wav=speaker_wav,
                output_dir=output_dir,
                language=language,
                prefix=input_file.stem,
                speed=speed,
            )
        else:
            output_path = self.generate(
                text=text,
                speaker_wav=speaker_wav,
                output_path=output_dir / f"{input_file.stem}.wav" if output_dir else None,
                language=language,
                speed=speed,
            )
            return [output_path]

    def precompute_speaker(self, audio_path: Path | str) -> None:
        """
        Pre-compute and cache speaker embedding for faster generation later.

        Args:
            audio_path: Path to reference audio
        """
        audio_path = Path(audio_path)
        console.print(f"[blue]Pre-computing speaker embedding for: {audio_path.name}[/blue]")
        self._get_speaker_embedding(audio_path)
        console.print("[green]Speaker embedding cached[/green]")

    def list_models(self) -> list[str]:
        """List available TTS models."""
        from TTS.utils.manage import ModelManager

        manager = ModelManager()
        return list(manager.list_models())

    def unload_model(self) -> None:
        """Unload the model to free memory."""
        if self._model is not None:
            del self._model
            self._model = None
            self._speaker_cache.clear()
            torch.cuda.empty_cache()
            console.print("[yellow]Model unloaded[/yellow]")
