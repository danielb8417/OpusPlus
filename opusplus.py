import argparse
from concurrent.futures import ThreadPoolExecutor
import ctypes
import math
import os
import shutil
import struct
import subprocess
import sys
import time
from collections import Counter, deque

import numpy as np

RATE = 48000
CHANNELS = 2
OPUS_APPLICATION_AUDIO = 2049

OPUS_SET_BITRATE_REQUEST = 4002
OPUS_SET_MAX_BANDWIDTH_REQUEST = 4004
OPUS_SET_VBR_REQUEST = 4006
OPUS_SET_BANDWIDTH_REQUEST = 4008
OPUS_SET_COMPLEXITY_REQUEST = 4010
OPUS_SET_FORCE_CHANNELS_REQUEST = 4022
OPUS_SET_SIGNAL_REQUEST = 4024
OPUS_GET_LOOKAHEAD_REQUEST = 4027
OPUS_SET_PHASE_INVERSION_DISABLED_REQUEST = 4046

# Experimental libopus force-mode control.
OPUS_SET_FORCE_MODE_REQUEST = 11002
OPUS_MODE_SILK_ONLY = 1000
OPUS_MODE_HYBRID = 1001
OPUS_MODE_CELT_ONLY = 1002

OPUS_AUTO = -1000
OPUS_SIGNAL_VOICE = 3001
OPUS_SIGNAL_MUSIC = 3002
OPUS_BANDWIDTH_WIDEBAND = 1103
OPUS_BANDWIDTH_SUPERWIDEBAND = 1104
OPUS_BANDWIDTH_FULLBAND = 1105

MIN_BITRATE_BPS = 500
MAX_BITRATE_BPS = 512000
HYBRID_MIN_FRAME_BITRATE_BPS = 18000  # Raise target when Hybrid wins (>16 kbps)
OGG_OVERHEAD_ESTIMATE_BPS = 250
MAX_PACKET_BYTES = 1275 * 3
SWB_MONO_GUARD_KBPS = 12.0
STEREO_MIN_BITRATE_BPS = 10000


def vprint(args, *values, **kwargs):
    if args.verbose:
        print(*values, **kwargs)


def auto_mono_enabled(args):
    """Use a genuine mono stream at <=16 kbps, unless already requested."""
    return bool(args.pure_mono or args.bitrate <= 16.0)


def output_channels(args):
    return 1 if auto_mono_enabled(args) else CHANNELS


def resolve_codec_strategy(args):
    if args.codec_strategy != "auto":
        return args.codec_strategy
    return "celt"


# ============================================================
# libopus loading
# ============================================================

def load_opus(dll_path=None):
    import ctypes.util
    import glob
    import platform

    system = platform.system()
    base_dir = os.path.dirname(os.path.abspath(__file__))
    candidates = []

    if dll_path:
        candidates.append(dll_path)
    else:
        if system == "Windows":
            candidates.extend([
                os.path.join(base_dir, "opus.dll"),
                "opus.dll",
            ])

        library_name = ctypes.util.find_library("opus")
        if library_name:
            candidates.append(library_name)

        if system == "Darwin":
            for prefix in [
                "/opt/homebrew/opt/opus/lib",
                "/usr/local/opt/opus/lib",
                "/opt/homebrew/lib",
                "/usr/local/lib",
            ]:
                candidates.extend(
                    glob.glob(os.path.join(prefix, "libopus*.dylib"))
                )

        elif system == "Linux":
            candidates.extend(["libopus.so.0", "libopus.so"])

        elif system == "Windows":
            candidates.extend(["libopus-0.dll", "opus.dll"])

    candidates = list(dict.fromkeys(candidates))
    errors = []

    for candidate in candidates:
        try:
            return ctypes.CDLL(candidate)
        except (OSError, TypeError) as exc:
            errors.append(f"{candidate}: {exc}")

    details = "\n".join(errors) if errors else "No library candidates found."
    raise RuntimeError(
        f"Could not load Opus on {system}.\n"
        "Install the Opus library or provide its path with --opus-dll.\n"
        f"Attempted locations:\n{details}"
    )


def configure_opus_functions(opus):
    opus.opus_encoder_create.argtypes = [
        ctypes.c_int, ctypes.c_int, ctypes.c_int,
        ctypes.POINTER(ctypes.c_int),
    ]
    opus.opus_encoder_create.restype = ctypes.c_void_p

    opus.opus_encoder_destroy.argtypes = [ctypes.c_void_p]
    opus.opus_encoder_destroy.restype = None

    opus.opus_encode.argtypes = [
        ctypes.c_void_p,
        ctypes.POINTER(ctypes.c_int16),
        ctypes.c_int,
        ctypes.POINTER(ctypes.c_ubyte),
        ctypes.c_int,
    ]
    opus.opus_encode.restype = ctypes.c_int

    opus.opus_decoder_create.argtypes = [
        ctypes.c_int, ctypes.c_int, ctypes.POINTER(ctypes.c_int),
    ]
    opus.opus_decoder_create.restype = ctypes.c_void_p
    opus.opus_decoder_destroy.argtypes = [ctypes.c_void_p]
    opus.opus_decoder_destroy.restype = None
    opus.opus_decode.argtypes = [
        ctypes.c_void_p, ctypes.POINTER(ctypes.c_ubyte), ctypes.c_int,
        ctypes.POINTER(ctypes.c_int16), ctypes.c_int, ctypes.c_int,
    ]
    opus.opus_decode.restype = ctypes.c_int

    # Variadic function: specify the return type, but not argtypes.
    opus.opus_encoder_ctl.restype = ctypes.c_int

    if hasattr(opus, "opus_get_version_string"):
        opus.opus_get_version_string.argtypes = []
        opus.opus_get_version_string.restype = ctypes.c_char_p

    return opus


def opus_ctl(opus, encoder, request, value):
    result = opus.opus_encoder_ctl(
        ctypes.c_void_p(encoder),
        ctypes.c_int(request),
        ctypes.c_int(int(value)),
    )
    if result != 0:
        raise RuntimeError(
            f"libopus control {request} failed: {result}"
        )


def opus_ctl_pointer(opus, encoder, request, pointer):
    result = opus.opus_encoder_ctl(
        ctypes.c_void_p(encoder),
        ctypes.c_int(request),
        ctypes.cast(pointer, ctypes.c_void_p),
    )
    if result != 0:
        raise RuntimeError(
            f"libopus pointer control {request} failed: {result}"
        )


def create_encoder(opus, args, cbr=True):
    channels = output_channels(args)
    error = ctypes.c_int(0)

    encoder = opus.opus_encoder_create(
        RATE, channels, OPUS_APPLICATION_AUDIO, ctypes.byref(error)
    )

    if not encoder or error.value != 0:
        raise RuntimeError(f"Could not create encoder: {error.value}")

    try:
        opus_ctl(opus, encoder, OPUS_SET_VBR_REQUEST, 0 if cbr else 1)
        opus_ctl(
            opus, encoder, OPUS_SET_COMPLEXITY_REQUEST, args.complexity
        )
        signal = OPUS_SIGNAL_VOICE if args.mode == "voice" else OPUS_SIGNAL_MUSIC
        opus_ctl(opus, encoder, OPUS_SET_SIGNAL_REQUEST, signal)
        opus_ctl(
            opus, encoder, OPUS_SET_PHASE_INVERSION_DISABLED_REQUEST, 1
        )

        if channels == 2:
            opus_ctl(
                opus, encoder, OPUS_SET_FORCE_CHANNELS_REQUEST, 2
            )

        # Private/build-dependent control. Auto mode chooses a mode per frame.
        if args.mode == "music":
            opus_ctl(
                opus, encoder,
                OPUS_SET_FORCE_MODE_REQUEST,
                OPUS_MODE_CELT_ONLY,
            )

        lookahead = ctypes.c_int(0)
        opus_ctl_pointer(
            opus, encoder,
            OPUS_GET_LOOKAHEAD_REQUEST,
            ctypes.byref(lookahead),
        )

        return encoder, lookahead.value

    except Exception:
        opus.opus_encoder_destroy(ctypes.c_void_p(encoder))
        raise


# ============================================================
# Opus packet / TOC analysis
# ============================================================

def classify_opus_packet(packet):
    if not packet:
        return {
            "mode": "Invalid", "bandwidth": "Unknown",
            "config": None, "toc": None,
        }

    toc = packet[0]
    config = toc >> 3

    if config <= 3:
        mode, bandwidth = "SILK", "Narrowband"
    elif config <= 7:
        mode, bandwidth = "SILK", "Mediumband"
    elif config <= 11:
        mode, bandwidth = "SILK", "Wideband"
    elif config <= 13:
        mode, bandwidth = "Hybrid", "Superwideband"
    elif config <= 15:
        mode, bandwidth = "Hybrid", "Fullband"
    elif config <= 19:
        mode, bandwidth = "CELT", "Narrowband"
    elif config <= 23:
        mode, bandwidth = "CELT", "Wideband"
    elif config <= 27:
        mode, bandwidth = "CELT", "Superwideband"
    else:
        mode, bandwidth = "CELT", "Fullband"

    return {
        "mode": mode, "bandwidth": bandwidth,
        "config": config, "toc": toc,
    }


def analyze_opus_packets(packets):
    modes = Counter()
    bandwidths = Counter()
    combinations = Counter()
    configs = Counter()
    tocs = Counter()
    invalid = 0

    for packet in packets:
        info = classify_opus_packet(packet)

        if info["mode"] == "Invalid":
            invalid += 1
            continue

        modes[info["mode"]] += 1
        bandwidths[info["bandwidth"]] += 1
        combinations[(info["mode"], info["bandwidth"])] += 1
        configs[info["config"]] += 1
        tocs[info["toc"]] += 1

    return {
        "total": len(packets),
        "valid": len(packets) - invalid,
        "invalid": invalid,
        "mode_counts": modes,
        "bandwidth_counts": bandwidths,
        "combination_counts": combinations,
        "config_counts": configs,
        "toc_counts": tocs,
    }


def print_opus_packet_report(packets):
    report = analyze_opus_packets(packets)
    valid = report["valid"]

    print("\nOPUS PACKET ANALYSIS")
    print("=" * 48)
    print(f"Total audio packets: {report['total']}")
    print(f"Valid TOC packets:   {valid}")

    if report["invalid"]:
        print(f"Invalid/empty:       {report['invalid']}")

    if valid == 0:
        print("No valid Opus packets to analyze.")
        return report

    def print_counts(title, counts, labels):
        print(f"\n{title}")
        for label in labels:
            count = counts.get(label, 0)
            print(
                f"  {label:<18} {count:>8,} "
                f"({100.0 * count / valid:6.2f}%)"
            )

    print_counts(
        "CODING MODES", report["mode_counts"],
        ["SILK", "Hybrid", "CELT"]
    )
    print_counts(
        "CODED BANDWIDTH", report["bandwidth_counts"],
        ["Narrowband", "Mediumband", "Wideband",
         "Superwideband", "Fullband"]
    )

    print("\nMODE + BANDWIDTH")
    combinations = [
        ("SILK", "Narrowband"),
        ("SILK", "Mediumband"),
        ("SILK", "Wideband"),
        ("Hybrid", "Superwideband"),
        ("Hybrid", "Fullband"),
        ("CELT", "Narrowband"),
        ("CELT", "Wideband"),
        ("CELT", "Superwideband"),
        ("CELT", "Fullband"),
    ]

    for mode, bandwidth in combinations:
        count = report["combination_counts"].get((mode, bandwidth), 0)
        print(
            f"  {mode:<8} {bandwidth:<16} {count:>8,} "
            f"({100.0 * count / valid:6.2f}%)"
        )

    print("\nTOC CONFIGURATIONS")
    for config in sorted(report["config_counts"]):
        count = report["config_counts"][config]
        print(
            f"  Config {config:>2}: {count:>8,} "
            f"({100.0 * count / valid:6.2f}%)"
        )

    print("\nMOST COMMON TOC BYTES")
    for toc, count in report["toc_counts"].most_common(10):
        print(
            f"  0x{toc:02X}: {count:>8,} "
            f"({100.0 * count / valid:6.2f}%)"
        )

    print(
        "\nNote: TOC bandwidth is coded bandwidth, not a measurement "
        "of the original audio's frequency content."
    )
    print(
        "Note: TOC mode analysis does not establish whether stereo "
        "channel coupling was used."
    )

    return report


# ============================================================
# Decode and frame preparation
# ============================================================

def find_ffmpeg():
    path = shutil.which("ffmpeg")
    if not path:
        raise RuntimeError("FFmpeg was not found on PATH.")
    return path


def decode_input(path, pure_mono=False):
    # FFmpeg downmixes to one channel before analysis/encoding when mono is on.
    channels = 1 if pure_mono else 2

    command = [
        find_ffmpeg(), "-hide_banner", "-loglevel", "error",
        "-i", path, "-map", "0:a:0", "-vn", "-sn", "-dn",
        "-f", "s16le", "-acodec", "pcm_s16le",
        "-ar", str(RATE), "-ac", str(channels), "pipe:1",
    ]

    process = subprocess.Popen(
        command, stdout=subprocess.PIPE, stderr=subprocess.PIPE
    )
    pcm = bytearray()

    try:
        while True:
            chunk = process.stdout.read(1024 * 1024)
            if not chunk:
                break
            pcm.extend(chunk)

        stderr = process.stderr.read()
        code = process.wait()

        if code != 0:
            raise RuntimeError(
                "FFmpeg decoding failed:\n"
                + stderr.decode("utf-8", errors="replace")
            )
    finally:
        if process.poll() is None:
            process.kill()
            process.wait()

    sample_width = 2 * channels
    pcm = pcm[:len(pcm) - len(pcm) % sample_width]

    if not pcm:
        raise RuntimeError("FFmpeg decoded no audio samples.")

    return np.frombuffer(bytes(pcm), dtype=np.int16).copy()


def make_frames(pcm, frame_samples, channels):
    values_per_frame = frame_samples * channels
    frames = []

    for start in range(0, len(pcm), values_per_frame):
        chunk = pcm[start:start + values_per_frame]
        actual_values = len(chunk)

        if actual_values < values_per_frame:
            padded = np.zeros(values_per_frame, dtype=np.int16)
            padded[:actual_values] = chunk
            chunk = padded

        frames.append((chunk, actual_values // channels))

    return frames


# ============================================================
# Analysis-only per-frame peak normalization
# ============================================================

def normalize_analysis_frames(frames, args, channels):
    if not args.vbr_normalize or args.vbr_normalize_level == 0:
        return [frame.copy() for frame, _ in frames]

    intensity = args.vbr_normalize_level / 10.0
    normalized = []

    for frame, actual_samples in frames:
        x = frame.astype(np.float32).copy()
        valid_values = actual_samples * channels
        valid = x[:valid_values]

        if valid.size == 0:
            normalized.append(frame.copy())
            continue

        peak = float(np.max(np.abs(valid)))

        if peak <= args.vbr_normalize_gate:
            normalized.append(frame.copy())
            continue

        target_gain_db = max(
            0.0, 20.0 * math.log10(32768.0 / peak)
        )
        gain = 10.0 ** ((target_gain_db * intensity) / 20.0)
        x *= gain

        normalized.append(
            np.clip(x, -32768.0, 32767.0).astype(np.int16)
        )

    return normalized


# ============================================================
# Mono, phase-inversion and SWB detection
# ============================================================

def analyze_mode(frame, previous_mode, args):
    channels = output_channels(args)
    samples = np.asarray(frame, dtype=np.float32)

    phase_inverted = False
    correlation = 1.0
    stereo = None

    if channels == 1:
        mid = samples
        mid_rms = float(np.sqrt(np.mean(mid * mid) + 1e-12))
        side_db = -120.0
        mono = True

    else:
        stereo = samples.reshape(-1, 2)
        left, right = stereo[:, 0], stereo[:, 1]

        left_rms = float(np.sqrt(np.mean(left * left) + 1e-12))
        right_rms = float(np.sqrt(np.mean(right * right) + 1e-12))

        denominator = left_rms * right_rms

        if denominator > 1.0:
            correlation = float(
                np.mean(left * right) / denominator
            )
            correlation = max(-1.0, min(1.0, correlation))
        else:
            correlation = 0.0

        level_balance = (
            min(left_rms, right_rms)
            / max(left_rms, right_rms, 1e-9)
        )

        phase_inverted = (
            correlation <= args.phase_invert_correlation
            and level_balance >= 0.5
            and min(left_rms, right_rms) >= args.silence_threshold
        )

        mid = (left + right) * 0.5
        side = (left - right) * 0.5

        mid_rms = float(np.sqrt(np.mean(mid * mid) + 1e-12))
        side_rms = float(np.sqrt(np.mean(side * side) + 1e-12))

        side_db = 20.0 * math.log10(
            max(side_rms, 1e-9) / max(mid_rms, 1e-9)
        )

        if phase_inverted:
            mono = False
        elif previous_mode is not None and previous_mode["mono"]:
            mono = side_db < args.mono_exit_db
        else:
            mono = side_db < args.mono_enter_db

    if channels == 2:
        left = stereo[:, 0]
        right = stereo[:, 1]

        window = np.hanning(len(left)).astype(np.float32)
        left_spectrum = np.fft.rfft(left * window)
        right_spectrum = np.fft.rfft(right * window)

        power = (
            np.abs(left_spectrum) ** 2
            + np.abs(right_spectrum) ** 2
        )
    else:
        window = np.hanning(len(mid)).astype(np.float32)
        spectrum = np.fft.rfft(mid * window)
        power = np.abs(spectrum) ** 2

    frequencies = np.fft.rfftfreq(len(mid), d=1.0 / RATE)

    low = float(np.sum(power[frequencies < 12000.0]))
    high = float(np.sum(power[frequencies >= 12000.0]))

    high_db = 10.0 * math.log10(
        max(high, 1e-30) / max(low, 1e-30)
    )

    if previous_mode is not None and previous_mode["swb"]:
        swb = high_db < args.swb_exit_db
    else:
        swb = high_db < args.swb_enter_db

    if channels == 2:
        signal_rms = float(
            np.sqrt(np.mean(stereo * stereo) + 1e-12)
        )
    else:
        signal_rms = mid_rms

    if signal_rms < args.silence_threshold:
        mono = True
        swb = True
        phase_inverted = False

    if args.pure_mono:
        mono = True
        phase_inverted = False

    return {
        "mono": bool(mono),
        "swb": bool(swb),
        "phase_inverted": bool(phase_inverted),
        "side_db": side_db,
        "high_db": high_db,
        "correlation": correlation,
    }


def analyze_all_modes(frames, args):
    modes = []
    previous = None

    for frame, _ in frames:
        mode = analyze_mode(frame, previous, args)
        modes.append(mode)
        previous = mode

    return modes


def mode_multiplier(mode, args):
    multiplier = 1.0

    if mode["swb"]:
        multiplier *= args.swb_multiplier

    if mode.get("phase_inverted", False):
        return multiplier

    if mode["mono"] and not args.pure_mono:
        bitrate_after_swb = args.bitrate * multiplier

        if bitrate_after_swb >= SWB_MONO_GUARD_KBPS:
            multiplier *= args.mono_multiplier

    return multiplier


# ============================================================
# Activity and transient analysis
# ============================================================

def analyze_frame(frame, channels):
    x = np.asarray(frame, dtype=np.float32).reshape(-1, channels)

    if len(x) < 3:
        return 0.45, 0.0

    rms = float(np.sqrt(np.mean(x * x) + 1.0))
    delta = np.diff(x, axis=0)
    mean_delta = float(np.mean(np.abs(delta)) + 1.0)

    relative_delta = mean_delta / max(rms, 16.0)
    level_activity = math.log10(1.0 + rms / 500.0)
    detail_activity = math.log10(1.0 + relative_delta * 20.0)

    activity = max(
        0.0,
        min(2.5, 0.20 * level_activity + 0.80 * detail_activity),
    )

    second_delta = np.diff(x, n=2, axis=0)
    curvature = float(np.mean(np.abs(second_delta)) + 1.0)
    transient_ratio = curvature / max(mean_delta, 1.0)

    half = max(1, len(x) // 2)
    rms_a = float(np.sqrt(np.mean(x[:half] ** 2) + 1.0))
    rms_b = float(np.sqrt(np.mean(x[half:] ** 2) + 1.0))

    level_jump = abs(math.log((rms_b + 16.0) / (rms_a + 16.0)))
    curvature_score = max(
        0.0, min((transient_ratio - 1.0) / 5.0, 1.0)
    )
    jump_score = max(0.0, min(level_jump / 1.5, 1.0))

    transient = 0.65 * curvature_score + 0.35 * jump_score

    return activity, max(0.0, min(transient, 1.0))


# ============================================================
# Adaptive bitrate controller
# ============================================================

class BitrateController:
    def __init__(self, target_kbps, intensity, channels):
        self.target_bps = float(target_kbps) * 1000.0
        self.target_kbps = float(target_kbps)
        self.intensity = int(intensity)
        self.channels = channels
        self.total_packet_bytes = 0
        self.total_samples = 0

    def bitrate_for_frame(self, analysis_frame, multiplier):
        packet_target_bps = max(
            MIN_BITRATE_BPS,
            self.target_bps * multiplier - OGG_OVERHEAD_ESTIMATE_BPS,
        )

        if self.intensity == 0:
            return int(packet_target_bps)

        activity, transient = analyze_frame(
            analysis_frame, self.channels
        )

        # Scale low-rate VBR strength against 32 kbps (not 16 kbps).
        # At 16 kbps this is 50%; at 8 kbps it is 25%; at 32 kbps it is full.
        low_rate_scale = min(1.0, self.target_kbps / 32.0)
        strength = (self.intensity / 4.0) * low_rate_scale

        activity_factor = (
            1.0 + strength * ((activity - 0.45) * 0.75)
        )
        transient_factor = 1.0 + strength * 1.5 * transient

        activity_factor = max(0.75, min(1.50, activity_factor))
        transient_factor = max(1.0, min(1.40, transient_factor))

        if self.total_samples <= 0 or self.total_packet_bytes <= 0:
            feedback_factor = 1.0
        else:
            elapsed = self.total_samples / RATE
            target_bytes = self.target_bps * elapsed / 8.0
            ratio = target_bytes / max(self.total_packet_bytes, 1)

            # Soften long-term feedback changes at low target rates.
            feedback_strength = 0.35 * low_rate_scale
            feedback_factor = ratio ** feedback_strength
            feedback_factor = max(0.85, min(1.20, feedback_factor))

        requested = (
            packet_target_bps
            * activity_factor
            * transient_factor
            * feedback_factor
        )

        return int(max(
            MIN_BITRATE_BPS,
            min(MAX_BITRATE_BPS, requested),
        ))

    def record_packet(self, packet_bytes, frame_samples):
        self.total_packet_bytes += int(packet_bytes)
        self.total_samples += int(frame_samples)


# ============================================================
# Bandwidth selection
# ============================================================

def select_bandwidth(args, mode):
    # Explicit override modes ignore the usual low-bitrate bandwidth policy.
    # These are encoder requests; libopus may still internally limit the
    # effective bandwidth when the requested bitrate is extremely low.
    if args.bandwidth_mode == "wideband":
        return OPUS_BANDWIDTH_WIDEBAND
    if args.bandwidth_mode == "swb":
        return OPUS_BANDWIDTH_SUPERWIDEBAND
    if args.bandwidth_mode == "fullband":
        return OPUS_BANDWIDTH_FULLBAND
    if args.bandwidth_mode == "auto":
        return (
            OPUS_BANDWIDTH_SUPERWIDEBAND
            if mode["swb"] else OPUS_BANDWIDTH_FULLBAND
        )

    # Default policy: at or below 6 kbps, request CELT wideband.
    if args.bitrate <= 6.0:
        return OPUS_BANDWIDTH_WIDEBAND

    # Above 6 and below 10 kbps, default to SWB on every frame.
    if args.bitrate < 10.0:
        if not args.adaptive_swb:
            return OPUS_BANDWIDTH_SUPERWIDEBAND
        return (
            OPUS_BANDWIDTH_SUPERWIDEBAND
            if mode["swb"] else OPUS_BANDWIDTH_FULLBAND
        )

    if args.codec_strategy == "swb" or mode["swb"]:
        return OPUS_BANDWIDTH_SUPERWIDEBAND
    return OPUS_BANDWIDTH_FULLBAND


def select_max_bandwidth(args):
    # Explicit override modes also lift the script's normal low-rate ceiling.
    if args.bandwidth_mode in ("auto", "fullband"):
        return OPUS_BANDWIDTH_FULLBAND
    if args.bandwidth_mode == "swb":
        return OPUS_BANDWIDTH_SUPERWIDEBAND
    if args.bandwidth_mode == "wideband":
        return OPUS_BANDWIDTH_WIDEBAND

    if args.bitrate <= 6.5:
        return OPUS_BANDWIDTH_WIDEBAND
    if args.bitrate < 16.0 and not args.adaptive_swb:
        return OPUS_BANDWIDTH_SUPERWIDEBAND
    return OPUS_BANDWIDTH_FULLBAND


# ============================================================
# Lightweight speech/music detector for --mode auto
# ============================================================

def detect_speech_frame(frame, channels):
    """Conservative heuristic detector; returns True for speech-like frames.

    Uses periodicity, zero-crossing rate, and spectral flatness. This is not
    a trained classifier; singing/rap and speech over music can be ambiguous.
    """
    x = np.asarray(frame, dtype=np.float32)
    if channels == 2:
        x = x.reshape(-1, 2).mean(axis=1)
    else:
        x = x.reshape(-1)
    if x.size < 80:
        return False
    x -= np.mean(x)
    peak = float(np.max(np.abs(x))) if x.size else 0.0
    rms = float(np.sqrt(np.mean(x * x))) if x.size else 0.0
    if peak < 80.0 or rms < 25.0:
        return False

    # Speech commonly has moderate ZCR and some short-time periodicity.
    zcr = float(np.mean((x[1:] >= 0) != (x[:-1] >= 0)))
    # Autocorrelation over plausible voiced speech pitch periods (roughly
    # 70-400 Hz at 48 kHz). Normalize to avoid level dependence.
    max_lag = min(len(x) // 2, int(RATE / 70))
    min_lag = max(1, int(RATE / 400))
    if max_lag <= min_lag:
        return False
    # Downsample for cheaper autocorrelation while preserving pitch range.
    y = x[::3]
    y = y * np.hanning(len(y))
    denom = float(np.dot(y, y)) + 1e-9
    corr = np.correlate(y, y, mode="full")[len(y)-1:]
    lo = max(1, min_lag // 3)
    hi = min(len(corr) - 1, max_lag // 3)
    periodicity = float(np.max(corr[lo:hi+1]) / (corr[0] + 1e-9)) if hi >= lo else 0.0

    # Spectral flatness: very noise-like frames are less likely to be voiced
    # speech; this feature is deliberately weak because consonants are noisy.
    spec = np.abs(np.fft.rfft(x * np.hanning(len(x))))[1:] + 1e-8
    flatness = float(np.exp(np.mean(np.log(spec))) / (np.mean(spec) + 1e-9))
    return bool(periodicity >= 0.22 and zcr <= 0.32 and flatness <= 0.72)


def detect_speech_sequence(frames, channels):
    """Detect frames and smooth isolated decisions to reduce mode flapping."""
    raw = [detect_speech_frame(frame, channels) for frame, _ in frames]
    smoothed = []
    for i in range(len(raw)):
        left, right = max(0, i - 1), min(len(raw), i + 2)
        votes = sum(raw[left:right])
        smoothed.append(votes >= 2 if right - left >= 2 else raw[i])
    return smoothed


# ============================================================
# Encode packets
# ============================================================

def create_auto_candidates(opus, args, cbr=True):
    """Create independent stateful encoder/decoder pairs for mode trials."""
    candidates = []
    for name, forced_mode, signal in (
        ("silk", OPUS_MODE_SILK_ONLY, OPUS_SIGNAL_VOICE),
        ("hybrid", OPUS_MODE_HYBRID, OPUS_SIGNAL_VOICE),
        ("celt", OPUS_MODE_CELT_ONLY, OPUS_SIGNAL_MUSIC),
    ):
        err = ctypes.c_int(0)
        enc = opus.opus_encoder_create(
            RATE, output_channels(args), OPUS_APPLICATION_AUDIO,
            ctypes.byref(err),
        )
        if not enc or err.value != 0:
            continue
        dec = None
        try:
            opus_ctl(opus, enc, OPUS_SET_VBR_REQUEST, 0 if cbr else 1)
            opus_ctl(opus, enc, OPUS_SET_COMPLEXITY_REQUEST, args.complexity)
            opus_ctl(opus, enc, OPUS_SET_SIGNAL_REQUEST, signal)
            opus_ctl(opus, enc, OPUS_SET_PHASE_INVERSION_DISABLED_REQUEST, 1)
            opus_ctl(opus, enc, OPUS_SET_FORCE_MODE_REQUEST, forced_mode)
            derr = ctypes.c_int(0)
            dec = opus.opus_decoder_create(
                RATE, output_channels(args), ctypes.byref(derr)
            )
            if not dec or derr.value != 0:
                raise RuntimeError("candidate decoder creation failed")
            candidates.append({
                "name": name, "mode": forced_mode, "encoder": enc,
                "decoder": dec, "signal": signal,
            })
        except Exception:
            if dec:
                opus.opus_decoder_destroy(ctypes.c_void_p(dec))
            opus.opus_encoder_destroy(ctypes.c_void_p(enc))
    if not candidates:
        raise RuntimeError(
            "Auto mode could not initialize candidate encoders. "
            "This libopus build may not support the private force-mode control (11002)."
        )
    return candidates


def _multiresolution_spectrum(x, fft_size):
    """Return overlapping Hann-window magnitude spectra for (samples, channels)."""
    n, channels = x.shape
    if n <= fft_size:
        starts = [0]
    else:
        hop = max(1, fft_size // 2)
        starts = list(range(0, n - fft_size + 1, hop))
        if starts[-1] != n - fft_size:
            starts.append(n - fft_size)

    win = np.hanning(fft_size).reshape(-1, 1)
    spectra = []
    for start in starts:
        chunk = x[start:start + fft_size]
        if len(chunk) < fft_size:
            padded = np.zeros((fft_size, channels), dtype=np.float64)
            padded[:len(chunk)] = chunk
            chunk = padded
        spectra.append(np.abs(np.fft.rfft(chunk * win, axis=0)))
    return spectra


def candidate_distortion(original, decoded, channels):
    """Compare the actual decoded waveform plus its multi-resolution spectrum.

    Lower is better. Unlike the earlier spectral-only scorer, this explicitly
    measures sample-domain error and transient/slope error as well as spectral
    shape and loudness-envelope changes. It is a heuristic, not a perceptual
    MOS predictor.
    """
    n_values = min(len(original), len(decoded))
    n_frames = n_values // channels
    if n_frames < 8:
        return float("inf")
    n_values = n_frames * channels
    x = np.asarray(original[:n_values], dtype=np.float64).reshape(n_frames, channels)
    y = np.asarray(decoded[:n_values], dtype=np.float64).reshape(n_frames, channels)

    # Remove DC consistently before comparing waveform and spectral content.
    x = x - np.mean(x, axis=0, keepdims=True)
    y = y - np.mean(y, axis=0, keepdims=True)
    reference_rms = float(np.sqrt(np.mean(x * x)) + 1.0)

    # Direct waveform error, normalized so the score is not level-dependent.
    waveform = float(np.sqrt(np.mean((x - y) ** 2)) / reference_rms)

    # First-difference error emphasizes transients and consonant edges.
    if n_frames > 1:
        dx = np.diff(x, axis=0)
        dy = np.diff(y, axis=0)
        slope_rms = float(np.sqrt(np.mean(dx * dx)) + 1.0)
        transient = float(np.sqrt(np.mean((dx - dy) ** 2)) / slope_rms)
    else:
        transient = waveform

    sizes = sorted(set(min(n_frames, k) for k in (128, 256, 512, 1024)
                       if min(n_frames, k) >= 32))
    if not sizes:
        sizes = [n_frames]

    spectral_errors = []
    convergence_errors = []
    for size in sizes:
        x_specs = _multiresolution_spectrum(x, size)
        y_specs = _multiresolution_spectrum(y, size)
        for sx, sy in zip(x_specs, y_specs):
            lx = np.log1p(sx)
            ly = np.log1p(sy)
            # Normalize log-spectrum error against the reference spectrum.
            scale = float(np.mean(np.abs(lx)) + 1.0)
            spectral_errors.append(float(np.mean(np.abs(lx - ly)) / scale))
            denom = float(np.linalg.norm(sx)) + 1e-9
            convergence_errors.append(float(np.linalg.norm(sx - sy) / denom))

    spectral = float(np.mean(spectral_errors)) if spectral_errors else 0.0
    convergence = float(np.mean(convergence_errors)) if convergence_errors else 0.0

    block = max(8, min(96, n_frames // 4))
    env_errors = []
    for start in range(0, n_frames, block):
        xx = x[start:start + block]
        yy = y[start:start + block]
        if len(xx) == 0:
            continue
        rms_x = np.sqrt(np.mean(xx * xx, axis=0) + 1.0)
        rms_y = np.sqrt(np.mean(yy * yy, axis=0) + 1.0)
        env_errors.append(float(np.mean(np.abs(np.log(rms_x / rms_y)))))
    envelope = float(np.mean(env_errors)) if env_errors else 0.0

    # Clamp the secondary metrics so no one metric overwhelms the comparison.
    return (
        0.35 * min(waveform, 3.0)
        + 0.20 * min(transient, 3.0)
        + 0.25 * min(spectral, 3.0)
        + 0.15 * min(convergence, 3.0)
        + 0.05 * min(envelope, 3.0)
    )


def choose_candidate_mode(opus, candidates, frame, frame_samples, channels,
                          bitrate, bandwidth, max_bandwidth, frame_is_mono,
                          original_samples, args, executor=None):
    """Trial-encode eligible modes, optionally scoring candidates in parallel.

    Candidate encoders/decoders have independent state. We parallelize only
    the candidates for the current frame and wait for all results before the
    next frame, preserving each candidate's sequential codec history.
    """
    eligible = []
    for candidate in candidates:
        # Hard eligibility split by this frame's requested bitrate:
        # <=16 kbps: SILK + CELT; >16 kbps: Hybrid + CELT.
        # Hybrid above 16 kbps additionally requires mono/near-mono content.
        if bitrate <= 16000:
            if candidate["name"] == "hybrid":
                continue
        else:
            if candidate["name"] == "silk":
                continue
            if candidate["name"] == "hybrid" and not frame_is_mono:
                continue
        eligible.append(candidate)

    def score_one(candidate):
        pcm_ptr = frame.ctypes.data_as(ctypes.POINTER(ctypes.c_int16))
        outbuf = (ctypes.c_ubyte * MAX_PACKET_BYTES)()
        decoded_buf = (ctypes.c_int16 * (frame_samples * channels))()
        enc = candidate["encoder"]
        dec = candidate["decoder"]
        try:
            cbw = bandwidth
            cmax = max_bandwidth
            if candidate["name"] == "silk":
                cbw = min(cbw, OPUS_BANDWIDTH_WIDEBAND)
                cmax = min(cmax, OPUS_BANDWIDTH_WIDEBAND)
            elif candidate["name"] == "hybrid":
                cbw = min(cbw, OPUS_BANDWIDTH_SUPERWIDEBAND)
                cmax = min(cmax, OPUS_BANDWIDTH_SUPERWIDEBAND)
            opus_ctl(opus, enc, OPUS_SET_BANDWIDTH_REQUEST, cbw)
            opus_ctl(opus, enc, OPUS_SET_MAX_BANDWIDTH_REQUEST, cmax)
            opus_ctl(opus, enc, OPUS_SET_BITRATE_REQUEST, bitrate)
            opus_ctl(opus, enc, OPUS_SET_FORCE_CHANNELS_REQUEST,
                     1 if (channels == 2 and frame_is_mono) else channels)
            length = opus.opus_encode(
                ctypes.c_void_p(enc), pcm_ptr, frame_samples, outbuf,
                MAX_PACKET_BYTES,
            )
            if length <= 0:
                return None
            packet = (ctypes.c_ubyte * length).from_buffer_copy(bytes(outbuf[:length]))
            decoded_count = opus.opus_decode(
                ctypes.c_void_p(dec), packet, length, decoded_buf,
                frame_samples, 0,
            )
            if decoded_count <= 0:
                return None
            decoded = np.ctypeslib.as_array(decoded_buf)[:decoded_count * channels].copy()
            original = original_samples[:decoded_count * channels]
            score = candidate_distortion(original, decoded, channels)
            return (score, candidate["name"], candidate["mode"])
        except Exception:
            # A mode/bandwidth combination may be unsupported by this build.
            return None

    if executor is not None and len(eligible) > 1:
        results = list(executor.map(score_one, eligible))
    else:
        results = [score_one(candidate) for candidate in eligible]
    scores = [result for result in results if result is not None]

    if not scores:
        return OPUS_MODE_CELT_ONLY, {"celt": float("inf")}
    scores.sort(key=lambda item: item[0])
    return scores[0][2], {name: score for score, name, _ in scores}

def encode_packets(
    opus, args, frames, modes, analysis_frames,
    frame_samples, cbr=True, targets=None, auto_mode_choices=None,
    auto_mode_select=True
):
    channels = output_channels(args)

    encoder, pre_skip = create_encoder(opus, args, cbr=cbr)
    controller = BitrateController(
        args.bitrate, args.vbr_intensity, channels
    )

    packets = []
    rates = []
    output_buffer = (ctypes.c_ubyte * MAX_PACKET_BYTES)()
    candidates = create_auto_candidates(opus, args, cbr=cbr) if args.mode == "auto" else []
    # Candidate modes for a single frame can be evaluated concurrently.
    # The worker count is configurable and defaults to half the logical CPUs.
    # Stateful output encoding itself remains sequential to preserve Opus history.
    trial_executor = (
        ThreadPoolExecutor(max_workers=max(1, min(args.threads, len(candidates))))
        if args.mode == "auto" and auto_mode_select and len(candidates) > 1
        else None
    )
    auto_wins = Counter()
    last_forced_mode = None
    # Section-level voice/music classifier driven by the candidate scorer.
    # The rolling window contains each frame's raw lowest-distortion winner.
    section_win_history = deque(maxlen=max(
        1, int(round(args.section_window_ms / args.frame_ms))
    ))
    voice_section = False
    voice_enter = args.voice_enter_percent / 100.0
    voice_exit = args.voice_exit_percent / 100.0

    try:
        for i, (frame, actual_samples) in enumerate(frames):
            mode = modes[i]
            phase_inverted = mode.get("phase_inverted", False)

            bandwidth = select_bandwidth(args, mode)

            opus_ctl(
                opus, encoder,
                OPUS_SET_BANDWIDTH_REQUEST,
                bandwidth,
            )
            opus_ctl(
                opus, encoder,
                OPUS_SET_MAX_BANDWIDTH_REQUEST,
                select_max_bandwidth(args),
            )

            if targets is None:
                requested = controller.bitrate_for_frame(
                    analysis_frames[i],
                    mode_multiplier(mode, args),
                )
            else:
                requested = targets[i]

            requested = int(max(
                MIN_BITRATE_BPS,
                min(MAX_BITRATE_BPS, requested),
            ))

            # Active-frame bitrate floor: avoid extreme one-frame target dips.
            # This floors the bitrate REQUEST sent to libopus, not the actual
            # packet size (libopus VBR may still emit a smaller packet).
            # True silence is exempt so quiet gaps can remain efficiently coded.
            frame_peak = float(np.max(np.abs(frame))) if frame.size else 0.0
            frame_rms = float(np.sqrt(np.mean(frame.astype(np.float64) ** 2))) if frame.size else 0.0
            is_silence_frame = (
                frame_peak < args.silence_threshold
                or frame_rms < args.silence_threshold
            )
            if not is_silence_frame:
                requested = max(
                    requested,
                    int(args.bitrate * 1000.0 / 3.0),
                )

            # Preserve the requested phase-inverted stereo floor.
            if phase_inverted and channels == 2:
                requested = max(
                    requested,
                    int(args.phase_invert_floor_kbps * 1000),
                )

            # At <=16 kbps, automatically use mono coding for ordinary
            # content to concentrate the limited bits. Above 16 kbps, use
            # the mono/mostly-mono detector. Phase-inverted material stays
            # stereo so its side-channel information is not collapsed.
            frame_is_mono = (
                channels == 2
                and not phase_inverted
                and (args.bitrate <= 16.0 or mode["mono"])
            )

            if args.mode == "auto":
                if auto_mode_select:
                    # Candidate competition is enabled only for the final
                    # encode. Calibration passes leave Opus in automatic mode.
                    if auto_mode_choices is not None and i < len(auto_mode_choices):
                        chosen_mode = auto_mode_choices[i]
                        trial_scores = {}
                    else:
                        raw_winner, trial_scores = choose_candidate_mode(
                            opus, candidates, frame, frame_samples, channels,
                            requested, bandwidth, select_max_bandwidth(args),
                            frame_is_mono, frame, args, trial_executor,
                        )
                        raw_name = {
                            OPUS_MODE_SILK_ONLY: "silk",
                            OPUS_MODE_HYBRID: "hybrid",
                            OPUS_MODE_CELT_ONLY: "celt",
                        }.get(raw_winner, "celt")
                        section_win_history.append(
                            raw_name in ("silk", "hybrid")
                        )
                        voice_ratio = (
                            sum(section_win_history) / len(section_win_history)
                            if section_win_history else 0.0
                        )
                        if not voice_section and voice_ratio >= voice_enter:
                            voice_section = True
                            vprint(
                                args,
                                f"Section switch -> VOICE at frame {i} "
                                f"({voice_ratio * 100:.1f}% SILK/Hybrid wins)",
                            )
                        elif voice_section and voice_ratio <= voice_exit:
                            voice_section = False
                            vprint(
                                args,
                                f"Section switch -> MUSIC at frame {i} "
                                f"({voice_ratio * 100:.1f}% SILK/Hybrid wins)",
                            )

                        if voice_section:
                            # In a voice-classified section, choose the best
                            # eligible SILK/Hybrid candidate by waveform score.
                            voice_scores = [
                                (score, name) for name, score in trial_scores.items()
                                if name in ("silk", "hybrid") and math.isfinite(score)
                            ]
                            if voice_scores:
                                voice_scores.sort(key=lambda item: item[0])
                                chosen_mode = {
                                    "silk": OPUS_MODE_SILK_ONLY,
                                    "hybrid": OPUS_MODE_HYBRID,
                                }[voice_scores[0][1]]
                            else:
                                chosen_mode = raw_winner
                        else:
                            # Music sections deliberately use CELT-only mode.
                            chosen_mode = OPUS_MODE_CELT_ONLY
                        if auto_mode_choices is not None:
                            auto_mode_choices.append(chosen_mode)

                    # Change the private force-mode control only when the selected
                    # mode actually changes. Reissuing this private control on every
                    # frame can disturb encoder state in some libopus builds and may
                    # create an audible dip at CELT/Hybrid boundaries.
                    if chosen_mode != last_forced_mode:
                        opus_ctl(opus, encoder, OPUS_SET_FORCE_MODE_REQUEST, chosen_mode)
                        last_forced_mode = chosen_mode
                    chosen_name = {
                        OPUS_MODE_SILK_ONLY: "silk",
                        OPUS_MODE_HYBRID: "hybrid",
                        OPUS_MODE_CELT_ONLY: "celt",
                    }.get(chosen_mode, "unknown")
                    auto_wins[chosen_name] += 1
                else:
                    # Calibration-only pass: don't run the candidate scorer or
                    # force a mode. Let libopus choose normally for rate tuning.
                    pass
            elif args.mode == "voice":
                forced_mode = (
                    OPUS_MODE_SILK_ONLY
                    if bandwidth == OPUS_BANDWIDTH_WIDEBAND
                    else OPUS_MODE_HYBRID
                )
                opus_ctl(opus, encoder, OPUS_SET_FORCE_MODE_REQUEST, forced_mode)

            # If Hybrid wins the final-iteration mode competition, raise this
            # frame's encoder target to 18 kbps. Opus bitrate control is
            # encoder-wide and VBR packet sizes can still vary, so this is a
            # per-frame target request, not a guarantee that the packet itself
            # will contain more than 16 kbps worth of bits.
            if (
                args.mode == "auto"
                and auto_mode_select
                and chosen_mode == OPUS_MODE_HYBRID
            ):
                requested = max(requested, HYBRID_MIN_FRAME_BITRATE_BPS)

            # Stereo floor is intentionally disabled in this settings variant.
            # STEREO_MIN_BITRATE_BPS remains available if the floor is re-enabled.
            requested = min(MAX_BITRATE_BPS, requested)

            opus_ctl(
                opus, encoder,
                OPUS_SET_BITRATE_REQUEST,
                requested,
            )

            if channels == 2:
                # Allow mono coding only when the detector identifies a
                # mono/mostly-mono frame. Phase-inverted material is kept
                # stereo to avoid destructive cancellation.
                opus_ctl(
                    opus, encoder,
                    OPUS_SET_FORCE_CHANNELS_REQUEST,
                    1 if frame_is_mono else 2,
                )

            pcm_ptr = frame.ctypes.data_as(
                ctypes.POINTER(ctypes.c_int16)
            )

            packet_length = opus.opus_encode(
                ctypes.c_void_p(encoder),
                pcm_ptr,
                frame_samples,
                output_buffer,
                MAX_PACKET_BYTES,
            )

            if packet_length < 0:
                raise RuntimeError(
                    f"opus_encode failed: {packet_length}"
                )

            packet = bytes(output_buffer[:packet_length])
            packets.append(packet)
            rates.append(packet_length * 8.0 * RATE / frame_samples)

            controller.record_packet(packet_length, frame_samples)

    finally:
        if trial_executor is not None:
            trial_executor.shutdown(wait=True)
        opus.opus_encoder_destroy(ctypes.c_void_p(encoder))
        for candidate in candidates:
            opus.opus_encoder_destroy(ctypes.c_void_p(candidate["encoder"]))
            opus.opus_decoder_destroy(ctypes.c_void_p(candidate["decoder"]))
        if args.mode == "auto":
            vprint(args, "Auto candidate wins: " + ", ".join(
                f"{name}={count}" for name, count in auto_wins.items()
            ))

    return packets, rates, sum(len(p) for p in packets), pre_skip


# ============================================================
# Ogg writer
# ============================================================

def make_ogg_crc_table():
    table = []

    for i in range(256):
        value = i << 24

        for _ in range(8):
            if value & 0x80000000:
                value = (
                    ((value << 1) ^ 0x04C11DB7) & 0xFFFFFFFF
                )
            else:
                value = (value << 1) & 0xFFFFFFFF

        table.append(value)

    return table


OGG_CRC_TABLE = make_ogg_crc_table()


def ogg_crc(data):
    crc = 0

    for byte in data:
        crc = (
            ((crc << 8) & 0xFFFFFFFF)
            ^ OGG_CRC_TABLE[((crc >> 24) ^ byte) & 0xFF]
        )

    return crc


def packet_lacing(packet):
    return [255] * (len(packet) // 255) + [len(packet) % 255]


class OggWriter:
    def __init__(self, fileobj):
        self.file = fileobj
        self.serial = int.from_bytes(os.urandom(4), "little")
        self.sequence = 0
        self.pending_packets = []
        self.pending_segments = 0
        self.pending_granule = 0
        self.pending_samples = 0

    def write_page(self, packets, granule, header_type=0):
        lacing = []
        bodies = []

        for packet in packets:
            lacing.extend(packet_lacing(packet))
            bodies.append(packet)

        if len(lacing) > 255:
            raise ValueError("Ogg page exceeds 255 segments")

        header = bytearray(struct.pack(
            "<4sBBqIIIB",
            b"OggS", 0, header_type, granule,
            self.serial, self.sequence, 0, len(lacing),
        ))

        header.extend(bytes(lacing))
        page = header + b"".join(bodies)

        struct.pack_into("<I", page, 22, ogg_crc(page))
        self.file.write(page)
        self.sequence += 1

    def write_header_packet(self, packet, granule=0, header_type=0):
        self.write_page([packet], granule, header_type)

    def add_audio_packet(self, packet, granule, duration_samples):
        segments = len(packet_lacing(packet))

        if self.pending_packets and (
            self.pending_segments + segments > 200
            or self.pending_samples >= RATE
        ):
            self.flush_audio()

        self.pending_packets.append(packet)
        self.pending_segments += segments
        self.pending_granule = granule
        self.pending_samples += duration_samples

    def flush_audio(self, eos=False):
        if not self.pending_packets:
            return

        self.write_page(
            self.pending_packets,
            self.pending_granule,
            4 if eos else 0,
        )

        self.pending_packets = []
        self.pending_segments = 0
        self.pending_granule = 0
        self.pending_samples = 0

    def finish(self):
        self.flush_audio(eos=True)


def make_opus_head(pre_skip, channels=2):
    return (
        b"OpusHead"
        + struct.pack("<BBHIhB", 1, channels, pre_skip, RATE, 0, 0)
    )


def make_opus_tags():
    vendor = b"OpusPlus"

    return (
        b"OpusTags"
        + struct.pack("<I", len(vendor))
        + vendor
        + struct.pack("<I", 0)
    )


def write_ogg(
    path, packets, pre_skip, total_samples,
    frame_samples, channels
):
    with open(path, "wb") as output_file:
        ogg = OggWriter(output_file)

        ogg.write_header_packet(
            make_opus_head(pre_skip, channels), 0, 2
        )
        ogg.write_header_packet(make_opus_tags(), 0)

        for i, packet in enumerate(packets):
            decoded_end = min(
                (i + 1) * frame_samples, total_samples
            )

            ogg.add_audio_packet(
                packet,
                pre_skip + decoded_end,
                min(
                    frame_samples,
                    max(0, total_samples - i * frame_samples),
                ),
            )

        ogg.finish()


# ============================================================
# Two-pass bitrate targeting
# ============================================================

def transform_pass1_pattern(
    pass1_rates, adjustment, offset_bps, multiplier
):
    result = []

    for rate in pass1_rates:
        value = (
            rate + offset_bps
            if adjustment == "add"
            else rate * multiplier
        )

        result.append(int(max(
            MIN_BITRATE_BPS,
            min(MAX_BITRATE_BPS, round(value)),
        )))

    return result


def measure_file(path, duration):
    size = os.path.getsize(path)
    rate = size * 8.0 / duration if duration > 0 else 0.0
    return size, rate


def run_two_pass(
    opus, args, frames, modes, analysis_frames,
    frame_samples, total_samples
):
    duration = total_samples / RATE

    average_multiplier = float(np.mean([
        mode_multiplier(mode, args) for mode in modes
    ]))

    target_file_bps = args.bitrate * 1000.0

    vprint(args, "\nPASS 1: OpusPlus custom-VBR analysis (libopus CBR)")

    # Pass 1 is bitrate calibration only. Candidate mode competition is
    # deliberately deferred until the final encode.
    auto_mode_choices = None
    # Keep libopus internal VBR disabled at every bitrate. OpusPlus performs
    # custom frame-by-frame VBR by changing the CBR target for each frame.
    pass1_cbr = True
    packets, pass1_rates, _, _ = encode_packets(
        opus, args, frames, modes, analysis_frames, frame_samples,
        cbr=pass1_cbr, auto_mode_choices=None, auto_mode_select=False,
    )

    del packets
    pass1_mean = float(np.mean(pass1_rates))

    vprint(
        args,
        f"Pass 1 mean packet rate: {pass1_mean / 1000:.3f} kbps",
    )
    vprint(
        args,
        f"Average mode multiplier: {average_multiplier:.3f}x",
    )
    vprint(
        args,
        f"Effective target:        {args.bitrate:.3f} kbps",
    )

    target_packet_bps = max(
        MIN_BITRATE_BPS,
        target_file_bps - OGG_OVERHEAD_ESTIMATE_BPS,
    )

    if args.pass_adjust == "add":
        offset_bps = target_packet_bps - pass1_mean
        multiplier = 1.0
    else:
        offset_bps = 0.0
        multiplier = (
            target_packet_bps / pass1_mean
            if pass1_mean > 0 else 1.0
        )

    best_path = args.output + ".opusplus-best.tmp"
    best_error = float("inf")
    best_info = None
    best_packets = None
    channels = output_channels(args)

    vprint(
        args,
        f"\nPASS 2: {args.pass_adjust} adjustment; "
        "calibrating custom-VBR targets with libopus CBR",
    )

    try:
        for iteration in range(args.pass_iterations):
            targets = transform_pass1_pattern(
                pass1_rates,
                args.pass_adjust,
                offset_bps,
                multiplier,
            )

            packets, _, packet_bytes, pre_skip = encode_packets(
                opus, args, frames, modes, analysis_frames,
                frame_samples, cbr=True, targets=targets,
                auto_mode_choices=None, auto_mode_select=False,
            )

            write_ogg(
                args.output, packets, pre_skip, total_samples,
                frame_samples, channels,
            )

            size, file_rate = measure_file(args.output, duration)
            packet_rate = packet_bytes * 8.0 / duration
            error_bps = target_file_bps - file_rate

            vprint(
                args,
                f"Iteration {iteration + 1}/{args.pass_iterations}: "
                f"file={file_rate / 1000:.3f} kbps, "
                f"audio={packet_rate / 1000:.3f} kbps, "
                f"error={error_bps / 1000:+.3f} kbps",
            )

            if abs(error_bps) < best_error:
                best_error = abs(error_bps)
                shutil.copyfile(args.output, best_path)

                best_info = (
                    size, file_rate, packet_rate, iteration + 1
                )
                best_packets = packets

            if abs(error_bps) <= args.pass_tolerance_kbps * 1000.0:
                break

            if args.pass_adjust == "add":
                offset_bps += error_bps
            elif file_rate > 0:
                multiplier *= target_file_bps / file_rate

            offset_bps = max(
                -MAX_BITRATE_BPS,
                min(MAX_BITRATE_BPS, offset_bps),
            )
            multiplier = max(0.05, min(20.0, multiplier))

        # Final encode: run per-frame waveform-based mode competition here and
        # nowhere in the calibration passes. The final output always comes
        # from this selected-mode encode, not a CELT/libopus calibration file.
        final_targets = transform_pass1_pattern(
            pass1_rates, args.pass_adjust, offset_bps, multiplier
        )
        final_packets, _, final_packet_bytes, final_pre_skip = encode_packets(
            opus, args, frames, modes, analysis_frames, frame_samples,
            cbr=True, targets=final_targets, auto_mode_choices=None,
            auto_mode_select=True,
        )
        write_ogg(
            args.output, final_packets, final_pre_skip, total_samples,
            frame_samples, channels,
        )
        final_size, final_file_rate = measure_file(args.output, duration)
        final_packet_rate = final_packet_bytes * 8.0 / duration
        best_info = (
            final_size, final_file_rate, final_packet_rate,
            args.pass_iterations + 1,
        )
        best_packets = final_packets
        best_error = abs(target_file_bps - final_file_rate)
        shutil.copyfile(args.output, best_path)
        vprint(
            args,
            f"Final mode-selection encode: file={final_file_rate / 1000:.3f} kbps, "
            f"audio={final_packet_rate / 1000:.3f} kbps, "
            f"error={(target_file_bps - final_file_rate) / 1000:+.3f} kbps",
        )

    finally:
        if best_info is not None and os.path.isfile(best_path):
            shutil.copyfile(best_path, args.output)

        try:
            os.remove(best_path)
        except OSError:
            pass

    if best_info is None:
        size, file_rate = measure_file(args.output, duration)
        packet_rate = 0.0
        best_iteration = 1
        best_packets = []
    else:
        size, file_rate, packet_rate, best_iteration = best_info

    return {
        "duration": duration,
        "file_size": size,
        "file_rate": file_rate,
        "packet_rate": packet_rate,
        "pass1_mean": pass1_mean,
        "best_iteration": best_iteration,
        "effective_target": args.bitrate,
        "average_multiplier": average_multiplier,
        "packets": best_packets,
    }


def run_single_pass(
    opus, args, frames, modes, analysis_frames,
    frame_samples, total_samples
):
    duration = total_samples / RATE
    channels = output_channels(args)

    packets, rates, packet_bytes, pre_skip = encode_packets(
        opus, args, frames, modes, analysis_frames, frame_samples, cbr=True
    )

    write_ogg(
        args.output, packets, pre_skip, total_samples,
        frame_samples, channels,
    )

    size, file_rate = measure_file(args.output, duration)

    average_multiplier = float(np.mean([
        mode_multiplier(mode, args) for mode in modes
    ]))

    return {
        "duration": duration,
        "file_size": size,
        "file_rate": file_rate,
        "packet_rate": packet_bytes * 8.0 / duration,
        "pass1_mean": float(np.mean(rates)),
        "best_iteration": 1,
        "effective_target": args.bitrate,
        "average_multiplier": average_multiplier,
        "packets": packets,
    }


# ============================================================
# Main workflow
# ============================================================

def encode(args):
    if not os.path.isfile(args.input):
        raise FileNotFoundError(f"Input not found: {args.input}")

    if not 0.5 <= args.bitrate <= 512:
        raise ValueError("Bitrate must be between 0.5 and 512 kbps (Opus minimum is 500 bps)")

    if not 24 <= args.phase_invert_floor_kbps <= 28:
        raise ValueError(
            "Phase-inverted bitrate floor must be between 24 and 28 kbps"
        )

    if not -1.0 <= args.phase_invert_correlation <= 0.0:
        raise ValueError(
            "Phase-inversion correlation threshold must be between -1 and 0"
        )

    requested_strategy = args.codec_strategy
    args.codec_strategy = resolve_codec_strategy(args)

    if args.codec_strategy == "celt" and args.pure_mono:
        # CELT supports mono, so this is allowed.
        pass

    if not 0 <= args.complexity <= 10:
        raise ValueError("Complexity must be between 0 and 10")

    if not 0 <= args.vbr_intensity <= 10:
        raise ValueError("VBR intensity must be between 0 and 10")

    if not 1 <= args.pass_iterations <= 10:
        raise ValueError("Pass iterations must be between 1 and 10")

    if args.pass_tolerance_kbps < 0:
        raise ValueError("Pass tolerance cannot be negative")

    if not 0.05 <= args.mono_multiplier <= 1.0:
        raise ValueError(
            "Mono multiplier must be between 0.05 and 1.0"
        )

    if not 0.05 <= args.swb_multiplier <= 1.0:
        raise ValueError(
            "SWB multiplier must be between 0.05 and 1.0"
        )

    if not 0 <= args.vbr_normalize_level <= 10:
        raise ValueError(
            "Normalization level must be an integer from 0 to 10"
        )

    if args.vbr_normalize_gate < 0:
        raise ValueError("Normalization gate cannot be negative")

    channels = output_channels(args)
    frame_samples = int(RATE * args.frame_ms / 1000.0)
    started = time.time()

    print(f"Input:          {args.input}")
    print(f"Output:         {args.output}")
    print("Encoder:        OpusPlus")
    print(f"Channels:       {'mono' if channels == 1 else 'stereo'}")
    if args.bitrate <= 16.0 and not args.pure_mono:
        print("WARNING: AUTO MONO ACTIVATED — target is <=16 kbps.")
        print("         Input will be downmixed and encoded as a true mono Opus stream (OpusHead channel count = 1).")
    elif args.pure_mono:
        print("WARNING: PURE MONO ENABLED — output will be a true mono Opus stream (OpusHead channel count = 1).")
    print(f"Target bitrate: {args.bitrate:g} kbps")
    print(f"Strategy:       {args.codec_strategy.upper()}")
    print(f"Pass mode:      {args.pass_mode}")

    vprint(
        args,
        "\nREQUESTED CODING MODE: " + (
            {"voice": "SILK/HYBRID (forced voice)", "auto": "THREE-CANDIDATE MODE TRIALS", "music": "CELT-ONLY (music)"}[args.mode]
        ),
    )
    vprint(args, "Bandwidth forcing:      standard libopus bandwidth controls")
    vprint(args, "Mode forcing:            uses private libopus force-mode request; build-dependent")
    vprint(args, "Hybrid bitrate band:    removed")
    vprint(args, "Active-frame bitrate floor: 1/3 target (requested; silence exempt)")
    vprint(args, "Channel policy:         true mono stream at <=16 kbps; stereo above 16 kbps unless --pure-mono")
    vprint(args, f"Frame duration:         {args.frame_ms:g} ms")
    vprint(args, f"Signal mode:            {args.mode.upper()}")
    vprint(args, f"VBR intensity:          {args.vbr_intensity}/10")
    vprint(args, f"Complexity:              {args.complexity}/10")
    vprint(args, f"Pure mono:               {'ON' if args.pure_mono else 'OFF'}")
    vprint(args, "Phase inversion:         detection enabled")
    vprint(
        args,
        f"Phase-inversion floor:   {args.phase_invert_floor_kbps:.1f} kbps",
    )
    vprint(
        args,
        f"Correlation threshold:   {args.phase_invert_correlation:.2f}",
    )
    vprint(
        args,
        f"VBR normalization:       {'ON' if args.vbr_normalize else 'OFF'}",
    )
    vprint(
        args,
        f"Normalization level:     {args.vbr_normalize_level}/10",
    )
    vprint(
        args,
        f"Normalization gate:      {args.vbr_normalize_gate} PCM units",
    )
    vprint(
        args,
        f"SWB bitrate multiplier:  {args.swb_multiplier:.3f}x",
    )
    vprint(
        args,
        f"Mono bitrate multiplier: {args.mono_multiplier:.3f}x",
    )
    vprint(
        args,
        f"Mono reduction guard:    {SWB_MONO_GUARD_KBPS:.1f} kbps",
    )
    vprint(args, "Low-rate VBR strength:   scaled proportionally below 32 kbps")
    if args.bandwidth_mode != "default":
        low_rate_bandwidth = f"override: {args.bandwidth_mode.upper()}"
    elif args.bitrate <= 6.0:
        low_rate_bandwidth = "default: CELT wideband (<= 6 kbps)"
    elif args.bitrate < 10.0:
        low_rate_bandwidth = (
            "default: detector-adaptive SWB/FB"
            if args.adaptive_swb else "default: SWB on every frame"
        )
    else:
        low_rate_bandwidth = "default: detector/strategy controlled"
    vprint(args, f"Bandwidth policy:        {low_rate_bandwidth}")
    if args.bandwidth_mode != "default" and args.bitrate <= 6.0:
        print(
            "WARNING: bandwidth override requested at <=6 kbps. "
            "libopus may still limit the effective bandwidth at this rate."
        )

    if requested_strategy == "auto":
        vprint(args, "Automatic rule:          CELT-only at all target bitrates")

    print("Decoding audio...")
    pcm = decode_input(args.input, pure_mono=(channels == 1))

    total_samples = len(pcm) // channels
    frames = make_frames(pcm, frame_samples, channels)
    analysis_frames = normalize_analysis_frames(frames, args, channels)
    modes = analyze_all_modes(frames, args)

    mono_count = sum(mode["mono"] for mode in modes)
    swb_count = sum(mode["swb"] for mode in modes)
    both_count = sum(mode["mono"] and mode["swb"] for mode in modes)
    phase_inverted_count = sum(
        mode.get("phase_inverted", False) for mode in modes
    )

    opus = configure_opus_functions(load_opus(args.opus_dll))

    version = (
        opus.opus_get_version_string().decode("ascii", errors="replace")
        if hasattr(opus, "opus_get_version_string")
        else "unknown"
    )

    vprint(args, f"libopus:                 {version}")
    vprint(args, f"Frames:                   {len(frames)}")
    vprint(args, f"Duration:                 {total_samples / RATE:.3f} sec")

    vprint(args, "\nPER-FRAME MODE ANALYSIS")
    vprint(args, f"Mono-detected frames:     {mono_count}/{len(frames)}")
    vprint(args, f"SWB detector frames:      {swb_count}/{len(frames)}")
    vprint(args, f"Mono + SWB frames:        {both_count}/{len(frames)}")
    vprint(
        args,
        f"Phase-inverted frames:    {phase_inverted_count}/{len(frames)}",
    )

    print("Encoding...")

    if args.pass_mode == "two-pass":
        result = run_two_pass(
            opus, args, frames, modes, analysis_frames,
            frame_samples, total_samples,
        )
    else:
        result = run_single_pass(
            opus, args, frames, modes, analysis_frames,
            frame_samples, total_samples,
        )

    print("\nOpusPlus encoding complete.")
    print(f"Duration:       {result['duration']:.3f} sec")
    print(f"Output size:    {result['file_size']:,} bytes")
    print(f"Audio bitrate:  {result['packet_rate'] / 1000:.3f} kbps")
    print(f"File bitrate:   {result['file_rate'] / 1000:.3f} kbps")
    print(f"Target bitrate: {result['effective_target']:.3f} kbps")
    print(
        f"Target error:   "
        f"{result['file_rate'] / 1000 - result['effective_target']:+.3f} kbps"
    )

    if args.verbose:
        print(
            f"Average multiplier: {result['average_multiplier']:.3f}x"
        )

        if args.pass_mode == "two-pass":
            print(f"Best iteration:     {result['best_iteration']}")

        print_opus_packet_report(result["packets"])
        print(
            f"Pass 1 mean packet rate: "
            f"{result['pass1_mean'] / 1000:.3f} kbps"
        )
        print(f"Elapsed time: {time.time() - started:.2f} sec")


# ============================================================
# Command-line interface
# ============================================================

def main():
    parser = argparse.ArgumentParser(
        description=(
            "OpusPlus adaptive encoder with CELT-only mode, "
            "low-rate VBR smoothing, adaptive channel coding, "
            "phase-inversion protection, two-pass targeting, "
            "and verbose packet analysis."
        )
    )

    parser.add_argument("input", help="Input audio file")
    parser.add_argument("output", help="Output .opus file")
    default_threads = max(1, (os.cpu_count() or 2) // 2)
    parser.add_argument(
        "--threads", type=int, default=default_threads,
        help=(
            f"Worker threads for parallel candidate evaluation "
            f"(default: half of logical CPUs = {default_threads}; use 1 to disable parallel scoring)"
        ),
    )

    parser.add_argument(
        "-b", "--bitrate", type=float, default=18,
        help="Target bitrate in kbps (0.5-512; Opus minimum is 500 bps)",
    )
    parser.add_argument(
        "--pure-mono", action="store_true",
        help="Force a genuine single-channel Opus stream at any bitrate",
    )

    parser.add_argument(
        "--pass-mode", choices=["live", "two-pass"],
        default="two-pass",
    )
    parser.add_argument(
        "--pass-adjust", choices=["add", "multiply"],
        default="multiply",
    )
    parser.add_argument("--pass-iterations", type=int, default=4)
    parser.add_argument(
        "--pass-tolerance-kbps", type=float, default=0.5
    )

    parser.add_argument(
        "--mono-multiplier", type=float, default=0.75
    )
    parser.add_argument(
        "--swb-multiplier", type=float, default=0.75,
        help="SWB bitrate multiplier; default 0.75x",
    )

    parser.add_argument("--mono-enter-db", type=float, default=-51.0)
    parser.add_argument("--mono-exit-db", type=float, default=-51.0)

    parser.add_argument(
        "--swb-enter-db", type=float, default=-54.0,
        help="Enter SWB detection below this high/low energy ratio",
    )
    parser.add_argument(
        "--swb-exit-db", type=float, default=-54.0,
        help="Remain in SWB while below this high/low energy ratio",
    )

    parser.add_argument(
        "--phase-invert-correlation",
        type=float,
        default=-0.8,
        help="Detect phase-inverted stereo at or below this correlation",
    )
    parser.add_argument(
        "--phase-invert-floor-kbps",
        type=float,
        default=26.0,
        help="Requested bitrate floor for phase-inverted stereo (24-28 kbps)",
    )

    parser.add_argument("--silence-threshold", type=float, default=8.0)

    parser.add_argument(
        "--vbr-normalize", action="store_true",
        help="Normalize frames for VBR analysis only",
    )
    parser.add_argument(
        "--vbr-normalize-level", type=int,
        choices=range(11), default=10,
        help="Normalization intensity from 0 to 10",
    )
    parser.add_argument(
        "--vbr-normalize-gate", type=float, default=8.0,
        help="Leave frames below this peak unchanged",
    )

    parser.add_argument(
        "--vbr-intensity", type=int,
        choices=range(11), default=10,
    )
    parser.add_argument(
        "--complexity", type=int,
        choices=range(11), default=10,
    )
    parser.add_argument(
        "--frame-ms", type=float,
        choices=[2.5, 5, 10, 20, 40, 60],
        default=20,
    )

    parser.add_argument(
        "--section-window-ms", type=float, default=400.0,
        help="Rolling window for auto voice/music section classification (default: 400 ms)",
    )
    parser.add_argument(
        "--voice-enter-percent", type=float, default=35.0,
        help="Enter voice sections when SILK/Hybrid win share reaches this percentage (default: 35)",
    )
    parser.add_argument(
        "--voice-exit-percent", type=float, default=35.0,
        help="Return to music sections when SILK/Hybrid win share falls to this percentage (default: 35)",
    )

    parser.add_argument(
        "--mode", choices=["auto", "music", "voice"],
        default="AUTO",
        help=(
            "Coding mode: music forces CELT-only; voice forces SILK/Hybrid; "
            "auto (default) trial-encodes candidates and uses a rolling "
            "SILK/Hybrid win-rate to classify voice-like sections versus music sections"
        ),
    )
    parser.add_argument(
        "--codec-strategy",
        choices=["auto", "swb", "celt"],
        default="auto",
        help=(
            "Music-mode CELT strategy: auto/celt force CELT-only; "
            "swb forces CELT-only with SWB bandwidth limit. Ignored in voice mode."
        ),
    )
    parser.add_argument(
        "--adaptive-swb", action="store_true",
        help=(
            "Below 16 kbps, use SWB only on frames flagged by the SWB "
            "detector and allow fullband on other frames; default is all SWB"
        ),
    )
    parser.add_argument(
        "--bandwidth-mode",
        choices=["default", "auto", "swb", "fullband", "wideband"],
        default="default",
        help=(
            "Bandwidth policy: default follows bitrate rules; auto chooses "
            "SWB or fullband per frame at ANY bitrate; swb/fullband/wideband "
            "request that bandwidth even at very low bitrates"
        ),
    )

    parser.add_argument("--opus-dll", default=None)
    parser.add_argument(
        "-v", "--verbose", action="store_true",
        help="Show detailed encoder diagnostics and packet analysis",
    )

    args = parser.parse_args()
    if args.threads < 1:
        parser.error("--threads must be at least 1")
    if args.section_window_ms <= 0:
        parser.error("--section-window-ms must be greater than zero")
    if not (0.0 <= args.voice_exit_percent < args.voice_enter_percent <= 100.0):
        parser.error("thresholds must satisfy 0 <= --voice-exit-percent < --voice-enter-percent <= 100")

    try:
        encode(args)
    except KeyboardInterrupt:
        print("\nCancelled.", file=sys.stderr)
        sys.exit(130)
    except Exception as exc:
        print(f"\nERROR: {exc}", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()
