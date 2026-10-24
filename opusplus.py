import argparse
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor
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

VBR_TRANSIENT_MULTIPLIER = 0.6

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
OPUS_BANDWIDTH_NARROWBAND = 1101
OPUS_BANDWIDTH_MEDIUMBAND = 1102
OPUS_BANDWIDTH_WIDEBAND = 1103
OPUS_BANDWIDTH_SUPERWIDEBAND = 1104
OPUS_BANDWIDTH_FULLBAND = 1105

MIN_BITRATE_BPS = 500
MAX_BITRATE_BPS = 512000
# Any frame requested as Hybrid is raised to at least this bitrate, but only
# when the overall encoder target is >= HYBRID_FLOOR_MIN_TARGET_KBPS.
#
# NOTE: libopus (measured on 1.4) silently turns a *CBR* Hybrid frame into
# SILK-wideband when its bitrate request is below ~15 kbps. That does not
# happen with libopus VBR enabled, so Hybrid frames are always encoded with
# the encoder's VBR flag ON (see set_frame_vbr()). With VBR on, Hybrid is kept
# at any requested bitrate; libopus itself then needs roughly 8.7 kbps (mono)
# / 9.5-10 kbps (stereo) minimum for a real Hybrid frame.
HYBRID_MIN_FRAME_BITRATE_BPS = 14000
# libopus Hybrid with VBR misbehaves when asked for very low rates: requests
# of about 1 kbps or less yield empty 1-byte packets, and SWB requests below
# ~5 kbps (stereo) can produce maximum-size ~1276-byte packets (~510 kbps).
# Hybrid requests are therefore never sent below this value. Real Hybrid
# output is ~8.7 kbps or more regardless, so nothing is lost by the clamp.
HYBRID_VBR_MIN_REQUEST_BPS = 6000
HYBRID_FLOOR_MIN_TARGET_KBPS = 16.0
# At or above this overall target, encode CELT-only: no Hybrid trials (auto
# mode) and no forced Hybrid (voice mode).
CELT_ONLY_MIN_TARGET_KBPS = 20.0
# At or BELOW this overall target, auto and voice modes encode CELT-only too.
# Not applied in --mode hybrid or --mode silk (explicit forced modes).
CELT_ONLY_MAX_TARGET_KBPS = 10.0
OGG_OVERHEAD_ESTIMATE_BPS = 250
# Exact-ish Ogg container accounting used by the real-time (live) controller:
# OpusHead page (27 + 1 + 19) + OpusTags page (27 + 1 + 24), one lacing byte
# per 255 bytes of every packet (plus one), and a 27-byte header per page
# (the writer starts a new page about every second).
OGG_HEADER_BYTES = 99
OGG_PAGE_HEADER_BYTES = 27
MAX_PACKET_BYTES = 1275 * 3
SWB_MONO_GUARD_KBPS = 12.0
STEREO_MIN_BITRATE_BPS = 17000
# libopus SILK silently drops to a lower bandwidth when the per-frame bitrate
# request is too low (measured on 1.4): wideband needs >= 8 kbps, mediumband
# >= 7 kbps; below ~7 kbps it falls to narrowband. In --mode silk every
# frame is requested at no less than the value for the chosen bandwidth.
SILK_HOLD_BANDWIDTH_MIN_BPS = {
    OPUS_BANDWIDTH_WIDEBAND: 8000,
    OPUS_BANDWIDTH_MEDIUMBAND: 7000,
    OPUS_BANDWIDTH_NARROWBAND: 0,
}
# Bandwidth names/configs for SILK-only packets (TOC config // 4).
SILK_BANDWIDTH_NAMES = {
    OPUS_BANDWIDTH_NARROWBAND: "Narrowband",
    OPUS_BANDWIDTH_MEDIUMBAND: "Mediumband",
    OPUS_BANDWIDTH_WIDEBAND: "Wideband",
}

def vprint(args, *values, **kwargs):
    if args.verbose:
        print(*values, **kwargs)


def auto_mono_enabled(args):
    """Use a genuine mono stream at <=16 kbps, unless already requested.

    --force-stereo disables the automatic <=16 kbps mono downmix and keeps a
    genuine stereo stream (and stereo coding on every frame) at any bitrate.
    """
    if args.force_stereo:
        return False
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
        # Hybrid must run with libopus VBR on or it falls back to SILK at
        # low rates; every other mode keeps the custom-VBR CBR behaviour.
        opus_ctl(
            opus, encoder, OPUS_SET_VBR_REQUEST,
            1 if (args.mode == "hybrid" or not cbr) else 0,
        )
        opus_ctl(
            opus, encoder, OPUS_SET_COMPLEXITY_REQUEST, args.complexity
        )
        # Voice signal hint for voice and hybrid modes; music otherwise.
        signal = (
            OPUS_SIGNAL_VOICE
            if args.mode in ("voice", "hybrid", "silk")
            else OPUS_SIGNAL_MUSIC
        )
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
        elif args.mode == "hybrid":
            # Force Hybrid from the very first frame so the encoder never
            # starts in another mode.
            opus_ctl(
                opus, encoder,
                OPUS_SET_FORCE_MODE_REQUEST,
                OPUS_MODE_HYBRID,
            )
        elif args.mode == "silk":
            # Explicitly forced SILK-only (never chosen automatically).
            opus_ctl(
                opus, encoder,
                OPUS_SET_FORCE_MODE_REQUEST,
                OPUS_MODE_SILK_ONLY,
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


def _ffmpeg_pcm(path, channels, seek=None, length=None):
    """Decode (a part of) the first audio stream to 48 kHz s16 PCM."""
    command = [find_ffmpeg(), "-hide_banner", "-loglevel", "error"]
    # -ss / -t before -i are input options: fast, sample-accurate seeking.
    if seek is not None:
        command += ["-ss", str(seek)]
    if length is not None:
        command += ["-t", str(length)]
    command += [
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
    return np.frombuffer(bytes(pcm), dtype=np.int16).copy()


def decode_input(path, pure_mono=False):
    # FFmpeg downmixes to one channel before analysis/encoding when mono is on.
    channels = 1 if pure_mono else 2
    pcm = _ffmpeg_pcm(path, channels)
    if not pcm.size:
        raise RuntimeError("FFmpeg decoded no audio samples.")
    return pcm


def probe_duration(path):
    """Audio duration in seconds via ffprobe, or None if unavailable."""
    ffprobe = shutil.which("ffprobe")
    if not ffprobe:
        return None
    for entry in ("stream=duration", "format=duration"):
        try:
            result = subprocess.run(
                [ffprobe, "-v", "error", "-select_streams", "a:0",
                 "-show_entries", entry, "-of", "default=nw=1:nk=1", path],
                capture_output=True, text=True, timeout=60,
            )
            value = float(result.stdout.strip().splitlines()[0])
            if value > 0:
                return value
        except Exception:
            continue
    return None


# Seams between parallel-decoded segments must match the overlap exactly
# (lossless sources) or to within this many LSB (resampled/lossy decoders).
DECODE_SEAM_TOLERANCE_LSB = 2
DECODE_SEAM_PAD_SECONDS = 1
DECODE_SEAM_PROBE_FRAMES = 8192
DECODE_SEAM_MAX_SHIFT = 16


def decode_input_parallel(path, pure_mono=False, threads=2,
                          segment_seconds=60):
    """Decode a long file as overlapping time segments in parallel.

    Segment boundaries are whole seconds (so the resampler grid is the same
    as in a single decode), and every segment is decoded with 1 s of extra
    audio on each side. The overlap lets each seam be verified: the
    neighbouring decodes must agree sample-for-sample (any 1-sample slip is
    found and corrected, anything else is rejected). Returns None when the
    parallel path can't be used or a seam does not verify; the caller then
    does a normal single decode.
    """
    channels = 1 if pure_mono else 2
    duration = probe_duration(path)
    segment = int(max(1, segment_seconds))
    if not duration or threads < 2 or duration < 2 * segment:
        return None

    count = int(math.ceil(duration / segment))
    pad = DECODE_SEAM_PAD_SECONDS
    jobs = []
    for k in range(count):
        start = 0 if k == 0 else k * segment - pad
        length = None if k == count - 1 else (k + 1) * segment + pad - start
        jobs.append((start, length))

    try:
        with ThreadPoolExecutor(max_workers=min(threads, count)) as pool:
            decoded = list(pool.map(
                lambda job: _ffmpeg_pcm(path, channels, job[0], job[1]),
                jobs,
            ))
    except Exception:
        return None
    if any(d.size == 0 for d in decoded):
        return None

    origins = [jobs[k][0] * RATE for k in range(count)]
    window = DECODE_SEAM_PROBE_FRAMES
    for k in range(1, count):
        previous, current = decoded[k - 1], decoded[k]
        # Search the overlap around this seam for the loudest probe window.
        seam = k * segment * RATE
        lo = max(origins[k] + DECODE_SEAM_MAX_SHIFT,
                 origins[k - 1]) + RATE // 10
        hi = min(
            origins[k - 1] + len(previous) // channels,
            origins[k] + len(current) // channels,
        ) - window - DECODE_SEAM_MAX_SHIFT - RATE // 10
        if hi <= lo:
            return None
        best_start, best_energy = None, -1.0
        position = lo
        while position <= hi:
            index = (position - origins[k - 1]) * channels
            energy = float(np.abs(
                previous[index:index + window * channels].astype(np.int32)
            ).sum())
            if energy > best_energy:
                best_start, best_energy = position, energy
            position += window
        if best_start is None:
            return None
        index = (best_start - origins[k - 1]) * channels
        reference = previous[index:index + window * channels].astype(np.int32)

        best_shift, best_diff = None, None
        for shift in range(-DECODE_SEAM_MAX_SHIFT, DECODE_SEAM_MAX_SHIFT + 1):
            index = (best_start - (origins[k] + shift)) * channels
            if index < 0 or index + window * channels > len(current):
                continue
            candidate = current[index:index + window * channels].astype(np.int32)
            diff = int(np.max(np.abs(candidate - reference)))
            if best_diff is None or diff < best_diff:
                best_shift, best_diff = shift, diff
        if best_diff is None or best_diff > DECODE_SEAM_TOLERANCE_LSB:
            return None
        # A near-silent probe cannot tell shifts apart; prefer no shift.
        if best_energy < window * channels * 4 and best_diff == 0:
            best_shift = 0
        origins[k] += best_shift
        del seam

    pieces = []
    for k in range(count):
        first = 0 if k == 0 else k * segment * RATE
        if k == count - 1:
            last = origins[k] + len(decoded[k]) // channels
        else:
            last = (k + 1) * segment * RATE
        begin = (first - origins[k]) * channels
        end = (last - origins[k]) * channels
        if begin < 0 or end > len(decoded[k]) or end < begin:
            return None
        pieces.append(decoded[k][begin:end])
    return np.concatenate(pieces)


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
        # Analysis frames are read-only, so no copy is needed.
        return [frame for frame, _ in frames]

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

    if mode["mono"] and not args.pure_mono and not args.force_stereo:
        bitrate_after_swb = args.bitrate * multiplier

        if bitrate_after_swb >= SWB_MONO_GUARD_KBPS:
            multiplier *= args.mono_multiplier

    return multiplier


def apply_hybrid_floor(requested, args):
    """Raise a Hybrid frame's requested bitrate to HYBRID_MIN_FRAME_BITRATE_BPS
    (14 kbps) when the overall encoder target is 16 kbps or higher.

    Below a 16 kbps target the requested bitrate is returned unchanged;
    Hybrid is still used there (libopus VBR keeps it from dropping to SILK).
    """
    if args.bitrate >= HYBRID_FLOOR_MIN_TARGET_KBPS:
        return max(requested, HYBRID_MIN_FRAME_BITRATE_BPS)
    return requested


def set_frame_vbr(opus, encoder, state, want_vbr):
    """Switch libopus' VBR flag only when it changes.

    Hybrid frames need VBR on (CBR Hybrid below ~15 kbps becomes SILK);
    CELT/SILK frames keep the CBR + custom per-frame target behaviour.
    """
    want = 1 if want_vbr else 0
    if state.get("vbr") != want:
        opus_ctl(opus, encoder, OPUS_SET_VBR_REQUEST, want)
        state["vbr"] = want


def celt_only_target(args):
    """True when auto/voice modes use only CELT for the whole encode.

    That is the case at or above CELT_ONLY_MIN_TARGET_KBPS (20) and at or
    below CELT_ONLY_MAX_TARGET_KBPS (10). Never true in the explicit forced
    modes --mode hybrid / --mode silk.
    """
    if args.mode in ("hybrid", "silk"):
        return False
    return (
        args.bitrate >= CELT_ONLY_MIN_TARGET_KBPS
        or args.bitrate <= CELT_ONLY_MAX_TARGET_KBPS
    )


# ============================================================
# Activity and transient analysis
# ============================================================

def _transient_score_for_signal(x):
    """Score fast attacks in one frequency-isolated signal."""
    if len(x) < 3:
        return 0.0

    delta = np.diff(x, axis=0)
    mean_delta = float(np.mean(np.abs(delta)) + 1.0)

    # Curvature catches sharp waveform edges and percussion attacks.
    second_delta = np.diff(x, n=2, axis=0)
    curvature = float(np.mean(np.abs(second_delta)) + 1.0)
    curvature_score = max(
        0.0, min(1.0, (curvature / max(mean_delta, 1.0) - 1.0) / 4.0)
    )

    # Short sub-frame windows preserve brief attacks that a whole-frame
    # average could otherwise hide.
    window_size = max(1, len(x) // 8)
    window_rms = []
    for start in range(0, len(x), window_size):
        chunk = x[start:start + window_size]
        if len(chunk):
            window_rms.append(float(np.sqrt(np.mean(chunk * chunk) + 1.0)))

    onset_score = 0.0
    peak_ratio_score = 0.0
    if len(window_rms) >= 2:
        levels = np.asarray(window_rms, dtype=np.float64)
        rises = levels[1:] / np.maximum(levels[:-1], 24.0)
        onset_score = float(np.max(np.clip((rises - 1.15) / 2.0, 0.0, 1.0)))
        baseline = float(np.median(levels) + 24.0)
        peak_ratio = float(np.max(levels) / baseline)
        peak_ratio_score = max(0.0, min(1.0, (peak_ratio - 1.5) / 3.0))

    half = max(1, len(x) // 2)
    rms_a = float(np.sqrt(np.mean(x[:half] ** 2) + 1.0))
    rms_b = float(np.sqrt(np.mean(x[half:] ** 2) + 1.0))
    level_jump = abs(math.log((rms_b + 16.0) / (rms_a + 16.0)))
    jump_score = max(0.0, min(1.0, level_jump / 1.2))

    score = (
        0.25 * curvature_score
        + 0.20 * jump_score
        + 0.25 * onset_score
        + 0.15 * peak_ratio_score
        + 0.15 * max(onset_score, peak_ratio_score)
    )
    return max(0.0, min(1.0, score))


def analyze_frame(frame, channels):
    """Measure general activity and frequency-weighted transient energy.

    Transient weighting: bass (<250 Hz) = 0.0, mids (250-4000 Hz) = 0.25,
    treble (>4000 Hz) = 1.0. General activity remains broadband and unchanged.
    """
    x = np.asarray(frame, dtype=np.float32).reshape(-1, channels)

    if len(x) < 3:
        return 0.45, 0.0

    rms = float(np.sqrt(np.mean(x * x) + 1.0))
    delta = np.diff(x, axis=0)
    mean_delta = float(np.mean(np.abs(delta)) + 1.0)
    relative_delta = mean_delta / max(rms, 16.0)
    level_activity = math.log10(1.0 + rms / 500.0)
    detail_activity = math.log10(1.0 + relative_delta * 20.0)
    activity = max(0.0, min(2.5, 0.20 * level_activity + 0.80 * detail_activity))

    # Split the frame into frequency bands using an FFT. This affects only the
    # transient detector; the regular broadband activity score is untouched.
    spectrum = np.fft.rfft(x, axis=0)
    frequencies = np.fft.rfftfreq(len(x), d=1.0 / RATE)
    bass_mask = frequencies < 250.0
    mid_mask = (frequencies >= 250.0) & (frequencies <= 4000.0)
    treble_mask = frequencies > 4000.0

    bass_spec = spectrum * bass_mask[:, None]
    mid_spec = spectrum * mid_mask[:, None]
    treble_spec = spectrum * treble_mask[:, None]

    bass = np.fft.irfft(bass_spec, n=len(x), axis=0).astype(np.float32, copy=False)
    mids = np.fft.irfft(mid_spec, n=len(x), axis=0).astype(np.float32, copy=False)
    treble = np.fft.irfft(treble_spec, n=len(x), axis=0).astype(np.float32, copy=False)

    bass_transient = _transient_score_for_signal(bass)
    mid_transient = _transient_score_for_signal(mids)
    treble_transient = _transient_score_for_signal(treble)

    # Bass has no direct transient contribution; mids contribute lightly,
    # while treble attacks receive full weight. No normalization is applied,
    # so mids alone cannot outweigh a similarly strong treble transient.
    transient = (0.0 * bass_transient) + (0.15 * mid_transient) + (1.15 * treble_transient) * VBR+TRANSIENT_MULTIPLIER
    return activity, max(0.0, min(1.0, transient))



# ============================================================
# Batched, parallel frame analysis for the custom VBR
# ============================================================
#
# analyze_frame() above is the reference per-frame implementation. The
# analysis depends only on the input audio (never on encoder state), so it is
# computed ONCE for all frames, vectorized across frames, and spread over
# worker processes. The encoder passes then just look the values up.

def _transient_scores_batch(x):
    """Vectorized _transient_score_for_signal for x of shape (F, n, ch)."""
    frames_count, n, _ = x.shape
    if n < 3:
        return np.zeros(frames_count)

    mean_delta = np.abs(np.diff(x, axis=1)).mean(
        axis=(1, 2), dtype=np.float64) + 1.0
    curvature = np.abs(np.diff(x, n=2, axis=1)).mean(
        axis=(1, 2), dtype=np.float64) + 1.0
    curvature_score = np.clip(
        (curvature / np.maximum(mean_delta, 1.0) - 1.0) / 4.0, 0.0, 1.0
    )

    window_size = max(1, n // 8)
    levels = np.stack([
        np.sqrt(
            (x[:, start:start + window_size].astype(np.float64) ** 2)
            .mean(axis=(1, 2)) + 1.0
        )
        for start in range(0, n, window_size)
    ], axis=1)

    onset_score = np.zeros(frames_count)
    peak_ratio_score = np.zeros(frames_count)
    if levels.shape[1] >= 2:
        rises = levels[:, 1:] / np.maximum(levels[:, :-1], 24.0)
        onset_score = np.clip((rises - 1.15) / 2.0, 0.0, 1.0).max(axis=1)
        baseline = np.median(levels, axis=1) + 24.0
        peak_ratio = levels.max(axis=1) / baseline
        peak_ratio_score = np.clip((peak_ratio - 1.5) / 3.0, 0.0, 1.0)

    half = max(1, n // 2)
    rms_a = np.sqrt(
        (x[:, :half].astype(np.float64) ** 2).mean(axis=(1, 2)) + 1.0)
    rms_b = np.sqrt(
        (x[:, half:].astype(np.float64) ** 2).mean(axis=(1, 2)) + 1.0)
    level_jump = np.abs(np.log((rms_b + 16.0) / (rms_a + 16.0)))
    jump_score = np.clip(level_jump / 1.2, 0.0, 1.0)

    score = (
        0.25 * curvature_score
        + 0.20 * jump_score
        + 0.25 * onset_score
        + 0.15 * peak_ratio_score
        + 0.15 * np.maximum(onset_score, peak_ratio_score)
    )
    return np.clip(score, 0.0, 1.0)


def analyze_frames_batch(frames_2d, channels):
    """Vectorized analyze_frame for int16 frames of shape (F, n * channels).

    Returns (activity, transient) float arrays of length F.
    """
    frames_count = frames_2d.shape[0]
    x = frames_2d.astype(np.float32).reshape(frames_count, -1, channels)
    n = x.shape[1]

    if n < 3:
        return np.full(frames_count, 0.45), np.zeros(frames_count)

    rms = np.sqrt(
        (x.astype(np.float64) ** 2).mean(axis=(1, 2)) + 1.0)
    mean_delta = np.abs(np.diff(x, axis=1)).mean(
        axis=(1, 2), dtype=np.float64) + 1.0
    relative_delta = mean_delta / np.maximum(rms, 16.0)
    level_activity = np.log10(1.0 + rms / 500.0)
    detail_activity = np.log10(1.0 + relative_delta * 20.0)
    activity = np.clip(
        0.20 * level_activity + 0.80 * detail_activity, 0.0, 2.5)

    spectrum = np.fft.rfft(x, axis=1)
    frequencies = np.fft.rfftfreq(n, d=1.0 / RATE)
    mid_mask = (frequencies >= 250.0) & (frequencies <= 4000.0)
    treble_mask = frequencies > 4000.0

    mids = np.fft.irfft(
        spectrum * mid_mask[None, :, None], n=n, axis=1
    ).astype(np.float32, copy=False)
    treble = np.fft.irfft(
        spectrum * treble_mask[None, :, None], n=n, axis=1
    ).astype(np.float32, copy=False)

    # Bass has weight 0.0, so it is not computed at all.
    transient = (
        0.25 * _transient_scores_batch(mids)
        + 1.0 * _transient_scores_batch(treble)
    )
    return activity, np.clip(transient, 0.0, 1.0)


def _metrics_block_worker(payload):
    block, channels = payload
    return analyze_frames_batch(block, channels)


def compute_frame_metrics(frames, channels, threads=1, block_frames=128):
    """(activity, transient) per frame, batched and optionally multi-process.

    frames: list of equal-length 1-D int16 arrays. Returns a list of tuples.
    """
    if not frames:
        return []

    blocks = [
        np.stack(frames[start:start + block_frames])
        for start in range(0, len(frames), block_frames)
    ]
    results = None
    if threads > 1 and len(blocks) >= 4:
        try:
            with ProcessPoolExecutor(
                max_workers=min(threads, len(blocks))
            ) as pool:
                results = list(pool.map(
                    _metrics_block_worker,
                    [(block, channels) for block in blocks],
                ))
        except Exception:
            results = None
    if results is None:
        results = [analyze_frames_batch(b, channels) for b in blocks]

    activity = np.concatenate([r[0] for r in results])
    transient = np.concatenate([r[1] for r in results])
    return list(zip(activity.tolist(), transient.tolist()))


# ============================================================
# Adaptive bitrate controller
# ============================================================

class BitrateController:
    """Per-frame bitrate request.

    realtime=False is the original ("uncalibrated") controller: a fixed
    base rate shaped by activity and transients, with a weak, clamped
    feedback term. It is used by two-pass pass 1 (whose shape is rescaled
    afterwards) and by --pass-mode uncalibrated; it has no way to pull the
    average back to the target on its own.

    realtime=True is a closed-loop controller for --pass-mode live. After
    every frame it knows exactly how many bytes the output file has used,
    and sets the next frame's rate so that the file lands on the target
    bitrate within `horizon_s` seconds:

        rate = (budget(t + horizon) - bytes_so_far) / horizon

    The activity/transient/mode "shape" is applied on top, normalized by its
    own running average so that it redistributes bits between frames
    without changing the average the feedback is holding.
    """

    def __init__(self, target_kbps, intensity, channels,
                 low_rate_scaling=True, realtime=False, horizon_s=1.0,
                 shape_tau_s=5.0, frame_ms=20.0):
        self.realtime = bool(realtime)
        self.horizon_s = max(0.05, float(horizon_s))
        self.shape_tau_s = max(0.05, float(shape_tau_s))
        self.frame_s = float(frame_ms) / 1000.0
        self.lacing_bytes = 0
        self.shape_mean = None
        self.low_rate_scaling = bool(low_rate_scaling)
        self.target_bps = float(target_kbps) * 1000.0
        self.target_kbps = float(target_kbps)
        self.intensity = int(intensity)
        self.channels = channels
        self.total_packet_bytes = 0
        self.total_samples = 0
        # Tracks the carried-over transient decay envelope
        self.transient_envelope = 0.0

    def _realtime_bitrate(self, analysis_frames, index, multiplier):
        low_rate_scale = (
            min(1.0, self.target_kbps / 32.0)
            if self.low_rate_scaling else 1.0
        )
        strength = (self.intensity / 2.857142857142857) * low_rate_scale

        activity_factor = 1.0
        transient = 0.0
        
        if self.intensity > 0:
            if isinstance(analysis_frames, list):
                activity, transient = analysis_frames[index]
                next_transient = analysis_frames[index + 1][1] if index + 1 < len(analysis_frames) else 0.0
            else:
                activity, transient = analysis_frames
                next_transient = 0.0
            
            # Lookahead: combine current transient with 1/2 of the next frame's transient
            target_transient = max(transient, next_transient * 0.5)
            # Carry-over: merge the lookahead target with 1/2 of the previous frame's decaying envelope
            self.transient_envelope = max(target_transient, self.transient_envelope * 0.5)
            
            activity_factor = 1.0 + strength * ((activity - 0.45) * 0.75)
            activity_factor = max(0.75, min(1.50, activity_factor))

        # Split components mathematically
        non_transient = multiplier * activity_factor
        transient_component = non_transient * (strength * 2.0 * self.transient_envelope)
        
        # Fixed lowering on non-transient components everywhere
        shape = (non_transient * 0.80) + transient_component

        if self.shape_mean is None:
            self.shape_mean = shape
        shape_norm = shape / max(self.shape_mean, 1e-9)
        blend = min(1.0, self.frame_s / self.shape_tau_s)
        self.shape_mean += blend * (shape - self.shape_mean)

        # Closed-loop level calculations
        elapsed = self.total_samples / RATE
        file_bytes = (
            OGG_HEADER_BYTES + self.total_packet_bytes + self.lacing_bytes
            + OGG_PAGE_HEADER_BYTES * (int(elapsed) + 1)
        )
        budget_bytes = self.target_bps * (elapsed + self.horizon_s) / 8.0
        rate_bps = (budget_bytes - file_bytes) * 8.0 / self.horizon_s
        overhead_bps = 8.0 * (1.0 / self.frame_s + OGG_PAGE_HEADER_BYTES)
        payload_bps = rate_bps - overhead_bps
        payload_bps = max(
            0.15 * self.target_bps, min(4.0 * self.target_bps, payload_bps)
        )

        requested = payload_bps * shape_norm
        return int(max(MIN_BITRATE_BPS, min(MAX_BITRATE_BPS, requested)))

    def bitrate_for_frame(self, analysis_frames, index, multiplier):
        if self.realtime:
            return self._realtime_bitrate(analysis_frames, index, multiplier)

        packet_target_bps = max(
            MIN_BITRATE_BPS,
            self.target_bps * multiplier - OGG_OVERHEAD_ESTIMATE_BPS,
        )

        if self.intensity == 0:
            self.transient_envelope = 0.0
            return int(packet_target_bps)

        if isinstance(analysis_frames, list):
            activity, transient = analysis_frames[index]
            next_transient = analysis_frames[index + 1][1] if index + 1 < len(analysis_frames) else 0.0
        else:
            activity, transient = analysis_frames
            next_transient = 0.0

        low_rate_scale = (
            min(1.0, self.target_kbps / 32.0)
            if self.low_rate_scaling else 1.0
        )
        strength = (self.intensity / 2.857142857142857 / 3) * low_rate_scale

        # Lookahead + Carry-over combined envelope logic
        target_transient = max(transient, next_transient * 0.5)
        self.transient_envelope = max(target_transient, self.transient_envelope * 0.5)

        activity_factor = 1.0 + strength * ((activity - 0.45) * 0.75)
        activity_factor = max(0.75, min(1.50, activity_factor))

        non_transient = packet_target_bps * activity_factor
        transient_component = non_transient * (strength * 2.0 * self.transient_envelope)

        if self.total_samples <= 0 or self.total_packet_bytes <= 0:
            feedback_factor = 1.0
        else:
            elapsed = self.total_samples / RATE
            target_bytes = self.target_bps * elapsed / 8.0
            ratio = target_bytes / max(self.total_packet_bytes, 1)
            feedback_strength = 0.35 * low_rate_scale
            feedback_factor = ratio ** feedback_strength
            feedback_factor = max(0.85, min(1.20, feedback_factor))

        requested = ((non_transient * 0.80) + transient_component) * feedback_factor
        return int(max(MIN_BITRATE_BPS, min(MAX_BITRATE_BPS, requested)))

    def record_packet(self, packet_bytes, frame_samples):
        self.lacing_bytes += int(packet_bytes) // 255 + 1
        self.total_packet_bytes += int(packet_bytes)
        self.total_samples += int(frame_samples)




# ============================================================
# Bandwidth selection
# ============================================================

def silk_bandwidth(args):
    """Bandwidth for --mode silk (SILK maxes out at wideband)."""
    if args.bandwidth_mode == "narrowband":
        return OPUS_BANDWIDTH_NARROWBAND
    if args.bandwidth_mode == "mediumband":
        return OPUS_BANDWIDTH_MEDIUMBAND
    return OPUS_BANDWIDTH_WIDEBAND


def select_bandwidth(args, mode):
    # SILK mode: fixed NB / MB / WB as requested (default wideband).
    if args.mode == "silk":
        return silk_bandwidth(args)

    # Hybrid mode: libopus silently converts Hybrid to SILK-only whenever the
    # bandwidth is wideband or lower, so only SWB or fullband may be requested.
    if args.mode == "hybrid":
        if args.bandwidth_mode == "fullband":
            return OPUS_BANDWIDTH_FULLBAND
        if args.bandwidth_mode == "swb" or args.bitrate < 10.0:
            return OPUS_BANDWIDTH_SUPERWIDEBAND
        return (
            OPUS_BANDWIDTH_SUPERWIDEBAND
            if mode["swb"] else OPUS_BANDWIDTH_FULLBAND
        )

    # Explicit override modes ignore the usual low-bitrate bandwidth policy.
    # These are encoder requests; libopus may still internally limit the
    # effective bandwidth when the requested bitrate is extremely low.
    # Narrowband applies to CELT frames; a Hybrid frame (auto/voice between
    # 10 and 20 kbps) still has to be SWB or fullband, see
    # hybrid_frame_bandwidth().
    if args.bandwidth_mode == "narrowband":
        return OPUS_BANDWIDTH_NARROWBAND
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


def hybrid_frame_bandwidth(bandwidth):
    """Hybrid needs SWB or fullband; WB or lower turns it into SILK."""
    return max(bandwidth, OPUS_BANDWIDTH_SUPERWIDEBAND)


def select_max_bandwidth(args):
    if args.mode == "silk":
        return silk_bandwidth(args)

    # Hybrid mode: the ceiling must never be wideband or lower.
    if args.mode == "hybrid":
        return (
            OPUS_BANDWIDTH_SUPERWIDEBAND
            if args.bandwidth_mode == "swb" else OPUS_BANDWIDTH_FULLBAND
        )

    # Explicit override modes also lift the script's normal low-rate ceiling.
    if args.bandwidth_mode == "narrowband":
        return OPUS_BANDWIDTH_NARROWBAND
    if args.bandwidth_mode in ("auto", "fullband"):
        return OPUS_BANDWIDTH_FULLBAND
    if args.bandwidth_mode == "swb":
        return OPUS_BANDWIDTH_SUPERWIDEBAND
    if args.bandwidth_mode == "wideband":
        return OPUS_BANDWIDTH_WIDEBAND

    if args.bitrate <= 5.5:
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


def candidate_distortion_reference(original, decoded, channels):
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


# ------------------------------------------------------------
# Vectorized scoring with a per-frame cached reference
# ------------------------------------------------------------
# The original (un-coded) frame is the same for every candidate, so its
# spectra are computed once per frame and shared. All windows of a spectrum
# are transformed in one batched FFT instead of a Python loop.

def _stft_mag(x, fft_size):
    """Hann-windowed magnitude spectra of all overlapping windows.

    Same window layout as _multiresolution_spectrum(); returns an array of
    shape (windows, bins, channels).
    """
    n, channels = x.shape
    if n < fft_size:
        padded = np.zeros((fft_size, channels), dtype=x.dtype)
        padded[:n] = x
        x, n = padded, fft_size
    if n == fft_size:
        starts = [0]
    else:
        hop = max(1, fft_size // 2)
        starts = list(range(0, n - fft_size + 1, hop))
        if starts[-1] != n - fft_size:
            starts.append(n - fft_size)
    index = np.asarray(starts)[:, None] + np.arange(fft_size)[None, :]
    window = np.hanning(fft_size).reshape(1, -1, 1)
    return np.abs(np.fft.rfft(x[index] * window, axis=1))


def prepare_distortion_reference(original, channels):
    """Everything about the original frame that scoring reuses per candidate."""
    n_frames = len(original) // channels
    if n_frames < 8:
        return None
    x = np.asarray(
        original[:n_frames * channels], dtype=np.float64
    ).reshape(n_frames, channels)
    x = x - np.mean(x, axis=0, keepdims=True)

    reference_rms = float(np.sqrt(np.mean(x * x)) + 1.0)
    dx = np.diff(x, axis=0) if n_frames > 1 else None
    slope_rms = (
        float(np.sqrt(np.mean(dx * dx)) + 1.0) if dx is not None else 1.0
    )

    sizes = sorted(set(min(n_frames, k) for k in (128, 256, 512, 1024)
                       if min(n_frames, k) >= 32))
    if not sizes:
        sizes = [n_frames]
    specs = []
    for size in sizes:
        sx = _stft_mag(x, size)
        lx = np.log1p(sx)
        scale = np.mean(np.abs(lx), axis=(1, 2)) + 1.0
        norm = np.sqrt(np.sum(sx * sx, axis=(1, 2))) + 1e-9
        specs.append((size, sx, lx, scale, norm))

    block = max(8, min(96, n_frames // 4))
    env = []
    for start in range(0, n_frames, block):
        xx = x[start:start + block]
        if len(xx) == 0:
            continue
        env.append((start, np.sqrt(np.mean(xx * xx, axis=0) + 1.0)))

    return {
        "channels": channels, "n_frames": n_frames, "x": x,
        "reference_rms": reference_rms, "dx": dx, "slope_rms": slope_rms,
        "specs": specs, "block": block, "env": env,
    }


def distortion_vs_reference(ref, decoded):
    """Same metric as candidate_distortion_reference, using a cached reference."""
    channels = ref["channels"]
    n_frames = ref["n_frames"]
    y = np.asarray(
        decoded[:n_frames * channels], dtype=np.float64
    ).reshape(n_frames, channels)
    y = y - np.mean(y, axis=0, keepdims=True)
    x = ref["x"]

    waveform = float(np.sqrt(np.mean((x - y) ** 2)) / ref["reference_rms"])
    if n_frames > 1:
        dy = np.diff(y, axis=0)
        transient = float(
            np.sqrt(np.mean((ref["dx"] - dy) ** 2)) / ref["slope_rms"]
        )
    else:
        transient = waveform

    spectral_errors = []
    convergence_errors = []
    for size, sx, lx, scale, norm in ref["specs"]:
        sy = _stft_mag(y, size)
        ly = np.log1p(sy)
        spectral_errors.append(np.mean(np.abs(lx - ly), axis=(1, 2)) / scale)
        convergence_errors.append(
            np.sqrt(np.sum((sx - sy) ** 2, axis=(1, 2))) / norm
        )
    spectral = float(np.mean(np.concatenate(spectral_errors)))
    convergence = float(np.mean(np.concatenate(convergence_errors)))

    block = ref["block"]
    env_errors = []
    for start, rms_x in ref["env"]:
        yy = y[start:start + block]
        rms_y = np.sqrt(np.mean(yy * yy, axis=0) + 1.0)
        env_errors.append(float(np.mean(np.abs(np.log(rms_x / rms_y)))))
    envelope = float(np.mean(env_errors)) if env_errors else 0.0

    return (
        0.35 * min(waveform, 3.0)
        + 0.20 * min(transient, 3.0)
        + 0.25 * min(spectral, 3.0)
        + 0.15 * min(convergence, 3.0)
        + 0.05 * min(envelope, 3.0)
    )


def candidate_distortion(original, decoded, channels):
    """Drop-in replacement for candidate_distortion_reference()."""
    n_values = min(len(original), len(decoded))
    n_frames = n_values // channels
    if n_frames < 8:
        return float("inf")
    n_values = n_frames * channels
    ref = prepare_distortion_reference(original[:n_values], channels)
    return distortion_vs_reference(ref, decoded[:n_values])


def score_decoded(ref, original, decoded, channels):
    """Score one candidate's decoded frame (cached reference when it fits)."""
    if ref is not None and len(decoded) == ref["n_frames"] * channels:
        return distortion_vs_reference(ref, decoded)
    return candidate_distortion(original[:len(decoded)], decoded, channels)


# ------------------------------------------------------------
# Candidate trials
# ------------------------------------------------------------

def candidate_eligible(name, bitrate, frame_is_mono):
    # SILK is never an automatic candidate (only via --mode silk).
    # Hybrid is eligible at every bitrate (it runs with VBR on so libopus
    # does not fall back to SILK). Above 16 kbps, retain the existing
    # mono/near-mono restriction. CELT remains the fallback.
    if bitrate > 16000 and name == "hybrid" and not frame_is_mono:
        return False
    return True


def run_candidate_frame(opus, candidate, frame, frame_samples, channels,
                        bitrate, bandwidth, max_bandwidth, frame_is_mono,
                        args):
    """Trial-encode and decode ONE frame with one candidate.

    Returns the decoded int16 samples, or None if this mode/bandwidth
    combination is unsupported by the libopus build. Only the candidate's
    own encoder/decoder state is touched, so different candidates can run
    in different threads (the ctypes calls release the GIL).
    """
    pcm_ptr = frame.ctypes.data_as(ctypes.POINTER(ctypes.c_int16))
    outbuf = (ctypes.c_ubyte * MAX_PACKET_BYTES)()
    decoded_buf = (ctypes.c_int16 * (frame_samples * channels))()
    enc = candidate["encoder"]
    dec = candidate["decoder"]
    try:
        # Trial Hybrid at the same floored rate it would actually be
        # encoded at, so the mode competition is fair.
        trial_bitrate = bitrate
        if candidate["name"] == "hybrid":
            trial_bitrate = max(
                apply_hybrid_floor(bitrate, args),
                HYBRID_VBR_MIN_REQUEST_BPS,
            )

        cbw = bandwidth
        cmax = max_bandwidth
        if candidate["name"] == "hybrid":
            cbw = hybrid_frame_bandwidth(
                min(cbw, OPUS_BANDWIDTH_SUPERWIDEBAND)
            )
            cmax = hybrid_frame_bandwidth(
                min(cmax, OPUS_BANDWIDTH_SUPERWIDEBAND)
            )
            opus_ctl(opus, enc, OPUS_SET_VBR_REQUEST, 1)
        opus_ctl(opus, enc, OPUS_SET_BANDWIDTH_REQUEST, cbw)
        opus_ctl(opus, enc, OPUS_SET_MAX_BANDWIDTH_REQUEST, cmax)
        opus_ctl(opus, enc, OPUS_SET_BITRATE_REQUEST, trial_bitrate)
        opus_ctl(opus, enc, OPUS_SET_FORCE_CHANNELS_REQUEST,
                 1 if (channels == 2 and frame_is_mono) else channels)
        length = opus.opus_encode(
            ctypes.c_void_p(enc), pcm_ptr, frame_samples, outbuf,
            MAX_PACKET_BYTES,
        )
        if length <= 0:
            return None
        packet = (ctypes.c_ubyte * length).from_buffer_copy(
            bytes(outbuf[:length]))
        decoded_count = opus.opus_decode(
            ctypes.c_void_p(dec), packet, length, decoded_buf,
            frame_samples, 0,
        )
        if decoded_count <= 0:
            return None
        return np.ctypeslib.as_array(decoded_buf)[
            :decoded_count * channels].copy()
    except Exception:
        # A mode/bandwidth combination may be unsupported by this build.
        return None


def pick_winner(scores):
    """Lowest-distortion candidate. scores: {name: score} in candidate order."""
    modes = {"hybrid": OPUS_MODE_HYBRID, "celt": OPUS_MODE_CELT_ONLY}
    valid = [
        (score, name) for name, score in scores.items() if score is not None
    ]
    if not valid:
        return OPUS_MODE_CELT_ONLY, {"celt": float("inf")}
    valid.sort(key=lambda item: item[0])
    return modes[valid[0][1]], {name: score for score, name in valid}


def choose_candidate_mode(opus, candidates, frame, frame_samples, channels,
                          bitrate, bandwidth, max_bandwidth, frame_is_mono,
                          original_samples, args, executor=None):
    """Per-frame trial used when per-frame bitrates are not known ahead of
    time (single-pass 'live' mode). Candidates run concurrently in threads."""
    eligible = [
        c for c in candidates
        if candidate_eligible(c["name"], bitrate, frame_is_mono)
    ]

    def run(candidate):
        return run_candidate_frame(
            opus, candidate, frame, frame_samples, channels, bitrate,
            bandwidth, max_bandwidth, frame_is_mono, args,
        )

    if executor is not None and len(eligible) > 1:
        decoded = list(executor.map(run, eligible))
    else:
        decoded = [run(c) for c in eligible]

    reference = (
        prepare_distortion_reference(original_samples, channels)
        if any(d is not None for d in decoded) else None
    )
    scores = {}
    for candidate, dec in zip(eligible, decoded):
        if dec is not None:
            scores[candidate["name"]] = score_decoded(
                reference, original_samples, dec, channels)
    return pick_winner(scores)


def _score_frame_task(payload):
    """Worker: score every candidate's decoded frame against the original."""
    original, decoded_by_name, channels = payload
    if not decoded_by_name:
        return {}
    reference = prepare_distortion_reference(original, channels)
    return {
        name: score_decoded(reference, original, dec, channels)
        for name, dec in decoded_by_name.items()
    }


def compute_trial_scores(opus, args, frames, plan, frame_samples, cbr,
                         threads, block_frames=256):
    """Mode-trial scores for EVERY frame, computed in parallel.

    The candidate encoders never depend on the output encoder or on the mode
    decisions: each one just encodes and decodes the frames it is eligible
    for, one after another. So each candidate's whole chain runs in its own
    thread (block by block, to bound memory), and the expensive distortion
    scoring of each block is spread over worker processes. The result is
    the same as the per-frame trial, only computed ahead of time.

    plan[i] = (requested_bitrate, bandwidth, max_bandwidth, frame_is_mono).
    Returns a list of {name: score} dicts, one per frame.
    """
    channels = output_channels(args)
    candidates = create_auto_candidates(opus, args, cbr=cbr)
    total = len(frames)
    scores = [None] * total

    chain_pool = (
        ThreadPoolExecutor(max_workers=min(threads, len(candidates)))
        if threads > 1 and len(candidates) > 1 else None
    )
    score_pool = None
    if threads > 1 and total >= 64:
        try:
            score_pool = ProcessPoolExecutor(max_workers=threads)
        except Exception:
            score_pool = None

    def run_chain(candidate, first, last):
        out = {}
        for i in range(first, last):
            requested, bandwidth, max_bandwidth, mono = plan[i]
            if not candidate_eligible(candidate["name"], requested, mono):
                continue
            decoded = run_candidate_frame(
                opus, candidate, frames[i][0], frame_samples, channels,
                requested, bandwidth, max_bandwidth, mono, args,
            )
            if decoded is not None:
                out[i] = decoded
        return out

    try:
        for first in range(0, total, block_frames):
            last = min(total, first + block_frames)
            if chain_pool is not None:
                outputs = list(chain_pool.map(
                    lambda c: run_chain(c, first, last), candidates))
            else:
                outputs = [run_chain(c, first, last) for c in candidates]

            tasks = []
            for i in range(first, last):
                decoded = {
                    c["name"]: outputs[k][i]
                    for k, c in enumerate(candidates) if i in outputs[k]
                }
                tasks.append((frames[i][0], decoded, channels))

            results = None
            if score_pool is not None:
                try:
                    results = list(score_pool.map(
                        _score_frame_task, tasks, chunksize=8))
                except Exception:
                    score_pool.shutdown(wait=False, cancel_futures=True)
                    score_pool = None
            if results is None:
                results = [_score_frame_task(t) for t in tasks]
            scores[first:last] = results
    finally:
        if chain_pool is not None:
            chain_pool.shutdown(wait=True)
        if score_pool is not None:
            score_pool.shutdown(wait=True)
        for candidate in candidates:
            opus.opus_encoder_destroy(ctypes.c_void_p(candidate["encoder"]))
            opus.opus_decoder_destroy(ctypes.c_void_p(candidate["decoder"]))
    return scores


def frame_is_silent(frame, threshold):
    """True silence test used for the active-frame bitrate floor."""
    frame_peak = float(np.max(np.abs(frame))) if frame.size else 0.0
    frame_rms = float(np.sqrt(np.mean(frame.astype(np.float64) ** 2))) if frame.size else 0.0
    return frame_peak < threshold or frame_rms < threshold


def mark_silent_frames(frames, modes, threshold, block_frames=256):
    """Evaluate the silence test once per frame (batched) instead of in
    every encoder pass and every trial plan."""
    for first in range(0, len(frames), block_frames):
        block = np.stack([f for f, _ in frames[first:first + block_frames]])
        peak = np.max(np.abs(block), axis=1).astype(np.float64)
        rms = np.sqrt(np.mean(block.astype(np.float64) ** 2, axis=1))
        silent = (peak < threshold) | (rms < threshold)
        level_db = 20.0 * np.log10(np.maximum(rms, 1.0) / 32768.0)
        for offset, flag in enumerate(silent.tolist()):
            modes[first + offset]["silent"] = bool(flag)
            modes[first + offset]["level_db"] = float(level_db[offset])


def frame_floor_fraction(args, mode):
    """Minimum per-frame request, as a fraction of the target bitrate.

    Normal (active) frames are held at --active-floor so the rate cannot
    sag during ordinary material. Only very quiet frames may drop lower:
    the floor slides smoothly from --active-floor (frame level at or above
    --active-db) down to --inactive-floor (at or below --inactive-db).
    SWB + mono frames keep their reduced floor (2/3 of the active floor,
    i.e. 1/3 of the target at the default 0.5).
    """
    active = args.active_floor
    if mode.get("swb") and mode.get("mono"):
        active *= 2.0 / 3.0
    inactive = min(args.inactive_floor, active)

    level_db = mode.get("level_db")
    if level_db is None:
        return active
    span = max(args.active_db - args.inactive_db, 1e-6)
    t = min(1.0, max(0.0, (level_db - args.inactive_db) / span))
    t = t * t * (3.0 - 2.0 * t)
    return inactive + (active - inactive) * t


def plan_frame_request(args, channels, frame, mode, requested):
    """Clamp and floor one frame's requested bitrate; decide mono coding.

    Shared by the encode loop and by the ahead-of-time trial planning so
    both see exactly the same per-frame bitrate and channel decision.
    Returns (requested, frame_is_mono).
    """
    phase_inverted = mode.get("phase_inverted", False)
    requested = int(max(
        MIN_BITRATE_BPS,
        min(MAX_BITRATE_BPS, requested),
    ))

    # Active-frame bitrate floor: avoid extreme one-frame target dips.
    # This floors the bitrate REQUEST sent to libopus, not the actual
    # packet size (libopus VBR may still emit a smaller packet).
    # True silence is exempt so quiet gaps can remain efficiently coded.
    # The silence decision is made once per frame by mark_silent_frames().
    is_silence_frame = mode.get("silent")
    if is_silence_frame is None:
        is_silence_frame = frame_is_silent(frame, args.silence_threshold)
        
    if not is_silence_frame:
        requested = max(
            requested,
            int(args.bitrate * 1000.0 * frame_floor_fraction(args, mode)),
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
        and not args.force_stereo
        and (args.bitrate <= 16.0 or mode["mono"])
    )
    return requested, frame_is_mono


def encode_packets(
    opus, args, frames, modes, analysis_frames,
    frame_samples, cbr=True, targets=None, auto_mode_choices=None,
    auto_mode_select=True
):
    channels = output_channels(args)

    encoder, pre_skip = create_encoder(opus, args, cbr=cbr)
    controller = BitrateController(
        args.bitrate, args.vbr_intensity, channels,
        low_rate_scaling=not args.no_low_rate_scaling,
        realtime=(args.pass_mode == "live"),
        horizon_s=args.live_horizon_seconds,
        shape_tau_s=args.live_shape_seconds,
        frame_ms=args.frame_ms,
    )

    packets = []
    rates = []
    output_buffer = (ctypes.c_ubyte * MAX_PACKET_BYTES)()
    celt_only = celt_only_target(args)
    # Mode trials. When the per-frame bitrates are known ahead of time
    # (two-pass) they are computed for all frames up front, in parallel, by
    # compute_trial_scores(). Only single-pass live mode, whose bitrates
    # depend on the output encoder's own feedback, still trials frame by
    # frame, with the candidates running concurrently.
    trials_needed = (
        args.mode == "auto" and auto_mode_select and not celt_only
        and not (auto_mode_choices is not None
                 and len(auto_mode_choices) >= len(frames))
    )
    precomputed_trials = None
    if trials_needed and targets is not None:
        trial_plan = []
        for i, (frame, _) in enumerate(frames):
            requested, mono = plan_frame_request(
                args, channels, frame, modes[i], targets[i])
            trial_plan.append((
                requested, select_bandwidth(args, modes[i]),
                select_max_bandwidth(args), mono,
            ))
        precomputed_trials = compute_trial_scores(
            opus, args, frames, trial_plan, frame_samples, cbr,
            args.threads,
        )
    candidates = (
        create_auto_candidates(opus, args, cbr=cbr)
        if trials_needed and targets is None else []
    )
    trial_executor = (
        ThreadPoolExecutor(max_workers=max(1, min(args.threads, len(candidates))))
        if len(candidates) > 1 and args.threads > 1
        else None
    )
    auto_wins = Counter()
    # Tracks libopus' VBR flag so it is only changed when needed.
    vbr_state = {"vbr": 1 if (args.mode == "hybrid" or not cbr) else 0}
    # In hybrid mode the encoder was already forced to Hybrid at creation.
    last_forced_mode = {
        "hybrid": OPUS_MODE_HYBRID, "silk": OPUS_MODE_SILK_ONLY,
    }.get(args.mode)
    # Count packets that came out as something other than Hybrid (hybrid mode).
    non_hybrid = 0
    non_hybrid_frames = []
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

            # Change the call signature to pass 'analysis_frames' and the index 'i'
            if targets is None:
                requested = controller.bitrate_for_frame(
                    analysis_frames,
                    i,
                    mode_multiplier(mode, args),
                )
            else:
                requested = targets[i]

            requested, frame_is_mono = plan_frame_request(
                args, channels, frame, mode, requested)

            if args.mode == "auto":
                if auto_mode_select:
                    # Candidate competition is enabled only for the final
                    # encode. Calibration passes leave Opus in automatic mode.
                    if celt_only:
                        # Target >= 20 or <= 10 kbps: CELT-only, no mode trials.
                        chosen_mode = OPUS_MODE_CELT_ONLY
                        trial_scores = {}
                    elif auto_mode_choices is not None and i < len(auto_mode_choices):
                        chosen_mode = auto_mode_choices[i]
                        trial_scores = {}
                    else:
                        if precomputed_trials is not None:
                            raw_winner, trial_scores = pick_winner(
                                precomputed_trials[i])
                        else:
                            raw_winner, trial_scores = choose_candidate_mode(
                                opus, candidates, frame, frame_samples,
                                channels, requested, bandwidth,
                                select_max_bandwidth(args),
                                frame_is_mono, frame, args, trial_executor,
                            )
                        raw_name = {
                            OPUS_MODE_HYBRID: "hybrid",
                            OPUS_MODE_CELT_ONLY: "celt",
                        }.get(raw_winner, "celt")
                        section_win_history.append(raw_name == "hybrid")
                        voice_ratio = (
                            sum(section_win_history) / len(section_win_history)
                            if section_win_history else 0.0
                        )
                        if not voice_section and voice_ratio >= voice_enter:
                            voice_section = True
                            vprint(
                                args,
                                f"Section switch -> VOICE at frame {i} "
                                f"({voice_ratio * 100:.1f}% Hybrid wins)",
                            )
                        elif voice_section and voice_ratio <= voice_exit:
                            voice_section = False
                            vprint(
                                args,
                                f"Section switch -> MUSIC at frame {i} "
                                f"({voice_ratio * 100:.1f}% Hybrid wins)",
                            )

                        if voice_section:
                            # In a voice-classified section, choose the best
                            # eligible Hybrid candidate by waveform score.
                            voice_scores = [
                                (score, name) for name, score in trial_scores.items()
                                if name == "hybrid" and math.isfinite(score)
                            ]
                            if voice_scores:
                                voice_scores.sort(key=lambda item: item[0])
                                chosen_mode = OPUS_MODE_HYBRID
                            else:
                                chosen_mode = raw_winner
                        else:
                            # Music sections deliberately use CELT-only mode.
                            chosen_mode = OPUS_MODE_CELT_ONLY
                        if auto_mode_choices is not None:
                            auto_mode_choices.append(chosen_mode)

                    # Safety guard: SILK is never selected automatically.
                    if chosen_mode == OPUS_MODE_SILK_ONLY:
                        chosen_mode = OPUS_MODE_CELT_ONLY

                    # Change the private force-mode control only when the selected
                    # mode actually changes. Reissuing this private control on every
                    # frame can disturb encoder state in some libopus builds and may
                    # create an audible dip at CELT/Hybrid boundaries.
                    if chosen_mode != last_forced_mode:
                        opus_ctl(opus, encoder, OPUS_SET_FORCE_MODE_REQUEST, chosen_mode)
                        last_forced_mode = chosen_mode
                    chosen_name = {
                        OPUS_MODE_HYBRID: "hybrid",
                        OPUS_MODE_CELT_ONLY: "celt",
                    }.get(chosen_mode, "unknown")
                    auto_wins[chosen_name] += 1
                else:
                    # Calibration-only pass: don't run the candidate scorer or
                    # force a mode. Let libopus choose normally for rate tuning.
                    pass
            elif args.mode == "voice":
                # No SILK. CELT-only at targets <=10 or >=20 kbps; in between,
                # CELT for wideband or Hybrid for SWB/fullband.
                if celt_only:
                    forced_mode = OPUS_MODE_CELT_ONLY
                elif bandwidth == OPUS_BANDWIDTH_WIDEBAND:
                    forced_mode = OPUS_MODE_CELT_ONLY
                else:
                    forced_mode = OPUS_MODE_HYBRID
                    # Forced-Hybrid frames get the 14 kbps floor when the
                    # overall target is >= 16 kbps.
                    requested = apply_hybrid_floor(requested, args)
                opus_ctl(opus, encoder, OPUS_SET_FORCE_MODE_REQUEST, forced_mode)
            elif args.mode == "silk":
                # Explicit SILK-only at the NB/MB/WB bandwidth requested.
                # Keep the request high enough that libopus does not lower
                # the SILK bandwidth by itself.
                requested = max(
                    requested,
                    SILK_HOLD_BANDWIDTH_MIN_BPS[silk_bandwidth(args)],
                )
                if last_forced_mode != OPUS_MODE_SILK_ONLY:
                    opus_ctl(
                        opus, encoder,
                        OPUS_SET_FORCE_MODE_REQUEST, OPUS_MODE_SILK_ONLY,
                    )
                    last_forced_mode = OPUS_MODE_SILK_ONLY
            elif args.mode == "hybrid":
                # Always Hybrid: no SILK, no CELT, at every bitrate. The
                # bandwidth selected above is guaranteed SWB or fullband and
                # libopus VBR is on, so low rates cannot drop to SILK.
                requested = apply_hybrid_floor(requested, args)
                if last_forced_mode != OPUS_MODE_HYBRID:
                    opus_ctl(
                        opus, encoder,
                        OPUS_SET_FORCE_MODE_REQUEST, OPUS_MODE_HYBRID,
                    )
                    last_forced_mode = OPUS_MODE_HYBRID

            # If Hybrid wins the final-iteration mode competition, raise the
            # frame's encoder target to the Hybrid floor (14 kbps) when the
            # overall target is >= 16 kbps. Opus bitrate control is
            # encoder-wide and VBR packet sizes can still vary, so this is
            # a per-frame target request, not a packet-size guarantee.
            if (
                args.mode == "auto"
                and auto_mode_select
                and chosen_mode == OPUS_MODE_HYBRID
            ):
                requested = apply_hybrid_floor(requested, args)

            # libopus codes the first frame after a SILK/Hybrid -> CELT switch
            # in the PREVIOUS mode (it carries a redundant CELT transition
            # frame). Keep that transition frame Hybrid (VBR on) rather than
            # letting it come out as SILK.
            frame_is_hybrid = (
                args.mode == "hybrid"
                or (args.mode == "voice" and forced_mode == OPUS_MODE_HYBRID)
                or (args.mode == "auto" and auto_mode_select
                    and chosen_mode == OPUS_MODE_HYBRID)
            )
            if (
                packets
                and (args.mode == "voice"
                     or (args.mode == "auto" and auto_mode_select))
            ):
                current_mode = (
                    forced_mode if args.mode == "voice" else chosen_mode
                )
                if current_mode == OPUS_MODE_CELT_ONLY:
                    previous_kind = classify_opus_packet(packets[-1])["mode"]
                    if previous_kind == "Hybrid":
                        # Transition frame is coded as Hybrid: VBR must be on.
                        frame_is_hybrid = True

            if frame_is_hybrid:
                requested = max(requested, HYBRID_VBR_MIN_REQUEST_BPS)
                # Hybrid needs SWB/fullband and libopus VBR on, otherwise
                # libopus silently converts it to SILK-wideband.
                hybrid_bw = hybrid_frame_bandwidth(bandwidth)
                if hybrid_bw != bandwidth:
                    opus_ctl(
                        opus, encoder, OPUS_SET_BANDWIDTH_REQUEST, hybrid_bw
                    )
                    opus_ctl(
                        opus, encoder, OPUS_SET_MAX_BANDWIDTH_REQUEST,
                        max(select_max_bandwidth(args), hybrid_bw),
                    )
            if args.mode != "hybrid":
                set_frame_vbr(opus, encoder, vbr_state,
                              frame_is_hybrid or not cbr)

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
            rates.append(packet_length * 8.0 / (frame_samples / RATE))

            # Verify that libopus really produced the forced mode (and, for
            # SILK, the requested NB/MB/WB bandwidth).
            if args.mode in ("hybrid", "silk"):
                info = classify_opus_packet(packet)
                wanted = "Hybrid" if args.mode == "hybrid" else "SILK"
                bad = info["mode"] != wanted
                if args.mode == "silk" and not bad:
                    bad = (
                        info["bandwidth"]
                        != SILK_BANDWIDTH_NAMES[silk_bandwidth(args)]
                    )
                if bad:
                    non_hybrid += 1
                    if len(non_hybrid_frames) < 10:
                        non_hybrid_frames.append(i)

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

    if args.mode in ("hybrid", "silk") and auto_mode_select:
        label = "Hybrid" if args.mode == "hybrid" else "SILK"
        if non_hybrid:
            print(
                f"WARNING: {non_hybrid}/{len(packets)} packets were NOT "
                f"{label} (first frames: {non_hybrid_frames}). This libopus "
                "build may not honor the private force-mode control (11002), "
                "or the frame size/bitrate forced a different mode."
            )
        else:
            vprint(args, f"{label} check: all {len(packets)} packets are {label}.")

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


# Calibration passes only need to MEASURE the bitrate of a given set of
# per-frame targets, so they can be split into time chunks that encode
# concurrently in worker processes. Each chunk first encodes a short
# pre-roll (the frames just before it) to warm up the codec state, and
# those pre-roll packets are discarded. The result is an estimate that is
# within a few hundredths of a kbps of the sequential pass; the final
# encode that is actually kept is always one sequential, exact pass.
CHUNK_PREROLL_MS = 500.0
MIN_CHUNK_FRAMES = 300

_WORKER_OPUS = None


def _encode_chunk_worker(payload):
    global _WORKER_OPUS
    (args, frames, modes, analysis, targets,
     frame_samples, cbr, preroll) = payload
    if _WORKER_OPUS is None:
        _WORKER_OPUS = configure_opus_functions(load_opus(args.opus_dll))
    packets, rates, _, pre_skip = encode_packets(
        _WORKER_OPUS, args, frames, modes, analysis, frame_samples,
        cbr=cbr, targets=targets, auto_mode_choices=None,
        auto_mode_select=False,
    )
    return packets[preroll:], rates[preroll:], pre_skip


def encode_packets_chunked(pool, args, frames, modes, analysis_frames,
                           frame_samples, targets, cbr=True):
    """Parallel calibration encode. Returns None if it cannot be used."""
    total = len(frames)
    chunks = min(args.threads, total // MIN_CHUNK_FRAMES)
    if pool is None or chunks < 2:
        return None
    preroll_frames = max(1, int(round(CHUNK_PREROLL_MS / args.frame_ms)))
    bounds = [round(k * total / chunks) for k in range(chunks + 1)]
    payloads = []
    for k in range(chunks):
        first, last = bounds[k], bounds[k + 1]
        start = max(0, first - preroll_frames)
        payloads.append((
            args, frames[start:last], modes[start:last],
            analysis_frames[start:last], targets[start:last],
            frame_samples, cbr, first - start,
        ))
    try:
        results = list(pool.map(_encode_chunk_worker, payloads))
    except Exception:
        return None
    packets, rates = [], []
    for chunk_packets, chunk_rates, _ in results:
        packets.extend(chunk_packets)
        rates.extend(chunk_rates)
    return packets, rates, sum(len(p) for p in packets), results[0][2]


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

    # Chunked calibration is exact only when the calibration pass is CBR
    # (auto, music, silk): measured identical to the sequential pass. In
    # voice/hybrid mode the calibration pass runs libopus VBR, whose
    # internal state takes much longer than a pre-roll to settle (chunked
    # estimates came out 0.2-0.5 kbps low), so those stay sequential.
    chunk_pool = None
    if (
        args.threads > 1
        and args.mode not in ("voice", "hybrid")
        and len(frames) >= 2 * MIN_CHUNK_FRAMES
    ):
        try:
            chunk_pool = ProcessPoolExecutor(max_workers=args.threads)
        except Exception:
            chunk_pool = None

    try:
        for iteration in range(args.pass_iterations):
            targets = transform_pass1_pattern(
                pass1_rates,
                args.pass_adjust,
                offset_bps,
                multiplier,
            )

            chunked = encode_packets_chunked(
                chunk_pool, args, frames, modes, analysis_frames,
                frame_samples, targets, cbr=True,
            )
            if chunked is None:
                chunked = encode_packets(
                    opus, args, frames, modes, analysis_frames,
                    frame_samples, cbr=True, targets=targets,
                    auto_mode_choices=None, auto_mode_select=False,
                )
            packets, _, packet_bytes, pre_skip = chunked

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

        # Final encode(s): per-frame waveform-based mode competition runs here
        # and nowhere in the calibration passes. Because the selected modes
        # (Hybrid especially) cost more or less than the calibration passes
        # predicted, the final encode is itself calibrated: the mode choices
        # from the first final encode are cached and reused, and only the
        # per-frame targets are rescaled until the file lands on the target.
        # The output always comes from a selected-mode encode.
        final_choices = [] if args.mode == "auto" else None
        final_best = None
        # Never overshoot: aim half a tolerance below the target and accept
        # only results in [target - tolerance, target].
        tolerance_bps = args.pass_tolerance_kbps * 1000.0
        aim_file_bps = max(
            MIN_BITRATE_BPS, target_file_bps - tolerance_bps * 0.5
        )
        for final_iteration in range(args.pass_iterations):
            final_targets = transform_pass1_pattern(
                pass1_rates, args.pass_adjust, offset_bps, multiplier
            )
            (final_packets, _, final_packet_bytes,
             final_pre_skip) = encode_packets(
                opus, args, frames, modes, analysis_frames, frame_samples,
                cbr=True, targets=final_targets,
                auto_mode_choices=final_choices, auto_mode_select=True,
            )
            write_ogg(
                args.output, final_packets, final_pre_skip, total_samples,
                frame_samples, channels,
            )
            final_size, final_file_rate = measure_file(args.output, duration)
            final_packet_rate = final_packet_bytes * 8.0 / duration
            final_error_bps = target_file_bps - final_file_rate

            vprint(
                args,
                f"Final encode {final_iteration + 1}/{args.pass_iterations}: "
                f"file={final_file_rate / 1000:.3f} kbps, "
                f"audio={final_packet_rate / 1000:.3f} kbps, "
                f"error={final_error_bps / 1000:+.3f} kbps",
            )

            # Rank non-overshooting results first, then by closeness.
            rank = (final_error_bps < 0, abs(final_error_bps))
            if final_best is None or rank < final_best[0]:
                final_best = (
                    rank,
                    (final_size, final_file_rate, final_packet_rate,
                     args.pass_iterations + final_iteration + 1),
                    final_packets,
                )
                shutil.copyfile(args.output, best_path)

            if 0.0 <= final_error_bps <= tolerance_bps:
                break

            if args.pass_adjust == "add":
                offset_bps += aim_file_bps - final_file_rate
            elif final_file_rate > 0:
                multiplier *= aim_file_bps / final_file_rate

            offset_bps = max(
                -MAX_BITRATE_BPS, min(MAX_BITRATE_BPS, offset_bps),
            )
            multiplier = max(0.05, min(20.0, multiplier))

        best_rank, best_info, best_packets = final_best
        best_error = best_rank[1]
        vprint(
            args,
            f"Final mode-selection encode kept: "
            f"file={best_info[1] / 1000:.3f} kbps, "
            f"error={(target_file_bps - best_info[1]) / 1000:+.3f} kbps",
        )

    finally:
        if chunk_pool is not None:
            chunk_pool.shutdown(wait=True)
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

    if not 0 <= args.vbr_intensity <= 20:
        raise ValueError("VBR intensity must be between 0 and 20")

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

    if not 0.0 <= args.inactive_floor <= args.active_floor <= 1.0:
        raise ValueError(
            "Floors must satisfy 0 <= --inactive-floor <= --active-floor <= 1"
        )
    if not args.inactive_db < args.active_db:
        raise ValueError("--inactive-db must be lower than --active-db")

    if not 0 <= args.vbr_normalize_level <= 10:
        raise ValueError(
            "Normalization level must be an integer from 0 to 10"
        )

    if args.vbr_normalize_gate < 0:
        raise ValueError("Normalization gate cannot be negative")

    if args.bandwidth_mode == "mediumband" and args.mode != "silk":
        raise ValueError(
            "--bandwidth-mode mediumband is only available with --mode silk "
            "(CELT has no mediumband; libopus would widen it to wideband)"
        )
    if args.bandwidth_mode == "narrowband" and args.mode == "hybrid":
        raise ValueError(
            "Hybrid needs superwideband or fullband; --bandwidth-mode "
            "narrowband is available with --mode silk and for CELT frames "
            "(music/auto/voice)"
        )
    if args.force_stereo and args.pure_mono:
        raise ValueError("--force-stereo and --pure-mono cannot be combined")

    if args.mode == "silk":
        if args.bandwidth_mode in ("auto", "swb", "fullband"):
            raise ValueError(
                "SILK-only tops out at wideband; use --bandwidth-mode "
                "narrowband, mediumband or wideband (default: wideband)"
            )
        if args.frame_ms < 10:
            raise ValueError(
                "SILK requires --frame-ms >= 10 "
                "(frames shorter than 10 ms force CELT-only)"
            )
        silk_floor = SILK_HOLD_BANDWIDTH_MIN_BPS[silk_bandwidth(args)]
        if args.bitrate * 1000.0 < silk_floor:
            print(
                f"WARNING: libopus SILK needs about {silk_floor / 1000:g} kbps "
                f"to stay "
                f"{SILK_BANDWIDTH_NAMES[silk_bandwidth(args)].lower()}; "
                f"frames are raised to that, so the file will exceed the "
                f"{args.bitrate:g} kbps target. Use --bandwidth-mode "
                "narrowband for lower rates."
            )

    if args.mode == "hybrid":
        if args.bitrate < 9.0:
            print(
                "WARNING: libopus Hybrid needs roughly 8.7 kbps (mono) / "
                "9.5-10 kbps (stereo) at minimum, so the file will come out "
                f"above the {args.bitrate:g} kbps target."
            )
        if args.bandwidth_mode == "wideband":
            raise ValueError(
                "Hybrid requires superwideband or fullband; "
                "--bandwidth-mode wideband would make libopus fall back to SILK"
            )
        if args.frame_ms < 10:
            raise ValueError(
                "Hybrid requires --frame-ms >= 10 "
                "(frames shorter than 10 ms force CELT-only)"
            )

    channels = output_channels(args)
    frame_samples = int(RATE * args.frame_ms / 1000.0)
    started = time.time()

    print(f"Input:          {args.input}")
    print(f"Output:         {args.output}")
    print("Encoder:        OpusPlus")
    print(f"Channels:       {'mono' if channels == 1 else 'stereo'}")
    if args.force_stereo:
        print("NOTE: FORCE STEREO ENABLED — output will be a stereo Opus stream with stereo coding on every frame.")
    elif args.bitrate <= 16.0 and not args.pure_mono:
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
            {
                "voice": "HYBRID/CELT (voice, no SILK)",
                "silk": "SILK-ONLY (forced; "
                        + SILK_BANDWIDTH_NAMES[silk_bandwidth(args)].upper() + ")",
                "auto": "THREE-CANDIDATE MODE TRIALS",
                "music": "CELT-ONLY (music)",
                "hybrid": "HYBRID-ONLY (forced; SWB/fullband only)",
            }[args.mode]
        ),
    )
    vprint(args, "Bandwidth forcing:      standard libopus bandwidth controls")
    vprint(args, "Mode forcing:            uses private libopus force-mode request; build-dependent")
    vprint(args, "SILK:                   never automatic (use --mode silk)")
    vprint(args, f"Stereo:                 {'forced' if args.force_stereo else 'auto (mono at <=16 kbps)'}")
    if args.bitrate >= HYBRID_FLOOR_MIN_TARGET_KBPS:
        vprint(
            args,
            f"Hybrid frame floor:     {HYBRID_MIN_FRAME_BITRATE_BPS / 1000:.1f} kbps "
            f"(active; target >= {HYBRID_FLOOR_MIN_TARGET_KBPS:g} kbps)",
        )
    else:
        vprint(
            args,
            f"Hybrid frame floor:     inactive (target < {HYBRID_FLOOR_MIN_TARGET_KBPS:g} kbps)",
        )
    vprint(
        args,
        f"Active-frame floor:     {args.active_floor:g} x target (SWB+mono: "
        f"{args.active_floor * 2.0 / 3.0:.3g} x); falls to "
        f"{min(args.inactive_floor, args.active_floor):g} x between "
        f"{args.active_db:g} and {args.inactive_db:g} dBFS; digital silence exempt",
    )
    vprint(
        args,
        "Channel policy:         "
        + ("forced stereo (--force-stereo)" if args.force_stereo
           else "true mono stream at <=16 kbps; stereo above 16 kbps unless --pure-mono"),
    )
    vprint(args, f"Frame duration:         {args.frame_ms:g} ms")
    vprint(args, f"Signal mode:            {args.mode.upper()}")
    vprint(args, f"VBR intensity:          {args.vbr_intensity}/10")
    if celt_only_target(args):
        vprint(
            args,
            f"Mode policy:            CELT-only (target <= {CELT_ONLY_MAX_TARGET_KBPS:g} or >= {CELT_ONLY_MIN_TARGET_KBPS:g} kbps)",
        )
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
    vprint(
        args,
        "Low-rate VBR strength:   "
        + ("full strength at all targets (scaling disabled)"
           if args.no_low_rate_scaling
           else "scaled proportionally below 32 kbps"),
    )
    if args.mode == "silk":
        low_rate_bandwidth = (
            f"silk: {SILK_BANDWIDTH_NAMES[silk_bandwidth(args)].upper()}"
        )
    elif args.mode == "hybrid":
        low_rate_bandwidth = (
            "hybrid: SWB/fullband only (wideband and below are never requested)"
        )
    elif args.bandwidth_mode != "default":
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
    if args.mode not in ("hybrid", "silk") and args.bandwidth_mode != "default" and args.bitrate <= 6.0:
        print(
            "WARNING: bandwidth override requested at <=6 kbps. "
            "libopus may still limit the effective bandwidth at this rate."
        )

    if requested_strategy == "auto":
        vprint(args, "Automatic rule:          CELT-only at all target bitrates")

    print("Decoding audio...")
    pcm = None
    decode_threads = args.threads if args.decode_threads == 0 else args.decode_threads
    if decode_threads > 1:
        pcm = decode_input_parallel(
            args.input, pure_mono=(channels == 1), threads=decode_threads,
            segment_seconds=args.decode_segment_seconds,
        )
        vprint(
            args,
            f"Parallel decode:         {'used' if pcm is not None else 'not used (single decode)'}"
            f" ({decode_threads} threads)",
        )
    if pcm is None:
        pcm = decode_input(args.input, pure_mono=(channels == 1))

    total_samples = len(pcm) // channels
    frames = make_frames(pcm, frame_samples, channels)
    analysis_frames = normalize_analysis_frames(frames, args, channels)
    # VBR frame analysis depends only on the audio, so it is computed once
    # (batched, multi-process) instead of inside every encoder pass.
    if args.vbr_intensity == 0:
        analysis_frames = [(0.45, 0.0)] * len(analysis_frames)
    else:
        analysis_frames = compute_frame_metrics(
            analysis_frames, channels, threads=args.threads
        )
    modes = analyze_all_modes(frames, args)
    mark_silent_frames(frames, modes, args.silence_threshold)

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
            "forced Hybrid mode, low-rate VBR smoothing, adaptive channel "
            "coding, phase-inversion protection, two-pass targeting, "
            "and verbose packet analysis."
        )
    )

    parser.add_argument("input", help="Input audio file")
    parser.add_argument("output", help="Output .opus file")
    default_threads = max(1, (os.cpu_count() or 2) // 2)
    parser.add_argument(
        "--threads", type=int, default=default_threads,
        help=(
            f"Worker threads/processes for parallel decode, VBR frame "
            f"analysis, auto-mode trials and calibration passes "
            f"(default: half of logical CPUs = {default_threads}; use 1 to run everything serially)"
        ),
    )

    parser.add_argument(
        "--decode-threads", type=int, default=0,
        help=(
            "Parallel input decode: split the file into overlapping time "
            "segments decoded concurrently, with every seam verified "
            "against the overlap (falls back to a normal decode if any "
            "seam doesn't match). 0 = use --threads (default), 1 = off. "
            "Only kicks in for files longer than two segments"
        ),
    )
    parser.add_argument(
        "--decode-segment-seconds", type=int, default=60,
        help="Segment length in whole seconds for parallel decode (default: 60)",
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
        "--force-stereo", action="store_true",
        help=(
            "Keep a genuine stereo stream at any bitrate: disables the "
            "automatic mono downmix at <=16 kbps and the mono-frame detector"
        ),
    )

    parser.add_argument(
        "--pass-mode", choices=["live", "uncalibrated", "two-pass"],
        default="two-pass",
        help=(
            "two-pass: calibrate the whole file (default, most accurate); "
            "live: single pass with a real-time closed-loop controller that "
            "adjusts every frame to hold the target bitrate; uncalibrated: "
            "single pass with the original open-loop controller and weak "
            "feedback (no target guarantee)"
        ),
    )
    parser.add_argument(
        "--live-horizon-seconds", type=float, default=1.0,
        help=(
            "Live mode: the controller aims to be back on the target "
            "bitrate within this many seconds (smaller = tighter, less "
            "room to shape bits; default 1.0)"
        ),
    )
    parser.add_argument(
        "--live-shape-seconds", type=float, default=5.0,
        help=(
            "Live mode: averaging time of the activity/transient weight "
            "that frames are compared against (default 5.0)"
        ),
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
        "--swb-enter-db", type=float, default=-45.0,
        help="Enter SWB detection below this high/low energy ratio",
    )
    parser.add_argument(
        "--swb-exit-db", type=float, default=-45.0,
        help="Remain in SWB while below this high/low energy ratio",
    )

    parser.add_argument(
        "--phase-invert-correlation",
        type=float,
        default=-0.7,
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
        "--active-floor", type=float, default=0.5,
        help=(
            "Minimum per-frame bitrate request for normal frames, as a "
            "fraction of the target (default 0.5; SWB+mono frames use 2/3 "
            "of it). Raise it to stop the rate sagging during ordinary "
            "material"
        ),
    )
    parser.add_argument(
        "--inactive-floor", type=float, default=0.1,
        help=(
            "Minimum request, as a fraction of the target, for very quiet "
            "frames (default 0.1). Only frames below --active-db get a "
            "floor lower than --active-floor"
        ),
    )
    parser.add_argument(
        "--active-db", type=float, default=-40.0,
        help="Frame RMS level (dBFS) at/above which the full active floor applies (default -40)",
    )
    parser.add_argument(
        "--inactive-db", type=float, default=-60.0,
        help="Frame RMS level (dBFS) at/below which the inactive floor applies (default -60)",
    )

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
        choices=range(21), default=10,
    )
    parser.add_argument(
        "--no-low-rate-scaling", action="store_true",
        help=(
            "Disable the proportional reduction of VBR strength below "
            "32 kbps. VBR stays fully active at the configured intensity "
            "at any target bitrate."
        ),
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
        "--voice-enter-percent", type=float, default=40.01,
        help="Enter voice sections when Hybrid win share reaches this percentage (default: 40)",
    )
    parser.add_argument(
        "--voice-exit-percent", type=float, default=40.0,
        help="Return to music sections when Hybrid win share falls to this percentage (default: 40)",
    )

    parser.add_argument(
        "--mode", choices=["auto", "music", "voice", "hybrid", "silk"],
        default="auto",
        help=(
            "Coding mode: music forces CELT-only; voice uses Hybrid/CELT "
            "(no SILK; CELT-only at <=10 or >=20 kbps); hybrid forces Hybrid "
            "on every frame at any bitrate (SWB/fullband only, requires "
            "--frame-ms >= 10); silk forces SILK-only (the only way to get "
            "SILK; see --bandwidth-mode narrowband/mediumband/wideband, "
            "requires --frame-ms >= 10); auto trial-encodes Hybrid vs CELT "
            "and uses a rolling Hybrid win-rate to classify voice-like "
            "sections versus music sections (CELT-only at <=10 or >=20 kbps)"
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
        choices=[
            "default", "auto", "swb", "fullband", "wideband",
            "narrowband", "mediumband",
        ],
        default="default",
        help=(
            "Bandwidth policy: default follows bitrate rules; auto chooses "
            "SWB or fullband per frame at ANY bitrate; swb/fullband/wideband "
            "request that bandwidth even at very low bitrates; narrowband "
            "(4 kHz) works for SILK and for CELT frames (music/auto/voice; "
            "Hybrid frames stay SWB+); mediumband (6 kHz) is SILK only "
            "(--mode silk, SILK default: wideband)"
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
