import argparse
import ctypes
import math
import os
import shutil
import struct
import subprocess
import sys
import time
from collections import Counter

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
OPUS_MODE_CELT_ONLY = 1002

OPUS_AUTO = -1000
OPUS_SIGNAL_VOICE = 3001
OPUS_SIGNAL_MUSIC = 3002
OPUS_BANDWIDTH_WIDEBAND = 1103
OPUS_BANDWIDTH_SUPERWIDEBAND = 1104
OPUS_BANDWIDTH_FULLBAND = 1105

MIN_BITRATE_BPS = 6000
MAX_BITRATE_BPS = 512000
OGG_OVERHEAD_ESTIMATE_BPS = 250
MAX_PACKET_BYTES = 1275 * 3
SWB_MONO_GUARD_KBPS = 12.0
STEREO_MIN_BITRATE_BPS = 17000


def vprint(args, *values, **kwargs):
    if args.verbose:
        print(*values, **kwargs)


def output_channels(args):
    return 1 if args.pure_mono else CHANNELS


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


def create_encoder(opus, args, cbr=False):
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
        opus_ctl(
            opus, encoder, OPUS_SET_SIGNAL_REQUEST, OPUS_SIGNAL_MUSIC
        )
        opus_ctl(
            opus, encoder, OPUS_SET_PHASE_INVERSION_DISABLED_REQUEST, 1
        )

        if channels == 2:
            opus_ctl(
                opus, encoder, OPUS_SET_FORCE_CHANNELS_REQUEST, 2
            )

        # This is an experimental/private libopus control in many builds.
        # If unsupported, fail clearly instead of claiming CELT is forced.
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

        # At 12 kbps this is 75% of normal strength.
        # At 8 kbps it is 50%; at 16 kbps and above it is unchanged.
        low_rate_scale = min(1.0, self.target_kbps / 16.0)
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
    # At or below 6.5 kbps, use CELT wideband on every frame.
    # CELT-only is forced separately by create_encoder().
    if args.bitrate <= 6.5:
        return OPUS_BANDWIDTH_WIDEBAND

    # Above 6.5 and below 16 kbps, default to SWB on every frame.
    # The optional adaptive mode permits FB except on SWB-detected frames.
    if args.bitrate < 16.0:
        if not args.adaptive_swb:
            return OPUS_BANDWIDTH_SUPERWIDEBAND
        return (
            OPUS_BANDWIDTH_SUPERWIDEBAND
            if mode["swb"] else OPUS_BANDWIDTH_FULLBAND
        )

    # At 16 kbps and above, retain the existing detector/strategy behavior.
    if args.codec_strategy == "swb" or mode["swb"]:
        return OPUS_BANDWIDTH_SUPERWIDEBAND

    return OPUS_BANDWIDTH_FULLBAND


def select_max_bandwidth(args):
    # Apply the same hard ceiling as the per-frame request. Without this,
    # libopus may choose a wider bandwidth than the requested one.
    if args.bitrate <= 6.5:
        return OPUS_BANDWIDTH_WIDEBAND
    if args.bitrate < 16.0 and not args.adaptive_swb:
        return OPUS_BANDWIDTH_SUPERWIDEBAND
    return OPUS_BANDWIDTH_FULLBAND


# ============================================================
# Encode packets
# ============================================================

def encode_packets(
    opus, args, frames, modes, analysis_frames,
    frame_samples, cbr=False, targets=None
):
    channels = output_channels(args)

    encoder, pre_skip = create_encoder(opus, args, cbr=cbr)
    controller = BitrateController(
        args.bitrate, args.vbr_intensity, channels
    )

    packets = []
    rates = []
    output_buffer = (ctypes.c_ubyte * MAX_PACKET_BYTES)()

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

            # Preserve the requested phase-inverted stereo floor.
            if phase_inverted and channels == 2:
                requested = max(
                    requested,
                    int(args.phase_invert_floor_kbps * 1000),
                )

            # Keep genuinely stereo frames at or above 17 kbps so adaptive
            # VBR does not starve the stereo image. Frames detected as mono
            # or mostly mono may use the lower adaptive bitrate.
            frame_is_mono = channels == 2 and mode["mono"] and not phase_inverted
            if channels == 2 and not frame_is_mono:
                requested = max(requested, STEREO_MIN_BITRATE_BPS)

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
        opus.opus_encoder_destroy(ctypes.c_void_p(encoder))

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

    vprint(args, "\nPASS 1: adaptive VBR analysis")

    packets, pass1_rates, _, _ = encode_packets(
        opus, args, frames, modes, analysis_frames, frame_samples
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
        "replaying pass-1 pattern",
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
        opus, args, frames, modes, analysis_frames, frame_samples
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

    if not 6 <= args.bitrate <= 512:
        raise ValueError("Bitrate must be between 6 and 512 kbps")

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
    print(f"Target bitrate: {args.bitrate:g} kbps")
    print(f"Strategy:       {args.codec_strategy.upper()}")
    print(f"Pass mode:      {args.pass_mode}")

    vprint(args, "\nREQUESTED CODING MODE: CELT-ONLY")
    vprint(args, "Force-mode control:     experimental libopus request")
    vprint(args, "Hybrid bitrate band:    removed")
    vprint(args, "Stereo bitrate floor:   17.0 kbps (stereo output only)")
    vprint(args, "Stereo channel policy:  always stereo; no detector collapse")
    vprint(args, f"Frame duration:         {args.frame_ms:g} ms")
    vprint(args, f"Signal mode:            MUSIC")
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
    vprint(args, "Low-rate VBR strength:   scaled proportionally below 16 kbps")
    if args.bitrate <= 6.5:
        low_rate_bandwidth = "forced CELT wideband (<= 6.5 kbps)"
    elif args.bitrate < 16.0:
        low_rate_bandwidth = (
            "detector-adaptive SWB/FB"
            if args.adaptive_swb else "SWB on every frame"
        )
    else:
        low_rate_bandwidth = "Detector/strategy controlled"
    vprint(args, f"Bandwidth policy:        {low_rate_bandwidth}")

    if requested_strategy == "auto":
        vprint(args, "Automatic rule:          CELT-only at all target bitrates")

    print("Decoding audio...")
    pcm = decode_input(args.input, pure_mono=args.pure_mono)

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
            "OpusPlus adaptive encoder with experimental CELT-only mode, "
            "low-rate VBR smoothing, adaptive channel coding, "
            "phase-inversion protection, two-pass targeting, "
            "and verbose packet analysis."
        )
    )

    parser.add_argument("input", help="Input audio file")
    parser.add_argument("output", help="Output .opus file")

    parser.add_argument(
        "-b", "--bitrate", type=float, default=18,
        help="Target bitrate in kbps",
    )
    parser.add_argument(
        "--pure-mono", action="store_true",
        help="Encode a genuine single-channel Opus stream",
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

    parser.add_argument("--mono-enter-db", type=float, default=-45.0)
    parser.add_argument("--mono-exit-db", type=float, default=-45.0)

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
        default=-0.5,
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
        "--mode", choices=["auto", "music", "voice"],
        default="music",
        help="Signal hint; CELT strategy uses MUSIC",
    )
    parser.add_argument(
        "--codec-strategy",
        choices=["auto", "swb", "celt"],
        default="auto",
        help=(
            "auto: request CELT-only at all bitrates; "
            "swb: CELT-only with SWB bandwidth limit; "
            "celt: CELT-only with detector-controlled bandwidth"
        ),
    )
    parser.add_argument(
        "--adaptive-swb", action="store_true",
        help=(
            "Below 16 kbps, use SWB only on frames flagged by the SWB "
            "detector and allow fullband on other frames; default is all SWB"
        ),
    )

    parser.add_argument("--opus-dll", default=None)
    parser.add_argument(
        "-v", "--verbose", action="store_true",
        help="Show detailed encoder diagnostics and packet analysis",
    )

    args = parser.parse_args()

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
