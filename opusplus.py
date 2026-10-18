
import argparse
import ctypes
import math
import os
import shutil
import struct
import subprocess
import sys
import time

import numpy as np


# ============================================================
# OpusPlus
#
# Requirements:
#   Python, numpy, FFmpeg, opus.dll
#
# Features:
#   - Per-frame mono/stereo coding decisions
#   - Per-frame SWB/fullband bandwidth decisions
#   - Mono bitrate multiplier: 0.80 by default
#   - SWB bitrate multiplier: 0.80 by default
#   - Adaptive VBR first pass
#   - Activity/transient-aware bitrate allocation
#   - Two-pass bitrate-pattern adjustment
#   - Pass 2 repeats Pass 1's per-frame mode decisions
#   - Ogg Opus output
#
# The Ogg Opus stream remains configured for stereo output.
# Mono-coded frames are decoded to dual-mono stereo by the decoder.
# ============================================================

RATE = 48000
CHANNELS = 2
OPUS_APPLICATION_AUDIO = 2049

OPUS_SET_BITRATE_REQUEST = 4002
OPUS_SET_MAX_BANDWIDTH_REQUEST = 4004
OPUS_SET_VBR_REQUEST = 4006
OPUS_SET_COMPLEXITY_REQUEST = 4010
OPUS_SET_FORCE_CHANNELS_REQUEST = 4022
OPUS_SET_SIGNAL_REQUEST = 4024
OPUS_GET_LOOKAHEAD_REQUEST = 4027

OPUS_AUTO = -1000
OPUS_SIGNAL_VOICE = 3001
OPUS_SIGNAL_MUSIC = 3002

OPUS_BANDWIDTH_SUPERWIDEBAND = 1104
OPUS_BANDWIDTH_FULLBAND = 1105

MIN_BITRATE_BPS = 6000
MAX_BITRATE_BPS = 512000
OGG_OVERHEAD_ESTIMATE_BPS = 250
MAX_PACKET_BYTES = 1275 * 3


# ============================================================
# libopus
# ============================================================

def load_opus(dll_path=None):
    candidates = (
        [dll_path]
        if dll_path
        else [
            os.path.join(
                os.path.dirname(os.path.abspath(__file__)),
                "opus.dll",
            ),
            "opus.dll",
        ]
    )

    last_error = None

    for candidate in candidates:
        try:
            return ctypes.CDLL(candidate)
        except OSError as exc:
            last_error = exc

    raise RuntimeError(
        "Could not load opus.dll. Put it beside opusplus.py, "
        "add it to PATH, or use --opus-dll.\n"
        f"Last error: {last_error}"
    )


def configure_opus_functions(opus):
    opus.opus_encoder_create.argtypes = [
        ctypes.c_int,
        ctypes.c_int,
        ctypes.c_int,
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

    # libopus_encoder_ctl is variadic.
    opus.opus_encoder_ctl.argtypes = [
        ctypes.c_void_p,
        ctypes.c_int,
    ]
    opus.opus_encoder_ctl.restype = ctypes.c_int

    if hasattr(opus, "opus_get_version_string"):
        opus.opus_get_version_string.restype = ctypes.c_char_p

    return opus


def opus_ctl(opus, encoder, request, value):
    result = opus.opus_encoder_ctl(
        encoder,
        ctypes.c_int(request),
        ctypes.c_int(int(value)),
    )

    if result != 0:
        raise RuntimeError(
            f"libopus control {request} failed: {result}"
        )


def create_encoder(opus, args, cbr=False):
    error = ctypes.c_int(0)

    # Always create a two-channel encoder. We can request mono coding
    # for individual frames while retaining a stereo-configured stream.
    encoder = opus.opus_encoder_create(
        RATE,
        CHANNELS,
        OPUS_APPLICATION_AUDIO,
        ctypes.byref(error),
    )

    if not encoder or error.value != 0:
        raise RuntimeError(
            f"Could not create encoder: {error.value}"
        )

    try:
        opus_ctl(
            opus,
            encoder,
            OPUS_SET_VBR_REQUEST,
            0 if cbr else 1,
        )

        opus_ctl(
            opus,
            encoder,
            OPUS_SET_COMPLEXITY_REQUEST,
            args.complexity,
        )

        signal = {
            "auto": OPUS_AUTO,
            "music": OPUS_SIGNAL_MUSIC,
            "voice": OPUS_SIGNAL_VOICE,
        }[args.mode]

        opus_ctl(
            opus,
            encoder,
            OPUS_SET_SIGNAL_REQUEST,
            signal,
        )

        # Initial values; updated for each frame before encoding.
        opus_ctl(
            opus,
            encoder,
            OPUS_SET_MAX_BANDWIDTH_REQUEST,
            OPUS_BANDWIDTH_FULLBAND,
        )

        opus_ctl(
            opus,
            encoder,
            OPUS_SET_FORCE_CHANNELS_REQUEST,
            2,
        )

        lookahead = ctypes.c_int(0)

        result = opus.opus_encoder_ctl(
            encoder,
            ctypes.c_int(OPUS_GET_LOOKAHEAD_REQUEST),
            ctypes.byref(lookahead),
        )

        pre_skip = lookahead.value if result == 0 else 312

        return encoder, pre_skip

    except Exception:
        opus.opus_encoder_destroy(encoder)
        raise


# ============================================================
# FFmpeg input decoding
# ============================================================

def find_ffmpeg():
    path = shutil.which("ffmpeg")

    if not path:
        raise RuntimeError(
            "FFmpeg was not found on PATH."
        )

    return path


def decode_input(path):
    command = [
        find_ffmpeg(),
        "-hide_banner",
        "-loglevel", "error",
        "-i", path,
        "-map", "0:a:0",
        "-vn", "-sn", "-dn",
        "-f", "s16le",
        "-acodec", "pcm_s16le",
        "-ar", str(RATE),
        "-ac", "2",
        "pipe:1",
    ]

    process = subprocess.Popen(
        command,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
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

    pcm = pcm[:len(pcm) - len(pcm) % 4]

    if not pcm:
        raise RuntimeError("FFmpeg decoded no audio samples")

    return np.frombuffer(bytes(pcm), dtype=np.int16).copy()


# ============================================================
# Frame preparation
# ============================================================

def make_frames(pcm, frame_samples):
    values_per_frame = frame_samples * CHANNELS
    frames = []

    for start in range(0, len(pcm), values_per_frame):
        chunk = pcm[start:start + values_per_frame]
        actual_values = len(chunk)

        if actual_values < values_per_frame:
            padded = np.zeros(values_per_frame, dtype=np.int16)
            padded[:actual_values] = chunk
            chunk = padded

        frames.append(
            (chunk, actual_values // CHANNELS)
        )

    return frames


# ============================================================
# Per-frame spectral and stereo analysis
# ============================================================

def analyze_mode(frame, previous_mode, args):
    """
    Return (mono, swb).

    Mono:
        Compare L-R side energy with L+R mid energy.

    Bandwidth:
        Measure energy at/above 12 kHz relative to lower frequencies.

    Hysteresis:
        Use separate thresholds when switching into and out of a mode.
    """

    stereo = np.asarray(frame, dtype=np.float32).reshape(-1, 2)

    left = stereo[:, 0]
    right = stereo[:, 1]

    mid = (left + right) * 0.5
    side = (left - right) * 0.5

    mid_rms = float(np.sqrt(np.mean(mid * mid) + 1e-12))
    side_rms = float(np.sqrt(np.mean(side * side) + 1e-12))

    side_db = 20.0 * math.log10(
        max(side_rms, 1e-9) / max(mid_rms, 1e-9)
    )

    # A frame must have very little side information to use mono.
    # Thresholds are configurable for experimentation.
    if previous_mode is not None and previous_mode["mono"]:
        mono = side_db < args.mono_exit_db
    else:
        mono = side_db < args.mono_enter_db

    # Use the mono-like mid signal for bandwidth detection so that
    # a wide stereo effect alone does not trigger fullband.
    window = np.hanning(len(mid)).astype(np.float32)

    spectrum = np.fft.rfft(mid * window)
    power = np.abs(spectrum) ** 2

    frequencies = np.fft.rfftfreq(
        len(mid),
        d=1.0 / RATE,
    )

    low = float(np.sum(power[frequencies < 12000.0]))
    high = float(np.sum(power[frequencies >= 12000.0]))

    high_db = 10.0 * math.log10(
        max(high, 1e-30) / max(low, 1e-30)
    )

    if previous_mode is not None and previous_mode["swb"]:
        swb = high_db < args.swb_exit_db
    else:
        swb = high_db < args.swb_enter_db

    # Silence has no useful evidence of stereo or high frequencies.
    if mid_rms < args.silence_threshold:
        mono = True
        swb = True

    return {
        "mono": bool(mono),
        "swb": bool(swb),
        "side_db": side_db,
        "high_db": high_db,
    }


def analyze_all_modes(frames, args):
    modes = []
    previous = None

    for frame, actual_samples in frames:
        mode = analyze_mode(frame, previous, args)
        modes.append(mode)
        previous = mode

    return modes


def mode_multiplier(mode, args):
    multiplier = 1.0

    if mode["mono"]:
        multiplier *= args.mono_multiplier

    if mode["swb"]:
        multiplier *= args.swb_multiplier

    return multiplier


# ============================================================
# Activity and transient analysis
# ============================================================

def analyze_frame(frame):
    x = np.asarray(frame, dtype=np.float32).reshape(-1, 2)

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

    level_jump = abs(
        math.log((rms_b + 16.0) / (rms_a + 16.0))
    )

    curvature_score = max(
        0.0,
        min((transient_ratio - 1.0) / 5.0, 1.0),
    )

    jump_score = max(
        0.0,
        min(level_jump / 1.5, 1.0),
    )

    transient = 0.65 * curvature_score + 0.35 * jump_score

    return activity, max(0.0, min(transient, 1.0))


# ============================================================
# Adaptive bitrate controller
# ============================================================

class BitrateController:
    def __init__(self, target_kbps, intensity):
        self.target_bps = float(target_kbps) * 1000.0
        self.intensity = int(intensity)

        self.total_packet_bytes = 0
        self.total_samples = 0

    def bitrate_for_frame(self, frame, multiplier):
        packet_target_bps = max(
            MIN_BITRATE_BPS,
            self.target_bps * multiplier - 250.0,
        )

        if self.intensity == 0:
            return int(packet_target_bps)

        activity, transient = analyze_frame(frame)
        strength = self.intensity / 10.0

        activity_factor = 1.0 + strength * (
            (activity - 0.45) * 0.90
        )

        transient_factor = 1.0 + strength * 0.40 * transient

        activity_factor = max(
            0.75,
            min(1.50, activity_factor),
        )

        transient_factor = max(
            1.0,
            min(1.40, transient_factor),
        )

        if self.total_samples <= 0 or self.total_packet_bytes <= 0:
            feedback_factor = 1.0
        else:
            elapsed = self.total_samples / RATE

            # Feedback tracks the nominal overall target. Per-frame
            # mode reductions are applied separately above.
            target_bytes = self.target_bps * elapsed / 8.0

            ratio = target_bytes / max(self.total_packet_bytes, 1)

            feedback_factor = max(
                0.70,
                min(1.40, ratio ** 0.35),
            )

        requested = (
            packet_target_bps
            * activity_factor
            * transient_factor
            * feedback_factor
        )

        return int(
            max(
                MIN_BITRATE_BPS,
                min(MAX_BITRATE_BPS, requested),
            )
        )

    def record_packet(self, packet_bytes, frame_samples):
        self.total_packet_bytes += int(packet_bytes)
        self.total_samples += int(frame_samples)


# ============================================================
# Encoding
# ============================================================

def encode_packets(
    opus,
    args,
    frames,
    modes,
    frame_samples,
    cbr=False,
    targets=None,
):
    encoder, pre_skip = create_encoder(opus, args, cbr=cbr)

    controller = BitrateController(
        args.bitrate,
        args.vbr_intensity,
    )

    packets = []
    rates = []
    packet_bytes_total = 0

    output_buffer = (ctypes.c_ubyte * MAX_PACKET_BYTES)()

    try:
        for i, (frame, actual_samples) in enumerate(frames):
            mode = modes[i]

            # Select channel coding and bandwidth for this frame.
            opus_ctl(
                opus,
                encoder,
                OPUS_SET_FORCE_CHANNELS_REQUEST,
                1 if mode["mono"] else 2,
            )

            opus_ctl(
                opus,
                encoder,
                OPUS_SET_MAX_BANDWIDTH_REQUEST,
                (
                    OPUS_BANDWIDTH_SUPERWIDEBAND
                    if mode["swb"]
                    else OPUS_BANDWIDTH_FULLBAND
                ),
            )

            if targets is None:
                requested = controller.bitrate_for_frame(
                    frame,
                    mode_multiplier(mode, args),
                )
            else:
                requested = targets[i]

            requested = int(
                max(
                    MIN_BITRATE_BPS,
                    min(MAX_BITRATE_BPS, requested),
                )
            )

            opus_ctl(
                opus,
                encoder,
                OPUS_SET_BITRATE_REQUEST,
                requested,
            )

            pcm_ptr = frame.ctypes.data_as(
                ctypes.POINTER(ctypes.c_int16)
            )

            packet_length = opus.opus_encode(
                encoder,
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
            packet_bytes_total += packet_length

            rates.append(
                packet_length * 8.0 * RATE / frame_samples
            )

            controller.record_packet(
                packet_length,
                frame_samples,
            )

    finally:
        opus.opus_encoder_destroy(encoder)

    return packets, rates, packet_bytes_total, pre_skip


# ============================================================
# Ogg Opus writer
# ============================================================

def make_ogg_crc_table():
    table = []

    for i in range(256):
        value = i << 24

        for _ in range(8):
            if value & 0x80000000:
                value = (
                    (value << 1) ^ 0x04C11DB7
                ) & 0xFFFFFFFF
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

        header = bytearray(
            struct.pack(
                "<4sBBqIIIB",
                b"OggS",
                0,
                header_type,
                granule,
                self.serial,
                self.sequence,
                0,
                len(lacing),
            )
        )

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


def make_opus_head(pre_skip):
    # A two-channel mapping-family-0 stream remains stereo-configured.
    return (
        b"OpusHead"
        + struct.pack(
            "<BBHIhB",
            1,
            2,
            pre_skip,
            RATE,
            0,
            0,
        )
    )


def make_opus_tags():
    vendor = b"OpusPlus"

    return (
        b"OpusTags"
        + struct.pack("<I", len(vendor))
        + vendor
        + struct.pack("<I", 0)
    )


def write_ogg(path, packets, pre_skip, total_samples, frame_samples):
    with open(path, "wb") as output_file:
        ogg = OggWriter(output_file)

        ogg.write_header_packet(
            make_opus_head(pre_skip),
            0,
            header_type=2,
        )

        ogg.write_header_packet(make_opus_tags(), 0)

        for i, packet in enumerate(packets):
            decoded_end = min(
                (i + 1) * frame_samples,
                total_samples,
            )

            ogg.add_audio_packet(
                packet,
                pre_skip + decoded_end,
                frame_samples,
            )

        ogg.finish()


# ============================================================
# Pass 2 bitrate-pattern transformation
# ============================================================

def transform_pass1_pattern(
    pass1_rates,
    adjustment,
    offset_bps,
    multiplier,
):
    result = []

    for rate in pass1_rates:
        value = (
            rate + offset_bps
            if adjustment == "add"
            else rate * multiplier
        )

        result.append(
            int(
                max(
                    MIN_BITRATE_BPS,
                    min(MAX_BITRATE_BPS, round(value)),
                )
            )
        )

    return result


def measure_file(path, duration):
    size = os.path.getsize(path)

    return size, (size * 8.0 / duration if duration > 0 else 0.0)


# ============================================================
# Two-pass encoding
# ============================================================

def run_two_pass(opus, args, frames, modes, frame_samples, total_samples):
    duration = total_samples / RATE

    multipliers = [
        mode_multiplier(mode, args)
        for mode in modes
    ]

    average_multiplier = float(np.mean(multipliers))
    effective_target = args.bitrate * average_multiplier
    target_file_bps = effective_target * 1000.0

    print("\nPASS 1: adaptive VBR with per-frame mode switching")

    packets, pass1_rates, packet_bytes, pre_skip = encode_packets(
        opus,
        args,
        frames,
        modes,
        frame_samples,
        cbr=False,
    )

    del packets

    pass1_mean = float(np.mean(pass1_rates))

    print(f"Pass 1 mean packet rate: {pass1_mean / 1000:.3f} kbps")
    print(f"Average mode multiplier: {average_multiplier:.3f}x")
    print(f"Effective target:        {effective_target:.3f} kbps")

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
            if pass1_mean > 0
            else 1.0
        )

    best_path = args.output + ".opusplus-best.tmp"
    best_error = float("inf")
    best_info = None

    print(f"\nPASS 2: {args.pass_adjust} adjustment; replaying mode pattern")

    try:
        for iteration in range(args.pass_iterations):
            targets = transform_pass1_pattern(
                pass1_rates,
                args.pass_adjust,
                offset_bps,
                multiplier,
            )

            packets, rates, packet_bytes, pre_skip = encode_packets(
                opus,
                args,
                frames,
                modes,
                frame_samples,
                cbr=True,
                targets=targets,
            )

            write_ogg(
                args.output,
                packets,
                pre_skip,
                total_samples,
                frame_samples,
            )

            size, file_rate = measure_file(args.output, duration)
            packet_rate = packet_bytes * 8.0 / duration
            error_bps = target_file_bps - file_rate

            print(
                f"Iteration {iteration + 1}/{args.pass_iterations}: "
                f"file={file_rate / 1000:.3f} kbps, "
                f"audio={packet_rate / 1000:.3f} kbps, "
                f"error={error_bps / 1000:+.3f} kbps"
            )

            if abs(error_bps) < best_error:
                best_error = abs(error_bps)
                shutil.copyfile(args.output, best_path)

                best_info = (
                    size,
                    file_rate,
                    packet_rate,
                    iteration + 1,
                )

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
    else:
        size, file_rate, packet_rate, best_iteration = best_info

    return {
        "duration": duration,
        "file_size": size,
        "file_rate": file_rate,
        "packet_rate": packet_rate,
        "pass1_mean": pass1_mean,
        "best_iteration": best_iteration,
        "effective_target": effective_target,
        "average_multiplier": average_multiplier,
    }


# ============================================================
# Single-pass encoding
# ============================================================

def run_single_pass(opus, args, frames, modes, frame_samples, total_samples):
    duration = total_samples / RATE

    packets, rates, packet_bytes, pre_skip = encode_packets(
        opus,
        args,
        frames,
        modes,
        frame_samples,
        cbr=False,
    )

    write_ogg(
        args.output,
        packets,
        pre_skip,
        total_samples,
        frame_samples,
    )

    size, file_rate = measure_file(args.output, duration)

    average_multiplier = float(np.mean([
        mode_multiplier(mode, args)
        for mode in modes
    ]))

    return {
        "duration": duration,
        "file_size": size,
        "file_rate": file_rate,
        "packet_rate": packet_bytes * 8.0 / duration,
        "pass1_mean": float(np.mean(rates)),
        "best_iteration": 1,
        "effective_target": args.bitrate * average_multiplier,
        "average_multiplier": average_multiplier,
    }


# ============================================================
# Main
# ============================================================

def encode(args):
    if not os.path.isfile(args.input):
        raise FileNotFoundError(f"Input not found: {args.input}")

    if not 6 <= args.bitrate <= 512:
        raise ValueError("Bitrate must be between 6 and 512 kbps")

    if not 0 <= args.complexity <= 10:
        raise ValueError("Complexity must be between 0 and 10")

    if not 0 <= args.vbr_intensity <= 10:
        raise ValueError("VBR intensity must be between 0 and 10")

    if not 1 <= args.pass_iterations <= 10:
        raise ValueError("Pass iterations must be between 1 and 10")

    if args.pass_tolerance_kbps < 0:
        raise ValueError("Pass tolerance cannot be negative")

    if not 0.05 <= args.mono_multiplier <= 1.0:
        raise ValueError("Mono multiplier must be between 0.05 and 1.0")

    if not 0.05 <= args.swb_multiplier <= 1.0:
        raise ValueError("SWB multiplier must be between 0.05 and 1.0")

    frame_samples = int(RATE * args.frame_ms / 1000.0)

    started = time.time()

    print(f"Input:  {args.input}")
    print(f"Output: {args.output}")
    print("Decoding to stereo 48 kHz PCM...")

    pcm = decode_input(args.input)
    total_samples = len(pcm) // 2
    frames = make_frames(pcm, frame_samples)

    # Detect all frame modes once, then replay them in Pass 2.
    modes = analyze_all_modes(frames, args)

    mono_count = sum(mode["mono"] for mode in modes)
    swb_count = sum(mode["swb"] for mode in modes)
    both_count = sum(mode["mono"] and mode["swb"] for mode in modes)

    opus = configure_opus_functions(load_opus(args.opus_dll))

    if hasattr(opus, "opus_get_version_string"):
        version = opus.opus_get_version_string().decode(
            "ascii",
            errors="replace",
        )
    else:
        version = "unknown"

    print(f"Encoder:            OpusPlus")
    print(f"libopus:            {version}")
    print(f"Reference bitrate:  {args.bitrate:.3f} kbps")
    print(f"Frame duration:     {args.frame_ms:g} ms")
    print(f"Frames:             {len(frames)}")
    print(f"Duration:           {total_samples / RATE:.3f} sec")
    print(f"Signal mode:        {args.mode}")
    print(f"VBR intensity:      {args.vbr_intensity}/10")
    print(f"Complexity:         {args.complexity}/10")

    print("\nPER-FRAME MODE ANALYSIS")
    print(f"Mono-coded frames:  {mono_count}/{len(frames)}")
    print(f"SWB frames:         {swb_count}/{len(frames)}")
    print(f"Mono + SWB frames:  {both_count}/{len(frames)}")
    print(f"Mono multiplier:    {args.mono_multiplier:.3f}x")
    print(f"SWB multiplier:     {args.swb_multiplier:.3f}x")

    if args.pass_mode == "two-pass":
        result = run_two_pass(
            opus,
            args,
            frames,
            modes,
            frame_samples,
            total_samples,
        )
    else:
        result = run_single_pass(
            opus,
            args,
            frames,
            modes,
            frame_samples,
            total_samples,
        )

    print("\nOpusPlus encoding complete.")
    print(f"Duration:             {result['duration']:.3f} sec")
    print(f"Output size:           {result['file_size']:,} bytes")
    print(f"Audio packet rate:     {result['packet_rate'] / 1000:.3f} kbps")
    print(f"Total file rate:       {result['file_rate'] / 1000:.3f} kbps")
    print(f"Reference target:      {args.bitrate:.3f} kbps")
    print(f"Effective target:      {result['effective_target']:.3f} kbps")
    print(f"Average multiplier:    {result['average_multiplier']:.3f}x")
    print(
        f"Difference from effective target: "
        f"{result['file_rate'] / 1000 - result['effective_target']:+.3f} kbps"
    )

    if args.pass_mode == "two-pass":
        print(f"Best iteration:        {result['best_iteration']}")

    print(f"Elapsed time:          {time.time() - started:.2f} sec")


# ============================================================
# CLI
# ============================================================

def main():
    parser = argparse.ArgumentParser(
        description=(
            "OpusPlus: adaptive per-frame bandwidth/channel coding "
            "with bitrate-pattern correction."
        )
    )

    parser.add_argument("input", help="Input audio file")
    parser.add_argument("output", help="Output .opus file")

    parser.add_argument(
        "-b", "--bitrate",
        type=float,
        default=32,
        help="Reference target bitrate in kbps (default: 32)",
    )

    parser.add_argument(
        "--pass-mode",
        choices=["live", "two-pass"],
        default="two-pass",
    )

    parser.add_argument(
        "--pass-adjust",
        choices=["add", "multiply"],
        default="multiply",
    )

    parser.add_argument(
        "--pass-iterations",
        type=int,
        default=4,
    )

    parser.add_argument(
        "--pass-tolerance-kbps",
        type=float,
        default=0.5,
    )

    parser.add_argument(
        "--mono-multiplier",
        type=float,
        default=0.80,
        help="Per-frame bitrate multiplier for mono coding (default: 0.80)",
    )

    parser.add_argument(
        "--swb-multiplier",
        type=float,
        default=0.80,
        help="Per-frame bitrate multiplier for SWB (default: 0.80)",
    )

    parser.add_argument(
        "--mono-enter-db",
        type=float,
        default=-45.0,
        help="Enter mono when side signal is below this dB level",
    )

    parser.add_argument(
        "--mono-exit-db",
        type=float,
        default=-41.0,
        help="Return to stereo when side signal exceeds this dB level",
    )

    parser.add_argument(
        "--swb-enter-db",
        type=float,
        default=-45.0,
        help="Enter SWB when high-band energy is below this dB level",
    )

    parser.add_argument(
        "--swb-exit-db",
        type=float,
        default=-41.0,
        help="Return to fullband when high-band energy exceeds this dB level",
    )

    parser.add_argument(
        "--silence-threshold",
        type=float,
        default=8.0,
        help="RMS threshold below which a frame is treated as silence",
    )

    parser.add_argument(
        "--vbr-intensity",
        type=int,
        choices=range(11),
        default=10,
    )

    parser.add_argument(
        "--complexity",
        type=int,
        choices=range(11),
        default=10,
    )

    parser.add_argument(
        "--frame-ms",
        type=float,
        choices=[2.5, 5, 10, 20, 40, 60],
        default=20,
    )

    parser.add_argument(
        "--mode",
        choices=["auto", "music", "voice"],
        default="music",
    )

    parser.add_argument("--opus-dll", default=None)

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
    try:
        main()
    except Exception:
        import traceback
        traceback.print_exc()
        sys.exit(1)
