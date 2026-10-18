
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
# opusplus.py
#
# Requirements:
#   - Python
#   - numpy
#   - ffmpeg available on PATH
#   - opus.dll (libopus)
#
# Features:
#   - Adaptive VBR first pass
#   - Actual per-packet bitrate measurements
#   - Activity and transient-aware bitrate allocation
#   - Automatic additive or multiplicative bitrate adjustment
#   - Iterative second-pass bitrate correction
#   - Per-frame CBR targets on subsequent passes
#   - Analysis-only normalization; original PCM is unchanged
#   - Stereo-aware bitrate floor
#   - Batched Ogg pages
# ============================================================

RATE = 48000
CHANNELS = 2
OPUS_APPLICATION_AUDIO = 2049

OPUS_SET_BITRATE_REQUEST = 4002
OPUS_SET_MAX_BANDWIDTH_REQUEST = 4004
OPUS_SET_VBR_REQUEST = 4006
OPUS_SET_COMPLEXITY_REQUEST = 4010
OPUS_SET_SIGNAL_REQUEST = 4024
OPUS_SET_FORCE_CHANNELS_REQUEST = 4022
OPUS_GET_LOOKAHEAD_REQUEST = 4027

OPUS_AUTO = -1000
OPUS_SIGNAL_VOICE = 3001
OPUS_SIGNAL_MUSIC = 3002
OPUS_BANDWIDTH_FULLBAND = 1105

OGG_OVERHEAD_ESTIMATE_BPS = 250
MIN_BITRATE_BPS = 6000
MAX_BITRATE_BPS = 512000
MAX_PACKET_BYTES = 1275 * 3


# ============================================================
# libopus loading and setup
# ============================================================

def load_opus(dll_path=None):
    if dll_path:
        candidates = [dll_path]
    else:
        candidates = [
            os.path.join(
                os.path.dirname(os.path.abspath(__file__)),
                "opus.dll",
            ),
            "opus.dll",
        ]

    last_error = None

    for candidate in candidates:
        try:
            return ctypes.CDLL(candidate)
        except OSError as exc:
            last_error = exc

    raise RuntimeError(
        "Could not load opus.dll. Put it beside opusplus.py, "
        "add it to PATH, or specify --opus-dll.\n"
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

    # opus_encoder_ctl is variadic. Declare the fixed arguments only.
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
            f"libopus encoder control {request} failed "
            f"with code {result}"
        )


def create_encoder(opus, args, cbr=False):
    error = ctypes.c_int(0)

    encoder = opus.opus_encoder_create(
        RATE,
        CHANNELS,
        OPUS_APPLICATION_AUDIO,
        ctypes.byref(error),
    )

    if not encoder or error.value != 0:
        raise RuntimeError(
            f"Could not create Opus encoder (error {error.value})"
        )

    try:
        # Pass 1 uses VBR. Pass 2 disables VBR and updates the target
        # bitrate between frames to follow the analyzed pattern.
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

        opus_ctl(opus, encoder, OPUS_SET_SIGNAL_REQUEST, signal)

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
# Ogg CRC and writer
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
    length = len(packet)
    return [255] * (length // 255) + [length % 255]


class OggWriter:
    """
    Writes Ogg Opus pages.

    Audio packets are batched into pages, normally around one second
    per page, subject to the Ogg page segment limit.
    """

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
        body_parts = []

        for packet in packets:
            lacing.extend(packet_lacing(packet))
            body_parts.append(packet)

        if len(lacing) > 255:
            raise ValueError("Ogg page exceeds the 255-segment limit")

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
        page = header + b"".join(body_parts)

        struct.pack_into("<I", page, 22, ogg_crc(page))

        self.file.write(page)
        self.sequence += 1

    def write_header_packet(self, packet, granule=0, header_type=0):
        self.write_page([packet], granule, header_type)

    def add_audio_packet(self, packet, granule, duration_samples):
        segments = len(packet_lacing(packet))

        if (
            self.pending_packets
            and self.pending_segments + segments > 200
        ):
            self.flush_audio()

        self.pending_packets.append(packet)
        self.pending_segments += segments
        self.pending_granule = granule
        self.pending_samples += duration_samples

        if self.pending_samples >= RATE:
            self.flush_audio()

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
        # The final audio page must carry the EOS flag.
        self.flush_audio(eos=True)


def make_opus_head(pre_skip):
    # Opus mapping family 0 supports mono and stereo.
    return (
        b"OpusHead"
        + struct.pack(
            "<BBHIhB",
            1,
            CHANNELS,
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


# ============================================================
# Input decoding
# ============================================================

def find_ffmpeg():
    path = shutil.which("ffmpeg")

    if not path:
        raise RuntimeError(
            "ffmpeg was not found on PATH. Install FFmpeg or add its "
            "bin directory to PATH."
        )

    return path


def decode_input(path):
    command = [
        find_ffmpeg(),
        "-hide_banner",
        "-loglevel", "error",
        "-i", path,
        "-map", "0:a:0",
        "-vn",
        "-sn",
        "-dn",
        "-f", "s16le",
        "-acodec", "pcm_s16le",
        "-ar", str(RATE),
        "-ac", str(CHANNELS),
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
        return_code = process.wait()

        if return_code != 0:
            raise RuntimeError(
                "FFmpeg decoding failed:\n"
                + stderr.decode("utf-8", errors="replace")
            )

    finally:
        if process.poll() is None:
            process.kill()
            process.wait()

    pcm = pcm[:len(pcm) - (len(pcm) % (CHANNELS * 2))]

    if not pcm:
        raise RuntimeError("FFmpeg decoded no audio samples")

    return np.frombuffer(bytes(pcm), dtype=np.int16).copy()


def make_frames(pcm, frame_samples):
    frame_values = frame_samples * CHANNELS
    frames = []

    for start in range(0, len(pcm), frame_values):
        chunk = pcm[start:start + frame_values]
        actual_values = len(chunk)

        if actual_values < frame_values:
            padded = np.zeros(frame_values, dtype=np.int16)
            padded[:actual_values] = chunk
            chunk = padded

        actual_samples = actual_values // CHANNELS
        frames.append((chunk, actual_samples))

    return frames


# ============================================================
# Hybrid activity and transient analysis
# ============================================================

def analyze_frame(frame):
    """
    Analysis only. Never modifies the PCM sent to libopus.

    Returns:
        activity: detail/complexity score
        transient: attack/rapid-change score
    """

    x = np.asarray(frame, dtype=np.float32).reshape(-1, CHANNELS)

    if len(x) < 3:
        return 0.45, 0.0

    rms = float(np.sqrt(np.mean(x * x) + 1.0))

    delta = np.diff(x, axis=0)
    mean_delta = float(np.mean(np.abs(delta)) + 1.0)

    relative_delta = mean_delta / max(rms, 16.0)

    level_activity = math.log10(1.0 + rms / 500.0)
    detail_activity = math.log10(1.0 + relative_delta * 20.0)

    activity = (
        0.20 * level_activity
        + 0.80 * detail_activity
    )

    activity = max(0.0, min(activity, 2.5))

    # Second differences help identify sharp attacks.
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

    transient = (
        0.65 * curvature_score
        + 0.35 * jump_score
    )

    return activity, max(0.0, min(transient, 1.0))


# ============================================================
# Pass 1 bitrate controller
# ============================================================

class BitrateController:
    def __init__(self, target_kbps, intensity):
        self.target_bps = float(target_kbps) * 1000.0
        self.intensity = int(intensity)

        self.packet_target_bps = max(
            MIN_BITRATE_BPS,
            self.target_bps - OGG_OVERHEAD_ESTIMATE_BPS,
        )

        self.total_packet_bytes = 0
        self.total_samples = 0

    def bitrate_for_frame(self, frame):
        if self.intensity == 0:
            return int(self.packet_target_bps)

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

            target_bytes = (
                self.packet_target_bps * elapsed / 8.0
            )

            ratio = target_bytes / max(
                self.total_packet_bytes, 1
            )

            feedback_factor = max(
                0.70,
                min(1.40, ratio ** 0.35),
            )

        requested = (
            self.packet_target_bps
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
# Encode a pass
# ============================================================

def encode_packets(opus, args, frames, frame_samples, cbr, targets=None):
    encoder, pre_skip = create_encoder(opus, args, cbr=cbr)

    controller = BitrateController(
        args.bitrate,
        args.vbr_intensity,
    )

    packets = []
    actual_bitrates = []
    packet_bytes_total = 0

    output_buffer = (ctypes.c_ubyte * MAX_PACKET_BYTES)()

    try:
        for i, (frame, actual_samples) in enumerate(frames):
            if targets is None:
                requested = controller.bitrate_for_frame(frame)
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
                    f"opus_encode failed with error {packet_length}"
                )

            packet = bytes(output_buffer[:packet_length])

            packets.append(packet)
            packet_bytes_total += packet_length

            # Actual packet rate based on nominal frame duration.
            actual_bitrates.append(
                packet_length * 8.0 * RATE / frame_samples
            )

            controller.record_packet(
                packet_length,
                frame_samples,
            )

    finally:
        opus.opus_encoder_destroy(encoder)

    return (
        packets,
        actual_bitrates,
        packet_bytes_total,
        pre_skip,
    )


# ============================================================
# Write Ogg Opus output
# ============================================================

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

            granule = pre_skip + decoded_end

            ogg.add_audio_packet(
                packet,
                granule,
                frame_samples,
            )

        ogg.finish()


# ============================================================
# Automatic Pass 2 pattern transformation
# ============================================================

def transform_pass1_pattern(
    pass1_bitrates,
    adjustment,
    offset_bps,
    multiplier,
):
    targets = []

    for rate in pass1_bitrates:
        if adjustment == "add":
            value = rate + offset_bps
        else:
            value = rate * multiplier

        value = int(
            max(
                MIN_BITRATE_BPS,
                min(MAX_BITRATE_BPS, round(value)),
            )
        )

        targets.append(value)

    return targets


def measure_file(path, duration):
    size = os.path.getsize(path)

    rate = (
        size * 8.0 / duration
        if duration > 0
        else 0.0
    )

    return size, rate


# ============================================================
# Two-pass encoding
# ============================================================

def run_two_pass(
    opus,
    args,
    frames,
    frame_samples,
    total_samples,
):
    duration = total_samples / RATE
    target_file_bps = args.bitrate * 1000.0

    print("\nPASS 1: adaptive VBR analysis")
    print("Encoding and measuring actual packet bitrates...")

    (
        pass1_packets,
        pass1_bitrates,
        pass1_bytes,
        pre_skip,
    ) = encode_packets(
        opus,
        args,
        frames,
        frame_samples,
        cbr=False,
    )

    mean_pass1 = float(np.mean(pass1_bitrates))
    pass1_packet_rate = pass1_bytes * 8.0 / duration

    print(
        f"Pass 1 audio packet rate: "
        f"{pass1_packet_rate / 1000:.3f} kbps"
    )

    print(
        f"Pass 1 mean frame rate:   "
        f"{mean_pass1 / 1000:.3f} kbps"
    )

    print(f"Pass 1 frames analyzed:   {len(pass1_bitrates)}")

    del pass1_packets

    # Estimate the initial audio-packet target. Later corrections use
    # the measured whole-file rate, including actual Ogg overhead.
    target_packet_bps = max(
        MIN_BITRATE_BPS,
        target_file_bps - OGG_OVERHEAD_ESTIMATE_BPS,
    )

    if args.pass_adjust == "add":
        offset_bps = target_packet_bps - mean_pass1
        multiplier = 1.0
    else:
        offset_bps = 0.0

        multiplier = (
            target_packet_bps / mean_pass1
            if mean_pass1 > 0
            else 1.0
        )

    print(
        f"\nPASS 2: automatic {args.pass_adjust} adjustment"
    )

    best_error = float("inf")
    best_path = args.output + ".opusplus-best.tmp"
    best_info = None

    try:
        for iteration in range(args.pass_iterations):
            targets = transform_pass1_pattern(
                pass1_bitrates,
                args.pass_adjust,
                offset_bps,
                multiplier,
            )

            (
                packets,
                second_rates,
                packet_bytes,
                second_pre_skip,
            ) = encode_packets(
                opus,
                args,
                frames,
                frame_samples,
                cbr=True,
                targets=targets,
            )

            write_ogg(
                args.output,
                packets,
                second_pre_skip,
                total_samples,
                frame_samples,
            )

            file_size, file_rate = measure_file(
                args.output,
                duration,
            )

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

                shutil.copyfile(
                    args.output,
                    best_path,
                )

                best_info = (
                    file_size,
                    file_rate,
                    packet_rate,
                    iteration + 1,
                )

            if abs(error_bps) <= args.pass_tolerance_kbps * 1000.0:
                break

            # Automatically correct the pattern using the actual output
            # file rate, rather than the nominal encoder targets.
            if args.pass_adjust == "add":
                offset_bps += error_bps
            else:
                if file_rate > 0:
                    multiplier *= target_file_bps / file_rate

            offset_bps = max(
                -MAX_BITRATE_BPS,
                min(MAX_BITRATE_BPS, offset_bps),
            )

            multiplier = max(
                0.05,
                min(20.0, multiplier),
            )

    finally:
        if best_info is not None and os.path.isfile(best_path):
            shutil.copyfile(best_path, args.output)

        try:
            os.remove(best_path)
        except OSError:
            pass

    if best_info is None:
        file_size, file_rate = measure_file(
            args.output,
            duration,
        )

        packet_rate = 0.0
        best_iteration = 1

    else:
        (
            file_size,
            file_rate,
            packet_rate,
            best_iteration,
        ) = best_info

    return {
        "duration": duration,
        "file_size": file_size,
        "file_rate": file_rate,
        "packet_rate": packet_rate,
        "frames": len(frames),
        "pass1_mean": mean_pass1,
        "best_iteration": best_iteration,
    }


# ============================================================
# Single-pass encoding
# ============================================================

def run_single_pass(
    opus,
    args,
    frames,
    frame_samples,
    total_samples,
):
    duration = total_samples / RATE

    (
        packets,
        rates,
        packet_bytes,
        pre_skip,
    ) = encode_packets(
        opus,
        args,
        frames,
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

    size, file_rate = measure_file(
        args.output,
        duration,
    )

    return {
        "duration": duration,
        "file_size": size,
        "file_rate": file_rate,
        "packet_rate": packet_bytes * 8.0 / duration,
        "frames": len(frames),
        "pass1_mean": float(np.mean(rates)),
        "best_iteration": 1,
    }


# ============================================================
# Main encoder
# ============================================================

def encode(args):
    if not os.path.isfile(args.input):
        raise FileNotFoundError(
            f"Input file not found: {args.input}"
        )

    if not 6 <= args.bitrate <= 512:
        raise ValueError(
            "Target bitrate must be between 6 and 512 kbps"
        )

    if not 0 <= args.complexity <= 10:
        raise ValueError(
            "Complexity must be between 0 and 10"
        )

    if not 0 <= args.vbr_intensity <= 10:
        raise ValueError(
            "VBR intensity must be between 0 and 10"
        )

    if not 1 <= args.pass_iterations <= 10:
        raise ValueError(
            "--pass-iterations must be between 1 and 10"
        )

    if args.pass_tolerance_kbps < 0:
        raise ValueError(
            "--pass-tolerance-kbps cannot be negative"
        )

    frame_samples = int(
        RATE * args.frame_ms / 1000.0
    )

    if frame_samples <= 0:
        raise ValueError("Invalid frame duration")

    started = time.time()

    print(f"Input:          {args.input}")
    print(f"Output:         {args.output}")
    print("Decoding input to stereo 48 kHz PCM...")

    pcm = decode_input(args.input)

    total_samples = len(pcm) // CHANNELS
    frames = make_frames(pcm, frame_samples)

    opus = configure_opus_functions(
        load_opus(args.opus_dll)
    )

    if hasattr(opus, "opus_get_version_string"):
        version = opus.opus_get_version_string().decode(
            "ascii",
            errors="replace",
        )
    else:
        version = "unknown"

    print(f"Encoder:        OpusPlus")
    print(f"libopus:        {version}")
    print(f"Target:         {args.bitrate:.2f} kbps total file")
    print(f"Pass mode:      {args.pass_mode}")
    print(f"Pass adjustment:{' ' + args.pass_adjust}")
    print(f"VBR intensity:  {args.vbr_intensity}/10")
    print(f"Complexity:     {args.complexity}/10")
    print(f"Signal mode:    {args.mode}")
    print("Channels:       stereo")
    print(f"Sample rate:    {RATE} Hz")
    print(f"Frame duration: {args.frame_ms:g} ms")
    print(f"Duration:       {total_samples / RATE:.3f} sec")
    print(f"Frames:         {len(frames)}")

    if args.pass_mode == "two-pass":
        result = run_two_pass(
            opus,
            args,
            frames,
            frame_samples,
            total_samples,
        )
    else:
        print("\nSingle-pass adaptive VBR encoding...")

        result = run_single_pass(
            opus,
            args,
            frames,
            frame_samples,
            total_samples,
        )

    print("\nOpusPlus encoding complete.")
    print(f"Duration:              {result['duration']:.3f} sec")
    print(f"Frames encoded:        {result['frames']}")
    print(
        f"Audio packet rate:     "
        f"{result['packet_rate'] / 1000:.3f} kbps"
    )
    print(
        f"Total Ogg file size:   "
        f"{result['file_size']:,} bytes"
    )
    print(
        f"Rate including Ogg:    "
        f"{result['file_rate'] / 1000:.3f} kbps"
    )
    print(f"Target bitrate:        {args.bitrate:.3f} kbps")
    print(
        f"Difference from target:"
        f" {result['file_rate'] / 1000 - args.bitrate:+.3f} kbps"
    )

    if args.pass_mode == "two-pass":
        print(
            f"Best output iteration: "
            f"{result['best_iteration']}"
        )

    print(
        f"Elapsed time:          "
        f"{time.time() - started:.2f} sec"
    )


# ============================================================
# CLI
# ============================================================

def main():
    parser = argparse.ArgumentParser(
        description=(
            "OpusPlus: adaptive Opus encoding with automatic "
            "two-pass bitrate-pattern adjustment."
        )
    )

    parser.add_argument(
        "input",
        help="Input audio file",
    )

    parser.add_argument(
        "output",
        help="Output .opus file",
    )

    parser.add_argument(
        "-b",
        "--bitrate",
        type=float,
        default=32,
        help="Target total-file bitrate in kbps (default: 32)",
    )

    parser.add_argument(
        "--pass-mode",
        choices=["live", "two-pass"],
        default="two-pass",
        help="Encoding mode (default: two-pass)",
    )

    parser.add_argument(
        "--pass-adjust",
        choices=["add", "multiply"],
        default="multiply",
        help=(
            "Pass 2 pattern transformation: add an automatic offset "
            "or apply an automatic multiplier (default: multiply)"
        ),
    )

    parser.add_argument(
        "--pass-iterations",
        type=int,
        default=4,
        help="Maximum Pass 2 correction iterations (default: 4)",
    )

    parser.add_argument(
        "--pass-tolerance-kbps",
        type=float,
        default=0.5,
        help=(
            "Stop when output is within this target error in kbps "
            "(default: 0.5)"
        ),
    )

    parser.add_argument(
        "--vbr-intensity",
        type=int,
        choices=range(0, 11),
        default=5,
        help="Activity/transient allocation strength, 0–10 (default: 5)",
    )

    parser.add_argument(
        "--complexity",
        type=int,
        choices=range(0, 11),
        default=10,
        help="libopus complexity, 0–10 (default: 10)",
    )

    parser.add_argument(
        "--frame-ms",
        type=float,
        choices=[2.5, 5, 10, 20, 40, 60],
        default=20,
        help="Opus frame duration in milliseconds (default: 20)",
    )

    parser.add_argument(
        "--mode",
        choices=["auto", "music", "voice"],
        default="auto",
        help="Encoder signal hint (default: auto)",
    )

    parser.add_argument(
        "--opus-dll",
        default=None,
        help="Path to opus.dll if not beside opusplus.py or on PATH",
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
    try:
        main()
    except Exception:
        import traceback
        traceback.print_exc()
        sys.exit(1)
