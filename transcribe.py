#!/usr/bin/env python3
"""
faster-whisper を使用した音声文字起こしスクリプト
タイムスタンプ付きで出力します
"""

import os
import sys
from pathlib import Path

# CUDA ライブラリを事前読み込み
def load_cuda_libraries():
    """CUDA ライブラリを ctypes で事前読み込み"""
    try:
        import ctypes
        import site

        site_packages = site.getsitepackages()

        for sp in site_packages:
            sp_path = Path(sp)
            cublas_lib = sp_path / "nvidia" / "cublas" / "lib" / "libcublas.so.12"
            cudnn_lib = sp_path / "nvidia" / "cudnn" / "lib" / "libcudnn.so.9"

            if cublas_lib.exists():
                ctypes.CDLL(str(cublas_lib))
                print(f"CUDA cuBLAS ライブラリを読み込みました")

            if cudnn_lib.exists():
                ctypes.CDLL(str(cudnn_lib))
                print(f"CUDA cuDNN ライブラリを読み込みました")

            if cublas_lib.exists() and cudnn_lib.exists():
                return True

        print("警告: CUDA ライブラリが見つかりません。")
        return False
    except Exception as e:
        print(f"警告: CUDA ライブラリの読み込みに失敗しました: {e}")
        return False

load_cuda_libraries()

from faster_whisper import WhisperModel
import torch
import torchaudio


def detect_speech_segments_with_vad(audio_path: str) -> list:
    """Silero VADで音声区間を検出

    Returns:
        音声区間のリスト [{"start": 開始秒, "end": 終了秒}, ...]
    """
    print("Silero VADで音声区間を検出中...")

    # Silero VADモデルをロード（ONNXモードでCPU実行）
    model, utils = torch.hub.load(
        repo_or_dir='snakers4/silero-vad',
        model='silero_vad',
        force_reload=False,
        onnx=True
    )

    (get_speech_timestamps, _, read_audio, *_) = utils

    # 音声ファイルを読み込み（16kHzにリサンプリング）
    print("VAD: 音声ファイル読み込み中...")
    wav = read_audio(audio_path, sampling_rate=16000)
    print(f"VAD: 音声読み込み完了 ({len(wav) / 16000:.1f}秒)")

    # 音声区間を検出
    print("VAD: 音声区間検出中...")
    speech_timestamps = get_speech_timestamps(
        wav,
        model,
        threshold=0.5,
        min_silence_duration_ms=1000,
        min_speech_duration_ms=250,
        sampling_rate=16000
    )

    # サンプル数を秒数に変換
    speech_segments = []
    for ts in speech_timestamps:
        start_sec = ts['start'] / 16000.0
        end_sec = ts['end'] / 16000.0
        speech_segments.append({"start": start_sec, "end": end_sec})

    print(f"VAD: 検出完了 - {len(speech_segments)}個の音声区間")
    return speech_segments


def format_timestamp(seconds: float) -> str:
    """秒数を HH:MM:SS 形式に変換"""
    hours = int(seconds // 3600)
    minutes = int((seconds % 3600) // 60)
    secs = int(seconds % 60)
    return f"{hours:02d}:{minutes:02d}:{secs:02d}"


def transcribe(audio_path: str, model_size: str = "large-v3-turbo", duration_limit: int = None, output_dir: str = None) -> None:
    """音声ファイルを文字起こしする

    Args:
        audio_path: 音声ファイルのパス
        model_size: Whisperモデルのサイズ
        duration_limit: 文字起こしする秒数の上限 (Noneの場合は全体)
        output_dir: 出力ディレクトリ (Noneの場合は音声ファイルと同じディレクトリ)
    """
    audio_file = Path(audio_path)
    if not audio_file.exists():
        print(f"エラー: ファイルが見つかりません: {audio_path}")
        sys.exit(1)

    print(f"モデル読み込み中: {model_size}")
    print(f"デバイス: GPU (CUDA)")
    model = WhisperModel(model_size, device="cuda", compute_type="float16")

    print(f"文字起こし中: {audio_file.name}")
    if duration_limit:
        print(f"制限時間: 最初の{duration_limit}秒")

    # ステップ1: Whisperで文字起こし（VAD有効）
    transcribe_kwargs = {
        "language": "ja",
        "beam_size": 5,
        "vad_filter": True,
    }
    if duration_limit:
        transcribe_kwargs["clip_timestamps"] = [0, duration_limit]

    segments, info = model.transcribe(audio_path, **transcribe_kwargs)

    print(f"\n検出言語: {info.language} (確率: {info.language_probability:.2%})")
    print(f"音声長: {info.duration:.2f}秒")

    # 処理する時間を決定
    total_duration = duration_limit if duration_limit else info.duration

    # ステップ2: Whisperセグメントを収集（進捗表示付き）
    print("\nWhisperセグメント収集中...")
    whisper_segments = []
    last_progress_percent = -1

    for segment in segments:
        seg_data = {
            "start": segment.start,
            "end": segment.end,
            "text": segment.text.strip()
        }
        whisper_segments.append(seg_data)

        # タイムスタンプと発話内容を表示
        start_str = format_timestamp(segment.start)
        end_str = format_timestamp(segment.end)
        print(f"[{start_str} -> {end_str}] {seg_data['text']}")

        # 進捗状況を表示（1%ごと）
        progress = int((segment.end / total_duration) * 100)
        if progress != last_progress_percent:
            elapsed_time = format_timestamp(segment.end)
            total_time = format_timestamp(total_duration)
            print(f"  → 進捗: {progress}% ({elapsed_time} / {total_time}) - {len(whisper_segments)}セグメント")
            last_progress_percent = progress

    print(f"\nWhisperセグメント数: {len(whisper_segments)}")

    # ステップ3: Silero VADで音声区間を検出
    vad_segments = detect_speech_segments_with_vad(audio_path)

    print("\n" + "=" * 60)
    print("VAD区間に基づいてセグメントをグルーピング")
    print("=" * 60 + "\n")

    # ステップ4: VAD区間に基づいてWhisperセグメントをグルーピング
    def find_vad_segment(timestamp, vad_segments):
        """タイムスタンプが属するVAD区間のインデックスを返す"""
        for i, vad_seg in enumerate(vad_segments):
            if vad_seg["start"] <= timestamp <= vad_seg["end"]:
                return i
        return None

    output_lines = []
    vad_grouped_segments = {}  # VAD区間インデックス -> Whisperセグメントリスト

    for ws in whisper_segments:
        # セグメントの中間点がどのVAD区間に属するか判定
        mid_point = (ws["start"] + ws["end"]) / 2
        vad_idx = find_vad_segment(mid_point, vad_segments)

        if vad_idx is not None:
            if vad_idx not in vad_grouped_segments:
                vad_grouped_segments[vad_idx] = []
            vad_grouped_segments[vad_idx].append(ws)

    # ステップ5: 各VAD区間内のセグメントを結合して出力
    print("\n" + "=" * 60)
    print("文字起こし結果 (VAD区間ごと)")
    print("=" * 60 + "\n")

    segment_count = 0
    for vad_idx in sorted(vad_grouped_segments.keys()):
        vad_seg = vad_segments[vad_idx]
        whisper_segs = vad_grouped_segments[vad_idx]

        # VAD区間内のWhisperセグメントを結合
        combined_text = " ".join([seg["text"] for seg in whisper_segs])
        start_time = whisper_segs[0]["start"]
        end_time = whisper_segs[-1]["end"]

        start_str = format_timestamp(start_time)
        end_str = format_timestamp(end_time)
        line = f"[{start_str} -> {end_str}] {combined_text}"

        print(f"VAD区間#{vad_idx + 1} ({len(whisper_segs)}個のセグメントを結合)")
        print(f"  VAD: [{format_timestamp(vad_seg['start'])} -> {format_timestamp(vad_seg['end'])}]")
        print(f"  結果: {line}\n")

        output_lines.append(line)
        segment_count += 1

    print(f"=" * 60)
    print(f"VAD区間数: {len(vad_segments)}")
    print(f"Whisperセグメント数: {len(whisper_segments)}")
    print(f"最終セグメント数: {segment_count}")

    # 完了メッセージ
    print(f"\n" + "=" * 60)
    print(f"文字起こし完了: {segment_count}セグメント処理しました")
    print("=" * 60)

    # 結果をファイルに保存
    if output_dir:
        output_dir_path = Path(output_dir)
        output_dir_path.mkdir(parents=True, exist_ok=True)
        output_path = output_dir_path / audio_file.with_suffix(".txt").name
    else:
        output_path = audio_file.with_suffix(".txt")

    with open(output_path, "w", encoding="utf-8") as f:
        f.write(f"ファイル: {audio_file.name}\n")
        f.write(f"モデル: {model_size}\n")
        f.write(f"言語: {info.language}\n")
        f.write(f"音声長: {info.duration:.2f}秒\n")
        if duration_limit:
            f.write(f"処理時間: 最初の{duration_limit}秒\n")
        f.write(f"セグメント数: {segment_count}\n")
        f.write("\n" + "=" * 60 + "\n\n")
        f.write("\n".join(output_lines))

    print(f"\n結果を保存しました: {output_path}")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="faster-whisperを使用した音声文字起こし")
    parser.add_argument("audio_file", help="音声ファイルのパス")
    parser.add_argument("duration_limit", type=int, nargs="?", help="文字起こしする秒数の上限")
    parser.add_argument("--output-dir", "-o", help="出力ディレクトリ")

    args = parser.parse_args()

    transcribe(args.audio_file, duration_limit=args.duration_limit, output_dir=args.output_dir)
