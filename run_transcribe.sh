#!/bin/bash
# 文字起こしスクリプトの実行用シェルスクリプト

# 使用方法を表示
if [ $# -eq 0 ]; then
    echo "使用方法: $0 <音声ファイル> [秒数制限]"
    echo ""
    echo "例:"
    echo "  $0 audio.mp3           # 全体を文字起こし"
    echo "  $0 audio.mp3 180       # 最初の3分だけ文字起こし"
    exit 1
fi

# uvを使ってスクリプトを実行
uv run python transcribe.py "$@"
