#!/bin/bash
# 文字起こしスクリプトの実行用シェルスクリプト
# sources内のすべてのmp3ファイルを処理し、artifacts内に成果物を保存

# スクリプトのディレクトリを取得
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
SOURCES_DIR="${SCRIPT_DIR}/sources"
ARTIFACTS_DIR="${SCRIPT_DIR}/artifacts"

# sourcesディレクトリが存在するか確認
if [ ! -d "$SOURCES_DIR" ]; then
    echo "エラー: sourcesディレクトリが見つかりません: $SOURCES_DIR"
    exit 1
fi

# artifactsディレクトリを作成
mkdir -p "$ARTIFACTS_DIR"

# mp3ファイルを検索
echo "sources内のmp3ファイルを検索中..."
shopt -s nullglob
MP3_FILES=("$SOURCES_DIR"/*.mp3)

if [ ${#MP3_FILES[@]} -eq 0 ]; then
    echo "エラー: sourcesディレクトリ内にmp3ファイルが見つかりません"
    exit 1
fi

echo "見つかったmp3ファイル: ${#MP3_FILES[@]}個"
echo ""

# 各mp3ファイルを処理
for mp3_file in "${MP3_FILES[@]}"; do
    # ファイル名（拡張子なし）を取得
    filename=$(basename "$mp3_file")
    basename_no_ext="${filename%.mp3}"

    echo "=========================================="
    echo "処理中: $filename"
    echo "=========================================="

    # artifacts内に出力ディレクトリを作成
    output_dir="${ARTIFACTS_DIR}/${basename_no_ext}"
    mkdir -p "$output_dir"

    echo "出力先: $output_dir"
    echo ""

    # 文字起こしを実行（秒数制限は第1引数で指定可能）
    if [ -n "$1" ]; then
        uv run python transcribe.py "$mp3_file" "$1" --output-dir "$output_dir"
    else
        uv run python transcribe.py "$mp3_file" --output-dir "$output_dir"
    fi

    echo ""
    echo "完了: $filename -> $output_dir"
    echo ""
done

echo "=========================================="
echo "すべての処理が完了しました"
echo "処理したファイル: ${#MP3_FILES[@]}個"
echo "成果物の保存先: $ARTIFACTS_DIR"
echo "=========================================="
