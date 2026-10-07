#!/bin/zsh
set -e
cd "$(dirname "$0")"

echo "Oil Red O Aorta QC — macOS v4"

if ! command -v python3 >/dev/null 2>&1; then
  osascript -e 'display dialog "未检测到 Python 3。请先安装 Python 3.10–3.12。" buttons {"OK"} default button "OK" with icon stop'
  exit 1
fi
if [ ! -d ".venv" ]; then python3 -m venv .venv; fi
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
python OilRed_Aorta_QC_macOS_v4.py
