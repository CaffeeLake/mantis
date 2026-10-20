#!/bin/bash
# Copyright 2026 Google LLC
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     https://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

set -e

if [ -z "$1" ]; then
    echo "Usage: ./run.sh <directory_or_file_to_scan> [extra_args...]"
    exit 1
fi

TARGET="$1"
# Anchor relative path to absolute without dereferencing symlinks.
# POSIX `case` is used instead of bash's [[ ]] so anchoring still applies when the
# script is invoked as `sh run.sh`, which would otherwise skip this branch entirely.
if [ -e "$TARGET" ] || [ -L "$TARGET" ]; then
    case "$TARGET" in
        /*) ;;
        *) TARGET="$(pwd)/$TARGET" ;;
    esac
fi
shift || true

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
cd "$SCRIPT_DIR"

if [ ! -d ".venv" ]; then
    echo "Virtual environment not found. Please run ./install.sh first."
    exit 1
fi

source .venv/bin/activate
export PYTHONUNBUFFERED=1
python3 scripts/launch.py "$TARGET" "$@"
