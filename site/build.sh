#!/usr/bin/env bash
# Copy the canonical markdown into the deployable site.
#
# The docs site has no content of its own on purpose: README.md and docs/*.md
# stay the single source of truth, and the site renders them at page load. That
# means the site can never drift from the repository it documents.
set -euo pipefail

root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
out="$root/site/public/content"

mkdir -p "$out"
cp "$root/README.md"            "$out/readme.md"
cp "$root/docs/autograd.md"     "$out/autograd.md"
cp "$root/docs/architecture.md" "$out/architecture.md"

echo "site/build.sh: copied $(ls -1 "$out" | wc -l | tr -d ' ') markdown files into site/public/content/"
ls -1 "$out" | sed 's/^/  · /'
