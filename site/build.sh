#!/usr/bin/env bash
# Prepare the deployable site.
#
# 1. Copy the canonical markdown in. The site ships no prose of its own:
#    README.md and docs/*.md stay the single source of truth and the pages
#    render them at load, so the site cannot drift from the repository.
#
# 2. Emit a real HTML file for each doc route. An earlier version used a
#    vercel.json rewrite (/docs/:slug -> /docs.html) instead, but cleanUrls
#    308-redirects /docs.html to /docs, which broke the chain and 404'd every
#    doc page. Real files are one hop shorter and need no routing rules at all.
set -euo pipefail

root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
pub="$root/site/public"
out="$pub/content"

mkdir -p "$out" "$pub/docs"
cp "$root/README.md"            "$out/readme.md"
cp "$root/docs/autograd.md"     "$out/autograd.md"
cp "$root/docs/architecture.md" "$out/architecture.md"

for slug in readme autograd architecture; do
  cp "$pub/docs.html" "$pub/docs/$slug.html"
done

echo "site/build.sh:"
echo "  content:  $(ls -1 "$out" | tr '\n' ' ')"
echo "  routes:   /docs $(ls -1 "$pub/docs" | sed 's/\.html//' | sed 's|^|/docs/|' | tr '\n' ' ')"
