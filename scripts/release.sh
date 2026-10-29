#!/usr/bin/env bash
# Cut a release: bump the version in pyproject.toml and ebt_rust/Cargo.toml, commit, tag vX.Y.Z
# and push. The tag push triggers .github/workflows/release.yml, which builds the sdist and the
# wheels and publishes them to PyPI.
#
#   scripts/release.sh 0.1.1
#   RELEASE_DRY_RUN=1 scripts/release.sh 0.1.1   # only bump the files, no git
set -euo pipefail

VERSION="${1:?usage: scripts/release.sh X.Y.Z}"
[[ "$VERSION" =~ ^[0-9]+\.[0-9]+\.[0-9]+([a-z0-9.]*)?$ ]] || { echo "bad version: $VERSION"; exit 1; }
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT"

if [ -z "${RELEASE_DRY_RUN:-}" ]; then
    [ -z "$(git status --porcelain)" ] || { echo "working tree not clean; commit or stash first"; exit 1; }
    git rev-parse -q --verify "refs/tags/v$VERSION" >/dev/null && { echo "tag v$VERSION already exists"; exit 1; }
fi

python3 - "$VERSION" <<'EOF'
import re, sys, pathlib
v = sys.argv[1]
for path in ('pyproject.toml', 'ebt_rust/Cargo.toml'):
    p = pathlib.Path(path)
    s, n = re.subn(r'^(version\s*=\s*")[^"]*(")', lambda m: m.group(1) + v + m.group(2), p.read_text(), count=1, flags=re.M)
    assert n == 1, f'no version line in {path}'
    p.write_text(s)
    print(f'{path}: version = "{v}"')
EOF
# refresh the ebt_rust entry in Cargo.lock without touching dependency versions
(cd ebt_rust && (cargo metadata --format-version 1 --offline >/dev/null 2>&1 || cargo metadata --format-version 1 >/dev/null))

if [ -n "${RELEASE_DRY_RUN:-}" ]; then
    echo "dry run: files bumped, nothing committed"
    exit 0
fi

git add pyproject.toml ebt_rust/Cargo.toml ebt_rust/Cargo.lock
if git diff --cached --quiet; then
    echo "version already $VERSION in the files; tagging the current commit"
else
    git commit -m "Release v$VERSION"
fi
git tag -a "v$VERSION" -m "v$VERSION"
git push
git push origin "v$VERSION"
echo "pushed v$VERSION: the Release workflow is building and publishing it (see the Actions tab)"
