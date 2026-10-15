#!/bin/sh
# Release a squashed snapshot of $RELEASE_REF (default HEAD) to the public repo.
#
# Each release creates ONE fresh commit on top of public/$PUBLIC_BRANCH whose
# tree is the snapshot, so the public history stays linear and the private
# history is never uploaded. In the snapshot, AGENTS.md is replaced by the
# contents of AGENTS.public.md (the sanitized sample).
#
# Usage: scripts/release-public.sh "release message"
# Env:   PUBLIC_REMOTE (default: public)  PUBLIC_BRANCH (default: main)
#        RELEASE_REF   (default: HEAD)
set -eu

REMOTE=${PUBLIC_REMOTE:-public}
BRANCH=${PUBLIC_BRANCH:-main}
REF=${RELEASE_REF:-HEAD}
MSG=${1:?"usage: $0 <release message>"}

git remote get-url "$REMOTE" >/dev/null 2>&1 || {
    echo "no '$REMOTE' remote; add it first:" >&2
    echo "  git remote add $REMOTE https://github.com/shoutliberte/Suzuku-Engine-q4.git" >&2
    exit 1
}

if [ -n "$(git status --porcelain)" ]; then
    echo "warning: worktree is dirty; snapshot uses the committed state of $REF" >&2
fi

PARENT=$(git ls-remote "$REMOTE" "refs/heads/$BRANCH" | cut -f1)
if [ -n "$PARENT" ]; then
    git fetch -q "$REMOTE" "$BRANCH"
fi

INDEX=$(mktemp)
trap 'rm -f "$INDEX"' EXIT
GIT_INDEX_FILE=$INDEX git read-tree "$REF"
GIT_INDEX_FILE=$INDEX git rm -q --cached --ignore-unmatch AGENTS.md AGENTS.public.md
BLOB=$(git hash-object -w AGENTS.public.md)
GIT_INDEX_FILE=$INDEX git update-index --add --cacheinfo 100644,"$BLOB",AGENTS.md
TREE=$(GIT_INDEX_FILE=$INDEX git write-tree)

SRC=$(git rev-parse --short "$REF")
if [ -n "$PARENT" ]; then
    COMMIT=$(git commit-tree "$TREE" -p "$PARENT" -m "$MSG" -m "Source: q4@$SRC")
else
    COMMIT=$(git commit-tree "$TREE" -m "$MSG" -m "Source: q4@$SRC")
fi

git push "$REMOTE" "$COMMIT:refs/heads/$BRANCH"
git update-ref refs/heads/public-release "$COMMIT"
echo "released $COMMIT -> $REMOTE/$BRANCH (snapshot of $SRC)"
