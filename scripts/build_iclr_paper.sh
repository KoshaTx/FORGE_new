#!/usr/bin/env bash
# Build the new ICLR 2027 paper.
#
# The official style files are vendored in manuscript/latex/iclr2027/ so the build never needs
# network access. TEXINPUTS must be absolute: a relative path resolves against the caller's working
# directory, which silently breaks the build from anywhere but manuscript/.
#
#   ./scripts/build_iclr_paper.sh          build
#   ./scripts/build_iclr_paper.sh clean    remove build products
set -euo pipefail

REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
DOC="FORGE_ICLR2027_paper"
STYLE="${REPO}/manuscript/latex/iclr2027"

cd "${REPO}/manuscript"

if [[ "${1:-}" == "clean" ]]; then
  latexmk -C "${DOC}.tex" >/dev/null 2>&1 || true
  rm -f "${DOC}".{aux,log,out,fls,fdb_latexmk,blg,bbl}
  echo "cleaned"
  exit 0
fi

export TEXINPUTS=".:${STYLE}//:"
export BIBINPUTS=".:${STYLE}//:"
export BSTINPUTS=".:${STYLE}//:"

if latexmk -pdf -interaction=nonstopmode -halt-on-error "${DOC}.tex" >/tmp/iclr_build.log 2>&1; then
  pages=$(pdfinfo "${DOC}.pdf" | awk '/^Pages:/ {print $2}')
  # Count \TODO{ USES, not the macro definition, which also contains the string "TODO:".
  todos=$(grep -o '\\TODO{' "${DOC}.tex" | wc -l | tr -d ' ')
  pending=$(grep -o 'PENDING{[^}]*}' "${DOC}.tex" | sort -u | wc -l | tr -d ' ')
  echo "built ${DOC}.pdf  —  ${pages} pages, ${todos} TODO markers, ${pending} distinct pending numbers"
  # Overfull boxes are the usual cause of ICLR margin violations, so surface them rather than
  # leaving them buried in the log.
  if grep -q 'Overfull \\hbox' "${DOC}.log"; then
    echo "warning: overfull hboxes present (possible margin violations):"
    grep 'Overfull \\hbox' "${DOC}.log" | head -5
  fi
else
  echo "BUILD FAILED"
  grep -E '^!|LaTeX Error|Undefined control sequence' "${DOC}.log" | head -10
  exit 1
fi
