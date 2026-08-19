#!/usr/bin/env bash
# Build a self-contained zip of the ICLR paper for upload to Overleaf.
#
# Two things make this non-trivial and are the reason this is a script rather than a one-off zip.
#
#   1. The style files are vendored under manuscript/latex/iclr2027/ and found locally only because
#      scripts/build_iclr_paper.sh sets TEXINPUTS. Overleaf sets no TEXINPUTS, and does not reliably
#      search subdirectories for .sty/.bst, so the style files are FLATTENED to the archive root
#      next to the main .tex.
#
#   2. \graphicspath is {{figures/}} and the generated .tex files are \input by bare name, so both
#      must sit at the archive root too. Only the figures the paper actually includes are shipped;
#      the archive is not a copy of the figures directory.
#
# The bundle is built from the tracked sources, then compiled inside the staging directory with no
# TEXINPUTS set, which is the only honest test that Overleaf will compile it.
set -euo pipefail

REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
DOC="FORGE_ICLR2027_paper"
STAGE="$(mktemp -d)"
OUT="${REPO}/manuscript/${DOC}_overleaf.zip"
trap 'rm -rf "${STAGE}"' EXIT

cd "${REPO}/manuscript"

# --- main source and the generated .tex files it \inputs ---------------------
cp "${DOC}.tex" "${STAGE}/"
for f in generated_supplement.tex generated_product_structures.tex generated_appendix_tables.tex; do
  [[ -f "$f" ]] && cp "$f" "${STAGE}/"
done

# --- bibliography -----------------------------------------------------------
cp forge.bib "${STAGE}/"

# --- style files, flattened to the root so Overleaf finds them --------------
for f in latex/iclr2027/*.sty latex/iclr2027/*.bst latex/iclr2027/*.tex; do
  [[ -f "$f" ]] && cp "$f" "${STAGE}/"
done

# --- only the figures the document actually includes ------------------------
# Downsampled and alpha-flattened on the way in. The originals are ~3900 px wide and are placed at
# textwidth, about 5.5 in, which is roughly 700 DPI. That is five times more than print needs, and
# every pass re-embeds it: a full pdflatex pass costs about 6.9 s against 0.8 s in draftmode, so the
# images are essentially the entire compile time and the reason Overleaf's free-plan timeout is hit.
# They also carry an alpha channel nothing uses. 300 DPI on an RGB image is visually identical here
# and compiles in a fraction of the time.
TARGET_DPI=300
mkdir -p "${STAGE}/figures"
grep -oE '\\includegraphics(\[[^]]*\])?\{[^}]*\}' "${DOC}.tex" \
  | sed -E 's/.*\{([^}]*)\}/\1/' | sort -u > "${STAGE}/.wanted"
count=0
while IFS= read -r rel; do
  [[ -z "$rel" ]] && continue
  for cand in "figures/${rel}" "figures/${rel}.pdf" "figures/${rel}.png" "${rel}"; do
    if [[ -f "$cand" ]]; then
      mkdir -p "${STAGE}/figures/$(dirname "${rel}")"
      dest="${STAGE}/figures/${rel}"
      [[ "$cand" == *.png && "$rel" != *.png ]] && dest="${dest}.png"
      [[ "$cand" == *.pdf && "$rel" != *.pdf ]] && dest="${dest}.pdf"
      if [[ "$cand" == *.png ]]; then
        python3 - "$cand" "$dest" "$TARGET_DPI" <<'PYIMG'
import sys
from PIL import Image
src, dst, dpi = sys.argv[1], sys.argv[2], int(sys.argv[3])
im = Image.open(src)
# Flatten any alpha onto white; the figures are line art on white and nothing composites them.
if im.mode in ("RGBA", "LA", "P"):
    im = im.convert("RGBA")
    flat = Image.new("RGB", im.size, (255, 255, 255))
    flat.paste(im, mask=im.split()[-1])
    im = flat
else:
    im = im.convert("RGB")
# Cap the long edge at what the target DPI needs for a 5.5 in text column, plus headroom for the
# taller portrait pages, which are placed at full width and so are bounded by width, not height.
cap = int(5.5 * dpi * 1.05)
if max(im.size) > cap:
    scale = cap / max(im.size)
    im = im.resize((max(1, round(im.width * scale)), max(1, round(im.height * scale))),
                   Image.LANCZOS)
im.save(dst, "PNG", optimize=True, dpi=(dpi, dpi))
# Flattening a small RGBA line-art image can grow the file. Keep whichever is smaller; the RGB
# version still wins on compile time, so only fall back when the original is also RGB.
import os, shutil
orig = Image.open(src)
if os.path.getsize(dst) > os.path.getsize(src) and orig.mode == "RGB":
    shutil.copyfile(src, dst)
PYIMG
      else
        cp "$cand" "$dest"
      fi
      count=$((count + 1))
      break
    fi
  done
done < "${STAGE}/.wanted"
rm -f "${STAGE}/.wanted"

cat > "${STAGE}/README.txt" <<'TXT'
FORGE - ICLR 2027 submission source

Main file:    FORGE_ICLR2027_paper.tex
Compiler:     pdfLaTeX
Bibliography: BibTeX

Compile twice. A prebuilt .bbl is included, so bibtex does not need to run.

Style files (iclr2027_conference.sty/.bst, fancyhdr, natbib) sit at the top
level rather than in a subfolder, since Overleaf doesnt search subfolders
for them. Leave them where they are.

These three are written by scripts and get overwriten on regeneration:
  generated_supplement.tex
  generated_product_structures.tex
  generated_appendix_tables.tex
Edit the surrounding prose in the main .tex instead.

Figures here are downsampled to 300 dpi to keep compile time down - originals
are in manuscript/figures/.

Red TODO/PENDING markers are placeholders for the final refit and the
prospective campaign. They are meant to be visible.

Uncomment \iclrfinalcopy for the camera-ready.
TXT

cd "${STAGE}"
echo "compiling in the staging directory with NO TEXINPUTS set, the way Overleaf will:"
if pdflatex -interaction=nonstopmode -halt-on-error "${DOC}.tex" >/dev/null 2>&1 \
   && bibtex "${DOC}" >/dev/null 2>&1 \
   && pdflatex -interaction=nonstopmode -halt-on-error "${DOC}.tex" >/dev/null 2>&1 \
   && pdflatex -interaction=nonstopmode -halt-on-error "${DOC}.tex" >/dev/null 2>&1; then
  pages=$(pdfinfo "${DOC}.pdf" | awk '/^Pages:/ {print $2}')
  undef=$(grep -c "Citation .* undefined\|Reference .* undefined" "${DOC}.log" || true)
  echo "  OK: ${pages} pages, ${undef} undefined citations/references"
  singlepass=$( { /usr/bin/time -p pdflatex -interaction=nonstopmode "${DOC}.tex" >/dev/null; } 2>&1 \
                | awk '/^real/ {print $2}' )
  echo "  one full pass: ${singlepass}s  (Overleaf free plan allows roughly 20s total)"
else
  echo "  FAILED. The archive would not compile on Overleaf. Errors:"
  grep -E '^!|LaTeX Error|Undefined control sequence' "${DOC}.log" | head -8
  exit 1
fi
# Keep the .bbl. Overleaf's free plan timed out before bibtex ran, which left every citation
# undefined; shipping the .bbl means citations resolve on the first pdflatex pass whether or not
# bibtex gets a chance to run.
rm -f "${DOC}".{aux,log,out,blg,pdf,fls,fdb_latexmk,toc}

rm -f "${OUT}"
zip -q -r "${OUT}" . -x '.*'
echo
echo "wrote $(basename "${OUT}")  ($(du -h "${OUT}" | cut -f1), ${count} figures)"
unzip -Z1 "${OUT}" | sort | sed 's/^/  /'
