#!/bin/bash
# Compile all standalone TikZ .tex files to PNG images
# Requires: pdflatex, pdftoppm (or convert from ImageMagick)
# Install on Fedora: sudo dnf install texlive-scheme-full poppler-utils
# Usage: cd website/web/tikz && bash compile_tikz.sh

OUTDIR="../img"
mkdir -p "$OUTDIR"

for texfile in *.tex; do
  [ -f "$texfile" ] || continue
  base="${texfile%.tex}"
  echo "Compiling $texfile ..."
  pdflatex -interaction=nonstopmode "$texfile" > /dev/null 2>&1
  if [ -f "${base}.pdf" ]; then
    pdftoppm -png -r 300 -singlefile "${base}.pdf" "$OUTDIR/$base"
    echo "  -> $OUTDIR/${base}.png"
  else
    echo "  FAILED: no PDF produced"
  fi
  rm -f "${base}.aux" "${base}.log" "${base}.pdf"
done

echo "Done. PNGs are in $OUTDIR/"
