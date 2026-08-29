# FORGE, Generative and Experimental Perspectives for Biomolecular Design workshop at NeurIPS 2026

`FORGE_NEURIPS2026_paper.tex` is an independent NeurIPS-formatted copy of the current GEM short
paper. The GEM source remains in `paper/v1_neuripsgem/` and is not imported by this build.

Build in place:

```bash
cd paper/v1_neurips
latexmk -pdf -interaction=nonstopmode -halt-on-error FORGE_NEURIPS2026_paper.tex
```

## Venue contract

The manuscript uses the anonymous double-blind workshop configuration:

```tex
\usepackage[dblblindworkshop]{neurips_2026}
\workshoptitle{Generative and Experimental Perspectives for Biomolecular Design}
```

Do not add `final` or `preprint` for an anonymous submission. The first-page notice identifies the
full workshop name and NeurIPS 2026 as the destination.

The unmodified official files were downloaded from:

`https://media.neurips.cc/Conferences/NeurIPS2026/Formatting_Instructions_For_NeurIPS_2026.zip`

| File | SHA-256 |
| --- | --- |
| `neurips_2026.sty` | `c3fc2894e83d2517ca18b66741d6c595986d97957dc08ec08bb2125a7ec4555a` |
| `NEURIPS_2026_reference_template.tex` | `cf4cee7991665306d1daaa3985be4feec7f8889d6d072ffa12f99a8e1537d797` |

## Numerical provenance

Every quantitative statement continues to expand from the generated, hash-pinned files under
`paper/v1/generated/`. Figures resolve through `\graphicspath{{./}{../v1/}}`. Do not manually edit a
reported number in the NeurIPS source; reissue its generating artifact so every paper build remains
consistent.
