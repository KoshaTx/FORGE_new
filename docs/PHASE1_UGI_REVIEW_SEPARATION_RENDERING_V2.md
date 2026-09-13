# Common-scale rendering amendment before packet selection

Date: 2026-09-09. This amends only the rendering implementation in
`PHASE1_UGI_REVIEW_SEPARATION_PROTOCOL.md`; the v1 protocol and configuration remain preserved.
No real packet has been selected or rendered and no new human outcomes have been inspected.

Synthetic preflight found that directly using an RDKit flexible canvas with `scalingFactor=24`
does not guarantee an actual 24-pixel-per-coordinate transform. Canvas rounding and fitting can
change the scale between molecules. Prescribing the option alone is insufficient for the
common-scale requirement. The original implementation is therefore not admitted.

Use the same RDKit depiction preparation and drawing options, with these additions:

- Obtain intrinsic dimensions through `GetMolSize` using a flexible sizing drawer.
- Compute the coordinate envelope at the target scale, accounting for the 0.04 padding fraction.
- For each dimension, take the ceiling of the larger intrinsic/envelope requirement and add
  two 36-pixel label margins.
- Draw into that explicit canvas with `fixedBondLength=24`, `scalingFactor=24`,
  `fixedFontSize=18`, `bondLineWidth=2.2`, `padding=0.04`, and metadata disabled.
- Verify the actual transform on every bond using conformer coordinates and `GetDrawCoords`.
  Require scale 24 within a numerical floating-point tolerance, finite canvas/atom bounds and
  unchanged canonical constitution. Fail if a molecule is shrunk to fit; do not loosen this check.

RDKit describes `fixedBondLength` as a cap that can still shrink if the canvas is too small.
The explicit canvas sizing and observed-coordinate check jointly enforce the intended scale;
the option by itself is not treated as proof. See the
[RDKit drawing options](https://www.rdkit.org/docs/source/rdkit.Chem.Draw.rdMolDraw2D.html).

The browser continues to use intrinsic pixels and common pair-level zoom with both complete
graphs retained. All selection seeds, freshness exclusions, populations, randomized assignments,
question fields, abstentions, original promotion gates and interpretation limits are unchanged.
The synthetic failure and corrected preflight are retained with their source and input hashes.
