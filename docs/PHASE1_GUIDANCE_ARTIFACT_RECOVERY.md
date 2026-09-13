# Guidance artifact recovery — 2026-09-11

## Fifth target material observation qualified — 2026-09-13

The sixth acquisition pass found qualifying **supplier-reported availability for exact stearolic
acid**. [Accel Scientific EC003WL5](https://www.accelsci.com/pro/506-24-1) displays **250 mg, 1 g
and 5 g** packs at **98%** purity, each labeled **Global Stock**. Its displayed SMILES, CAS
506-24-1, formula C18H32O2 and visually inspected structure drawing agree with the frozen graph.
The derived full InChIKey matches `RGTIBVZDHOMOKC-UHFFFAOYSA-N`. Its same-domain
[shipping page](https://www.accelsci.com/support/shipping_methods) explicitly lists the United
States as a destination. These jointly support an exact terminal observation of global stock
offered for US shipment; US warehouse stock, bottle counts and a delivery date remain unconfirmed.

The supplier terms advise inquiry for confirmed current availability and contain inconsistent
website/jurisdiction references. These are preserved in the review; the terms are not used as
stock evidence. The public listing is an observation, not an independently verified inventory or
a guaranteed delivery. The observed 98% grade is not represented as the historical MCE 99.90%
grade. No new purity threshold was introduced: the frozen terminal contract requires exact
identity, recorded item purity, explicit stock/shipping evidence and the existing expiry policy.

The new observation passes the **unchanged C18 terminal contract**. Its original observation
time is **2026-09-13T21:09:01.311431Z** and its 30-day expiry is
**2026-10-13T21:09:01.311431Z**. Combined with the four earlier still-valid observations, this
brings the total to **five of five identified target material identities** at the recorded
assessment. The earlier cyclohexylamine observation retains its shorter September 20 expiry.
This does not renew earlier observations, reassess complete routes or qualify the combined source.

The negative checks are retained. AstaTech H36846's public quantity-one availability endpoint
returns **Back order** for both 1 g and 5 g. Biorbyt's four variants have zero stock and BackOrder
offers. Alfa Chemistry's formula/mass conflict with the target identity. Other listings and access
failures remain unqualified. Across **37** public GET attempts, **23** returned HTTP 200, **six**
HTTP 403, **two** HTTP 501 and **six** transport failures. HTTP success alone is not admission.

The read-only historical source check still fails with
`third_wave_head_terminals input pin mismatch: supplier_page`. No original missing hash was
recovered in this pass. The next work is a separately versioned combined source and exact-route
qualification, followed by structured-value and zero-guidance checks. Model realism has not been
reevaluated or improved by this evidence acquisition.

The acquisition report, terminal observation, review and source check are under
`results/phase1/ugi_guidance_evidence_acquisition_v6/`. Report SHA-256 is
`3dc6468dee3401c1410a341388327d12d93d395d1e2e2b29048f3639f2188176`; terminal-observation SHA-256
is `d9b00e385153a2dc852fa910f90a0ba7584ad9574f6df9c8603cdb97da97c146`.
Exact raw HTML and drawings are preserved in 24 compressed assets. Restore and replay with:

```bash
UV_CACHE_DIR=/private/tmp/forge-uv-cache uv run python \
  results/phase1/ugi_guidance_evidence_acquisition_v6/qualify.py --restore
UV_CACHE_DIR=/private/tmp/forge-uv-cache uv run python \
  results/phase1/ugi_guidance_evidence_acquisition_v6/qualify.py --verify
UV_CACHE_DIR=/private/tmp/forge-uv-cache uv run pytest -q \
  results/phase1/ugi_guidance_evidence_acquisition_v6/test_qualify.py \
  tests/test_current_sampler_preflight.py tests/test_pinned_sources_are_tracked.py
```

All **20 focused tests** pass, including **12** new terminal checks. A development attempt with
`+00:00` timestamps was rejected because the unchanged C18 contract requires `Z`; its source,
observation and failing test report are preserved. Only the new observation serializer was fixed.
No gates, historical pins, model weights or sampling policies changed. No supplier messages,
purchases, candidate selection, training, generation or guidance calls occurred.

Repository-wide validation was rerun: all **30** vendor assets verify; `make test` reports
**2,338 passed, 167 failed, 21 setup errors and 92 skipped/xfail** across 2,618 cases. These totals
match the prior full run; no claim is made that every failure was diagnosed. The new terminal tests
are explicitly covered by the 20-test focused run. Black/Ruff and staged whitespace checks pass.
The global Phase 1 definition of done remains **unmet**. Exact logs and JUnit reports are compressed
under `validation_logs/`; `validation.json` has SHA-256
`14c76e1c12326193e2948c342fc71f12d9a1ba6e5fafa4d2dd575c97b42a2be8`.

## Primary article gap closed — 2026-09-13

The user supplied the original Oppolzer article, DOI **10.1021/jo000463n**. Its exact bytes are
preserved as `results/phase1/ugi_guidance_primary_article_audit_v1/jo000463n.pdf` (107,059 bytes;
SHA-256 `b510d36a265e2c0edcf46ad707c79e82498d0a27d8497168fdc99c4f039262df`). All five article
pages and the relevant supplement pages with their neighbors were rendered and visually inspected.
This resolves the primary-procedure and scheme access gap recorded in the acquisition passes below.
It does not reproduce or replace the missing historical article-transcription hash.

The exact **17-octadecyn-1-ol (7d) → 17-octadecynal (8d)** oxidation has a complete evidence chain:
Scheme 2 on journal page 4766, Table 1 row d on page 4767, the general Parikh–Doering procedure
on page 4769, and the exact substrate/product entries on SI pages 3 and 5. Table 1 and the SI
both report **52% isolated yield** for 8d. The source procedure agrees with the frozen C18
configuration. Its generic **10 mmol** scale is kept separate from the product-specific
**0.72 g** isolation; no unstated compound-specific batch scale is inferred.

The unchanged registry transform has one eligible site and reconstructs exactly the source target.
All **19 heavy atoms** retain their identities, the terminal alkyne and carbon skeleton are preserved,
and the sole bond-order change converts the alcohol to the aldehyde. The substrate/product formulas
are C18H34O and C18H32O; the two-hydrogen difference belongs to the oxidation system, whose reagent
byproducts are not modeled. All **100** seeded atom-order permutations preserve reconstruction and
the repository adjudicator's disposition. The evidence is **`admit_exact`**, basis
**`exact_executed_characterized`**, for this **single L2 step only**.

The saved audit rejects a wrong substrate, a wrong target, an altered source hash, incomplete
procedure linkage, pending visual review, and a substituted yield. **17** existing focused tests and
**five** new adversarial checks pass, as do Black/Ruff and all **30** vendored-asset checks. Read-only
replay reproduces the scientific payload. A development-time pin-wrapper failure is retained with
its original source and result; the corrected caller preserves the strict verifier contract.
The previous failing full-suite result remains current; no new repository-wide pass is claimed.

The current remaining requirements are qualifying stearolic-acid terminal evidence (or a separately
qualified route to that material) and combined source/route/value qualification. This article adds
no availability observation and does not requalify the complete C18 route or authorize production
guidance. Existing gates, historical pins, model weights and sampling policy are unchanged.

```bash
UV_CACHE_DIR=/private/tmp/forge-uv-cache uv run python \
  results/phase1/ugi_guidance_primary_article_audit_v1/audit.py --verify
UV_CACHE_DIR=/private/tmp/forge-uv-cache uv run pytest -q \
  results/phase1/ugi_guidance_primary_article_audit_v1/test_audit.py
```

The article and original SI are retained as exact source assets. The structured review and report
are in `results/phase1/ugi_guidance_primary_article_audit_v1/`; result SHA-256 is
`77a759c2d66166ae72f9bfe7d32a779a06d251ff46d1cbcd5c5cdc7865eea98a` and validation SHA-256 is
`b2987c97c057d64398a8c6dac320ebc5d9354de423a6f85b9e5713b9a30001da`.

## Further public-source search — 2026-09-13

The fifth pass found a new **stearolic-acid global-stock lead**, but adds **zero qualified
terminal observations**. The total remains **four of five identified target materials**. Aaron
Chemicals AR003V4D currently lists 250 mg and 1 g packs at **80%** purity and positive prices.
Its exact graph, full InChIKey and formula match the frozen stearolic-acid identity, and the
supplier drawing was visually inspected. The older search-index entry's 98% packs are not used
as current evidence. The retrieved product and supporting ordering/terms pages do not establish
item-specific US fulfillment. No new purity threshold is introduced; the observed grade is
preserved and is not represented as the historical 99.90% material.

BenchChem's page says in stock but requires a quote for pricing and availability and lacks
item-specific purity and US fulfillment evidence. Apollo's table has stock headings but no pack
rows. ChemScene and TargetMol returned HTTP 403 and 412 respectively; the Sigma/Combi-Blocks
request ended without a response. These are abstentions, not evidence of chemical impossibility.

The author's University of Geneva publication page provides an abstract and the same restricted
repository link. The publisher's open Figshare record contains only `jo000463n_si_001.pdf`, the
supplement already recovered. The original main article remains unavailable through the checked
public sources. An accessible licensed PDF of DOI **10.1021/jo000463n** is needed for the outstanding
procedure/scheme inspection under the chemistry-evidence workflow quoted below. No author contact,
purchase, credential use or access-control bypass occurred.

Eleven public requests produced eight HTTP-200 responses, two HTTP-error responses and one
transport failure. Four descriptive identity/stock controls pass, and the acquisition script
passes Black/Ruff. Sources and the abstaining result replay offline:

```bash
UV_CACHE_DIR=/private/tmp/forge-uv-cache uv run python \
  results/phase1/ugi_guidance_evidence_acquisition_v5/preserve.py --restore
UV_CACHE_DIR=/private/tmp/forge-uv-cache uv run python \
  results/phase1/ugi_guidance_evidence_acquisition_v5/preserve.py --verify
```

The report is `results/phase1/ugi_guidance_evidence_acquisition_v5/result.json`; exact raw assets
are archived under `provenance/recovery/ugi_guidance_evidence_v5/assets/`. Previous evidence and
gates remain unchanged. No route qualification or guidance run occurred, and repository-wide tests
were not rerun for this acquisition-only pass. The prior failing full-suite result below remains
the latest; global Phase 1 definition of done is still unmet.

## Additional supplier qualification — 2026-09-13

The fourth pass qualifies **three additional exact terminal observations**, bringing the total
to **four of the five identified target material identities** at the recorded assessment time.
This updates the availability status in the earlier section below. These are fresh, separately
versioned observations; the original recovery inventory remains **30/55 recovered and 25
unavailable**. No complete routes have been reassessed and guidance remains blocked.

| Material | Qualified fresh evidence | Limit |
| --- | --- | --- |
| Cyclohexylamine | Prior v3 Chem-Impex record, five same-day SKUs | Original seven-day lifetime retained; expires September 20. |
| Heptadecanal | BLD BD305903, positive US stock flags for 250 mg, 1 g and 5 g; 97% | Public flags indicate stock presence, not bottle counts. |
| 1-Bromotridecane | BLD BD57579, positive US stock flags and priced packs at 1 g, 5 g and 25 g; 98% | The 100 g row has no positive public price and is excluded. |
| Protected propargyl alcohol | Fisher AC295060250, 25 g, 98%, positive supplier fulfillment quote | Supplier fulfillment, not Fisher warehouse inventory. |
| Stearolic acid | Identity established; no qualifying current US availability in checked sources | BLD has zero stock flags; the additional Fisher 5 g listing is backordered. |

BLD's ordinary public guest API supplies the dynamic stock observations. Its unchanged display
script maps the US presence flag to the in-stock label; the static HTML placeholder is not used
as stock evidence. Identical duplicate rows are collapsed and conflicting duplicates rejected.
The three new observations preserve the original corresponding **30-day** lifetime. Reverification
reproduces the recorded assessment and does not extend expiry. Fisher's reference ZIP **10001** is
explicitly diagnostic and is not the user's address.

The protected-alcohol identifier discrepancy was resolved without ignoring part of an InChIKey:
the complete supplier key is reproduced by the nonstandard FixedH InChI representation, which
round-trips to the exact frozen graph. The standard key, full supplier key and drawing review are
preserved. All **17** qualification checks pass, covering positive evidence, zero-stock rejection,
incorrect identities, duplicate conflicts, atom ordering and expired/future observations.

Twenty-four public GET requests and four guest availability lookups produced **23 HTTP-200 and
five HTTP-403 responses**. HTTP success is not admission. The fetched Tebubio page's stock text is
inside a hidden container and does not establish US availability. The Combi-Blocks search requires
login. No claim of universal stearolic-acid unavailability follows from these bounded checks.

An open primary [Organic Syntheses paper](https://www.orgsyn.org/demo.aspx?prep=CV4P0851) was
recovered and all four PDF pages visually inspected. It describes an exact stearolic-acid route
from oleic acid. It is a separate computational route lead: registry qualification and its own
terminal evidence are pending. It does not replace the frozen C18 route or justify changing
planner budgets. OpenAlex and Semantic Scholar return no open full-text location for the original
Oppolzer article. Its main procedure and schemes still need primary-source visual authentication,
as required by the [chemistry-evidence skill](../.claude/skills/forge-adjudicate-chemistry-evidence/SKILL.md):
“Render the cited chemistry pages and neighboring pages to images. Inspect schemes visually.”
The available SI and article transcription do not complete that check.

The next source version still requires the remaining evidence and combined source/route/value
qualification. The earlier frozen-loader failure remains preserved; no frozen pin, gate,
production module, model weight or sampling policy changed in this pass. No generation, training,
guidance experiment, candidate selection, supplier contact, purchase or paid computation occurred.

All **30** vendor assets verify and **eight** focused provenance/preflight tests pass. All four
acquisition Python files pass Black/Ruff, and the JavaScript request helper passes syntax checking.
The previous v3 full suite remains the latest repository-wide result: **2,338 passed, 167 failed,
21 setup errors and 92 skipped/xfail**. It was not rerun for this acquisition-only follow-up;
global Phase 1 definition of done remains unmet.

The **29** exact raw source/log assets are archived in **1,922,000 compressed bytes** under
`provenance/recovery/ugi_guidance_evidence_v4/assets/`. On a fresh checkout, restore the v3 inputs
using the command below in the v3 section, then restore and verify this pass:

```bash
UV_CACHE_DIR=/private/tmp/forge-uv-cache uv run python \
  results/phase1/ugi_guidance_evidence_acquisition_v4/archive_sources.py --restore
UV_CACHE_DIR=/private/tmp/forge-uv-cache uv run python \
  results/phase1/ugi_guidance_evidence_acquisition_v4/qualify_terminals.py --verify
UV_CACHE_DIR=/private/tmp/forge-uv-cache uv run python \
  results/phase1/ugi_guidance_evidence_acquisition_v4/closeout.py --verify
```

The qualification and summary replay exactly against their pinned inputs. The current report is
`results/phase1/ugi_guidance_evidence_acquisition_v4/result.json`, SHA-256
`8fdccd1d57ff8341fdf8a5206b140ee8352cd99fe55e7022d2ec75e744bf806a`.

## Fresh evidence acquisition — 2026-09-13

The third pass obtains a **separate new evidence package**. It does not replace the original
historical files or change their expected hashes. The original direct-input recovery inventory
remains **30 authenticated and 25 unavailable**. Five additional bounded PubChem endpoint
candidates also failed to reproduce the historical identity hash.

The initial eighteen public-source retrievals produced nine HTTP-200 responses, seven retained HTTP-error
responses and two transport failures. HTTP success alone is not evidence admission. Fresh PubChem
records for all **five required terminal identities** match the surviving configurations by
canonical graph, formula and InChIKey, including an atom-order check. An unrelated extra CID
returned in one request is explicitly excluded. All five earlier availability snapshots have
expired. Twenty acquisition files, including receipts, indexed-page observations and access
failures, are preserved in a **566,183-byte** compressed archive under
`provenance/recovery/ugi_guidance_evidence_v3/assets/`.

| Required evidence | Obtained in this pass | Remaining qualification gap |
| --- | --- | --- |
| Cyclohexylamine | Direct Chem-Impex page, fresh identity match, five same-day SKUs | The single terminal passes the existing supplier validator; integration into a separately versioned cumulative source remains outstanding. |
| Heptadecanal | Exact PubChem identity; TCI/Fisher listing and live guest availability reply | Direct TCI access returns 403. Fisher's checked SKU reports a backorder and supplier shipping estimate, not positive available quantity. |
| Protected propargyl alcohol | Exact identity, Fisher listing and live guest availability replies | The 25 g pack reports positive available quantity and an in-stock label; this new observation still requires separate source qualification. The 5 g and 100 g packs report backorders. |
| 1-Bromotridecane | Exact identity, Fisher listing and live guest availability replies | Both checked Sigma/Fisher pack sizes report backorders and supplier shipping estimates. |
| Stearolic acid | Exact identity, Thermo specification, MCE/Thermo Fisher listings and live guest availability replies | Both checked listings report backorders or supplier estimates. The specification establishes identity, not stock. |
| Oppolzer C18 oxidation procedure | New full article transcription and the already authenticated original ACS SI | The transcription differs from the historical hash and omits scheme images; primary-article visual verification remains incomplete. The university's published-version file requires login. |

The fresh cyclohexylamine terminal qualification preserves the original **seven-day** availability
lifetime. It checks each declared SKU against embedded supplier availability and same-day labels.
The existing validator rejects a rehashed archive with stock labels removed and an incorrect
registry graph; separate expiry checks reject expired and future-dated observations. This admits
**one terminal-material record only**, with **zero new route qualifications**. It does not refresh
the other terminals, import old cumulative closure counts, or enable guidance.

Follow-up inspection of Fisher's public product-page JavaScript identified a read-only guest
availability service. Eight exact-SKU lookups used reference ZIP **10001**, explicitly not the
user's address, and made no cart changes. Seven replies report zero available quantity with a
backorder or supplier estimate. **AC295060250**, the 25 g protected-propargyl-alcohol pack, reports
one available unit, zero backordered units, an `In Stock` display and a 2026-09-13 shipping date.
The source's `stockedItem=false` and generic lead-time field are retained; the response explicitly
says the material ships from the supplier, so this is not a claim of Fisher warehouse inventory.
The individual requests, raw replies, source scripts and identity listings are preserved under
`results/phase1/ugi_guidance_evidence_acquisition_v3/`. The summary is
`supplier_stock_summary.json`. Backorders are not treated as chemical impossibility, and this
bounded check does not exhaust suppliers.

ACS SI pages 1–6 and the new Thermo specification were rendered and visually inspected. SI page 5
contains the exact 17-octadecynal 8d characterization and reported 52% yield. The new transcription
contains the general oxidation procedure, but it has not completed the primary-article scheme
check. The Thermo PDF prints a 2026-09-14 date despite retrieval on 2026-09-13 UTC; that discrepancy
is preserved and the document is not used as an availability observation.

The remaining visual check follows the repository
[chemistry-evidence skill](../.claude/skills/forge-adjudicate-chemistry-evidence/SKILL.md):
“Render the cited chemistry pages and neighboring pages to images. Inspect schemes visually.”
The accessible article transcription does not contain the scheme images needed for that check.

The frozen cumulative source was exercised again and still fails at
`third_wave_head_terminals input pin mismatch: supplier_page`. No generation, training, candidate
selection, supplier contact, purchase, or paid computation occurred. The next evidence needed for
a new source version is qualifying availability for the three materials whose checked listings
report backorders, qualification of the positive protected-alcohol observation, and the complete
primary article. The new source, route outputs and value qualifications must
then be rebuilt and checked before the passive guidance-contrast experiment. Exact historical
recovery instead requires the original source records and missing code revision; the 17 missing
result artifacts depend on those inputs. Neither path permits changing frozen pins in place.

Acquisition receipts and the original-inventory recheck are in
`provenance/recovery/ugi_guidance_evidence_acquisition_v3.json`. The later terminal qualification
is `results/phase1/ugi_guidance_evidence_acquisition_v3/cyclohexylamine_terminal_qualification.json`;
it supersedes only the acquisition review's pending cyclohexylamine disposition. The initial
acquisition record remains unchanged. Nine follow-up raw pages, scripts and test logs are also
stored as compressed artifacts with exact raw hashes in
`provenance/recovery/ugi_guidance_followup_archive_v3.json`. On a fresh checkout, restore the
29 archived source/log files before running the read-only verification commands:

```bash
UV_CACHE_DIR=/private/tmp/forge-uv-cache uv run python \
  results/phase1/ugi_guidance_evidence_acquisition_v3/archive_followups.py --restore
```

Restoration refuses to overwrite different existing bytes. Verify the initial archived package with:

```bash
UV_CACHE_DIR=/private/tmp/forge-uv-cache uv run python \
  results/phase1/ugi_guidance_evidence_acquisition_v3/collect_report.py --verify
```

Final validation: all **30** vendor assets verify. The focused run has **19 passed and three
historical third-wave tests failed**, owing to unavailable original source paths/bytes; no checks
were skipped or weakened. All nine acquisition Python files pass Black and Ruff. The full suite
has **2,338 passed, 167 failed, 21 setup errors and 92 skipped/xfail** across **2,618 cases**. Its
failing-test identity/outcome set is identical to the preceding recovery run. Global Phase 1
definition of done remains unmet. The final source, stock, qualification and validation receipt is
`results/phase1/ugi_guidance_evidence_acquisition_v3/closeout.json`, SHA-256
`ad03b77e4964dc3b08440eb06a4c28eeed8e15e7ba666338419f4dd87b1e540c`.

```bash
UV_CACHE_DIR=/private/tmp/forge-uv-cache uv run python \
  results/phase1/ugi_guidance_evidence_acquisition_v3/closeout.py --verify
```

## Current status after exact replay

**30 of the original 55 unavailable input identities now authenticate; 25 remain unavailable.**
The first pass recovered 19 files. The second pass restored another 11 identities in that inventory,
four additional result artifacts and three additional historical source files. In total, the second
pass restored **15 result artifacts and three source files**, each at its original SHA-256.

The previously blocking `product_gap_ledger.csv.gz` is restored at
`a8ddbce7284de247b17c7e6cd017f8f1d86a16eb11f567214322377e17f4ae48`. All eight of its frozen
inputs were present and authenticated. Replaying the existing audit also reproduced the exact
original result and both companion priority ledgers. This establishes a recovery path that the
first pass had not tested; an external backup was not necessary for these deterministic outputs.

The same method recovered the targeted aldehyde audit, exact-overlay diagnostic, role-gap audit,
and first two head-terminal audits. The overlay required the original source code, Python **3.14.2**
and RDKit **2026.03.4** in an isolated environment. Current code emits a different assessment schema
and was rejected. The original code then differed only in 200 source-locator strings carrying the
old workstation root. Recovery serialization restored that historical namespace, retaining the
unchanged byte-identity and software-version checks. The head results also required their original
absolute `inputs.*.path` metadata. Scientific fields, input hashes and final expected hashes were
unchanged; the complete restored output bytes, including metadata, match the original pins.

The three additional source files came from Git commit
`cc947f5af590084cfb3a1c1eaa0e358883f995af` and were appended to the content-addressed archive.
Active implementation files were not changed. Original failed replay attempts and mismatched
candidates remain separate from admitted outputs.

The actual cumulative source loader now clears the restored chain and stops at
`third_wave_head_terminals input pin mismatch: supplier_page`:

```
data/source_cache/phase1_ugi3_third_wave_head_terminals/chemimpex_03599_2026-08-01.html
sha256: b8473f5749707bdb9adecca9a017132333d7aa14607ad3a7efc60006b0f9b518
```

The current supplier page was retrieved as a recovery candidate and rejected because its bytes
differ. A public web-archive lookup returned no matching 2026 captures. No present-day stock claim
or reconstructed observation was substituted. Guidance therefore remains blocked on historical
source authentication. Replaying historical registry assessments is not a new generation run or
an experiment demonstrating a guidance benefit.

The permanent receipt is `provenance/recovery/ugi_guidance_replayed_artifacts_v2.json`. Replay
scripts, isolated historical code, candidate outputs, input pins and failed attempts are under
`results/phase1/ugi_guidance_artifact_recovery_v2/`; its `remaining_inputs.json` supersedes the
first-pass inventory. No external drive or Time Machine backup was available during the expanded
search; the local mirror directory contained no repository backup and the available Modal workspace
had only the previously searched environment.

All 18 restored files are staged with byte-identical copies in the Git index. The 15 result files
total **196,515 bytes** and are explicitly tracked despite the general results-directory ignore rule.
All **166** implementation-pin checks across successful replay receipts pass.

Second-pass validation: all **30** vendored assets verify and **13** focused tests pass. The full
suite reports **2,338 passed, 167 failed, 21 setup errors and 92 skipped/xfail** across **2,618 cases**.
There are no new failing test identities. Restoring the exact-overlay artifact resolves one previous
failure; global Phase 1 definition of done remains unmet. The final report is
`results/phase1/ugi_guidance_artifact_recovery_v2/result.json`, SHA-256
`dc43d8645b6412a86c59c83a75dbb02492d21abc5342250c47f9e7ba9d82007e`.

## First-pass record

The following records the earlier recovery state; the exact replay above supersedes its blocker.

Recovered **19 of the 55 distinct unavailable path/hash identities** in the preflight's direct
input inventory: ten historical Python source files and nine original PDFs. Every restored file
matches its unchanged expected SHA-256. No historical result ledger has been recovered.

The ten source files were recovered from Git commit
`cc947f5af590084cfb3a1c1eaa0e358883f995af` and added to the existing content-addressed archive in
`provenance/frozen-code/`. All previous archive entries were preserved. Active Python modules
were not replaced with obsolete source code. The archived identities are available through
`experiments._runtime.historical.resolve_pinned_input`.

The nine PDFs were retrieved from their original public publisher, author, patent or supplier
repositories. They were restored under their original `data/source_cache/` paths. The permanent
download receipt at `provenance/recovery/ugi_guidance_public_documents_v1.json` records the exact
URLs, sizes and unchanged hashes. Alternative endpoints that returned errors were retained in
the recovery log; different bytes were never admitted. This includes the previously rejected
source-conflict patent: restoring its bytes does not change its rejected evidence status.

## First-pass blocker

**36 identities remain unavailable:** 28 historical result artifacts, seven cached source records
and one older `synthesis.py` revision. The missing code identity was already recorded as inherited
drift; its exception was not broadened. These counts describe the original direct-input inventory,
not complete transitive source closure. The 27 missing cumulative-loader paths also remain missing.

The real route loader now passes the recovered code and literature pins and stops at the
`product_gap_ledger` input:

```
results/phase1/ugi3_l2_coverage_priority_audit_v1/product_gap_ledger.csv.gz
sha256: a8ddbce7284de247b17c7e6cd017f8f1d86a16eb11f567214322377e17f4ae48
```

The first pass identified an original backup of the missing evidence/result directories as its next
input; the subsequent replay above recovered part of that chain without a backup.
`results/phase1/ugi_guidance_artifact_recovery_v1/remaining_inputs.json` lists each exact path and
expected hash. Recover those bytes, then reauthenticate the source. Do not regenerate plausible
replacement ledgers, change expected hashes, replace historical availability observations with
current stock information or substitute broader dossier-resolution labels.

## Search and validation records

The search covered the current and older FORGE checkouts, the two local worktrees, reachable Git
history, unreachable Git objects, candidate local artifact files, the current original GitHub
repository, GitHub releases and Actions artifacts, and input-directory metadata from surviving
FORGE Modal jobs. Some Downloads paths were inaccessible under macOS filesystem permissions.
No claim is made that every possible backup has been searched.

All search receipts, downloads, errors, source-loader recheck and validation reports are under
`results/phase1/ugi_guidance_artifact_recovery_v1/`. Original missing-input reports remain unchanged.
The recovery itself performs no model training, molecular generation, candidate route assessment,
oracle calls, paid compute job, candidate selection or evidence re-adjudication. Guidance remains
blocked on source authentication; this is a partial artifact recovery, not a guidance result.

All **30** vendored assets verify and **12** focused archive, source-provenance and preflight tests
pass. The full repository suite reports **2,337 passed, 168 failed, 21 setup errors and 92
skipped/xfail** across **2,618 cases**. Its failing test-identity/outcome-kind set exactly matches
the preserved pre-recovery run: no failures were added or removed. Repository-wide Phase 1
definition of done remains unmet. No tests or gates were relaxed.

The final recovery report is `results/phase1/ugi_guidance_artifact_recovery_v1/result.json`, SHA-256
`059703ab4afb201a830dc1956c83b31787b5ae45d401b8afd9bdcaea388dd76b`. It verifies all restored hashes,
preservation of previous archive entries, and the test comparison, and pins the recovery receipts.
