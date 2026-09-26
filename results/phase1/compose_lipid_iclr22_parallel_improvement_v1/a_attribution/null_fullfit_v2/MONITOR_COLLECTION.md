# Local monitor and streaming collector

Qualified locally only. No network, GPU, submission or retry call was made. The pinned launcher, supervisor, fit wrapper, request and training source are unchanged.

Readiness: `monitor_collection_readiness_v1.json`, SHA256 `1e9e7d74e14c61daab344ec3e2093ce686784c60b1e4b6e563306bd2faf29436`.

All 18 current fixture cases pass. The first run passed 17; its one failed assertion counted only duplicated checkpoints and omitted identical staging/inventory metadata. That test expectation was corrected, the affected case passed, and the original failing XML remains preserved. No helper implementation changed between those test runs. CPU time was not measured directly; this qualification used only small local async byte-stream fixtures, within the 120-CPU-second authorization. No expensive validation was repeated.

After root authorizes and submits the exact run, observe once with:

```sh
.venv/bin/python results/phase1/compose_lipid_iclr22_parallel_improvement_v1/a_attribution/null_fullfit_v2/monitor_progress_v1.py
```

Each invocation writes a new durable directory under `monitor_observations_v1/`, including observed files, missing-file records, network errors and validation errors. Missing/unreadable/malformed event files produce `null` progress, not zero. Successful empty event files can report zero. Reads are separate snapshots; these counts are monitoring evidence, not final atomic checkpoint validation. Polling and the remote job lifetime are independent. There is no submission API or automatic retry in the monitor.

After a terminal observation, collect its artifacts with:

```sh
.venv/bin/python results/phase1/compose_lipid_iclr22_parallel_improvement_v1/a_attribution/null_fullfit_v2/collect_stream_v2.py --observation results/phase1/compose_lipid_iclr22_parallel_improvement_v1/a_attribution/null_fullfit_v2/monitor_observations_v1/<timestamp>
```

For success, the collector requires the function-returned outer result pin and the exact `fit/result.json` pin before collecting its nested artifacts. Failed execution artifacts can be collected separately and never count as scientific success. A distinct terminal catalog requires an explicit new `--record <filename>` rather than overwriting an earlier collection.

Artifacts stream through `volume.read_file_into_fileobj.aio` into unique partial files, with a 120-second deadline per remote file. Verified hashes are atomically renamed into `collected/`. The manifest is written after each artifact. Equal hashes are copied locally without duplicate downloads; partial/corrupt files and failure records remain preserved. Repeating the command is an explicit read-only collection resume: completed files are rechecked, missing files are downloaded, and a completed receipt remains byte-identical. A changed completed file produces a separate validation-error record and fails.

Collection does not admit training results. Root still authenticates full state, exposure, draw ledgers and final counters before evaluation. Fresh paid retry authorization is handled separately by root; this readiness receipt grants none.
