# portfolio publication boundary

`portfolio-notebooks.json` is an explicit publication gate for mhaider.dev. only entries marked `publishable: true` are imported. each pins a notebook path, a full source revision, and a SHA-256 hash of the original bytes.

editing a research notebook alone does not publish it. after reviewing a notebook for publication, commit its source, update the manifest revision and matching hashes, then commit the manifest. the portfolio checks for approved changes hourly and exports saved outputs without executing cells. private drafts and unlisted notebooks are never imported. CUDA notebooks are excluded from the portfolio by editorial choice.

keep central research arguments, implementation work, and original outputs in their canonical notebooks. this directory contains publication metadata only.
