# Gateway

Available: bounded local L0 batch identity/validation adapter. No network transport or privacy protection.

## Ownership

`gateway/batch.py`, `cli.py`; shared vocabulary in `contracts/gateway.py`.

## Input/output boundary

Versioned query/ranking/evidence envelope -> unchanged valid payload + status/errors, or sanitized error envelope.

## Extension direction

Partner schema/version/provenance acceptance is pending. L1–L3 are reserved and rejected; actual rewriting belongs to abstractor. Origin labels and structural validation do not establish source authenticity or data rights.

See [architecture](../architecture/overview.md), [contracts](../contracts/data.md) and the [component index](README.md).
