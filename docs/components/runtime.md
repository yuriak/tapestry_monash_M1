# Runtime

Available: pinned asset acquisition/verification, config resolution, source identity, local Transformer loader and scheduler GPU guard.

## Ownership

`runtime/assets.py`, `configuration.py`, `provenance.py`, `transformers.py`, `devices.py`.

## Input/output boundary

Reviewed component config + selected environment + verified local assets -> explicit runtime/model handles and provenance.

## Extension direction

The loader imports model frameworks lazily. Core data/contracts/gateway paths must remain usable in the CPU profile without Torch. Runtime does not choose research recipes or embed site-specific accounts/paths. Future backends may add reviewed device/serving adapters.

See [architecture](../architecture/overview.md), [contracts](../contracts/data.md) and the [component index](README.md).
