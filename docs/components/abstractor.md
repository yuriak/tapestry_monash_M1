# Abstractor

Planned extension area; no rewriting, PII removal or privacy defense is implemented.

## Ownership

Future implementations live under `abstractor/`; the directory currently declares ownership only.

## Input/output boundary

Future input translator and evidence/training-example transforms will consume explicit records and produce versioned abstractions with retained provenance.

## Extension direction

Keep abstraction level, transformation recipe/model, cached outputs and utility/privacy evaluation traceable. Gateway handles batch I/O. Named privacy claims require attack-specific validation; a reserved level or schema is not a privacy mechanism.

See [architecture](../architecture/overview.md), [contracts](../contracts/data.md) and the [component index](README.md).
