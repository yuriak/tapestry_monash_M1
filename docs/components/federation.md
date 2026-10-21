# Federation

Planned integration area; no live federation, partner backend or server is bundled.

## Ownership

Future framework adapters live under `federation/`.

## Input/output boundary

Explicit shared/local parameter manifests, round state and validated artifacts -> framework-specific exchange/checkpoint/communication records.

## Extension direction

Separate allowed shared parameters from node-local heads/experts. Review Slakshna/Flower interfaces, resume semantics, checksum/exclusion tests and partner acceptance before implementing an adapter. Preserve third-party Git semantics and isolate incompatible stacks. Never infer network readiness from local schema tests.

See [architecture](../architecture/overview.md), [contracts](../contracts/data.md) and the [component index](README.md).
