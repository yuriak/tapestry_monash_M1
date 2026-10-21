# Generator

Planned integration area; no generator serving/training or MoE router is implemented.

## Ownership

Future implementations live under `generator/`; no generator framework is installed for this placeholder.

## Input/output boundary

Reranked evidence/abstractions + jurisdiction + generation config -> cited answer, evidence references and explicit generation metadata.

## Extension direction

Support reviewed model/serving backends, LoRA adaptation, shared/node expert grouping and routing experiments as the programme proceeds. Keep evaluation independently usable. Isolate incompatible serving/training dependencies through a reviewed independent environment rather than upgrading the reranker environment implicitly.

See [architecture](../architecture/overview.md), [contracts](../contracts/data.md) and the [component index](README.md).
