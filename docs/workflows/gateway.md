# Local L0 batch validation

Read the [batch contract](../contracts/gateway.md) and [gateway component boundary](../components/gateway.md).

```bash
./scripts/run --environment cpu gateway tests/fixtures/gateway/request.jsonl outputs/gateway-response.jsonl --collection demo
```

The fixtures are synthetic. A valid response preserves query/ranking/evidence and adds explicit status/errors. Empty ranking plus empty evidence is `no_evidence`; absent evidence for nonempty candidates is an error. Per-record errors must be inspected by callers; malformed JSON, duplicate batch IDs, budgets and output conflicts fail the batch. There is no model, network or partner session.

L0 provides no privacy protection. L1/L2/L3 are reserved, rejected labels until actual abstractor methods and privacy/utility validation are introduced. Future approved partners must confirm namespaces, source provenance, rank semantics, byte budgets and transport policy; this local adapter is not an accepted partner API.
