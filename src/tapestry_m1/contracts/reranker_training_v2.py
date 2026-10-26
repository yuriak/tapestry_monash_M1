"""Explicit real-data opt-in for assumed negatives; v1 remains unchanged."""
import copy
from tapestry_m1.core.io import ContractError
from tapestry_m1.contracts.reranker_training import validate_groups as validate_v1

SCHEMA='m1.reranker.training.v2'


def validate_groups(groups):
    strict=copy.deepcopy(groups)
    for original,row in zip(groups,strict):
        if original.get('schema_version')!=SCHEMA or original.get('synthetic') is not False:
            raise ContractError('real assumed-negative v2 schema required')
        source_ids=set(original.get('query_source_doc_ids',[]))
        if not source_ids:raise ContractError('query source-document identities required')
        used_sources=set()
        for index,(c,checked) in enumerate(zip(original.get('candidates',[]),row.get('candidates',[]))):
            if c.get('split')!=original['split']:raise ContractError('candidate borrowed from another split')
            if used_sources.intersection(c.get('source_doc_ids',[])):raise ContractError('same-source candidates repeat within group')
            used_sources.update(c.get('source_doc_ids',[]))
            if not c.get('source_doc_ids'):raise ContractError('candidate source identities required')
            if index:
                if c.get('supervision')!='assumed' or c.get('ambiguous') is not True:
                    raise ContractError('unjudged negatives require explicit assumed/ambiguous provenance')
                if source_ids & set(c['source_doc_ids']):raise ContractError('same-source negative excluded')
                checked['ambiguous']=False
            elif c.get('supervision')!='source_positive' or c.get('ambiguous') is not False:
                raise ContractError('source.text positive must retain explicit supervision')
        row['schema_version']='m1.reranker.training.v1'
    validate_v1(strict)
    # Check all aliases, not only the representative source_doc_id used in v1.
    seen={}
    for g in groups:
        for c in g['candidates']:
            for source in c['source_doc_ids']:
                if source in seen and seen[source]!=g['split']:raise ContractError('source alias leaks across splits')
                seen[source]=g['split']
    return groups
