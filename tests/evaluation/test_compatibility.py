from tapestry_m1.core.io import records


def test_compatibility_ties_are_a_separate_view(benchmark,rankings,tmp_path):
    from tapestry_m1.evaluation.retrieval.compatibility import compatibility_view
    original=rankings.read_bytes()
    result=compatibility_view(benchmark,rankings,tmp_path/'compatibility')
    assert result['definition']['gain']=='linear' and result['definition']['mrr_depth']==100
    assert rankings.read_bytes()==original
    assert next(records(tmp_path/'compatibility/rankings.jsonl'))['candidates'][0]['doc_id']=='c::b'
