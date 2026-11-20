from newsum.data.pipeline import normalize, token_segment_ids, make_document


def test_normalization():
    assert normalize('Hello\n    world') == 'Hello world'


def test_token_segments():
    spans = [(0, 11), (12, 24)]
    offsets = [(0,0),(0,5),(6,11),(12,16),(17,23),(0,0)]
    assert token_segment_ids(offsets, spans, 64) == [-1,0,0,1,1,-1]


def test_multi_news_separator():
    source, ref = make_document({'document':'a ||||| b','summary':'brief'}, 'multi_news')
    assert '[DOC_SEP]' in source
    assert ref == 'brief'
