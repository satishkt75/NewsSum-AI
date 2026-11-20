from newsum.evaluation.stats import holm_adjust, paired_bootstrap

def test_holm():
    assert holm_adjust([.01,.04,.02]) == [.03,.04,.04]

def test_bootstrap():
    p = paired_bootstrap([2,3,4],[1,2,3],iterations=100)
    assert p['mean_difference'] == 1.0
    assert p['ci95'][0] == 1.0
