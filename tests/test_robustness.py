import pytest

from folate_twin.robustness import build_jobs, mean_sd, paired_t, r2_against, rmse

T_CRIT_DF4 = 2.7764451051977987   # two-sided 5 % critical value of Student's t, 4 dof


def test_job_matrix():
    jobs = build_jobs(10)
    labels = [j.label for j in jobs]
    assert len(jobs) == 10 + 24 + 16          # multi-seed + LOTPO + LOO
    assert len(set(labels)) == len(labels)
    assert "lotpo:D_nophys:node3:both" in labels
    assert "loo:D:node0:tumour" in labels


def test_summary_statistics():
    m, sd = mean_sd([1.0, 2.0, 3.0, 4.0])
    assert m == 2.5 and sd == pytest.approx(1.2909944487358056)
    assert r2_against([1.0, 2.0, 3.0], [1.0, 2.0, 3.0]) == 1.0
    assert rmse([0.0, 0.0], [3.0, 4.0]) == pytest.approx(12.5 ** 0.5)


def test_paired_t_matches_student_t():
    d = [1.0, -1.0, 0.5, -0.5, 0.0]
    t, p, df = paired_t(d, [0.0] * 5)
    assert df == 4 and t == pytest.approx(0.0, abs=1e-12) and p == pytest.approx(1.0)

    shift = T_CRIT_DF4 * mean_sd(d)[1] / 5 ** 0.5
    t, p, _ = paired_t([x + shift for x in d], [0.0] * 5)
    assert t == pytest.approx(T_CRIT_DF4)
    assert p == pytest.approx(0.05, abs=1e-9)
