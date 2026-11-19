from fpl_model import paths
from fpl_model.site_export.build_public_site import strip_solver


def test_strip_solver_removes_every_solver_piece():
    html = (paths.SITE / "index.html").read_text(encoding="utf-8")
    out = strip_solver(html)
    assert "solver" not in out.lower()
    assert 'id="view-fixtures"' in out and '<script src="app.js"></script>' in out
