from pathlib import Path

from caribbean_short_term_forecast.src.inspect_data import should_ignore


def test_macos_appledouble_files_are_ignored_at_any_depth() -> None:
    assert should_ignore(Path("F:/data/._top.nc"))
    assert should_ignore(Path("F:/data/._folder/real.nc"))
    assert should_ignore(Path("F:/data/folder/._real.nc"))
    assert not should_ignore(Path("F:/data/folder/real.nc"))
