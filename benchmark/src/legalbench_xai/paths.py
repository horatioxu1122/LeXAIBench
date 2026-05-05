from pathlib import Path


def repo_root() -> Path:
    return Path(__file__).resolve().parents[3]


def benchmark_root() -> Path:
    return repo_root() / "benchmark"


def resolve_repo_path(value: str) -> Path:
    path = Path(value)
    if path.is_absolute():
        return path
    return repo_root() / value


def repo_relative_path(path: Path) -> str:
    try:
        return str(path.resolve().relative_to(repo_root()))
    except ValueError:
        return str(path.resolve())
