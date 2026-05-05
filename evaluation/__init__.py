__all__ = ["run_benchmark"]


def __getattr__(name):
    if name == "run_benchmark":
        from evaluation.runner import run_benchmark

        return run_benchmark
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
