"""Dashboard package exports.

Import heavy UI dependencies lazily so core modules remain importable in
minimal test environments where optional dashboard deps are not installed.
"""

__all__ = ["run_dashboard", "launch_dashboard"]


def run_dashboard(*args, **kwargs):
    from dashboard.app import run_dashboard as _run_dashboard

    return _run_dashboard(*args, **kwargs)


def launch_dashboard(*args, **kwargs):
    from dashboard.launch_dashboard import launch_dashboard as _launch_dashboard

    return _launch_dashboard(*args, **kwargs)
