"""Backend package initializer.

This file makes the ``backend`` directory a proper Python package so that
imports such as ``backend.app`` work when the project is executed with
``uvicorn backend.app.main:app``.
"""