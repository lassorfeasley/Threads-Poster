"""Throughline Admin: one place to watch every customer account.

A separate web app (``python run.py admin``; hosted at admin.continuity.marketing)
that only the operator can open. It reads every workspace's database read-only
(app/admin/sources.py), asks Fly for the state of each machine (app/admin/fly.py)
and turns both into health, pipeline, decision and results views
(app/admin/collect.py). It never writes to a workspace.
"""
