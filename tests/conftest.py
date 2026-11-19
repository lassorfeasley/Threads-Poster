"""Suite-wide test setup.

Machine settings in a workspace .env must not leak into tests: app.config loads
those files at import, but never overrides a variable that's already set, so
setting it here (before any app import) wins. Tests that need one of these on
turn it on themselves (e.g. tests/test_shared_docs.py).
"""
import os

# Sharing reads documents from the database; the tests' throwaway databases
# don't have them. The dashboard login calls Supabase.
for name in ("SHARED_WORKSPACE", "DASHBOARD_AUTH"):
    os.environ[name] = ""
