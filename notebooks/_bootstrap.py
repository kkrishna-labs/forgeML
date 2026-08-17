# Databricks notebook source
# MAGIC %md
# MAGIC # _bootstrap
# MAGIC
# MAGIC Shared setup for every ForgeML notebook: find the repo checkout and put
# MAGIC `src/` on `sys.path`.
# MAGIC
# MAGIC Notebooks call this with `%run ./_bootstrap`. Two reasons it lives here
# MAGIC rather than being copy-pasted into each one:
# MAGIC
# MAGIC * **`%run` takes a path relative to the calling notebook**, so it resolves
# MAGIC   correctly no matter where the workspace put the Git folder — which is
# MAGIC   exactly the problem this file exists to solve.
# MAGIC * five copies of the same forty lines would drift, and the one that drifts
# MAGIC   is the one you are not looking at.

# COMMAND ----------

import os
import sys


def find_repo_root() -> str:
    """Locate the checked-out repo, wherever this workspace put it.

    Databricks Git folders land in different places depending on workspace
    vintage and settings::

        /Workspace/Repos/<repo>             legacy, unscoped
        /Workspace/Repos/<email>/<repo>     legacy, user-scoped
        /Workspace/Users/<email>/<repo>     current default

    Hardcoding one of them is how you get `ModuleNotFoundError: No module named
    'forgeml'` four cells into a notebook. Derive it from the calling notebook's
    own path instead, and verify the package is really there before trusting the
    answer — a path that merely exists is not proof it is the right one.
    """
    candidates: list[str] = []

    try:
        context = dbutils.notebook.entry_point.getDbutils().notebook().getContext()  # noqa: F821
        node = "/Workspace" + context.notebookPath().get()
        # Walk up from the notebook. Depth 6 covers /Workspace/Users/<email>/
        # <repo>/notebooks/<name> with room to spare.
        for _ in range(6):
            node = os.path.dirname(node)
            if node in ("", "/", "/Workspace"):
                break
            candidates.append(node)
    except Exception as exc:  # noqa: BLE001 - fall back rather than fail here
        print(f"could not read the notebook context ({exc}); trying fixed paths")

    candidates += ["/Workspace/Repos/forgeML", "/Workspace/Repos/forgeml"]

    for root in candidates:
        if os.path.exists(os.path.join(root, "src", "forgeml", "__init__.py")):
            return root

    raise RuntimeError(
        "Could not find the forgeML checkout. Looked in:\n  "
        + "\n  ".join(candidates)
        + "\n\nLocate it with:\n"
        "  %sh find /Workspace -maxdepth 6 -type d -name forgeml 2>/dev/null"
    )


REPO_ROOT = find_repo_root()

# Guard on the path actually inserted, not on REPO_ROOT — otherwise re-running
# prepends "/src" again on every execution.
if f"{REPO_ROOT}/src" not in sys.path:
    sys.path.insert(0, f"{REPO_ROOT}/src")

print(f"repo root : {REPO_ROOT}")
