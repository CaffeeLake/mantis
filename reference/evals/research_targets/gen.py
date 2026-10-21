# Copyright 2026 Google LLC
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     https://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Seeded research targets: generated corpora for an honest researcher eval.

One hand-written 26-file target cannot tell improvement from noise: it fits
in a single context window (so whole-file reading always "works"), it has no
hard negatives (so precision measures only the match phrases), and anything
tuned against it is tuned against the only thing that measures it. This
module generates targets instead:

* ``generate(seed)`` is a pure function of its arguments: same seed, same
  bytes. Different seeds differ in module names, routes, parameter names,
  which bug variants appear, and where.
* Every target carries **seeded** bugs (what recall counts) and **decoys**
  (the same sink behind a real sanitiser, the same code with no caller, the
  same pattern under ``tests/`` or ``vendor/``). A report that matches a decoy
  is a certain false positive and is scored as one.
* ``size`` pads the tree with benign modules until read-everything cannot
  fit in one context, which is the condition under which navigation tools
  are supposed to matter.
* **Holdout is by construction.** Seeds in ``HOLDOUT_SEEDS`` are for one-shot
  confirmation only; ``assert_not_holdout`` refuses to build any
  proposer/reflector/tuning context on them. One bug class and one framework
  skin never appear in dev seeds, so a change that merely learned the dev
  templates shows up on holdout.

Usage::

    python3 evals/research_targets/gen.py --seed 1001 --size medium --out /tmp/t1001
    python3 evals/research_eval.py --target /tmp/t1001 --toolset structural ...
    python3 evals/research_targets/gen.py --list dev

The generated tree must read like ordinary application code: no comment may
name a flaw, a CWE, a decoy, or a sanitiser. ``tests/test_research_targets.py``
greps every generated file for such words. Flaw documentation lives only in
the emitted ``ground_truth.json``, which the eval jail excludes.
"""

from __future__ import annotations

import argparse
import json
import random
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple

GROUND_TRUTH_BASENAME = "ground_truth.json"
FORMAT_VERSION = "2.0"

# --- Holdout by construction ----------------------------------------------------------

DEV_SEEDS = range(1000, 1100)
HOLDOUT_SEEDS = range(9000, 9100)
# Never generated for a dev seed; always eligible for a holdout seed. A harness
# that only learned the dev templates loses here.
HOLDOUT_ONLY_CLASSES = frozenset({"xss"})
# Framework skins. Only Flask is implemented; the holdout-only skin mechanism is
# in place for when a second skin (e.g. Express) lands, and stays empty until
# then rather than relabelling Flask trees.
SKINS: Tuple[str, ...] = ("flask",)
HOLDOUT_ONLY_SKINS: frozenset = frozenset()


class HoldoutViolation(RuntimeError):
    """Raised when a holdout seed is about to be used for anything but confirmation."""


def seed_split(seed: int) -> str:
    if seed in DEV_SEEDS:
        return "dev"
    if seed in HOLDOUT_SEEDS:
        return "holdout"
    return "other"


def assert_not_holdout(seed: int, purpose: str) -> None:
    """Refuses holdout seeds for tuning, proposing, reflecting, or dev gates.

    The holdout set is run once per change, after dev. Anything that reads a
    holdout target to *improve* the harness burns it. This is called at the
    construction of every such context, so the rule is enforced by code, not
    by convention.
    """
    if seed_split(seed) == "holdout":
        raise HoldoutViolation(
            f"seed {seed} is a holdout seed and may not be used for {purpose}; "
            f"holdout targets are for one-shot confirmation only"
        )


# --- Vocabulary ------------------------------------------------------------------------

_DOMAINS = [
    "orders", "invoices", "shipments", "tenants", "coupons", "webhooks", "exports",
    "reports", "profiles", "payments", "catalog", "inventory", "tickets", "audits",
    "notes", "sessions", "uploads", "billing", "ledgers", "quotes", "returns",
    "subscriptions", "devices", "alerts", "feeds", "labels", "regions", "vendors",
    "contracts", "shifts", "rosters", "badges", "surveys", "campaigns", "drafts",
]
_QUERY_PARAMS = ["q", "name", "term", "filter", "label", "code", "ref", "key", "tag"]
_PATH_PARAMS = ["path", "file", "doc", "attachment", "asset", "template"]
_URL_PARAMS = ["url", "callback", "target", "source", "endpoint", "feed"]
_REDIRECT_PARAMS = ["next", "return_to", "continue", "back", "dest"]
_SIZES: Dict[str, Tuple[int, int]] = {
    # benign modules, approximate functions per module
    "small": (18, 4),
    "medium": (110, 7),
    "large": (380, 9),
}

_BANNED_IN_TREE = (
    "vulnerab", "cwe-", "decoy", "injection", "traversal", "exploit", "insecure",
    "unsafe", "ground truth", "sanitiz", "seeded", "holdout", "attacker", "malicious",
)


# --- Bug templates (Flask skin) ---------------------------------------------------------
#
# Each template renders one bug (or its sanitised twin) as a route -> service
# -> store chain. ``hops`` is 1 (all in the route file), 2 (route + service)
# or 3 (route + service + store). ``safe=True`` renders the decoy_sanitized
# twin: identical shape, real mitigation at the sink.


@dataclass
class Rendered:
    files: Dict[str, str]
    chain: List[str]          # files on the data-flow path, route first
    sink: str                 # "file:function"
    source: str               # "file:/route"


@dataclass
class BugClass:
    name: str
    cwe: str
    phrases: List[str]
    render: Callable[["_Names", int, bool, random.Random], Rendered]
    # Whether a no-caller twin (decoy_dead) makes sense for this class.
    dead_ok: bool = True
    # Whether the pattern under tests/ or vendor/ is out of scope (decoy_vendor).
    # A hardcoded live secret is a finding wherever it sits, so it is not one.
    vendor_ok: bool = True


@dataclass
class _Names:
    mod: str
    route: str
    param: str
    fn: str
    store_fn: str
    blueprint: str = field(default="")

    def __post_init__(self) -> None:
        self.blueprint = self.blueprint or f"{self.mod}_bp"


_FLASK_ROUTE_HEAD = '''from flask import Blueprint, request, jsonify, abort

{imports}
{blueprint} = Blueprint("{mod}", __name__)


'''


def _route_file(n: _Names, body: str, imports: str = "") -> str:
    return _FLASK_ROUTE_HEAD.format(imports=imports, blueprint=n.blueprint, mod=n.mod) + body


def _chain(n: _Names, hops: int, route_body: Callable[[str], str], service_body: Callable[[str], str],
           store_body: str, sink_fn: str, safe: bool, extra_imports: Dict[str, str] | None = None) -> Rendered:
    """Builds a 1/2/3-hop chain. ``route_body(callee)`` and ``service_body(callee)``
    receive the expression to call for the next hop; ``store_body`` is the sink."""
    extra_imports = extra_imports or {}
    route_path = f"web/{n.mod}_routes.py"
    svc_path = f"services/{n.mod}.py"
    store_path = f"dal/{n.mod}_store.py"
    files: Dict[str, str] = {}
    if hops == 1:
        imports = extra_imports.get("route", "")
        files[route_path] = _route_file(n, route_body(None), imports)
        return Rendered(files, [route_path], f"{route_path}:{n.fn}", f"{route_path}:{n.route}")
    if hops == 2:
        files[route_path] = _route_file(
            n, route_body(f"{n.mod}_service.{n.fn}"),
            f"from services import {n.mod} as {n.mod}_service\n" + extra_imports.get("route", ""),
        )
        files[svc_path] = extra_imports.get("service", "") + service_body(None)
        return Rendered(files, [route_path, svc_path], f"{svc_path}:{n.fn}", f"{route_path}:{n.route}")
    files[route_path] = _route_file(
        n, route_body(f"{n.mod}_service.{n.fn}"),
        f"from services import {n.mod} as {n.mod}_service\n" + extra_imports.get("route", ""),
    )
    files[svc_path] = (
        f"from dal import {n.mod}_store\n" + extra_imports.get("service", "") + "\n\n"
        + service_body(f"{n.mod}_store.{n.store_fn}")
    )
    files[store_path] = extra_imports.get("store", "") + store_body
    return Rendered(files, [route_path, svc_path, store_path], f"{store_path}:{sink_fn}", f"{route_path}:{n.route}")


def _handler(n: _Names, arg_expr: str, call: str, result: str = "result") -> str:
    return (
        f'@{n.blueprint}.route("{n.route}")\n'
        f"def {n.fn}_view():\n"
        f"    value = {arg_expr}\n"
        f"    if value is None:\n"
        f"        abort(400)\n"
        f"    {result} = {call}\n"
        f"    return jsonify({{\"ok\": True, \"data\": {result}}})\n"
    )


# -- sql_injection
def _r_sqli(n: _Names, hops: int, safe: bool, rng: random.Random) -> Rendered:
    variant = rng.choice(["fstring", "format", "concat"])
    if safe:
        sink = (
            "import sqlite3\n\nDB_PATH = \"data/app.db\"\n\n\n"
            f"def {{fn}}(term):\n"
            "    conn = sqlite3.connect(DB_PATH)\n"
            "    try:\n"
            f"        rows = conn.execute(\"SELECT id, name FROM {n.mod} WHERE name LIKE ?\", (f\"%{{term}}%\",)).fetchall()\n"
            "    finally:\n"
            "        conn.close()\n"
            "    return [dict(id=r[0], name=r[1]) for r in rows]\n"
        )
    elif variant == "fstring":
        sink = (
            "import sqlite3\n\nDB_PATH = \"data/app.db\"\n\n\n"
            f"def {{fn}}(term):\n"
            "    conn = sqlite3.connect(DB_PATH)\n"
            "    try:\n"
            f"        rows = conn.execute(f\"SELECT id, name FROM {n.mod} WHERE name LIKE '%{{term}}%'\").fetchall()\n"
            "    finally:\n"
            "        conn.close()\n"
            "    return [dict(id=r[0], name=r[1]) for r in rows]\n"
        )
    elif variant == "format":
        sink = (
            "import sqlite3\n\nDB_PATH = \"data/app.db\"\n\n\n"
            f"def {{fn}}(term):\n"
            "    conn = sqlite3.connect(DB_PATH)\n"
            f"    query = \"SELECT id, name FROM {n.mod} WHERE name = '{{0}}'\".format(term)\n"
            "    try:\n"
            "        rows = conn.execute(query).fetchall()\n"
            "    finally:\n"
            "        conn.close()\n"
            "    return [dict(id=r[0], name=r[1]) for r in rows]\n"
        )
    else:
        sink = (
            "import sqlite3\n\nDB_PATH = \"data/app.db\"\n\n\n"
            f"def {{fn}}(term):\n"
            "    conn = sqlite3.connect(DB_PATH)\n"
            f"    query = \"SELECT id, name FROM {n.mod} WHERE code = '\" + term + \"'\"\n"
            "    try:\n"
            "        rows = conn.execute(query).fetchall()\n"
            "    finally:\n"
            "        conn.close()\n"
            "    return [dict(id=r[0], name=r[1]) for r in rows]\n"
        )

    def route_body(callee):
        if callee is None:
            return sink.replace("{fn}", n.fn) + "\n\n" + _handler(n, f'request.args.get("{n.param}")', f"{n.fn}(value)")
        return _handler(n, f'request.args.get("{n.param}")', f"{callee}(value)")

    def service_body(callee):
        if callee is None:
            return sink.replace("{fn}", n.fn)
        return f"def {n.fn}(term):\n    term = term.strip()\n    return {callee}(term)\n"

    return _chain(n, hops, route_body, service_body, sink.replace("{fn}", n.store_fn), n.store_fn, safe)


# -- command_injection
def _r_cmdi(n: _Names, hops: int, safe: bool, rng: random.Random) -> Rendered:
    variant = rng.choice(["shell", "os_system"])
    tool = rng.choice(["report-gen", "convert", "pdfinfo", "exiftool"])
    if safe:
        sink = (
            "import subprocess\n\n\n"
            f"def {{fn}}(name):\n"
            f"    proc = subprocess.run([\"{tool}\", \"--name\", name], capture_output=True, text=True, timeout=30)\n"
            "    return proc.stdout\n"
        )
    elif variant == "shell":
        sink = (
            "import subprocess\n\n\n"
            f"def {{fn}}(name):\n"
            f"    proc = subprocess.run(\"{tool} --name \" + name, shell=True, capture_output=True, text=True, timeout=30)\n"
            "    return proc.stdout\n"
        )
    else:
        sink = (
            "import os\n\n\n"
            f"def {{fn}}(name):\n"
            f"    status = os.system(f\"{tool} --name {{name}} > /tmp/{n.mod}.out\")\n"
            "    return {\"status\": status}\n"
        )

    def route_body(callee):
        src = f'request.args.get("{n.param}")'
        if callee is None:
            return sink.replace("{fn}", n.fn) + "\n\n" + _handler(n, src, f"{n.fn}(value)")
        return _handler(n, src, f"{callee}(value)")

    def service_body(callee):
        if callee is None:
            return sink.replace("{fn}", n.fn)
        return f"def {n.fn}(name):\n    if len(name) > 200:\n        raise ValueError(\"too long\")\n    return {callee}(name)\n"

    return _chain(n, hops, route_body, service_body, sink.replace("{fn}", n.store_fn), n.store_fn, safe)


# -- path_traversal
def _r_traversal(n: _Names, hops: int, safe: bool, rng: random.Random) -> Rendered:
    base = rng.choice(["/srv/documents", "/var/lib/app/files", "storage/attachments"])
    if safe:
        sink = (
            "import os\n\n"
            f"BASE_DIR = os.path.realpath(\"{base}\")\n\n\n"
            f"def {{fn}}(rel):\n"
            "    candidate = os.path.realpath(os.path.join(BASE_DIR, rel))\n"
            "    if os.path.commonpath([candidate, BASE_DIR]) != BASE_DIR:\n"
            "        raise PermissionError(\"outside base directory\")\n"
            "    with open(candidate, \"rb\") as fh:\n"
            "        return fh.read()\n"
        )
    else:
        sink = (
            "import os\n\n"
            f"BASE_DIR = \"{base}\"\n\n\n"
            f"def {{fn}}(rel):\n"
            "    with open(os.path.join(BASE_DIR, rel), \"rb\") as fh:\n"
            "        return fh.read()\n"
        )

    def route_body(callee):
        src = f'request.args.get("{n.param}")'
        if callee is None:
            return sink.replace("{fn}", n.fn) + "\n\n" + _handler(n, src, f"{n.fn}(value).decode('utf-8', 'replace')")
        return _handler(n, src, f"{callee}(value).decode('utf-8', 'replace')")

    def service_body(callee):
        if callee is None:
            return sink.replace("{fn}", n.fn)
        return f"def {n.fn}(rel):\n    rel = rel.lstrip('/')\n    return {callee}(rel)\n"

    return _chain(n, hops, route_body, service_body, sink.replace("{fn}", n.store_fn), n.store_fn, safe)


# -- ssrf
def _r_ssrf(n: _Names, hops: int, safe: bool, rng: random.Random) -> Rendered:
    host = rng.choice(["hooks.example.internal", "partners.example.com", "cdn.example.net"])
    if safe:
        sink = (
            "import urllib.parse\nimport requests\n\n"
            f"ALLOWED_HOSTS = {{\"{host}\"}}\n\n\n"
            f"def {{fn}}(url):\n"
            "    parsed = urllib.parse.urlparse(url)\n"
            "    if parsed.scheme != \"https\" or parsed.hostname not in ALLOWED_HOSTS:\n"
            "        raise ValueError(\"host not allowed\")\n"
            "    return requests.get(url, timeout=5, allow_redirects=False).text[:4096]\n"
        )
    else:
        sink = (
            "import requests\n\n\n"
            f"def {{fn}}(url):\n"
            "    return requests.get(url, timeout=5).text[:4096]\n"
        )

    def route_body(callee):
        src = f'request.args.get("{n.param}")'
        if callee is None:
            return sink.replace("{fn}", n.fn) + "\n\n" + _handler(n, src, f"{n.fn}(value)")
        return _handler(n, src, f"{callee}(value)")

    def service_body(callee):
        if callee is None:
            return sink.replace("{fn}", n.fn)
        return f"def {n.fn}(url):\n    url = url.strip()\n    return {callee}(url)\n"

    return _chain(n, hops, route_body, service_body, sink.replace("{fn}", n.store_fn), n.store_fn, safe)


# -- open_redirect
def _r_redirect(n: _Names, hops: int, safe: bool, rng: random.Random) -> Rendered:
    hops = 1  # the sink is the response; a chain adds nothing
    if safe:
        body = (
            "import urllib.parse\n\n\n"
            f"def {n.fn}(target):\n"
            "    parsed = urllib.parse.urlparse(target)\n"
            "    if parsed.scheme or parsed.netloc or not target.startswith('/') or target.startswith(('//', '/\\\\')):\n"
            "        return '/'\n"
            "    return target\n\n\n"
            f'@{n.blueprint}.route("{n.route}")\n'
            f"def {n.fn}_view():\n"
            f"    value = request.args.get(\"{n.param}\", \"/\")\n"
            f"    return redirect({n.fn}(value))\n"
        )
    else:
        body = (
            f"def {n.fn}(target):\n"
            "    return target or '/'\n\n\n"
            f'@{n.blueprint}.route("{n.route}")\n'
            f"def {n.fn}_view():\n"
            f"    value = request.args.get(\"{n.param}\", \"/\")\n"
            f"    return redirect({n.fn}(value))\n"
        )
    path = f"web/{n.mod}_routes.py"
    content = _route_file(n, body, "from flask import redirect\n")
    return Rendered({path: content}, [path], f"{path}:{n.fn}_view", f"{path}:{n.route}")


# -- insecure_deserialization
def _r_deser(n: _Names, hops: int, safe: bool, rng: random.Random) -> Rendered:
    variant = rng.choice(["pickle", "yaml"])
    if safe:
        sink = (
            "import json\n\n\n"
            f"def {{fn}}(blob):\n"
            "    data = json.loads(blob)\n"
            "    if not isinstance(data, dict):\n"
            "        raise ValueError(\"expected an object\")\n"
            "    return data\n"
        )
    elif variant == "pickle":
        sink = (
            "import base64\nimport pickle\n\n\n"
            f"def {{fn}}(blob):\n"
            "    return pickle.loads(base64.b64decode(blob))\n"
        )
    else:
        sink = (
            "import yaml\n\n\n"
            f"def {{fn}}(blob):\n"
            "    return yaml.load(blob, Loader=yaml.Loader)\n"
        )

    def route_body(callee):
        src = 'request.get_data(as_text=True)'
        if callee is None:
            return sink.replace("{fn}", n.fn) + "\n\n" + _handler(n, src, f"{n.fn}(value)")
        return _handler(n, src, f"{callee}(value)")

    def service_body(callee):
        if callee is None:
            return sink.replace("{fn}", n.fn)
        return f"def {n.fn}(blob):\n    if not blob:\n        raise ValueError(\"empty payload\")\n    return {callee}(blob)\n"

    return _chain(n, hops, route_body, service_body, sink.replace("{fn}", n.store_fn), n.store_fn, safe)


# -- weak_hash (password storage)
def _r_weakhash(n: _Names, hops: int, safe: bool, rng: random.Random) -> Rendered:
    algo = rng.choice(["md5", "sha1"])
    if safe:
        sink = (
            "import hashlib\nimport os\n\n\n"
            f"def {{fn}}(password):\n"
            "    salt = os.urandom(16)\n"
            "    digest = hashlib.pbkdf2_hmac(\"sha256\", password.encode(), salt, 200_000)\n"
            "    return salt.hex() + \":\" + digest.hex()\n"
        )
    else:
        sink = (
            "import hashlib\n\n\n"
            f"def {{fn}}(password):\n"
            f"    return hashlib.{algo}(password.encode()).hexdigest()\n"
        )

    def route_body(callee):
        src = 'request.form.get("password")'
        if callee is None:
            return sink.replace("{fn}", n.fn) + "\n\n" + _handler(n, src, f"{n.fn}(value)", "digest")
        return _handler(n, src, f"{callee}(value)", "digest")

    def service_body(callee):
        if callee is None:
            return sink.replace("{fn}", n.fn)
        return f"def {n.fn}(password):\n    if len(password) < 8:\n        raise ValueError(\"too short\")\n    return {callee}(password)\n"

    return _chain(n, hops, route_body, service_body, sink.replace("{fn}", n.store_fn), n.store_fn, safe)


# -- hardcoded_secret
def _r_secret(n: _Names, hops: int, safe: bool, rng: random.Random) -> Rendered:
    hops = min(hops, 2)
    token = "".join(rng.choice("0123456789abcdef") for _ in range(32))
    if safe:
        svc = (
            "import os\n\n\n"
            f"def {{fn}}(payload):\n"
            f"    key = os.environ[\"{n.mod.upper()}_API_KEY\"]\n"
            "    return {\"authorization\": \"Bearer \" + key, \"payload\": payload}\n"
        )
    else:
        svc = (
            f"API_KEY = \"sk_live_{token}\"\n\n\n"
            f"def {{fn}}(payload):\n"
            "    return {\"authorization\": \"Bearer \" + API_KEY, \"payload\": payload}\n"
        )

    def route_body(callee):
        src = f'request.args.get("{n.param}")'
        if callee is None:
            return svc.replace("{fn}", n.fn) + "\n\n" + _handler(n, src, f"{n.fn}(value)")
        return _handler(n, src, f"{callee}(value)")

    def service_body(callee):
        return svc.replace("{fn}", n.fn)

    return _chain(n, hops, route_body, service_body, "", n.fn, safe)


# -- weak_random (tokens)
def _r_weakrandom(n: _Names, hops: int, safe: bool, rng: random.Random) -> Rendered:
    hops = min(hops, 2)
    if safe:
        svc = (
            "import secrets\n\n\n"
            f"def {{fn}}(owner):\n"
            "    return owner + \":\" + secrets.token_urlsafe(32)\n"
        )
    else:
        svc = (
            "import random\nimport string\n\n\n"
            f"def {{fn}}(owner):\n"
            "    alphabet = string.ascii_letters + string.digits\n"
            "    return owner + \":\" + \"\".join(random.choice(alphabet) for _ in range(32))\n"
        )

    def route_body(callee):
        src = f'request.args.get("{n.param}")'
        if callee is None:
            return svc.replace("{fn}", n.fn) + "\n\n" + _handler(n, src, f"{n.fn}(value)", "token")
        return _handler(n, src, f"{callee}(value)", "token")

    def service_body(callee):
        return svc.replace("{fn}", n.fn)

    return _chain(n, hops, route_body, service_body, "", n.fn, safe)


# -- eval_injection
def _r_eval(n: _Names, hops: int, safe: bool, rng: random.Random) -> Rendered:
    if safe:
        sink = (
            "import ast\nimport operator\n\n"
            "_OPS = {ast.Add: operator.add, ast.Sub: operator.sub, ast.Mult: operator.mul}\n\n\n"
            "def _fold(node):\n"
            "    if isinstance(node, ast.Constant) and isinstance(node.value, (int, float)):\n"
            "        return node.value\n"
            "    if isinstance(node, ast.BinOp) and type(node.op) in _OPS:\n"
            "        return _OPS[type(node.op)](_fold(node.left), _fold(node.right))\n"
            "    raise ValueError(\"unsupported expression\")\n\n\n"
            f"def {{fn}}(expr):\n"
            "    return _fold(ast.parse(expr, mode=\"eval\").body)\n"
        )
    else:
        sink = (
            f"def {{fn}}(expr):\n"
            "    return eval(expr)\n"
        )

    def route_body(callee):
        src = f'request.args.get("{n.param}")'
        if callee is None:
            return sink.replace("{fn}", n.fn) + "\n\n" + _handler(n, src, f"{n.fn}(value)")
        return _handler(n, src, f"{callee}(value)")

    def service_body(callee):
        if callee is None:
            return sink.replace("{fn}", n.fn)
        return f"def {n.fn}(expr):\n    expr = expr.replace(' ', '')\n    return {callee}(expr)\n"

    return _chain(n, hops, route_body, service_body, sink.replace("{fn}", n.store_fn), n.store_fn, safe)


# -- xss (holdout-only class)
def _r_xss(n: _Names, hops: int, safe: bool, rng: random.Random) -> Rendered:
    hops = 1
    if safe:
        body = (
            f'@{n.blueprint}.route("{n.route}")\n'
            f"def {n.fn}_view():\n"
            f"    value = request.args.get(\"{n.param}\", \"\")\n"
            f"    return render_template_string(\"<h1>{{{{ label }}}}</h1>\", label=value)\n"
        )
    else:
        body = (
            f'@{n.blueprint}.route("{n.route}")\n'
            f"def {n.fn}_view():\n"
            f"    value = request.args.get(\"{n.param}\", \"\")\n"
            f"    return render_template_string(\"<h1>\" + value + \"</h1>\")\n"
        )
    path = f"web/{n.mod}_routes.py"
    content = _route_file(n, body, "from flask import render_template_string\n")
    return Rendered({path: content}, [path], f"{path}:{n.fn}_view", f"{path}:{n.route}")


BUG_CLASSES: Dict[str, BugClass] = {
    "sql_injection": BugClass("sql_injection", "CWE-89", ["sql injection", "sqli", "cwe-89", "sql query", "query interpolation"], _r_sqli),
    "command_injection": BugClass("command_injection", "CWE-78", ["command injection", "os command", "cwe-78", "shell=true", "os.system", "shell command"], _r_cmdi),
    "path_traversal": BugClass("path_traversal", "CWE-22", ["path traversal", "directory traversal", "cwe-22", "arbitrary file read", "../"], _r_traversal),
    "ssrf": BugClass("ssrf", "CWE-918", ["ssrf", "server-side request forgery", "server side request forgery", "cwe-918", "arbitrary url"], _r_ssrf),
    "open_redirect": BugClass("open_redirect", "CWE-601", ["open redirect", "cwe-601", "unvalidated redirect", "redirect to arbitrary"], _r_redirect, dead_ok=False),
    "insecure_deserialization": BugClass("insecure_deserialization", "CWE-502", ["deserializ", "pickle.loads", "yaml.load", "cwe-502", "untrusted data"], _r_deser),
    "weak_hash": BugClass("weak_hash", "CWE-328", ["md5", "sha1", "weak hash", "cwe-328", "cwe-327", "unsalted"], _r_weakhash),
    "hardcoded_secret": BugClass("hardcoded_secret", "CWE-798", ["hardcoded", "hard-coded", "cwe-798", "api key", "secret in source"], _r_secret, dead_ok=False, vendor_ok=False),
    "weak_random": BugClass("weak_random", "CWE-330", ["weak random", "insufficiently random", "cwe-330", "cwe-338", "random.choice", "predictable token"], _r_weakrandom, dead_ok=False),
    "eval_injection": BugClass("eval_injection", "CWE-95", ["eval(", "eval (", "dynamic eval", "unsafe eval", "code injection", "cwe-95", "cwe-94", "arbitrary code"], _r_eval),
    "xss": BugClass("xss", "CWE-79", ["xss", "cross-site scripting", "cross site scripting", "cwe-79", "html injection", "ssti", "template injection", "cwe-1336", "cwe-94"], _r_xss, dead_ok=False),
}


# --- Benign filler ----------------------------------------------------------------------

_BENIGN_FUNCS = [
    (
        "def {name}(items, page=1, per_page=25):\n"
        "    \"\"\"Returns one page of ``items`` plus paging metadata.\"\"\"\n"
        "    page = max(1, int(page))\n"
        "    per_page = min(100, max(1, int(per_page)))\n"
        "    start = (page - 1) * per_page\n"
        "    chunk = list(items)[start:start + per_page]\n"
        "    return {{\"items\": chunk, \"page\": page, \"per_page\": per_page, \"total\": len(items)}}\n"
    ),
    (
        "def {name}(record, fields=(\"id\", \"name\", \"updated_at\")):\n"
        "    \"\"\"Projects ``record`` onto the public field set.\"\"\"\n"
        "    return {{key: record.get(key) for key in fields if key in record}}\n"
    ),
    (
        "def {name}(value, default=0):\n"
        "    \"\"\"Parses an integer query parameter leniently.\"\"\"\n"
        "    try:\n"
        "        return int(str(value).strip())\n"
        "    except (TypeError, ValueError):\n"
        "        return default\n"
    ),
    (
        "def {name}(rows):\n"
        "    \"\"\"Groups rows by their ``status`` column.\"\"\"\n"
        "    groups = {{}}\n"
        "    for row in rows:\n"
        "        groups.setdefault(row.get(\"status\", \"unknown\"), []).append(row)\n"
        "    return groups\n"
    ),
    (
        "def {name}(amount_cents, currency=\"USD\"):\n"
        "    \"\"\"Formats a minor-unit amount for display.\"\"\"\n"
        "    sign = \"-\" if amount_cents < 0 else \"\"\n"
        "    whole, minor = divmod(abs(int(amount_cents)), 100)\n"
        "    return f\"{{sign}}{{whole:,}}.{{minor:02d}} {{currency}}\"\n"
    ),
    (
        "def {name}(timestamps):\n"
        "    \"\"\"Returns the latest ISO-8601 timestamp, or None.\"\"\"\n"
        "    cleaned = [t for t in timestamps if isinstance(t, str) and t]\n"
        "    return max(cleaned) if cleaned else None\n"
    ),
    (
        "def {name}(payload, required=(\"id\",)):\n"
        "    \"\"\"Lists the required keys missing from ``payload``.\"\"\"\n"
        "    return [key for key in required if key not in payload]\n"
    ),
    (
        "def {name}(text, width=72):\n"
        "    \"\"\"Soft-wraps ``text`` at word boundaries.\"\"\"\n"
        "    words, lines, current = text.split(), [], \"\"\n"
        "    for word in words:\n"
        "        if len(current) + len(word) + 1 > width and current:\n"
        "            lines.append(current)\n"
        "            current = word\n"
        "        else:\n"
        "            current = (current + \" \" + word).strip()\n"
        "    if current:\n"
        "        lines.append(current)\n"
        "    return \"\\n\".join(lines)\n"
    ),
    (
        "class {Name}Cache:\n"
        "    \"\"\"A bounded in-process cache with insertion-order eviction.\"\"\"\n"
        "\n"
        "    def __init__(self, capacity=256):\n"
        "        self.capacity = capacity\n"
        "        self._data = {{}}\n"
        "\n"
        "    def get(self, key, default=None):\n"
        "        return self._data.get(key, default)\n"
        "\n"
        "    def put(self, key, value):\n"
        "        if key not in self._data and len(self._data) >= self.capacity:\n"
        "            oldest = next(iter(self._data))\n"
        "            del self._data[oldest]\n"
        "        self._data[key] = value\n"
        "\n\n"
        "def {name}(*args, **kwargs):\n"
        "    return {Name}Cache()\n"
    ),
]

_BENIGN_ROUTE = (
    '@{bp}.route("/{mod}/{path}")\n'
    "def {fn}_view():\n"
    "    page = request.args.get(\"page\", 1)\n"
    "    per_page = request.args.get(\"per_page\", 25)\n"
    "    rows = {mod}_service.{fn}(page, per_page)\n"
    "    return jsonify(rows)\n"
)


def _benign_module(rng: random.Random, mod: str, n_funcs: int, with_routes: bool) -> Dict[str, str]:
    files: Dict[str, str] = {}
    names = [f"{rng.choice(['build', 'load', 'shape', 'merge', 'select', 'render', 'summarize', 'prepare'])}_{mod}_{i}" for i in range(n_funcs)]
    funcs = []
    for name in names:
        tmpl = rng.choice(_BENIGN_FUNCS)
        funcs.append(tmpl.format(name=name, Name=name.title().replace("_", "")))
    files[f"services/{mod}.py"] = f'"""Business logic for {mod}."""\n\n' + "\n\n".join(funcs) + "\n"
    if with_routes:
        bp = f"{mod}_bp"
        head = _FLASK_ROUTE_HEAD.format(imports=f"from services import {mod} as {mod}_service\n", blueprint=bp, mod=mod)
        routes = [
            _BENIGN_ROUTE.format(bp=bp, mod=mod, path=rng.choice(["list", "recent", "summary", "search"]), fn=names[0]),
        ]
        files[f"web/{mod}_routes.py"] = head + "\n\n".join(routes)
    return files


# --- Target assembly --------------------------------------------------------------------


@dataclass
class TargetSpec:
    seed: int
    size: str
    skin: str
    split: str
    seeded_classes: List[str]
    decoy_plan: List[Tuple[str, str]]   # (kind, class)
    benign_modules: int


def plan(seed: int, size: str = "medium", skin: Optional[str] = None) -> TargetSpec:
    """Decides what a target contains. Pure function of (seed, size, skin)."""
    if size not in _SIZES:
        raise ValueError(f"unknown size {size!r}; known: {sorted(_SIZES)}")
    rng = random.Random(f"plan:{seed}")
    split = seed_split(seed)
    skins = list(SKINS)
    if skin is None:
        dev_skins = [k for k in skins if k not in HOLDOUT_ONLY_SKINS]
        skin = dev_skins[0] if split != "holdout" else rng.choice(skins)
    if skin not in SKINS:
        raise ValueError(f"unknown skin {skin!r}; known: {sorted(SKINS)}")
    if split != "holdout" and skin in HOLDOUT_ONLY_SKINS:
        raise HoldoutViolation(f"skin {skin!r} is holdout-only and may not be generated for non-holdout seed {seed}")
    classes = [c for c in BUG_CLASSES if split == "holdout" or c not in HOLDOUT_ONLY_CLASSES]
    if split == "holdout":
        # Every holdout target carries at least one holdout-only class.
        must = sorted(HOLDOUT_ONLY_CLASSES)
        rest = [c for c in classes if c not in HOLDOUT_ONLY_CLASSES]
        rng.shuffle(rest)
        seeded = must + rest[: max(8, len(rest) - 2)]
    else:
        rest = list(classes)
        rng.shuffle(rest)
        seeded = rest[: max(8, len(rest) - 2)]
    # A few classes appear twice, as different variants in different modules.
    extra = rng.sample(seeded, k=min(3, len(seeded)))
    seeded = seeded + extra
    decoy_plan: List[Tuple[str, str]] = []
    pool = [c for c in seeded if c in BUG_CLASSES]
    for cls in rng.sample(pool, k=min(3, len(pool))):
        decoy_plan.append(("decoy_sanitized", cls))
    dead_pool = [c for c in pool if BUG_CLASSES[c].dead_ok]
    for cls in rng.sample(dead_pool, k=min(2, len(dead_pool))):
        decoy_plan.append(("decoy_dead", cls))
    vendor_pool = [c for c in pool if BUG_CLASSES[c].vendor_ok]
    for cls in rng.sample(vendor_pool, k=min(2, len(vendor_pool))):
        decoy_plan.append(("decoy_vendor", cls))
    benign_modules, _ = _SIZES[size]
    return TargetSpec(seed, size, skin, split, seeded, decoy_plan, benign_modules)


def _fresh_names(rng: random.Random, used: set, cls: str) -> _Names:
    while True:
        mod = rng.choice(_DOMAINS)
        if mod not in used:
            used.add(mod)
            break
    verb = rng.choice(["lookup", "fetch", "load", "resolve", "export", "preview", "import", "sync"])
    fn = f"{verb}_{mod}"
    if cls == "path_traversal":
        param = rng.choice(_PATH_PARAMS)
    elif cls == "ssrf":
        param = rng.choice(_URL_PARAMS)
    elif cls == "open_redirect":
        param = rng.choice(_REDIRECT_PARAMS)
    else:
        param = rng.choice(_QUERY_PARAMS)
    route = f"/api/{mod}/{rng.choice(['search', 'open', 'run', 'fetch', 'preview', 'export'])}"
    return _Names(mod=mod, route=route, param=param, fn=fn, store_fn=f"{verb}_{mod}_row")


def generate(seed: int, size: str = "medium", skin: Optional[str] = None) -> Tuple[Dict[str, str], Dict[str, Any]]:
    """Returns ``(tree, ground_truth)``. Deterministic in its arguments."""
    spec = plan(seed, size, skin)
    rng = random.Random(f"gen:{seed}:{size}:{spec.skin}")
    tree: Dict[str, str] = {}
    entries: List[Dict[str, Any]] = []
    used_mods: set = set()
    blueprints: List[Tuple[str, str]] = []  # (module path, blueprint var)

    def _emit(rendered: Rendered) -> None:
        for rel, content in rendered.files.items():
            if rel in tree:
                raise RuntimeError(f"generator collision on {rel} (seed {seed})")
            tree[rel] = content

    # Seeded bugs.
    for i, cls in enumerate(spec.seeded_classes):
        bug = BUG_CLASSES[cls]
        names = _fresh_names(rng, used_mods, cls)
        hops = rng.choice([1, 2, 3, 3, 2]) if cls not in ("open_redirect", "xss") else 1
        rendered = bug.render(names, hops, False, rng)
        _emit(rendered)
        blueprints.append((f"web.{names.mod}_routes", names.blueprint))
        entries.append({
            "id": f"{cls}-{names.mod}",
            "kind": "seeded",
            "files": rendered.chain,
            "sink": rendered.sink,
            "source": rendered.source,
            "cwe": bug.cwe,
            "class": cls,
            "cross_file": len(rendered.chain) > 1,
            "match_any": list(bug.phrases),
        })

    # Decoys.
    for kind, cls in spec.decoy_plan:
        bug = BUG_CLASSES[cls]
        names = _fresh_names(rng, used_mods, cls)
        if kind == "decoy_sanitized":
            rendered = bug.render(names, rng.choice([2, 3]), True, rng)
            _emit(rendered)
            blueprints.append((f"web.{names.mod}_routes", names.blueprint))
            files = rendered.chain
            sink = rendered.sink
        elif kind == "decoy_dead":
            # The same sink, unreferenced: no route, no import anywhere.
            rendered = bug.render(names, 1, False, rng)
            route_path = next(iter(rendered.files))
            body = rendered.files[route_path]
            dead_path = f"services/legacy_{names.mod}.py"
            # Strip the Flask route wrapper so nothing registers it.
            lines = body.splitlines()
            keep = [ln for ln in lines if not ln.startswith("from flask") and "Blueprint(" not in ln]
            text = "\n".join(keep).strip() + "\n"
            text = text.split(f"@{names.blueprint}.route")[0].rstrip() + "\n"
            tree[dead_path] = f'"""Retained for the {names.mod} migration; superseded by services/{names.mod}."""\n\n' + text
            files = [dead_path]
            sink = f"{dead_path}:{names.fn}"
        else:  # decoy_vendor
            rendered = bug.render(names, 1, False, rng)
            route_path = next(iter(rendered.files))
            body = rendered.files[route_path]
            where = rng.choice(["tests", "vendor"])
            if where == "tests":
                vendor_path = f"tests/test_{names.mod}_fixtures.py"
                text = (
                    "import pytest\n\n"
                    + "\n".join(ln for ln in body.splitlines() if not ln.startswith("from flask") and "Blueprint(" not in ln and not ln.startswith("@"))
                    + f"\n\n\ndef test_{names.fn}_smoke():\n    assert callable({names.fn})\n"
                )
            else:
                vendor_path = f"vendor/{names.mod}kit/{names.mod}kit/core.py"
                text = (
                    f'"""{names.mod}kit 0.3.1 -- bundled copy; do not edit."""\n\n'
                    + "\n".join(ln for ln in body.splitlines() if not ln.startswith("from flask") and "Blueprint(" not in ln and not ln.startswith("@"))
                    + "\n"
                )
            text = text.split(f"def {names.fn}_view")[0].rstrip() + "\n" if f"def {names.fn}_view" in text else text
            while "\n\n\n\n" in text:
                text = text.replace("\n\n\n\n", "\n\n\n")
            tree[vendor_path] = text
            files = [vendor_path]
            sink = f"{vendor_path}:{names.fn}"
        entries.append({
            "id": f"{kind}-{cls}-{names.mod}",
            "kind": kind,
            "files": files,
            "sink": sink,
            "source": "",
            "cwe": bug.cwe,
            "class": cls,
            "cross_file": len(files) > 1,
            "match_any": list(bug.phrases),
        })

    # Benign filler, sized.
    _, per_mod = _SIZES[size]
    n_benign = spec.benign_modules
    benign_mods: List[str] = []
    k = 0
    while len(benign_mods) < n_benign:
        base = _DOMAINS[k % len(_DOMAINS)]
        mod = base if k < len(_DOMAINS) else f"{base}_{k // len(_DOMAINS)}"
        k += 1
        if mod in used_mods:
            continue
        used_mods.add(mod)
        benign_mods.append(mod)
    for mod in benign_mods:
        with_routes = rng.random() < 0.6
        files = _benign_module(rng, mod, rng.randint(max(2, per_mod - 2), per_mod + 2), with_routes)
        for rel, content in files.items():
            if rel not in tree:
                tree[rel] = content
        if with_routes:
            blueprints.append((f"web.{mod}_routes", f"{mod}_bp"))

    # Application scaffold.
    tree["app.py"] = (
        '"""Application entry point."""\n\nfrom flask import Flask\n\n'
        + "".join(f"from {m} import {bp}\n" for m, bp in sorted(blueprints))
        + "\n\ndef create_app():\n    app = Flask(__name__)\n"
        + "".join(f"    app.register_blueprint({bp})\n" for _, bp in sorted(blueprints))
        + "    return app\n\n\nif __name__ == \"__main__\":\n    create_app().run()\n"
    )
    tree["config.py"] = (
        '"""Runtime configuration, read from the environment."""\n\nimport os\n\n'
        "DEBUG = os.environ.get(\"APP_DEBUG\", \"0\") == \"1\"\n"
        "DATABASE_PATH = os.environ.get(\"APP_DB\", \"data/app.db\")\n"
        "REQUEST_TIMEOUT = float(os.environ.get(\"APP_TIMEOUT\", \"5\"))\n"
    )
    for pkg in ("web", "services", "dal", "utils", "tests"):
        tree.setdefault(f"{pkg}/__init__.py", "")
    tree["README.md"] = f"# {spec.skin} service {seed}\n\nInternal service. Run `python app.py`.\n"
    tree["requirements.txt"] = "flask\nrequests\npyyaml\n"

    gt = {
        "format_version": FORMAT_VERSION,
        "target": f"generated/{spec.skin}/{seed}/{size}",
        "generator": {"seed": seed, "size": size, "skin": spec.skin, "split": spec.split},
        "notes": (
            "Generated by evals/research_targets/gen.py. 'seeded' entries are the bugs "
            "recall counts; 'decoy_*' entries are certain false positives."
        ),
        "findings": entries,
    }
    return tree, gt


def write_target(dest: Path, seed: int, size: str = "medium", skin: Optional[str] = None) -> Dict[str, Any]:
    """Writes the tree and ground_truth.json under ``dest``; returns the ground truth."""
    tree, gt = generate(seed, size, skin)
    dest = Path(dest)
    dest.mkdir(parents=True, exist_ok=True)
    for rel, content in tree.items():
        path = dest / rel
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(content, encoding="utf-8")
    (dest / GROUND_TRUTH_BASENAME).write_text(json.dumps(gt, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return gt


def banned_words_in_tree(tree: Dict[str, str]) -> List[Tuple[str, str]]:
    """Returns (path, word) pairs for any generated file that names a flaw."""
    hits: List[Tuple[str, str]] = []
    for rel, content in tree.items():
        low = content.lower()
        for word in _BANNED_IN_TREE:
            if word in low:
                hits.append((rel, word))
    return hits


def main(argv: Optional[Sequence[str]] = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--seed", type=int)
    parser.add_argument("--size", default="medium", choices=sorted(_SIZES))
    parser.add_argument("--skin", default=None)
    parser.add_argument("--out", type=Path)
    parser.add_argument("--list", choices=["dev", "holdout"], help="Print the seed range and exit.")
    parser.add_argument("--allow-holdout", action="store_true", help="Required to write a holdout seed.")
    args = parser.parse_args(argv)
    if args.list:
        rng = DEV_SEEDS if args.list == "dev" else HOLDOUT_SEEDS
        print(" ".join(str(s) for s in rng))
        return 0
    if args.seed is None or args.out is None:
        parser.error("--seed and --out are required unless --list is given")
    if seed_split(args.seed) == "holdout" and not args.allow_holdout:
        parser.error(f"seed {args.seed} is a holdout seed; pass --allow-holdout to write it for a confirmation run")
    gt = write_target(args.out, args.seed, args.size, args.skin)
    seeded = sum(1 for e in gt["findings"] if e["kind"] == "seeded")
    decoys = len(gt["findings"]) - seeded
    print(f"wrote {args.out}: seed={args.seed} size={args.size} skin={gt['generator']['skin']} "
          f"split={gt['generator']['split']} seeded={seeded} decoys={decoys}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
