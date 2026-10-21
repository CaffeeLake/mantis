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

"""The guest the reproducer runs PoCs in has no network and no host mounts.

The reproducer deliberately executes attacker payloads: command injection,
SSRF to attacker-chosen URLs, path traversal, deserialisation. Those are
only safe to run because the microVM is created with egress and ingress
denied and nothing from the host mounted in. This pins that creation call so
a refactor cannot quietly hand the guest a NAT uplink or a bind mount.
"""

from __future__ import annotations

import asyncio
import sys
import unittest
from pathlib import Path
from unittest import mock

_REF_ROOT = Path(__file__).resolve().parent.parent
if str(_REF_ROOT) not in sys.path:
    sys.path.insert(0, str(_REF_ROOT))


class _FakeFs:
    def __init__(self):
        self.dirs: list[str] = []

    async def mkdir(self, path):
        self.dirs.append(str(path))


class _FakeSandbox:
    def __init__(self):
        self.fs = _FakeFs()


class MicrosandboxCreationTest(unittest.TestCase):
    def _create_kwargs(self) -> dict:
        from core.environments import microsandbox_env as M

        env = M.MicrosandboxEnvironment.__new__(M.MicrosandboxEnvironment)
        env.target_path = ""
        env.image = "mantis-sandbox:latest"
        env.timeout = 30.0
        env._workdir = Path("/workspace")
        env._sb = None
        env._name = None
        env._stage_error = None
        env.is_initialized = False

        seen: dict = {}

        async def fake_create(name, **kwargs):
            seen["name"] = name
            seen.update(kwargs)
            return _FakeSandbox()

        with mock.patch.object(M.MsbSandbox, "create", side_effect=fake_create):
            asyncio.run(env.initialize())
        return seen

    def test_the_guest_is_created_with_all_network_denied(self):
        from microsandbox.types import Action, Network

        kwargs = self._create_kwargs()
        self.assertIn("network", kwargs, "the sandbox is created without saying what network it gets")
        net = kwargs["network"]
        self.assertEqual(net, Network.none())
        self.assertEqual(net.policy.default_egress, Action.DENY)
        self.assertEqual(net.policy.default_ingress, Action.DENY)
        self.assertEqual(net.policy.rules, ())
        self.assertEqual(dict(net.ports), {}, "no port of the guest is published to the host")

    def test_the_guest_gets_no_host_mounts_or_environment(self):
        kwargs = self._create_kwargs()
        for key in ("volumes", "mounts", "env", "environment", "privileged"):
            self.assertNotIn(key, kwargs, f"{key!r} would reach into the host from the guest")
        self.assertEqual(kwargs.get("pull_policy"), __import__("microsandbox").PullPolicy.NEVER,
                         "a run must not fetch images; the operator pulls what the guest runs")
        self.assertTrue(str(kwargs["name"]).startswith("mantis-msb-"))

    def test_the_source_says_the_same_thing(self):
        """A second, independent witness: the literal call site. If someone
        routes creation through a helper the test above still holds; this one
        catches a network argument that is computed rather than written."""
        import inspect

        from core.environments import microsandbox_env as M

        src = inspect.getsource(M.MicrosandboxEnvironment._ensure)
        self.assertIn("network=Network.none()", src)


if __name__ == "__main__":
    unittest.main()
