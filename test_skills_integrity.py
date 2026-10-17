"""Dependency-light integrity checks for the Mantis skills.

Runs in a bare clone with nothing installed:

    python test_skills_integrity.py

Checks the skill surface itself; the reference implementation has its own
suites under reference/, which require reference/install.sh.
"""

import ast
import pathlib
import re
import subprocess
import unittest

REPO_ROOT = pathlib.Path(__file__).resolve().parent
SKILLS_DIR = REPO_ROOT / "skills"
SKILLS = sorted(SKILLS_DIR.glob("*/SKILL.md"))
_REF_LINK = re.compile(
    r"\]\(((?:\.\./)?[\w\-./]*references/[\w\-./]+\.md)(?:#[^)]*)?\)"
)


class SkillFrontmatterTests(unittest.TestCase):
    def test_skills_discovered(self):
        self.assertGreater(len(SKILLS), 10)

    def test_frontmatter_declares_name_and_description(self):
        for skill in SKILLS:
            with self.subTest(skill=skill.parent.name):
                lines = skill.read_text(encoding="utf-8").splitlines()
                self.assertEqual(lines[0], "---")
                body = lines[1 : lines[1:].index("---") + 1]
                self.assertTrue(any(l.startswith("name:") for l in body))
                self.assertTrue(any(l.startswith("description:") for l in body))

    def test_frontmatter_name_matches_directory(self):
        for skill in SKILLS:
            with self.subTest(skill=skill.parent.name):
                text = skill.read_text(encoding="utf-8")
                self.assertIn(f"name: {skill.parent.name}", text)


class SkillsIndexTests(unittest.TestCase):
    """skills/README.md is what a reader sees first; it must name every skill
    and nothing else. reference/scripts/gen_skills_index.py regenerates it."""

    def test_index_lists_exactly_the_skills(self):
        index = (SKILLS_DIR / "README.md").read_text(encoding="utf-8")
        linked = set(re.findall(r"\]\(([\w\-]+)/SKILL\.md\)", index))
        present = {s.parent.name for s in SKILLS}
        self.assertEqual(linked, present,
                         "run: python3 reference/scripts/gen_skills_index.py .")

    def test_index_links_resolve(self):
        index = (SKILLS_DIR / "README.md").read_text(encoding="utf-8")
        for rel in re.findall(r"\]\(([^)]+)\)", index):
            with self.subTest(link=rel):
                self.assertTrue((SKILLS_DIR / rel).is_file(), rel)


class ReferenceLinkTests(unittest.TestCase):
    def test_reference_links_resolve_and_are_tracked(self):
        for skill in SKILLS:
            for m in _REF_LINK.finditer(skill.read_text(encoding="utf-8")):
                target = (skill.parent / m.group(1)).resolve()
                with self.subTest(skill=skill.parent.name, link=m.group(1)):
                    self.assertTrue(
                        target.is_file(), f"broken link: {m.group(1)}"
                    )
                    rc = subprocess.run(
                        [
                            "git",
                            "ls-files",
                            "--error-unmatch",
                            str(target.relative_to(REPO_ROOT)),
                        ],
                        cwd=REPO_ROOT,
                        capture_output=True,
                    ).returncode
                    self.assertEqual(
                        rc, 0, "target exists but is not git-tracked"
                    )


class TestLayoutTests(unittest.TestCase):
    """Every test module that runs itself must do so from its last statement.

    A `unittest.main()` guard in the middle of a file exits before the
    classes below it are even defined, so `python tests/test_x.py` reports
    a fraction of the suite as all of it. One file carried 34 silent test
    classes that way; this keeps the next one from getting past review.
    """

    TEST_FILES = sorted(
        list(REPO_ROOT.glob("test_*.py"))
        + list((REPO_ROOT / "reference").glob("test_*.py"))
        + list((REPO_ROOT / "reference" / "tests").glob("test_*.py"))
    )

    @staticmethod
    def _is_main_guard(node: ast.stmt) -> bool:
        if not isinstance(node, ast.If) or not isinstance(node.test, ast.Compare):
            return False
        left = node.test.left
        return isinstance(left, ast.Name) and left.id == "__name__"

    def test_main_guard_is_the_last_statement(self):
        self.assertGreater(len(self.TEST_FILES), 10)
        for path in self.TEST_FILES:
            with self.subTest(file=path.relative_to(REPO_ROOT).as_posix()):
                body = ast.parse(path.read_text(encoding="utf-8")).body
                guards = [i for i, n in enumerate(body) if self._is_main_guard(n)]
                if not guards:
                    continue
                self.assertEqual(
                    guards, [len(body) - 1],
                    f"{len(body) - 1 - guards[0]} top-level statement(s) follow the "
                    "__main__ guard and never run under direct execution",
                )


if __name__ == "__main__":
    unittest.main()
