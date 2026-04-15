"""
Security audit and hardening module for SandFish.

Provides:
- Code pattern analysis (eval/exec, shell=True, weak crypto, etc.)
- Configuration analysis (hardcoded secrets, weak defaults)
- Dependency vulnerability scan (best-effort, via `safety` if installed)
- File permission checks (POSIX only)
- File integrity hashes
"""

from __future__ import annotations

import hashlib
import json
import logging
import os
import re
import shutil
import subprocess
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Iterable, List, Optional


def _utcnow() -> datetime:
    return datetime.now(timezone.utc)


@dataclass
class SecurityFinding:
    """A single audit finding."""
    severity: str  # CRITICAL, HIGH, MEDIUM, LOW
    category: str  # dependency, code, config, runtime
    description: str
    file_path: Optional[str] = None
    line_number: Optional[int] = None
    remediation: Optional[str] = None
    timestamp: datetime = field(default_factory=_utcnow)


# Patterns that are nearly always wrong in production code.
_CODE_PATTERNS: List[tuple] = [
    # (severity, regex, description, remediation)
    (
        "CRITICAL",
        re.compile(r"(?<!\.)\beval\s*\("),
        "eval() detected",
        "Use ast.literal_eval() for trusted literals or a proper parser.",
    ),
    (
        "CRITICAL",
        re.compile(r"(?<!\.)\bexec\s*\("),
        "exec() detected",
        "Avoid runtime code execution; refactor to call known functions.",
    ),
    (
        "HIGH",
        re.compile(r"subprocess\.[A-Za-z_]+\([^)]*shell\s*=\s*True"),
        "subprocess called with shell=True",
        "Pass argument list and shell=False; quote with shlex if needed.",
    ),
    (
        "HIGH",
        re.compile(r"\bos\.system\s*\("),
        "os.system() detected",
        "Use subprocess.run([...], shell=False) instead.",
    ),
    (
        "HIGH",
        re.compile(r"pickle\.loads?\("),
        "pickle.load[s]() on untrusted data is unsafe",
        "Use json or msgpack for cross-process data.",
    ),
    (
        "MEDIUM",
        re.compile(r"yaml\.load\s*\((?![^)]*Loader\s*=)"),
        "yaml.load() without Loader argument is unsafe",
        "Use yaml.safe_load() instead.",
    ),
    (
        "MEDIUM",
        re.compile(r"hashlib\.(md5|sha1)\("),
        "Weak hash algorithm (MD5/SHA1)",
        "Use SHA-256 or BLAKE2 for non-legacy hashing.",
    ),
    (
        "LOW",
        re.compile(r"requests\.[a-z]+\([^)]*\)(?![^#\n]*timeout)"),
        "HTTP request without explicit timeout",
        "Add timeout=<seconds> to avoid indefinite hangs.",
    ),
    (
        "LOW",
        re.compile(r"verify\s*=\s*False"),
        "TLS verification disabled",
        "Remove verify=False; trust the system CA store.",
    ),
]


# Default exclusion globs (anywhere in the path).
_DEFAULT_EXCLUDED_DIR_NAMES = {
    "venv",
    ".venv",
    "env",
    "__pycache__",
    "build",
    "dist",
    "node_modules",
    ".git",
    ".tox",
    ".mypy_cache",
    ".pytest_cache",
    "site-packages",
    ".eggs",
}


class SecurityAuditor:
    """Run a security audit against a project directory."""

    def __init__(
        self,
        project_root: Path,
        excluded_dirs: Optional[Iterable[str]] = None,
    ):
        self.project_root = Path(project_root)
        self.findings: List[SecurityFinding] = []
        self.logger = logging.getLogger("sandfish.security")
        self._excluded_dirs = set(_DEFAULT_EXCLUDED_DIR_NAMES) | set(excluded_dirs or ())

    # ----- Entry point -----

    def run_full_audit(self) -> List[SecurityFinding]:
        self.findings = []
        self._audit_dependencies()
        self._audit_code_patterns()
        self._audit_configuration()
        self._audit_file_permissions()
        return self.findings

    # ----- Dependency check -----

    def _audit_dependencies(self) -> None:
        if not shutil.which("safety"):
            self.logger.info("`safety` CLI not on PATH; skipping dependency audit.")
            return

        # `safety scan` is the modern subcommand; fall back to `check` for older versions.
        for cmd in (["safety", "scan", "--output", "json"], ["safety", "check", "--json"]):
            try:
                result = subprocess.run(
                    cmd,
                    capture_output=True,
                    text=True,
                    timeout=120,
                    cwd=str(self.project_root),
                    check=False,
                )
            except FileNotFoundError:
                self.logger.info("`safety` not installed; skipping dependency audit.")
                return
            except subprocess.SubprocessError as exc:
                self.logger.warning("`safety` invocation failed: %s", exc)
                continue

            if result.returncode == 0 and not result.stdout.strip():
                # Modern safety prints empty output for clean repos.
                return
            if result.returncode == 0:
                # Try to parse JSON; even success can include advisories.
                try:
                    data = json.loads(result.stdout)
                except json.JSONDecodeError:
                    return
                vulns = data.get("vulnerabilities") or data.get("advisories") or []
                if not vulns:
                    return
                for vuln in vulns:
                    self.findings.append(
                        SecurityFinding(
                            severity="HIGH",
                            category="dependency",
                            description=str(vuln.get("advisory") or vuln),
                            remediation="Upgrade the affected package.",
                        )
                    )
                return
            # Non-zero exit → vulnerabilities probably found.
            self.findings.append(
                SecurityFinding(
                    severity="HIGH",
                    category="dependency",
                    description="`safety` reported vulnerable dependencies.",
                    remediation="Run `safety scan` for details and upgrade affected packages.",
                )
            )
            return

    # ----- Code scan -----

    def _audit_code_patterns(self) -> None:
        for file_path in self._iter_python_files():
            try:
                content = file_path.read_text(encoding="utf-8", errors="replace")
            except OSError as exc:
                self.logger.warning("Could not read %s: %s", file_path, exc)
                continue

            for line_num, line in enumerate(content.splitlines(), start=1):
                # Skip pure comments.
                stripped = line.strip()
                if stripped.startswith("#"):
                    continue
                self._check_line_patterns(line, file_path, line_num)

    def _check_line_patterns(self, line: str, file_path: Path, line_num: int) -> None:
        for severity, pattern, description, remediation in _CODE_PATTERNS:
            if pattern.search(line):
                # ast.literal_eval is the safe variant — don't flag it.
                if "ast.literal_eval" in line and "eval" in description:
                    continue
                self.findings.append(
                    SecurityFinding(
                        severity=severity,
                        category="code",
                        description=description,
                        file_path=str(file_path),
                        line_number=line_num,
                        remediation=remediation,
                    )
                )

    # ----- Config + filesystem -----

    def _audit_configuration(self) -> None:
        env_file = self.project_root / ".env"
        if not env_file.exists():
            return

        try:
            content = env_file.read_text(encoding="utf-8", errors="replace")
        except OSError:
            return

        for line in content.splitlines():
            line = line.strip()
            if not line or line.startswith("#") or "=" not in line:
                continue
            key, _, value = line.partition("=")
            key = key.strip().upper()
            value = value.strip().strip("'\"")

            if key in {"SECRET_KEY", "API_KEY", "JWT_SECRET", "SANDFISH_API_KEY"}:
                if not value or value.lower() in {
                    "default",
                    "changeme",
                    "secret",
                    "password",
                    "test",
                }:
                    self.findings.append(
                        SecurityFinding(
                            severity="HIGH",
                            category="config",
                            description=f"{key} has a weak/default value in .env",
                            file_path=str(env_file),
                            remediation="Generate a strong secret: openssl rand -hex 32",
                        )
                    )
                elif len(value) < 16:
                    self.findings.append(
                        SecurityFinding(
                            severity="MEDIUM",
                            category="config",
                            description=f"{key} is shorter than 16 characters",
                            file_path=str(env_file),
                            remediation="Use at least 32 characters of entropy.",
                        )
                    )

    def _audit_file_permissions(self) -> None:
        # POSIX-only check; Windows reports symbolic perms differently.
        if os.name != "posix":
            return

        for filename in (".env", "secrets.yaml", "secrets.yml"):
            target = self.project_root / filename
            if not target.exists():
                continue
            mode = target.stat().st_mode & 0o777
            if mode & 0o077:  # any group/world bits set
                self.findings.append(
                    SecurityFinding(
                        severity="MEDIUM",
                        category="config",
                        description=f"{target.name} permissions are too permissive ({oct(mode)})",
                        file_path=str(target),
                        remediation=f"chmod 600 {target}",
                    )
                )

    # ----- Iteration helpers -----

    def _iter_python_files(self) -> Iterable[Path]:
        for path in self.project_root.rglob("*.py"):
            if any(part in self._excluded_dirs for part in path.parts):
                continue
            yield path

    # ----- Reporting -----

    def generate_report(self) -> str:
        sev_order = {"CRITICAL": 0, "HIGH": 1, "MEDIUM": 2, "LOW": 3}
        sorted_findings = sorted(
            self.findings, key=lambda f: (sev_order.get(f.severity, 99), f.file_path or "")
        )

        lines = [
            "# SandFish Security Audit Report",
            f"Generated: {_utcnow().isoformat()}",
            f"Project: {self.project_root}",
            "",
            "## Summary",
            f"Total findings: {len(self.findings)}",
            f"Critical: {sum(1 for f in self.findings if f.severity == 'CRITICAL')}",
            f"High: {sum(1 for f in self.findings if f.severity == 'HIGH')}",
            f"Medium: {sum(1 for f in self.findings if f.severity == 'MEDIUM')}",
            f"Low: {sum(1 for f in self.findings if f.severity == 'LOW')}",
            "",
            "## Findings",
        ]

        if not sorted_findings:
            lines.append("\nNo issues detected.")
            return "\n".join(lines)

        for finding in sorted_findings:
            lines.append(f"\n### {finding.severity}: {finding.description}")
            lines.append(f"Category: {finding.category}")
            if finding.file_path:
                location = finding.file_path
                if finding.line_number:
                    location += f":{finding.line_number}"
                lines.append(f"Location: {location}")
            if finding.remediation:
                lines.append(f"Remediation: {finding.remediation}")

        return "\n".join(lines)


def verify_code_integrity(file_path: Path) -> str:
    """Return SHA-256 hex digest of a file."""
    sha256 = hashlib.sha256()
    with open(file_path, "rb") as fp:
        for chunk in iter(lambda: fp.read(8192), b""):
            sha256.update(chunk)
    return sha256.hexdigest()


def run_security_audit(project_root: str) -> List[SecurityFinding]:
    """Convenience wrapper: run a full audit and return findings."""
    return SecurityAuditor(Path(project_root)).run_full_audit()
