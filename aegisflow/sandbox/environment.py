"""
AegisFlow Sandbox Execution Environment.

This module provides isolated execution boundaries for tools and agents,
ensuring security-first code execution and file system access.
Inspired by DeerFlow's sandbox design.
"""

import os
import subprocess
import logging
from typing import Dict, Any, Optional

logger = logging.getLogger(__name__)

class Sandbox:
    """
    Isolated execution environment for agent operations.
    """

    def __init__(self, workspace_path: str = "./sandbox_workspace", isolation_level: str = "local"):
        """
        Initialize the sandbox.

        :param workspace_path: The root directory for sandbox operations.
        :param isolation_level: 'local' (directory bound) or 'container' (docker/podman bound).
        """
        self.workspace = os.path.abspath(workspace_path)
        self.isolation_level = isolation_level
        self._ensure_directories()

    def _ensure_directories(self) -> None:
        """Create the necessary isolated directories."""
        for d in ["workspace", "uploads", "outputs"]:
            os.makedirs(os.path.join(self.workspace, d), exist_ok=True)

    def _resolve_path(self, relative_path: str) -> str:
        """Safely resolve a path to ensure it stays within the workspace."""
        # Simple path traversal prevention
        target = os.path.abspath(os.path.join(self.workspace, relative_path))
        if not target.startswith(self.workspace):
            raise PermissionError(f"Path traversal detected: {relative_path} escapes workspace.")
        return target

    def write_file(self, relative_path: str, content: str) -> str:
        """Write a file safely within the sandbox."""
        safe_path = self._resolve_path(relative_path)
        os.makedirs(os.path.dirname(safe_path), exist_ok=True)
        with open(safe_path, "w") as f:
            f.write(content)
        return f"Successfully wrote {len(content)} bytes to {relative_path}"

    def read_file(self, relative_path: str) -> str:
        """Read a file safely from the sandbox."""
        safe_path = self._resolve_path(relative_path)
        if not os.path.exists(safe_path):
            raise FileNotFoundError(f"File not found in sandbox: {relative_path}")
        with open(safe_path, "r") as f:
            return f.read()

    def execute_command(self, command: str, timeout: int = 30) -> Dict[str, Any]:
        """
        Execute a shell command within the sandbox context.
        In 'container' mode, this would route to a containerized bash runtime.
        """
        if self.isolation_level != "local":
            return {"error": "Container isolation not fully implemented in this scaffold."}

        logger.warning("Executing host command in local sandbox mode.")

        try:
            # We strictly execute within the workspace directory
            result = subprocess.run(
                command,
                shell=True,
                cwd=self.workspace,
                capture_output=True,
                text=True,
                timeout=timeout
            )
            return {
                "stdout": result.stdout,
                "stderr": result.stderr,
                "exit_code": result.returncode
            }
        except subprocess.TimeoutExpired:
            return {"error": f"Command timed out after {timeout} seconds."}
        except Exception as e:
            return {"error": str(e)}
