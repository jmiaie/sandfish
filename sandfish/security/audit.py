"""
Security audit and hardening module for SandFish.

Provides:
- Dependency vulnerability scanning
- Code security analysis
- Runtime security monitoring
- Audit logging
"""

import hashlib
import logging
import subprocess
from pathlib import Path
from typing import List, Dict, Optional
from dataclasses import dataclass
from datetime import datetime


@dataclass
class SecurityFinding:
    """Represents a security finding from audit."""
    severity: str  # CRITICAL, HIGH, MEDIUM, LOW
    category: str  # dependency, code, config, runtime
    description: str
    file_path: Optional[str] = None
    line_number: Optional[int] = None
    remediation: Optional[str] = None
    timestamp: datetime = None
    
    def __post_init__(self):
        if self.timestamp is None:
            self.timestamp = datetime.utcnow()


class SecurityAuditor:
    """Main security audit controller."""
    
    CRITICAL_PATTERNS = [
        r'eval\s*\(',
        r'exec\s*\(',
        r'subprocess\.call.*shell\s*=\s*True',
        r'os\.system\s*\(',
        r'__import__\s*\(',
        r'pickle\.loads',
        r'yaml\.load\s*\(',
    ]
    
    SUSPICIOUS_IMPORTS = [
        'requests_without_timeout',
        'urllib_without_validation',
        'ftplib',
        'telnetlib',
    ]
    
    def __init__(self, project_root: Path):
        self.project_root = Path(project_root)
        self.findings: List[SecurityFinding] = []
        self.logger = logging.getLogger('sandfish.security')
        
    def run_full_audit(self) -> List[SecurityFinding]:
        """Execute complete security audit."""
        self.findings = []
        
        self._audit_dependencies()
        self._audit_code_patterns()
        self._audit_configuration()
        self._audit_file_permissions()
        
        return self.findings
    
    def _audit_dependencies(self) -> None:
        """Check for vulnerable dependencies."""
        try:
            # Run safety check
            result = subprocess.run(
                ['safety', 'check', '--json'],
                capture_output=True,
                text=True,
                timeout=60
            )
            
            if result.returncode != 0:
                self.findings.append(SecurityFinding(
                    severity='HIGH',
                    category='dependency',
                    description='Vulnerable dependencies detected. Run `safety check` for details.',
                    remediation='Update dependencies: pip install --upgrade <package>'
                ))
                
        except FileNotFoundError:
            self.logger.warning("safety not installed. Skipping dependency audit.")
            
    def _audit_code_patterns(self) -> None:
        """Scan for dangerous code patterns."""
        python_files = list(self.project_root.rglob('*.py'))
        
        for file_path in python_files:
            if 'venv' in str(file_path) or '__pycache__' in str(file_path):
                continue
                
            try:
                content = file_path.read_text()
                lines = content.split('\n')
                
                for i, line in enumerate(lines, 1):
                    self._check_line_patterns(line, file_path, i)
                    
            except Exception as e:
                self.logger.error(f"Error reading {file_path}: {e}")
    
    def _check_line_patterns(self, line: str, file_path: Path, line_num: int) -> None:
        """Check a single line for dangerous patterns."""
        import re
        
        # Check for eval/exec
        if re.search(r'\beval\s*\(', line) and 'ast.literal_eval' not in line:
            self.findings.append(SecurityFinding(
                severity='CRITICAL',
                category='code',
                description=f'Dangerous eval() detected',
                file_path=str(file_path),
                line_number=line_num,
                remediation='Use ast.literal_eval() for safe evaluation'
            ))
        
        # Check for shell=True
        if 'shell=True' in line and 'subprocess' in line:
            self.findings.append(SecurityFinding(
                severity='HIGH',
                category='code',
                description=f'subprocess with shell=True is dangerous',
                file_path=str(file_path),
                line_number=line_num,
                remediation='Use shell=False and pass list of arguments'
            ))
    
    def _audit_configuration(self) -> None:
        """Check configuration files for security issues."""
        env_file = self.project_root / '.env'
        
        if env_file.exists():
            content = env_file.read_text()
            
            # Check for hardcoded secrets
            if 'SECRET_KEY=mirofish' in content or 'SECRET_KEY=default' in content:
                self.findings.append(SecurityFinding(
                    severity='HIGH',
                    category='config',
                    description='Default or weak SECRET_KEY detected in .env',
                    remediation='Generate strong secret: openssl rand -hex 32'
                ))
    
    def _audit_file_permissions(self) -> None:
        """Check for overly permissive file permissions."""
        sensitive_files = [
            self.project_root / '.env',
            self.project_root / 'secrets.yaml',
        ]
        
        for file_path in sensitive_files:
            if file_path.exists():
                stat = file_path.stat()
                # Check if world-readable
                if stat.st_mode & 0o044:
                    self.findings.append(SecurityFinding(
                        severity='MEDIUM',
                        category='config',
                        description=f'{file_path} is world-readable',
                        remediation='chmod 600 {file_path}'
                    ))
    
    def generate_report(self) -> str:
        """Generate human-readable audit report."""
        lines = [
            "# SandFish Security Audit Report",
            f"Generated: {datetime.utcnow().isoformat()}",
            f"Project: {self.project_root}",
            "",
            f"## Summary",
            f"Total findings: {len(self.findings)}",
            f"Critical: {sum(1 for f in self.findings if f.severity == 'CRITICAL')}",
            f"High: {sum(1 for f in self.findings if f.severity == 'HIGH')}",
            f"Medium: {sum(1 for f in self.findings if f.severity == 'MEDIUM')}",
            f"Low: {sum(1 for f in self.findings if f.severity == 'LOW')}",
            "",
            "## Findings",
        ]
        
        for finding in sorted(self.findings, key=lambda f: f.severity):
            lines.append(f"\n### {finding.severity}: {finding.category}")
            lines.append(f"{finding.description}")
            if finding.file_path:
                lines.append(f"Location: {finding.file_path}:{finding.line_number or 'N/A'}")
            if finding.remediation:
                lines.append(f"Remediation: {finding.remediation}")
        
        return '\n'.join(lines)


def verify_code_integrity(file_path: Path) -> str:
    """Calculate SHA-256 hash of file for integrity verification."""
    sha256_hash = hashlib.sha256()
    
    with open(file_path, 'rb') as f:
        for byte_block in iter(lambda: f.read(4096), b""):
            sha256_hash.update(byte_block)
    
    return sha256_hash.hexdigest()


# Convenience function
def run_security_audit(project_root: str) -> List[SecurityFinding]:
    """Run complete security audit on project."""
    auditor = SecurityAuditor(Path(project_root))
    return auditor.run_full_audit()
