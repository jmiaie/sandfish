import pytest
from aegisflow.memory import MemoryVault, KnowledgeGraph, PalaceNavigation
from aegisflow.sandbox import Sandbox
from aegisflow.orchestration import LeadOrchestrator, SubAgent
import os

def test_memory_vault_creation(tmp_path):
    vault = MemoryVault(path=str(tmp_path))
    assert os.path.exists(tmp_path / "brain")
    assert os.path.exists(tmp_path / "work")
    assert os.path.exists(tmp_path / "org")
    assert os.path.exists(tmp_path / "perf")

def test_knowledge_graph_addition():
    kg = KnowledgeGraph()
    kg.add_triple("Agent", "solves", "Task")
    assert len(kg.triples) == 1
    assert kg.query_entity("Agent")[0]["predicate"] == "solves"

def test_sandbox_path_traversal_prevention(tmp_path):
    sandbox = Sandbox(workspace_path=str(tmp_path))
    with pytest.raises(PermissionError):
        sandbox.read_file("../../../../etc/passwd")

def test_orchestrator_delegation(tmp_path):
    memory = MemoryVault(path=str(tmp_path / "vault"))
    sandbox = Sandbox(workspace_path=str(tmp_path / "sandbox"))
    lead = LeadOrchestrator(memory=memory, sandbox=sandbox)

    result = lead.delegate_and_run("Build a secure multi-agent system.")

    assert result["status"] == "success"
    assert len(result["details"]) == 3
    assert lead.sub_agents[0].status == "completed"
