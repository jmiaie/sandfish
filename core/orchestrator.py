"""
Swarm Orchestrator for SandFish.

Manages distributed multi-agent simulations with OMPA-native memory.
Replaces MiroFish's Zep-dependent orchestration with clean, auditable code.
"""

import asyncio
import uuid
from typing import Dict, List, Optional, Any, Callable
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
import logging

from ..memory.ompa_adapter import OMPAMemoryAdapter


class SimulationStatus(Enum):
    """Simulation execution states."""
    PENDING = "pending"
    RUNNING = "running"
    PAUSED = "paused"
    COMPLETED = "completed"
    FAILED = "failed"


@dataclass
class SimulationConfig:
    """Configuration for a simulation run."""
    name: str
    description: str
    max_rounds: int = 100
    num_agents: int = 10
    agent_types: List[str] = field(default_factory=lambda: ["default"])
    seed_data: Dict[str, Any] = field(default_factory=dict)
    enable_logging: bool = True
    checkpoint_interval: int = 10


@dataclass
class SimulationResult:
    """Results from a completed simulation."""
    simulation_id: str
    status: SimulationStatus
    rounds_completed: int
    final_state: Dict[str, Any]
    metrics: Dict[str, float]
    start_time: datetime
    end_time: Optional[datetime] = None
    error_message: Optional[str] = None


class SwarmOrchestrator:
    """
    Main orchestrator for multi-agent swarm simulations.
    
    Features:
    - Distributed agent management
    - Round-based simulation execution
    - OMPA-native memory and logging
    - Real-time metrics and monitoring
    - Checkpoint/resume capability
    """
    
    def __init__(self, memory_adapter: OMPAMemoryAdapter):
        """
        Initialize orchestrator.
        
        Args:
            memory_adapter: OMPA memory adapter instance
        """
        self.memory = memory_adapter
        self.logger = logging.getLogger('sandfish.orchestrator')
        
        # Active simulations
        self.simulations: Dict[str, Dict[str, Any]] = {}
        
        # Agent registry
        self.agents: Dict[str, 'BaseAgent'] = {}
        
        # Callbacks for events
        self.event_callbacks: List[Callable] = []
        
    def create_simulation(self, config: SimulationConfig) -> str:
        """
        Create a new simulation.
        
        Args:
            config: Simulation configuration
            
        Returns:
            Simulation ID
        """
        sim_id = str(uuid.uuid4())[:8]
        
        self.simulations[sim_id] = {
            'id': sim_id,
            'config': config,
            'status': SimulationStatus.PENDING,
            'current_round': 0,
            'agents': [],
            'start_time': None,
            'checkpoints': []
        }
        
        # Log to OMPA
        self.memory.record_event(
            event_type="SIMULATION_CREATED",
            description=f"Created simulation '{config.name}' with {config.num_agents} agents",
            metadata={
                'simulation_id': sim_id,
                'config': config.__dict__
            }
        )
        
        self.logger.info(f"Created simulation {sim_id}: {config.name}")
        return sim_id
    
    async def run_simulation(self, sim_id: str) -> SimulationResult:
        """
        Execute a simulation.
        
        Args:
            sim_id: Simulation ID
            
        Returns:
            Simulation results
        """
        if sim_id not in self.simulations:
            raise ValueError(f"Simulation {sim_id} not found")
        
        sim = self.simulations[sim_id]
        config = sim['config']
        
        # Initialize
        sim['status'] = SimulationStatus.RUNNING
        sim['start_time'] = datetime.utcnow()
        
        self.logger.info(f"Starting simulation {sim_id}")
        
        try:
            # Create agents
            await self._initialize_agents(sim_id, config)
            
            # Run rounds
            for round_num in range(config.max_rounds):
                sim['current_round'] = round_num
                
                # Execute round
                await self._execute_round(sim_id, round_num)
                
                # Checkpoint if needed
                if round_num % config.checkpoint_interval == 0:
                    await self._create_checkpoint(sim_id)
                
                # Emit progress event
                self._emit_event('round_complete', {
                    'simulation_id': sim_id,
                    'round': round_num,
                    'total_rounds': config.max_rounds
                })
                
                # Check for pause/stop
                if sim['status'] == SimulationStatus.PAUSED:
                    self.logger.info(f"Simulation {sim_id} paused at round {round_num}")
                    return self._create_partial_result(sim)
            
            # Complete
            sim['status'] = SimulationStatus.COMPLETED
            
        except Exception as e:
            sim['status'] = SimulationStatus.FAILED
            sim['error'] = str(e)
            self.logger.error(f"Simulation {sim_id} failed: {e}")
        
        # Generate results
        result = self._create_result(sim)
        
        # Log completion
        self.memory.record_event(
            event_type="SIMULATION_COMPLETED",
            description=f"Simulation '{config.name}' completed with status {result.status.value}",
            metadata={
                'simulation_id': sim_id,
                'rounds': result.rounds_completed,
                'metrics': result.metrics
            }
        )
        
        return result
    
    async def _initialize_agents(self, sim_id: str, config: SimulationConfig) -> None:
        """Initialize agents for simulation."""
        from ..agents.base_agent import create_agent
        
        sim = self.simulations[sim_id]
        
        for i in range(config.num_agents):
            agent_type = config.agent_types[i % len(config.agent_types)]
            agent = create_agent(
                agent_type=agent_type,
                agent_id=f"{sim_id}_agent_{i}",
                memory_adapter=self.memory
            )
            
            sim['agents'].append(agent.id)
            self.agents[agent.id] = agent
            
            # Initialize agent with seed data
            await agent.initialize(config.seed_data)
    
    async def _execute_round(self, sim_id: str, round_num: int) -> None:
        """Execute a single simulation round."""
        sim = self.simulations[sim_id]
        
        # Gather actions from all agents
        actions = []
        for agent_id in sim['agents']:
            agent = self.agents[agent_id]
            action = await agent.decide_action()
            actions.append((agent_id, action))
        
        # Execute actions
        for agent_id, action in actions:
            agent = self.agents[agent_id]
            await agent.execute_action(action)
        
        # Update shared state
        await self._update_shared_state(sim_id, actions)
    
    async def _update_shared_state(self, sim_id: str, actions: List[tuple]) -> None:
        """Update shared simulation state based on agent actions."""
        # This would update the OMPA knowledge graph
        # with new facts from the simulation
        pass
    
    async def _create_checkpoint(self, sim_id: str) -> None:
        """Create a simulation checkpoint."""
        sim = self.simulations[sim_id]
        
        checkpoint = {
            'round': sim['current_round'],
            'timestamp': datetime.utcnow().isoformat(),
            'agent_states': {
                agent_id: self.agents[agent_id].get_state()
                for agent_id in sim['agents']
            }
        }
        
        sim['checkpoints'].append(checkpoint)
        
        self.logger.debug(f"Created checkpoint for {sim_id} at round {checkpoint['round']}")
    
    def pause_simulation(self, sim_id: str) -> None:
        """Pause a running simulation."""
        if sim_id in self.simulations:
            self.simulations[sim_id]['status'] = SimulationStatus.PAUSED
            self.logger.info(f"Paused simulation {sim_id}")
    
    def resume_simulation(self, sim_id: str) -> None:
        """Resume a paused simulation."""
        if sim_id in self.simulations:
            self.simulations[sim_id]['status'] = SimulationStatus.RUNNING
            self.logger.info(f"Resumed simulation {sim_id}")
    
    def stop_simulation(self, sim_id: str) -> None:
        """Stop a simulation."""
        if sim_id in self.simulations:
            self.simulations[sim_id]['status'] = SimulationStatus.COMPLETED
            self.logger.info(f"Stopped simulation {sim_id}")
    
    def get_simulation_status(self, sim_id: str) -> Optional[Dict[str, Any]]:
        """Get current simulation status."""
        if sim_id not in self.simulations:
            return None
        
        sim = self.simulations[sim_id]
        return {
            'id': sim_id,
            'status': sim['status'].value,
            'current_round': sim['current_round'],
            'total_rounds': sim['config'].max_rounds,
            'num_agents': len(sim['agents']),
            'start_time': sim['start_time'].isoformat() if sim['start_time'] else None
        }
    
    def list_simulations(self) -> List[Dict[str, Any]]:
        """List all simulations."""
        return [
            {
                'id': sim_id,
                'name': sim['config'].name,
                'status': sim['status'].value,
                'round': sim['current_round']
            }
            for sim_id, sim in self.simulations.items()
        ]
    
    def on_event(self, callback: Callable) -> None:
        """Register event callback."""
        self.event_callbacks.append(callback)
    
    def _emit_event(self, event_type: str, data: Dict[str, Any]) -> None:
        """Emit event to all callbacks."""
        for callback in self.event_callbacks:
            try:
                callback(event_type, data)
            except Exception as e:
                self.logger.error(f"Event callback failed: {e}")
    
    def _create_result(self, sim: Dict[str, Any]) -> SimulationResult:
        """Create final result object."""
        config = sim['config']
        
        return SimulationResult(
            simulation_id=sim['id'],
            status=sim['status'],
            rounds_completed=sim['current_round'],
            final_state=self._aggregate_final_state(sim),
            metrics=self._calculate_metrics(sim),
            start_time=sim['start_time'],
            end_time=datetime.utcnow(),
            error_message=sim.get('error')
        )
    
    def _create_partial_result(self, sim: Dict[str, Any]) -> SimulationResult:
        """Create partial result for paused simulation."""
        return self._create_result(sim)
    
    def _aggregate_final_state(self, sim: Dict[str, Any]) -> Dict[str, Any]:
        """Aggregate final state from all agents."""
        return {
            agent_id: self.agents[agent_id].get_state()
            for agent_id in sim['agents']
        }
    
    def _calculate_metrics(self, sim: Dict[str, Any]) -> Dict[str, float]:
        """Calculate simulation metrics."""
        # Placeholder - would calculate actual metrics
        return {
            'completion_rate': sim['current_round'] / sim['config'].max_rounds,
            'agent_activity': len(sim['agents']),
            'checkpoint_count': len(sim['checkpoints'])
        }


# Convenience function
def create_orchestrator(vault_path: str) -> SwarmOrchestrator:
    """Factory function to create orchestrator."""
    memory = OMPAMemoryAdapter(vault_path)
    return SwarmOrchestrator(memory)
