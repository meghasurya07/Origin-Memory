"""
Origin Brain — Human-Like Memory Infrastructure for AI Agents

By Origin AI (https://originai.in)

A biologically-grounded memory system that gives AI agents the ability to
encode, consolidate, retrieve, and strategically forget information the way
the human brain does.
"""

__version__ = "0.5.0"

# Core
from .models import (
    MemoryTier,
    MemoryType,
    MemoryStatus,
    EpisodicMemory,
    SemanticMemory,
    ProceduralMemory,
    MemoryQuery,
    MemoryResult,
    ConsolidationResult,
    BrainConfig,
)
from .brain import Brain, EncodeResult, RecallResult

# Engines (v0.1-v0.2)
from .decay import DecayEngine
from .hippocampus import HippocampalEngine
from .consolidation import ConsolidationDaemon
from .router import MemoryRouter
from .reconsolidation import ReconsolidationEngine
from .schemas import SchemaEngine, MemorySchema
from .emotional import EmotionalModulator
from .interference import InterferenceDetector
from .prospective import ProspectiveMemoryEngine
from .metamemory import MetamemoryEngine

# New engines (v0.3 — The 4 Pillars)
from .temporal_context import TemporalContextEngine, TemporalContextConfig, ContextSnapshot
from .prediction import PredictionEngine, PredictionEngineConfig, PredictionResult
from .sleep import SleepEngine, SleepConfig, SleepReport

# The Breakthrough (v0.4 — Reconstructive Memory)
from .reconstruction import ReconstructionEngine, ReconstructionConfig, ReconstructedMemory, ReconstructionFidelity
from .pattern_completion import PatternCompletionEngine, PatternCompletionConfig, CompletionResult
from .working_memory import WorkingMemoryEngine, WorkingMemoryConfig, WorkingMemoryItem
from .embedding import EmbeddingEngine, EmbeddingConfig
from .storage import StorageBackend, InMemoryStorage, SQLiteStorage
from .neurogenesis import NeurogenesisEngine, NeurogenesisConfig, MemoryNeuron, NeuronState
from .neuromodulation import NeuromodulationEngine, NeuromodulationConfig, BrainMode
from .synaptic_plasticity import SynapticPlasticityEngine, PlasticityConfig, Synapse, SynapticTag
from .client import OriginBrainClient, MemoryItem

__all__ = [
    # Core API
    "Brain",
    "BrainConfig",
    "EncodeResult",
    "RecallResult",
    # Memory types
    "MemoryTier",
    "MemoryType",
    "MemoryStatus",
    "EpisodicMemory",
    "SemanticMemory",
    "ProceduralMemory",
    "MemoryQuery",
    "MemoryResult",
    "ConsolidationResult",
    # Engines (v0.1-v0.2)
    "DecayEngine",
    "HippocampalEngine",
    "ConsolidationDaemon",
    "MemoryRouter",
    "ReconsolidationEngine",
    "SchemaEngine",
    "MemorySchema",
    "EmotionalModulator",
    "InterferenceDetector",
    "ProspectiveMemoryEngine",
    "MetamemoryEngine",
    # New engines (v0.3 — The 4 Pillars)
    "TemporalContextEngine",
    "TemporalContextConfig",
    "ContextSnapshot",
    "PredictionEngine",
    "PredictionEngineConfig",
    "PredictionResult",
    "SleepEngine",
    "SleepConfig",
    "SleepReport",
]

