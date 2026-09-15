from __future__ import annotations

import logging
from datetime import datetime
from typing import Any, Callable, Dict, List, Literal, Optional, Union

from pydantic import BaseModel, Field

from .models import (
    BrainConfig, MemoryType, MemoryResult, EpisodicMemory, SemanticMemory, ProceduralMemory, MemoryTier
)
from .emotional import EmotionalModulator, EmotionalTag
from .reconsolidation import ReconsolidationEngine
from .interference import InterferenceDetector, InterferenceEvent
from .schemas import SchemaEngine, PREDEFINED_SCHEMAS
from .prospective import ProspectiveMemoryEngine, ProspectiveMemoryType, ProspectiveMemory
from .metamemory import MetamemoryEngine, MetamemoryAssessment
from .router import MemoryRouter
from .decay import DecayEngine
from .hippocampus import HippocampalEngine
from .consolidation import ConsolidationDaemon
from .temporal_context import TemporalContextEngine, TemporalContextConfig
from .prediction import PredictionEngine, PredictionEngineConfig, PredictionResult
from .sleep import SleepEngine, SleepConfig, SleepReport
from .reconstruction import ReconstructionEngine, ReconstructionConfig, ReconstructedMemory
from .pattern_completion import PatternCompletionEngine, PatternCompletionConfig
from .working_memory import WorkingMemoryEngine, WorkingMemoryConfig
from .neurogenesis import NeurogenesisEngine, NeurogenesisConfig
from .neuromodulation import NeuromodulationEngine, NeuromodulationConfig

logger = logging.getLogger(__name__)

def jaccard_similarity(text1: str, text2: str) -> float:
    set1 = set(text1.lower().split())
    set2 = set(text2.lower().split())
    if not set1 or not set2:
        return 0.0
    return len(set1.intersection(set2)) / len(set1.union(set2))

class EncodeResult(BaseModel):
    memory: Optional[EpisodicMemory] = None
    action: Literal['encoded', 'reconsolidated', 'duplicate', 'schema_matched']
    emotional_tag: Optional[EmotionalTag] = None
    interference_events: List[InterferenceEvent] = Field(default_factory=list)
    schema_match: Optional[str] = None
    prospective_intents: List[ProspectiveMemory] = Field(default_factory=list)

class RecallResult(BaseModel):
    results: List[MemoryResult]
    confidence: MetamemoryAssessment
    triggered_intents: List[ProspectiveMemory] = Field(default_factory=list)
    query_classification: List[MemoryType] = Field(default_factory=list)


class Brain:
    """
    Origin Brain - A human-like memory infrastructure for AI agents.
    Integrates all memory engines into a cohesive encode/recall pipeline.
    """

    def __init__(self, config: Union[BrainConfig, dict], embedding_fn: Optional[Callable[[str], List[float]]] = None):
        if isinstance(config, dict):
            self.config = BrainConfig(**config)
        else:
            self.config = config

        self.embedding_fn = embedding_fn

        # Initialize engines
        self.decay_engine = DecayEngine(
            model_type=self.config.decay_model,
            alpha=self.config.spaced_repetition_alpha,
            w=self.config.spaced_repetition_w
        )
        self.hippocampus = HippocampalEngine(
            config=self.config,
            decay_engine=self.decay_engine,
            embedding_fn=self.embedding_fn
        )
        self.consolidation = ConsolidationDaemon(
            hippocampus=self.hippocampus,
            config=self.config
        )
        self.router = MemoryRouter(config=self.config)
        self.reconsolidation = ReconsolidationEngine()
        self.schema_engine = SchemaEngine(predefined_schemas=PREDEFINED_SCHEMAS)
        self.emotional_modulator = EmotionalModulator()
        self.interference_detector = InterferenceDetector(similarity_fn=jaccard_similarity)
        self.prospective_engine = ProspectiveMemoryEngine()
        self.metamemory = MetamemoryEngine()

        # v0.3 Pillar engines
        self.temporal_context = TemporalContextEngine(
            config=TemporalContextConfig(dimension=self.config.embedding_dimension)
        )
        self.prediction_engine = PredictionEngine(
            config=PredictionEngineConfig()
        )
        self.sleep_engine = SleepEngine(config=SleepConfig())
        self.reconstruction = ReconstructionEngine(config=ReconstructionConfig())
        self.pattern_completion = PatternCompletionEngine(
            config=PatternCompletionConfig(beta=8.0)
        )
        self.working_memory = WorkingMemoryEngine(
            config=WorkingMemoryConfig(capacity=7)
        )
        
        # v0.5: Neurogenesis — dynamic neuron creation/destruction
        self.neurogenesis = NeurogenesisEngine(config=NeurogenesisConfig())
        
        # v0.5: Neuromodulation — ACh/DA/NE mode switching
        self.neuromodulation = NeuromodulationEngine(config=NeuromodulationConfig())

        # v0.5: Synaptic plasticity — STDP + Hebbian connection learning
        from .synaptic_plasticity import SynapticPlasticityEngine, PlasticityConfig
        self.plasticity = SynapticPlasticityEngine(config=PlasticityConfig())

        # v0.5: Engram tracking — memory trace formation
        from .engram import EngramEngine, EngramConfig
        self.engram_engine = EngramEngine(config=EngramConfig())

        # v0.5: Persistent storage
        if self.config.storage_path:
            from .storage import SQLiteStorage
            self._storage = SQLiteStorage(db_path=self.config.storage_path)
            # Auto-load existing memories on init
            self._load_from_storage()
        else:
            self._storage = None

        # Backward-compatible context buffer (delegates to working memory)
        self.context_buffer: Dict[str, Any] = {}

        logger.info(f"Origin Brain v0.5 initialized for agent: {self.config.agent_id}")

    @property
    def agent_id(self) -> str:
        return self.config.agent_id

    @property
    def episodic_count(self) -> int:
        return len(self.hippocampus)

    @property
    def semantic_count(self) -> int:
        return self.consolidation.get_statistics().get('semantic_count', 0)

    @property
    def procedural_count(self) -> int:
        return self.consolidation.get_statistics().get('procedural_count', 0)

    def encode(self, content: str, context: Optional[Dict[str, Any]] = None, salience: Optional[float] = None, memory_type: MemoryType = MemoryType.EPISODIC) -> EncodeResult:
        """
        Encodes new information into the brain.
        
        v0.3 Pipeline:
        1. Context buffer update
        2. Prediction error evaluation (Pillar 1) — gate encoding
        3. Emotional analysis
        4. Salience modulation (prediction × emotion)
        5. Reconsolidation check
        6. Schema matching
        7. Hippocampal encoding
        8. Temporal context binding (Pillar 2)
        9. Interference detection
        10. Prospective intent extraction
        """
        ctx = context or {}
        
        # 1. Add to context buffer
        self.context_buffer.update(ctx)
        
        # 2. Prediction error evaluation (Pillar 1: Surprise Gate)
        prediction = self.prediction_engine.evaluate(content, ctx)
        
        # 3. Emotional analysis
        emotional_tag = self.emotional_modulator.analyze(content, ctx)
        
        # 3b. Neuromodulation — signal novelty and surprise to modulate brain state
        if prediction.should_encode:
            self.neuromodulation.on_novelty(prediction.surprise_score)
        if prediction.surprise_score > 0.5:
            self.neuromodulation.on_surprise(prediction.surprise_score)
        if hasattr(emotional_tag, 'arousal') and emotional_tag.arousal > 0.7:
            self.neuromodulation.on_arousal(emotional_tag.arousal)
        
        # 4. Modulate salience using prediction error + emotion + neuromodulation
        base_salience = salience if salience is not None else 0.3
        # Prediction-boosted salience: surprising content gets higher salience
        prediction_boost = prediction.encoding_strength * 0.3
        # Neuromodulation-boosted: ACh/NE enhance encoding strength
        neuro_modifier = self.neuromodulation.get_encoding_modifier()
        boosted_salience = self.emotional_modulator.modulate_salience(
            base_salience + prediction_boost, emotional_tag
        )
        boosted_salience *= neuro_modifier
        # Clamp
        boosted_salience = min(1.0, boosted_salience)
        
        # 5. Reconsolidation check
        recon_result = self.reconsolidation.process_new_input(content, ctx, jaccard_similarity)
        if recon_result.action == 'reconsolidated':
            return EncodeResult(
                memory=None,
                action='reconsolidated',
                emotional_tag=emotional_tag,
            )

        # 6. Schema matching
        schema_matches = self.schema_engine.match_to_schema(content, ctx)
        if schema_matches and schema_matches[0][1] > 0.6:
            schema, score = schema_matches[0]
            instance = self.schema_engine.fill_schema(schema, content, ctx)
            
            # Store as semantic fact
            fact = SemanticMemory(
                key=schema.name,
                value=str(instance.filled_slots),
                confidence=score,
                category="schema_instance"
            )
            self.consolidation.store_semantic(fact)
            
            return EncodeResult(
                memory=None,
                action='schema_matched',
                emotional_tag=emotional_tag,
                schema_match=schema.name
            )

        # 7. Hippocampal encoding
        memory = self.hippocampus.encode(
            content=content, 
            context=ctx, 
            salience=boosted_salience, 
            memory_type=memory_type
        )
        
        if memory is None:
            return EncodeResult(
                memory=None,
                action='duplicate',
                emotional_tag=emotional_tag
            )

        # Store emotional tag and prediction metadata
        memory.metadata['emotional_valence'] = emotional_tag.valence.value
        memory.metadata['emotional_arousal'] = emotional_tag.arousal
        memory.metadata['surprise_score'] = prediction.surprise_score
        memory.metadata['encoding_strength'] = prediction.encoding_strength

        # 8. Temporal context binding (Pillar 2)
        # Create a simple feature vector from the memory's sparse embedding
        embedding = memory.embedding or []
        if not embedding:
            # Use hash-based feature for temporal binding
            words = content.lower().split()
            dim = self.config.embedding_dimension
            feature = [0.0] * dim
            for w in words:
                idx = hash(w) % dim
                feature[idx] += 1.0
            embedding = feature
        self.temporal_context.update(embedding, memory_id=memory.id)

        # 8b. Store pattern for CA3-like pattern completion
        self.pattern_completion.store(
            pattern=embedding,
            memory_id=memory.id,
            label=content[:50],
            strength=boosted_salience
        )

        # 9. Interference detection
        existing_memories = list(self.hippocampus._episodic_store.values())
        interference_events = self.interference_detector.detect(memory, existing_memories, jaccard_similarity)

        # 10. Prospective intents
        prospective_intents = []
        lower_content = content.lower()
        if 'remind me' in lower_content or "don't forget to" in lower_content or 'at ' in lower_content or 'when ' in lower_content:
            intent = self.prospective_engine.add(
                content=content,
                trigger_type=ProspectiveMemoryType.EVENT_BASED,
                trigger_condition=content, # simple fallback
                context=ctx
            )
            prospective_intents.append(intent)

        # 11. Decay neuromodulators toward baseline (transient effects)
        self.neuromodulation.decay_to_baseline()

        # 12. Persist to storage (if configured)
        self._persist_memory(memory)

        # 13. Allocate engram trace (neural ensemble for this memory)
        self.engram_engine.allocate_engram(memory.id)

        return EncodeResult(
            memory=memory,
            action='encoded',
            emotional_tag=emotional_tag,
            interference_events=interference_events,
            prospective_intents=prospective_intents
        )

    def recall(self, query: str, memory_types: Optional[List[MemoryType]] = None, top_k: int = 5) -> RecallResult:
        """
        Recalls information from the brain based on a query.
        """
        # 1. Classify query
        classified_types = self.router.classify_query(query)
        if memory_types:
            classified_types = list(set(classified_types + memory_types))
        # Always search episodic (primary store) unless explicitly excluded
        if MemoryType.EPISODIC not in classified_types:
            classified_types.append(MemoryType.EPISODIC)
            
        # 2. Build query
        memory_query = self.router.build_query(query, context=self.context_buffer, top_k=top_k)
        
        # 3. Retrieve from stores
        raw_results = []
        if MemoryType.EPISODIC in classified_types:
            raw_results.extend(self.hippocampus.retrieve(query, top_k=top_k, temporal_range=memory_query.temporal_range))
            
        if MemoryType.SEMANTIC in classified_types:
            semantic_mems = self.consolidation.retrieve_semantic(query, top_k=top_k)
            for sm in semantic_mems:
                raw_results.append(MemoryResult(
                    memory=sm,
                    relevance_score=sm.confidence,
                    tier=MemoryTier.SEMANTIC,
                    retrieval_latency_ms=0.0
                ))
                
        if MemoryType.PROCEDURAL in classified_types:
            proc_mems = self.consolidation.retrieve_procedural(query)
            for pm in proc_mems[:top_k]:
                raw_results.append(MemoryResult(
                    memory=pm,
                    relevance_score=1.0,
                    tier=MemoryTier.PROCEDURAL,
                    retrieval_latency_ms=0.0
                ))
                
        # 4. Rank results
        ranked_results = self.router.rank_results(raw_results, query)
        
        # 5. Reconsolidation on_retrieval for episodic
        for res in ranked_results:
            if isinstance(res.memory, EpisodicMemory):
                self.reconsolidation.on_retrieval(res.memory)
        
        # 5b. Testing effect: retrieval strengthens memory (access tracking)
        # Only strengthen the TOP result — like focused retrieval practice
        if ranked_results and isinstance(ranked_results[0].memory, EpisodicMemory):
            ranked_results[0].memory.on_retrieval()
            
            # 5c. Retrieval-Induced Forgetting (Anderson 1994)
            # Retrieving this memory INHIBITS competitors
            all_episodic = list(self.hippocampus._episodic_store.values())
            self.interference_detector.apply_retrieval_induced_forgetting(
                retrieved_memory=ranked_results[0].memory,
                all_memories=all_episodic,
                embedding_engine=self.hippocampus._semantic_engine,
                inhibition_rate=0.03,
                similarity_threshold=0.35,
            )
        
        # 5d. Engram reactivation: retrieval reactivates the neural trace
        for res in ranked_results:
            if isinstance(res.memory, EpisodicMemory):
                self.engram_engine.reactivate(res.memory.id)
                
        # 6. Metamemory assessment
        confidence = self.metamemory.assess(query, ranked_results)
        
        # 6b. Synaptic plasticity: co-recalled memories strengthen connections
        episodic_results = [
            r for r in ranked_results if isinstance(r.memory, EpisodicMemory)
        ]
        for i, r1 in enumerate(episodic_results):
            for r2 in episodic_results[i+1:]:
                # Find or create synapse between co-recalled memories
                existing = self.plasticity.find_synapses(
                    source_id=r1.memory.id, target_id=r2.memory.id
                )
                if existing:
                    syn = existing[0]
                else:
                    syn = self.plasticity.create_synapse(
                        r1.memory.id, r2.memory.id, initial_weight=0.3
                    )
                # Co-retrieval = Hebbian strengthening
                co_activation = min(r1.relevance_score, r2.relevance_score)
                self.plasticity.hebbian_update(syn.id, co_activation)
        
        # 7. Prospective memory triggers
        trigger_context = self.context_buffer.copy()
        trigger_context['query'] = query
        trigger_context['activity'] = 'recall'
        triggered_intents = self.prospective_engine.check_triggers(trigger_context)
        
        return RecallResult(
            results=ranked_results,
            confidence=confidence,
            triggered_intents=triggered_intents,
            query_classification=classified_types
        )

    def reconstruct_recall(
        self, 
        query: str, 
        memory_types: Optional[List[MemoryType]] = None, 
        top_k: int = 5,
        emotional_arousal: float = 0.5,
        emotional_valence: str = "neutral"
    ) -> List[ReconstructedMemory]:
        """
        Generative reconstructive recall — the breakthrough method.
        
        Instead of returning verbatim stored content, this method
        RECONSTRUCTS each retrieved memory based on:
        - Context overlap between encoding and current context
        - Current emotional state
        - Active schemas
        - Semantic knowledge
        
        This produces different outputs for the same query depending
        on the current cognitive state — exactly like human memory.
        
        Args:
            query: What to recall
            memory_types: Filter by memory type
            top_k: Max results
            emotional_arousal: Current arousal level [0,1]
            emotional_valence: "positive", "negative", or "neutral"
            
        Returns:
            List of ReconstructedMemory objects with fidelity metadata
        """
        # Step 1: Standard recall to get raw results
        recall_result = self.recall(query, memory_types=memory_types, top_k=top_k)
        
        # Step 2: Reconstruct each episodic result
        reconstructed = []
        for result in recall_result.results:
            if not isinstance(result.memory, EpisodicMemory):
                continue
            
            memory = result.memory
            
            # Compute context overlap: how similar is current context to encoding context?
            ctx_overlap = self._compute_context_overlap(memory)
            
            # Get semantic knowledge for enrichment
            semantic_facts = self._get_relevant_semantic(query)
            
            # Reconstruct
            recon = self.reconstruction.reconstruct(
                memory=memory,
                context_overlap=ctx_overlap,
                emotional_arousal=emotional_arousal,
                emotional_valence=emotional_valence,
                semantic_knowledge=semantic_facts,
                current_context=self.context_buffer
            )
            reconstructed.append(recon)
        
        return reconstructed
    
    def _compute_context_overlap(self, memory: EpisodicMemory) -> float:
        """
        Compute context overlap between current state and encoding context.
        
        Uses a hybrid approach:
        - TCM temporal similarity (how much temporal context has drifted)
        - Dict value comparison (how similar are the explicit context fields)
        - Blended: 50% TCM + 50% dict (when both available)
        """
        tcm_sim = None
        dict_sim = None
        
        # TCM-based temporal context similarity
        if memory.id in self.temporal_context._memory_to_snapshot:
            idx = self.temporal_context._memory_to_snapshot[memory.id]
            encoding_ctx = self.temporal_context._snapshots[idx].vector
            current_ctx = self.temporal_context.current_context
            dot = sum(a * b for a, b in zip(encoding_ctx, current_ctx))
            tcm_sim = max(0.0, min(1.0, (dot + 1.0) / 2.0))
        
        # Dict value comparison
        mem_ctx = memory.context or {}
        cur_ctx = self.context_buffer or {}
        if mem_ctx or cur_ctx:
            all_keys = set(mem_ctx.keys()) | set(cur_ctx.keys())
            if all_keys:
                matches = 0
                for k in all_keys:
                    v1 = str(mem_ctx.get(k, "")).lower()
                    v2 = str(cur_ctx.get(k, "")).lower()
                    if v1 and v2 and v1 == v2:
                        matches += 1
                dict_sim = matches / len(all_keys)
        
        # Blend available signals
        if tcm_sim is not None and dict_sim is not None:
            return 0.4 * tcm_sim + 0.6 * dict_sim  # Dict-weighted (more reliable for explicit context)
        elif tcm_sim is not None:
            return tcm_sim
        elif dict_sim is not None:
            return dict_sim
        else:
            return 0.5  # Neutral default

    def _get_relevant_semantic(self, query: str, top_k: int = 3) -> List[str]:
        """Get relevant semantic facts for reconstruction enrichment."""
        facts = self.consolidation.retrieve_semantic(query, top_k=top_k)
        return [f"{f.key}: {f.value}" for f in facts]
        
    def consolidate(self):
        """Manually trigger a legacy consolidation cycle."""
        return self.consolidation.run_cycle()

    def sleep(self, num_cycles: Optional[int] = None) -> SleepReport:
        """
        Run a multi-phase sleep consolidation cycle (v0.3 Pillar 4).
        
        Executes 3 phases per cycle:
        1. SWS: Prioritized replay + compression to semantic gist
        2. REM: Emotional decoupling + schema integration
        3. Pruning: Anderson-Schooler optimal forgetting
        
        Args:
            num_cycles: Number of sleep cycles (default: 3)
            
        Returns:
            SleepReport with per-phase details
        """
        episodic_list = list(self.hippocampus._episodic_store.values())
        
        def store_semantic_fn(sem: SemanticMemory):
            self.consolidation.store_semantic(sem)
        
        def evict_fn(memory_id: str) -> bool:
            if memory_id in self.hippocampus._episodic_store:
                del self.hippocampus._episodic_store[memory_id]
                self.hippocampus._embeddings.pop(memory_id, None)
                return True
            return False
        
        report = self.sleep_engine.run_sleep_cycle(
            episodic_memories=episodic_list,
            store_semantic_fn=store_semantic_fn,
            evict_fn=evict_fn,
            num_cycles=num_cycles
        )
        
        logger.info(
            f"Sleep cycle complete: consolidated={report.total_consolidated}, "
            f"evicted={report.total_evicted}, new_semantic={report.new_semantic_count}"
        )
        return report

    def store_fact(self, key: str, value: str, confidence: float = 1.0, category: str = "general") -> SemanticMemory:
        """Stores a semantic fact directly."""
        memory = SemanticMemory(
            key=key,
            value=value,
            category=category,
            confidence=confidence
        )
        self.consolidation.store_semantic(memory)
        return memory
        
    def store_procedure(self, name: str, description: str, steps: List[str], trigger_conditions: Optional[List[str]] = None) -> ProceduralMemory:
        """Stores a procedural memory directly."""
        memory = ProceduralMemory(
            name=name,
            description=description,
            steps=steps,
            trigger_conditions=trigger_conditions or []
        )
        self.consolidation.store_procedural(memory)
        return memory

    def get_context_buffer(self) -> Dict[str, Any]:
        """Returns the current working memory context buffer."""
        return self.context_buffer

    def clear_context(self) -> None:
        """Clears the working memory context buffer."""
        self.context_buffer.clear()
        
    def get_statistics(self) -> Dict[str, Any]:
        """Returns statistics across all brain systems."""
        hippo_stats = self.hippocampus.get_statistics()
        consol_stats = self.consolidation.get_statistics()
        return {
            "agent_id": self.config.agent_id,
            "version": "0.4.0",
            "episodic_count": self.episodic_count,
            "semantic_count": self.semantic_count,
            "procedural_count": self.procedural_count,
            "context_buffer_size": len(self.context_buffer),
            "hippocampus": hippo_stats,
            "consolidation": consol_stats,
            "schemas": self.schema_engine.get_statistics(),
            "prospective": self.prospective_engine.get_statistics(),
            "interference": self.interference_detector.get_statistics(),
            "reconsolidation": {
                "labile_count": self.reconsolidation.get_labile_count()
            },
            # v0.3 Pillar engines
            "temporal_context": self.temporal_context.get_statistics(),
            "prediction": self.prediction_engine.get_statistics(),
            "sleep": self.sleep_engine.get_statistics(),
            # v0.4 Breakthrough engines
            "reconstruction": self.reconstruction.get_statistics(),
            "pattern_completion": self.pattern_completion.get_statistics(),
            "working_memory": self.working_memory.get_statistics(),
        }
        
    def add_reminder(self, content: str, trigger_type: ProspectiveMemoryType, trigger_condition: str, trigger_time: Optional[datetime] = None) -> ProspectiveMemory:
        """Adds a prospective memory reminder."""
        return self.prospective_engine.add(
            content=content,
            trigger_type=trigger_type,
            trigger_condition=trigger_condition,
            trigger_time=trigger_time,
            context=self.context_buffer
        )
        
    def check_reminders(self, context: Optional[Dict[str, Any]] = None) -> List[ProspectiveMemory]:
        """Checks and triggers any pending prospective memories."""
        return self.prospective_engine.check_triggers(context or self.context_buffer)
        
    def get_emotional_summary(self) -> Dict[str, Any]:
        """Returns a summary of the emotional state based on episodic memories."""
        memories = list(self.hippocampus._episodic_store.values())
        return self.emotional_modulator.get_emotional_summary(memories)
        
    def get_interference_log(self) -> List[InterferenceEvent]:
        """Returns the log of detected interference events."""
        return self.interference_detector.get_event_log()
        
    def export_state(self) -> Dict[str, Any]:
        """Exports the entire brain state for persistence."""
        episodic_data = {
            mid: mem.model_dump(mode='json')
            for mid, mem in self.hippocampus._episodic_store.items()
        }
        semantic_data = {
            mid: mem.model_dump(mode='json')
            for mid, mem in self.consolidation._semantic_store.items()
        }
        procedural_data = {
            mid: mem.model_dump(mode='json')
            for mid, mem in self.consolidation._procedural_store.items()
        }
        return {
            "agent_id": self.config.agent_id,
            "episodic_count": self.episodic_count,
            "semantic_count": self.semantic_count,
            "procedural_count": self.procedural_count,
            "episodic_memories": episodic_data,
            "semantic_memories": semantic_data,
            "procedural_memories": procedural_data,
        }

    def import_state(self, state: Dict[str, Any]) -> None:
        """Imports brain state from a previously exported dict."""
        for mid, mdata in state.get("episodic_memories", {}).items():
            mem = EpisodicMemory(**mdata)
            self.hippocampus._episodic_store[mid] = mem
            # Rebuild sparse embedding for retrieval
            words = set(mem.content.lower().split())
            self.hippocampus._embeddings[mid] = words

        for mid, mdata in state.get("semantic_memories", {}).items():
            mem = SemanticMemory(**mdata)
            self.consolidation._semantic_store[mid] = mem

        for mid, mdata in state.get("procedural_memories", {}).items():
            mem = ProceduralMemory(**mdata)
            self.consolidation._procedural_store[mid] = mem

        logger.info(
            f"Imported state: {len(state.get('episodic_memories', {}))} episodic, "
            f"{len(state.get('semantic_memories', {}))} semantic, "
            f"{len(state.get('procedural_memories', {}))} procedural"
        )

    # ── Persistence ──────────────────────────────────────────────────

    def save(self) -> int:
        """
        Persist all current episodic memories to storage.
        
        Returns:
            Number of memories saved
        """
        if self._storage is None:
            logger.warning("No storage backend configured (set storage_path in BrainConfig)")
            return 0
        
        count = 0
        for mem in self.hippocampus._episodic_store.values():
            self._storage.save_memory(mem)
            count += 1
        
        logger.info(f"Saved {count} memories to storage")
        return count

    def _persist_memory(self, memory: EpisodicMemory):
        """Persist a single memory to storage (if configured)."""
        if self._storage is not None:
            self._storage.save_memory(memory)

    def _load_from_storage(self):
        """Load all memories from storage into hippocampus."""
        if self._storage is None:
            return
        
        memories = self._storage.load_all_memories()
        for mem in memories:
            self.hippocampus._episodic_store[mem.id] = mem
            # Rebuild sparse embedding for retrieval
            words = set(mem.content.lower().split())
            self.hippocampus._embeddings[mem.id] = words
            # Rebuild TF-IDF index
            self.hippocampus._semantic_engine.add_document(mem.id, mem.content)
        
        if memories:
            logger.info(f"Loaded {len(memories)} memories from storage")
