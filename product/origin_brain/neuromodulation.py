"""
Neuromodulation Engine — Chemical Mode Switching

Implements the three key neuromodulators that control memory:

1. **Acetylcholine (ACh)** — Encoding vs Retrieval mode switch
   - High ACh → encoding mode (new inputs prioritized)
   - Low ACh → retrieval mode (old memories prioritized)
   - Modulated by: novelty, attention, task demands
   
2. **Dopamine (DA)** — Reward/Surprise signal
   - Unexpected rewards → DA burst → stronger consolidation
   - Prediction errors drive DA release
   - Modulates: salience, long-term potentiation
   
3. **Norepinephrine (NE)** — Arousal/Urgency signal
   - High arousal → NE surge → flashbulb memories
   - Fight-or-flight → enhanced encoding
   - Modulates: attention, encoding strength

Brain analogue: Basal forebrain (ACh), VTA/SNc (DA), Locus coeruleus (NE)
Research: Hasselmo 1999 (ACh mode switching), Schultz 1997 (DA RPE),
          McGaugh 2000 (NE emotional modulation)
"""

from __future__ import annotations

import logging
import math
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional
from enum import Enum

from pydantic import BaseModel, Field

logger = logging.getLogger(__name__)


class BrainMode(str, Enum):
    """Current operating mode of the memory system."""
    ENCODING = "encoding"       # ACh-dominant: prioritize new inputs
    RETRIEVAL = "retrieval"     # ACh-low: prioritize old memories
    CONSOLIDATION = "consolidation"  # Sleep/offline processing
    ALERT = "alert"             # NE-dominant: high arousal state


class NeuromodulatorState(BaseModel):
    """Current levels of each neuromodulator (0.0 to 1.0)."""
    acetylcholine: float = 0.5    # Encoding mode gate
    dopamine: float = 0.3          # Reward/surprise signal
    norepinephrine: float = 0.2    # Arousal/urgency signal
    
    # Derived properties
    @property
    def encoding_drive(self) -> float:
        """How strongly the system favors encoding new info."""
        return self.acetylcholine * 0.7 + self.norepinephrine * 0.3
    
    @property
    def retrieval_drive(self) -> float:
        """How strongly the system favors retrieving old info."""
        return (1.0 - self.acetylcholine) * 0.8 + self.dopamine * 0.2
    
    @property
    def consolidation_drive(self) -> float:
        """How strongly memories should be consolidated."""
        return self.dopamine * 0.6 + (1.0 - self.norepinephrine) * 0.4
    
    @property
    def current_mode(self) -> BrainMode:
        """Determine the current brain operating mode."""
        drives = {
            BrainMode.ENCODING: self.encoding_drive,
            BrainMode.RETRIEVAL: self.retrieval_drive,
        }
        # High NE overrides to alert mode
        if self.norepinephrine > 0.8:
            return BrainMode.ALERT
        return max(drives, key=drives.get)


class NeuromodulationConfig(BaseModel):
    """Configuration for neuromodulation dynamics."""
    # Decay rates (how quickly levels return to baseline per step)
    ach_decay_rate: float = 0.05
    da_decay_rate: float = 0.08
    ne_decay_rate: float = 0.1
    
    # Baselines (resting levels)
    ach_baseline: float = 0.4
    da_baseline: float = 0.2
    ne_baseline: float = 0.15
    
    # Response magnitudes
    novelty_ach_boost: float = 0.3       # Novel input → ACh up
    surprise_da_boost: float = 0.4       # Prediction error → DA up
    arousal_ne_boost: float = 0.5        # Emotional/urgent → NE up
    reward_da_boost: float = 0.35        # Positive outcome → DA up
    
    # Encoding modulation
    encoding_strength_multiplier: float = 2.0  # How much ACh boosts encoding


class NeuromodulationEngine:
    """
    Manages the chemical state of the memory system.
    
    The neuromodulators don't directly encode/retrieve memories —
    they MODULATE how other engines operate:
    
    - High ACh → hippocampal encoding is boosted, retrieval suppressed
    - High DA → consolidation is boosted, salience increased
    - High NE → encoding of current moment is strongly enhanced (flashbulb)
    
    Usage:
        engine = NeuromodulationEngine()
        
        # Signal novel input
        engine.on_novelty(0.8)
        
        # Signal surprise (prediction error)
        engine.on_surprise(0.9)
        
        # Get current encoding strength modifier
        modifier = engine.get_encoding_modifier()
        
        # Get current brain mode
        mode = engine.state.current_mode
    """
    
    def __init__(self, config: Optional[NeuromodulationConfig] = None):
        self.config = config or NeuromodulationConfig()
        self.state = NeuromodulatorState(
            acetylcholine=self.config.ach_baseline,
            dopamine=self.config.da_baseline,
            norepinephrine=self.config.ne_baseline,
        )
        
        # History for analysis
        self._mode_history: List[Dict[str, Any]] = []
        self._total_signals: int = 0
        
        logger.info("NeuromodulationEngine initialized")
    
    def on_novelty(self, novelty_score: float):
        """
        Signal detection of novel input.
        
        Novelty → ACh release → encoding mode activated.
        This is Hasselmo's ACh mode switching theory:
        the brain detects "this is new" and shifts to encoding.
        
        Args:
            novelty_score: 0.0 (familiar) to 1.0 (completely novel)
        """
        self._total_signals += 1
        boost = novelty_score * self.config.novelty_ach_boost
        self.state.acetylcholine = min(1.0, self.state.acetylcholine + boost)
        
        # Mild NE increase with novelty (attention grabbed)
        ne_boost = novelty_score * 0.1
        self.state.norepinephrine = min(1.0, self.state.norepinephrine + ne_boost)
        
        logger.debug(f"Novelty signal ({novelty_score:.2f}): ACh→{self.state.acetylcholine:.2f}")
    
    def on_surprise(self, surprise_magnitude: float):
        """
        Signal a prediction error / surprise.
        
        Surprise → DA burst → memory consolidation enhanced.
        This is Schultz's reward prediction error theory:
        unexpected outcomes trigger dopamine release.
        
        Args:
            surprise_magnitude: 0.0 (expected) to 1.0 (completely unexpected)
        """
        self._total_signals += 1
        da_boost = surprise_magnitude * self.config.surprise_da_boost
        self.state.dopamine = min(1.0, self.state.dopamine + da_boost)
        
        # Surprise also grabs attention (mild ACh boost)
        ach_boost = surprise_magnitude * 0.1
        self.state.acetylcholine = min(1.0, self.state.acetylcholine + ach_boost)
        
        logger.debug(f"Surprise signal ({surprise_magnitude:.2f}): DA→{self.state.dopamine:.2f}")
    
    def on_arousal(self, arousal_level: float):
        """
        Signal high emotional arousal or urgency.
        
        Arousal → NE surge → flashbulb memory formation.
        This is McGaugh's emotional modulation theory:
        stressful/exciting events are remembered better.
        
        Args:
            arousal_level: 0.0 (calm) to 1.0 (extreme arousal/stress)
        """
        self._total_signals += 1
        ne_boost = arousal_level * self.config.arousal_ne_boost
        self.state.norepinephrine = min(1.0, self.state.norepinephrine + ne_boost)
        
        # High arousal also triggers DA (salience)
        da_boost = arousal_level * 0.15
        self.state.dopamine = min(1.0, self.state.dopamine + da_boost)
        
        logger.debug(f"Arousal signal ({arousal_level:.2f}): NE→{self.state.norepinephrine:.2f}")
    
    def on_reward(self, reward_value: float):
        """
        Signal positive outcome / reward.
        
        Reward → DA release → consolidation of associated memories.
        
        Args:
            reward_value: 0.0 (neutral) to 1.0 (high reward)
        """
        self._total_signals += 1
        da_boost = reward_value * self.config.reward_da_boost
        self.state.dopamine = min(1.0, self.state.dopamine + da_boost)
        
        logger.debug(f"Reward signal ({reward_value:.2f}): DA→{self.state.dopamine:.2f}")
    
    def decay_to_baseline(self):
        """
        Decay all neuromodulators toward their baselines.
        
        Should be called periodically (each processing step).
        Neuromodulator effects are transient — they return to baseline
        unless re-stimulated.
        """
        # Exponential decay toward baseline
        self.state.acetylcholine += (
            self.config.ach_baseline - self.state.acetylcholine
        ) * self.config.ach_decay_rate
        
        self.state.dopamine += (
            self.config.da_baseline - self.state.dopamine
        ) * self.config.da_decay_rate
        
        self.state.norepinephrine += (
            self.config.ne_baseline - self.state.norepinephrine
        ) * self.config.ne_decay_rate
        
        # Clamp
        self.state.acetylcholine = max(0.0, min(1.0, self.state.acetylcholine))
        self.state.dopamine = max(0.0, min(1.0, self.state.dopamine))
        self.state.norepinephrine = max(0.0, min(1.0, self.state.norepinephrine))
    
    def get_encoding_modifier(self) -> float:
        """
        Get the current encoding strength modifier.
        
        High ACh + high NE → strong encoding (2x-3x normal)
        Low ACh → weak encoding (0.5x normal)
        
        Returns:
            Multiplier for encoding strength (0.5 to 3.0)
        """
        base = 0.5 + self.state.encoding_drive * self.config.encoding_strength_multiplier
        
        # NE provides additional boost for flashbulb memories
        if self.state.norepinephrine > 0.7:
            base *= 1.5
        
        return min(3.0, max(0.5, base))
    
    def get_consolidation_modifier(self) -> float:
        """
        Get the current consolidation strength modifier.
        
        High DA → strong consolidation (important memories preserved)
        Low DA → weak consolidation (routine memories may be lost)
        
        Returns:
            Multiplier for consolidation (0.5 to 2.5)
        """
        return 0.5 + self.state.consolidation_drive * 2.0
    
    def get_retrieval_modifier(self) -> float:
        """
        Get the current retrieval accessibility modifier.
        
        Low ACh → retrieval mode (old memories easier to access)
        High ACh → encoding mode (retrieval suppressed)
        
        Returns:
            Multiplier for retrieval ease (0.5 to 2.0)
        """
        return 0.5 + self.state.retrieval_drive * 1.5
    
    def log_mode(self):
        """Record current mode for analysis."""
        self._mode_history.append({
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "mode": self.state.current_mode.value,
            "ach": round(self.state.acetylcholine, 3),
            "da": round(self.state.dopamine, 3),
            "ne": round(self.state.norepinephrine, 3),
        })
    
    def get_statistics(self) -> Dict[str, Any]:
        """Get neuromodulation statistics."""
        return {
            "current_mode": self.state.current_mode.value,
            "acetylcholine": round(self.state.acetylcholine, 3),
            "dopamine": round(self.state.dopamine, 3),
            "norepinephrine": round(self.state.norepinephrine, 3),
            "encoding_modifier": round(self.get_encoding_modifier(), 3),
            "consolidation_modifier": round(self.get_consolidation_modifier(), 3),
            "retrieval_modifier": round(self.get_retrieval_modifier(), 3),
            "total_signals": self._total_signals,
            "mode_transitions": len(self._mode_history),
        }
