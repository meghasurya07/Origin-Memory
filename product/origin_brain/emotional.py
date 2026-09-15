"""
Emotional Modulation of Memory - Analogous to the Amygdala's role in the brain.
Emotionally significant events are remembered better and longer. The amygdala enhances encoding strength and consolidation priority.
"""

import logging
from enum import Enum
from typing import Any, Dict, List, Optional

from pydantic import BaseModel

from .models import EpisodicMemory

logger = logging.getLogger(__name__)


class EmotionalValence(str, Enum):
    POSITIVE = 'positive'
    NEGATIVE = 'negative'
    NEUTRAL = 'neutral'
    URGENT = 'urgent'
    SURPRISING = 'surprising'


class EmotionalTag(BaseModel):
    valence: EmotionalValence
    arousal: float
    dominance: float
    confidence: float


class EmotionalModulator:
    """
    Modulates memory properties based on emotional significance.
    """

    def __init__(self, arousal_boost_factor: float = 1.5, negative_bias: float = 1.2):
        self.arousal_boost_factor = arousal_boost_factor
        self.negative_bias = negative_bias

    def analyze(self, content: str, context: Optional[Dict[str, Any]] = None) -> EmotionalTag:
        """
        Analyze the emotional content of text and return an EmotionalTag.
        """
        content_lower = content.lower()

        positive_words = {'great', 'excellent', 'love', 'happy', 'success', 'wonderful', 
                          'amazing', 'thank', 'pleased', 'congratulations', 'perfect'}
        negative_words = {'error', 'fail', 'problem', 'issue', 'bug', 'crash', 'wrong', 
                          'broken', 'hate', 'terrible', 'angry', 'frustrated', 'worried', 'disappointed'}
        urgent_words = {'urgent', 'asap', 'immediately', 'critical', 'emergency', 'deadline', 'now', 'hurry'}
        surprising_words = {'wow', 'unexpected', 'surprise', 'shocking', 'unbelievable', 
                            'incredible', 'never seen', 'first time'}

        pos_count = sum(1 for word in positive_words if word in content_lower)
        neg_count = sum(1 for word in negative_words if word in content_lower)
        urg_count = sum(1 for word in urgent_words if word in content_lower)
        sur_count = sum(1 for word in surprising_words if word in content_lower)

        counts = {
            EmotionalValence.URGENT: urg_count,
            EmotionalValence.SURPRISING: sur_count,
            EmotionalValence.NEGATIVE: neg_count,
            EmotionalValence.POSITIVE: pos_count
        }

        max_val = max(counts.values())
        if max_val == 0:
            valence = EmotionalValence.NEUTRAL
            confidence = 0.5
            arousal_base = 0.1
        else:
            valence = EmotionalValence.NEUTRAL
            for v, c in counts.items():
                if c == max_val:
                    valence = v
                    break
            total_matches = sum(counts.values())
            confidence = min(0.5 + (max_val / total_matches) * 0.5, 1.0)
            arousal_base = min(1.0, max_val * 0.2)

        exclamation_count = content.count('!')
        question_count = content.count('?')
        caps_count = sum(1 for c in content if c.isupper())
        length = len(content) or 1

        caps_ratio = caps_count / length
        punct_ratio = (exclamation_count + question_count) / max(1, content.count('.') + exclamation_count + question_count)

        arousal_score = min(1.0, arousal_base + (exclamation_count * 0.15) + (caps_ratio * 1.5) + (punct_ratio * 0.5))

        low_dom_words = {'please', 'could', 'help', 'might', 'maybe', 'not sure'}
        high_dom_words = {'must', 'do', 'will', 'require', 'command', 'force'}

        low_dom = sum(1 for w in low_dom_words if w in content_lower)
        high_dom = sum(1 for w in high_dom_words if w in content_lower)

        if high_dom > low_dom:
            dominance = 0.8
        elif low_dom > high_dom:
            dominance = 0.3
        else:
            dominance = 0.5

        return EmotionalTag(
            valence=valence,
            arousal=max(0.0, min(1.0, arousal_score)),
            dominance=dominance,
            confidence=confidence
        )

    def modulate_salience(self, base_salience: float, emotional_tag: EmotionalTag) -> float:
        """
        Adjust salience based on emotional content.
        """
        salience = base_salience
        salience *= (1.0 + emotional_tag.arousal * self.arousal_boost_factor * 0.5)

        if emotional_tag.valence == EmotionalValence.NEGATIVE:
            salience *= self.negative_bias

        if emotional_tag.valence == EmotionalValence.URGENT:
            salience = max(salience, 0.9)

        if emotional_tag.valence == EmotionalValence.SURPRISING:
            salience *= 1.3

        return max(0.0, min(1.0, salience))

    def modulate_stability(self, base_stability: float, emotional_tag: EmotionalTag) -> float:
        """
        Emotional events are more stable (resist forgetting).
        """
        stability = base_stability
        stability *= (1.0 + emotional_tag.arousal * 0.5)

        if emotional_tag.valence in (EmotionalValence.NEGATIVE, EmotionalValence.URGENT):
            stability *= 1.2

        return stability

    def should_create_flashbulb(self, emotional_tag: EmotionalTag) -> bool:
        """
        Flashbulb memories: extremely vivid, long-lasting memories of surprising/emotional events.
        """
        return emotional_tag.arousal > 0.8 and emotional_tag.valence in (
            EmotionalValence.SURPRISING,
            EmotionalValence.NEGATIVE,
            EmotionalValence.URGENT
        )

    def get_emotional_summary(self, memories: List[EpisodicMemory]) -> Dict[str, Any]:
        """
        Analyze a batch of memories and return emotional distribution.
        """
        counts = {
            'positive_count': 0,
            'negative_count': 0,
            'neutral_count': 0,
            'urgent_count': 0,
            'surprising_count': 0
        }

        total_arousal = 0.0
        total_dominance = 0.0
        most_emotional_memory = None
        max_arousal = -1.0
        valid_memories_count = 0

        for mem in memories:
            tag = getattr(mem, 'emotional_tag', None)
            
            # Analyze on the fly if no tag is present but content is
            if not tag and hasattr(mem, 'content'):
                tag = self.analyze(mem.content)
            elif isinstance(tag, dict):
                tag = EmotionalTag(**tag)
                
            if not tag:
                continue

            valid_memories_count += 1

            if tag.valence == EmotionalValence.POSITIVE:
                counts['positive_count'] += 1
            elif tag.valence == EmotionalValence.NEGATIVE:
                counts['negative_count'] += 1
            elif tag.valence == EmotionalValence.NEUTRAL:
                counts['neutral_count'] += 1
            elif tag.valence == EmotionalValence.URGENT:
                counts['urgent_count'] += 1
            elif tag.valence == EmotionalValence.SURPRISING:
                counts['surprising_count'] += 1

            total_arousal += tag.arousal
            total_dominance += tag.dominance

            if tag.arousal > max_arousal:
                max_arousal = tag.arousal
                most_emotional_memory = mem

        return {
            **counts,
            'avg_arousal': total_arousal / valid_memories_count if valid_memories_count else 0.0,
            'avg_dominance': total_dominance / valid_memories_count if valid_memories_count else 0.0,
            'most_emotional_memory': most_emotional_memory
        }
