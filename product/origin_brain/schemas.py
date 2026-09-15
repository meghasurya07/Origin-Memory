import uuid
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional, Tuple

from pydantic import BaseModel, Field

from .models import EpisodicMemory, SemanticMemory


class SchemaSlot(BaseModel):
    """A slot in a schema that can be filled with information."""
    name: str
    description: str
    slot_type: str
    required: bool = False
    default_value: Any = None
    examples: List[str] = Field(default_factory=list)


class MemorySchema(BaseModel):
    """A prior knowledge structure used to accelerate learning."""
    id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    name: str
    description: str
    slots: Dict[str, SchemaSlot]
    instance_count: int = 0
    confidence: float = 0.0
    created_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))
    updated_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))
    metadata: Dict[str, Any] = Field(default_factory=dict)


class SchemaInstance(BaseModel):
    """A filled-in instance of a MemorySchema."""
    id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    schema_id: str
    filled_slots: Dict[str, Any]
    source_memory_id: str
    confidence: float = 1.0
    created_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))


class SchemaEngine:
    """Engine for registering, matching, and filling schemas."""
    
    def __init__(self, predefined_schemas: Optional[List] = None):
        self._schemas: Dict[str, MemorySchema] = {}
        self._instances: Dict[str, SchemaInstance] = {}
        
        if predefined_schemas:
            for schema in predefined_schemas:
                if isinstance(schema, MemorySchema):
                    self._schemas[schema.id] = schema
                elif isinstance(schema, dict):
                    self.register_schema(
                        name=schema["name"],
                        description=schema.get("description", ""),
                        slots=schema.get("slots", {})
                    )

    def register_schema(self, name: str, description: str, slots: Dict[str, dict]) -> MemorySchema:
        """Creates and registers a new schema."""
        schema_slots = {}
        for slot_name, slot_info in slots.items():
            schema_slots[slot_name] = SchemaSlot(
                name=slot_name,
                description=slot_info.get("description", ""),
                slot_type=slot_info.get("slot_type", "string"),
                required=slot_info.get("required", False),
                default_value=slot_info.get("default_value", None)
            )
            
        schema = MemorySchema(
            name=name,
            description=description,
            slots=schema_slots
        )
        self._schemas[schema.id] = schema
        return schema

    def match_to_schema(self, content: str, context: Optional[Dict[str, Any]] = None) -> List[Tuple[MemorySchema, float]]:
        """Attempts to match input content to existing schemas."""
        matches = []
        content_lower = content.lower()
        context_keys = set(context.keys()) if context else set()
        
        for schema in self._schemas.values():
            score = 0.0
            total_factors = 1 + len(schema.slots)
            
            if schema.name.lower() in content_lower:
                score += 1.0
            elif context_keys and schema.name.lower() in context_keys:
                score += 0.8
                
            for slot_name in schema.slots.keys():
                if slot_name.lower() in content_lower:
                    score += 1.0
                elif slot_name in context_keys:
                    score += 0.5
                    
            normalized_score = score / total_factors
            if normalized_score > 0:
                matches.append((schema, min(normalized_score, 1.0)))
                
        matches.sort(key=lambda x: x[1], reverse=True)
        return matches

    def fill_schema(self, schema: MemorySchema, content: str, context: Optional[Dict[str, Any]] = None, source_memory_id: str = '') -> SchemaInstance:
        """Attempts to fill schema slots from the content and context."""
        filled_slots = {}
        content_lower = content.lower()
        context = context or {}
        
        # Simple extraction logic (placeholder for LLM-based extraction)
        for slot_name, slot in schema.slots.items():
            if slot_name in context:
                filled_slots[slot_name] = context[slot_name]
            elif slot.default_value is not None:
                filled_slots[slot_name] = slot.default_value
            elif slot.slot_type == 'boolean':
                filled_slots[slot_name] = slot_name.lower() in content_lower
            else:
                filled_slots[slot_name] = f"<extracted_{slot_name}>"
                
        instance = SchemaInstance(
            schema_id=schema.id,
            filled_slots=filled_slots,
            source_memory_id=source_memory_id,
            confidence=0.8
        )
        
        self._instances[instance.id] = instance
        
        schema.instance_count += 1
        schema.updated_at = datetime.now(timezone.utc)
        self._schemas[schema.id] = schema
        
        return instance

    def get_schema_by_name(self, name: str) -> Optional[MemorySchema]:
        """Lookup schema by its name."""
        for schema in self._schemas.values():
            if schema.name == name:
                return schema
        return None

    def discover_schemas(self, memories: List[EpisodicMemory], min_pattern_count: int = 3) -> List[MemorySchema]:
        """Analyze a collection of episodic memories to discover recurring patterns/schemas."""
        context_key_counts = {}
        for memory in memories:
            if hasattr(memory, 'context') and isinstance(memory.context, dict):
                for key in memory.context.keys():
                    context_key_counts[key] = context_key_counts.get(key, 0) + 1
                    
        discovered = []
        for key, count in context_key_counts.items():
            if count >= min_pattern_count:
                schema_name = f"auto_{key}"
                if not self.get_schema_by_name(schema_name):
                    schema = self.register_schema(
                        name=schema_name,
                        description=f"Auto-discovered schema from context key: {key}",
                        slots={
                            "value": {
                                "description": f"Value for {key}",
                                "slot_type": "string",
                                "required": False
                            }
                        }
                    )
                    discovered.append(schema)
                    
        return discovered

    def get_statistics(self) -> Dict[str, Any]:
        """Returns statistics about the loaded schemas and instances."""
        most_used = max(self._schemas.values(), key=lambda s: s.instance_count, default=None)
        
        avg_conf = 0.0
        if self._schemas:
            avg_conf = sum(s.confidence for s in self._schemas.values()) / len(self._schemas)
            
        return {
            "schema_count": len(self._schemas),
            "instance_count": len(self._instances),
            "most_used_schema": most_used.name if most_used else None,
            "avg_confidence": avg_conf
        }


# Predefined schemas module-level list
PREDEFINED_SCHEMAS = [
    {
        "name": "user_preference",
        "description": "A user's stated preference or interest",
        "slots": {
            "preference_type": {"description": "Type of preference", "slot_type": "string", "required": True},
            "value": {"description": "The specific preference", "slot_type": "string", "required": True},
            "strength": {"description": "How strong the preference is (0-1)", "slot_type": "number", "required": False}
        }
    },
    {
        "name": "meeting",
        "description": "A scheduled meeting or event",
        "slots": {
            "title": {"description": "Title of meeting", "slot_type": "string", "required": True},
            "datetime": {"description": "When it happens", "slot_type": "datetime", "required": True},
            "participants": {"description": "Who is attending", "slot_type": "string", "required": False},
            "location": {"description": "Where it is", "slot_type": "string", "required": False},
            "notes": {"description": "Meeting notes", "slot_type": "string", "required": False}
        }
    },
    {
        "name": "person",
        "description": "A person the user knows",
        "slots": {
            "name": {"description": "Person's name", "slot_type": "string", "required": True},
            "role": {"description": "Their job or role", "slot_type": "string", "required": False},
            "relationship": {"description": "Relationship to user", "slot_type": "string", "required": False},
            "contact": {"description": "Contact info", "slot_type": "string", "required": False},
            "notes": {"description": "Other notes", "slot_type": "string", "required": False}
        }
    },
    {
        "name": "task",
        "description": "A task or action item",
        "slots": {
            "title": {"description": "Task description", "slot_type": "string", "required": True},
            "status": {"description": "Current status", "slot_type": "string", "required": True},
            "priority": {"description": "Task priority", "slot_type": "string", "required": False},
            "deadline": {"description": "When it is due", "slot_type": "datetime", "required": False},
            "assignee": {"description": "Who is responsible", "slot_type": "string", "required": False}
        }
    },
    {
        "name": "fact",
        "description": "A factual statement",
        "slots": {
            "subject": {"description": "Subject of the fact", "slot_type": "string", "required": True},
            "predicate": {"description": "Relationship or property", "slot_type": "string", "required": True},
            "object": {"description": "Object or value", "slot_type": "string", "required": True},
            "source": {"description": "Where the fact came from", "slot_type": "string", "required": False},
            "confidence": {"description": "Confidence in the fact", "slot_type": "number", "required": False}
        }
    }
]
