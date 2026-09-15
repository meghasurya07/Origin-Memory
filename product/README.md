# Origin Brain SDK

**Human-Like Memory Infrastructure for AI Agents**

By [Origin AI](https://originai.in)

```bash
pip install -e ".[test]"
python -m pytest tests/ -v
```

```python
from origin_brain import Brain

brain = Brain(config={"agent_id": "my-agent"})
brain.encode("User prefers Python", salience=0.8)
results = brain.recall("What language?")
```

See the full documentation in `../research/`.
