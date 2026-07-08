"""
Conversation history store — simple JSON file per user.
Replaces ChromaDB for history (no embeddings needed for plain text storage).
"""
import os
import json
from typing import List, Dict

_PROJECT_ROOT   = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
_HISTORY_FILE   = os.path.join(_PROJECT_ROOT, "conversation_history.json")


def _load_all() -> Dict:
    if os.path.exists(_HISTORY_FILE):
        try:
            with open(_HISTORY_FILE, "r", encoding="utf-8") as f:
                return json.load(f)
        except Exception:
            pass
    return {}


def _save_all(data: Dict) -> None:
    with open(_HISTORY_FILE, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)


def save_history_to_chroma(user_id: str, history: List[Dict[str, str]]) -> None:
    """Save conversation history for a user."""
    data = _load_all()
    data[user_id] = history
    _save_all(data)


def load_history_from_chroma(user_id: str) -> List[Dict[str, str]]:
    """Load conversation history for a user. Returns empty list if not found."""
    data = _load_all()
    return data.get(user_id, [])


def delete_history_from_chroma(user_id: str) -> None:
    """Delete history for a user."""
    data = _load_all()
    if user_id in data:
        del data[user_id]
        _save_all(data)
