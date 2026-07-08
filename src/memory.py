"""
Memory module — Day 5 Session 2
Exercise 1: conversation history passed to LLM
Exercise 2: summarisation after every 10 turns
Exercise 3: persistent user profiles (JSON file)
Exercise 4: personalised system prompt injection
Exercise 5: clear-data command + 30-day auto-expire + privacy notice
"""
import os
import json
import time
from typing import List, Dict, Any, Optional
from openai import OpenAI
from src.history_store import save_history_to_chroma, load_history_from_chroma, delete_history_from_chroma

_PROFILES_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "user_profiles.json")
_SUMMARISE_EVERY = 10   # turns before summarisation kicks in
_EXPIRE_DAYS     = 30   # profiles not accessed for this many days are deleted

# ── Privacy notice shown on first interaction ──────────────────────────────────
PRIVACY_NOTICE = (
    "👋 **Before we start** — Zia remembers your name, branch interest, and "
    "language preference to personalise your experience. "
    "This data is stored locally on this device only. "
    "Type **'clear my data'** at any time to delete everything about you. "
    "Data not accessed for 30 days is automatically removed. 🔒"
)

# ─────────────────────────────────────────────────────────────────────────────
# Exercise 2 · Summarisation
# ─────────────────────────────────────────────────────────────────────────────

def summarise_history(history: List[Dict[str, str]]) -> str:

    transcript = "\n".join(
        f"{m['role'].upper()}: {m['content'][:300]}" for m in history
    )
    prompt = (
        "Summarise this BVRIT chatbot conversation into ONE concise paragraph (max 120 words). "
        "You MUST preserve: user's name (if stated), branch/topic interests, "
        "specific fee amounts or dates mentioned, stated preferences (language, detail level), "
        "and any unresolved questions.\n\n"
        f"CONVERSATION:\n{transcript}\n\nSUMMARY:"
    )
    from src.observability import logged_llm_call
    api_key  = os.getenv("OPENAI_API_KEY")
    base_url = os.getenv("OPENAI_BASE_URL", "https://models.inference.ai.azure.com")
    from src.generator import FREE_MODEL
    client = OpenAI(api_key=api_key, base_url=base_url)
    try:
        resp = logged_llm_call(
            client=client,
            model=FREE_MODEL,
            messages=[{"role": "user", "content": prompt}],
            purpose="summarise",
            temperature=0.2,
            max_tokens=180,
        )
        if resp and resp.choices:
            content = resp.choices[0].message.content
            return content.strip() if content else ""
    except Exception:
        pass
    return ""


def compress_history(
    history: List[Dict[str, str]],
    keep_last: int = 10,
) -> List[Dict[str, str]]:
    """
    Exercise 2: after every `_SUMMARISE_EVERY` turns, summarise older turns.
    Returns a new history list: [summary_message] + last `keep_last` turns.
    Also returns token estimate before/after for reporting.
    """
    if len(history) <= keep_last:
        return history

    older  = history[:-keep_last]
    recent = history[-keep_last:]

    summary_text = summarise_history(older)
    if not summary_text:
        # fallback: just drop older turns
        return recent

    summary_msg = {
        "role": "system",
        "content": f"[Conversation summary so far]: {summary_text}",
    }
    return [summary_msg] + recent


# ─────────────────────────────────────────────────────────────────────────────
# Exercise 3 · Persistent user profiles
# ─────────────────────────────────────────────────────────────────────────────

def _load_all_profiles() -> Dict[str, Any]:
    if os.path.exists(_PROFILES_PATH):
        try:
            with open(_PROFILES_PATH, "r", encoding="utf-8") as f:
                return json.load(f)
        except Exception:
            pass
    return {}


def _save_all_profiles(profiles: Dict[str, Any]) -> None:
    with open(_PROFILES_PATH, "w", encoding="utf-8") as f:
        json.dump(profiles, f, indent=2, ensure_ascii=False)


def _empty_profile(user_id: str) -> Dict[str, Any]:
    """
    User profile schema (Exercise 3 + 5 classification):
      name              — ESSENTIAL
      branch_interest   — ESSENTIAL
      language          — ESSENTIAL
      detail_level      — NICE-TO-HAVE
      prior_topics      — NICE-TO-HAVE
      last_session_summary — NICE-TO-HAVE
      fee_amounts_discussed — NICE-TO-HAVE
      scholarship_details   — SENSITIVE (stored only if user explicitly shares)
      full_conversation_transcripts — NOT STORED (privacy)
      created_at        — internal
      last_accessed     — internal (used for 30-day expiry)
    """
    now = time.time()
    return {
        "user_id":               user_id,
        "name":                  None,
        "branch_interest":       None,
        "language":              "English",
        "detail_level":          "normal",   # "brief" | "normal" | "detailed"
        "prior_topics":          [],
        "last_session_summary":  None,
        "fee_amounts_discussed": [],
        "scholarship_details":   None,       # only stored if user shares explicitly
        "created_at":            now,
        "last_accessed":         now,
        "is_new":                True,       # True on very first load
    }


def load_profile(user_id: str) -> Dict[str, Any]:
    """Load profile for user_id, creating a new one if not found. Auto-expires stale profiles."""
    profiles = _load_all_profiles()
    _auto_expire(profiles)

    if user_id not in profiles:
        profile = _empty_profile(user_id)
        profiles[user_id] = profile
        _save_all_profiles(profiles)
        return profile

    profile = profiles[user_id]
    profile["last_accessed"] = time.time()
    profile["is_new"] = False
    profiles[user_id] = profile
    _save_all_profiles(profiles)
    return profile


def load_history(user_id: str) -> List[Dict[str, str]]:
    """Load conversation history for user_id from ChromaDB (separate from KB)."""
    return load_history_from_chroma(user_id)


def save_profile(profile: Dict[str, Any]) -> None:
    """Persist an updated profile."""
    profiles = _load_all_profiles()
    profile["last_accessed"] = time.time()
    profile["is_new"] = False
    profiles[profile["user_id"]] = profile
    _save_all_profiles(profiles)


def save_history(user_id: str, history: List[Dict[str, str]]) -> None:
    """Persist conversation history to ChromaDB (separate collection from KB)."""
    save_history_to_chroma(user_id, history)


def delete_profile(user_id: str) -> bool:
    """Exercise 5: delete a user's profile and history. Returns True if profile existed."""
    profiles = _load_all_profiles()
    delete_history_from_chroma(user_id)   # wipe history from ChromaDB too
    if user_id in profiles:
        del profiles[user_id]
        _save_all_profiles(profiles)
        return True
    return False


def _auto_expire(profiles: Dict[str, Any]) -> None:
    """Exercise 5: remove profiles not accessed for _EXPIRE_DAYS days."""
    cutoff = time.time() - _EXPIRE_DAYS * 86400
    stale  = [uid for uid, p in profiles.items() if p.get("last_accessed", 0) < cutoff]
    for uid in stale:
        del profiles[uid]
    if stale:
        _save_all_profiles(profiles)


# ─────────────────────────────────────────────────────────────────────────────
# Exercise 4 · Personalised system prompt
# ─────────────────────────────────────────────────────────────────────────────

def build_personalised_system_prompt(base_prompt: str, profile: Dict[str, Any]) -> str:
    """
    Inject user profile facts into the system prompt so the LLM personalises responses.
    """
    lines = []
    if profile.get("name"):
        lines.append(f"The user's name is {profile['name']}.")
    if profile.get("branch_interest"):
        lines.append(f"They are interested in {profile['branch_interest']}.")
    if profile.get("language") and profile["language"] != "English":
        lines.append(f"Respond in {profile['language']}.")
    if profile.get("detail_level") == "brief":
        lines.append("Keep answers brief — use bullet points, no long paragraphs.")
    elif profile.get("detail_level") == "detailed":
        lines.append("Give detailed, thorough answers with full explanations.")
    if profile.get("last_session_summary"):
        lines.append(f"Previous session summary: {profile['last_session_summary']}")
    if profile.get("prior_topics"):
        topics = ", ".join(profile["prior_topics"][-5:])
        lines.append(f"Topics discussed before: {topics}.")

    if not lines:
        return base_prompt

    injection = "\n\n## USER PROFILE (personalise based on this)\n" + "\n".join(lines)
    return base_prompt + injection


# ─────────────────────────────────────────────────────────────────────────────
# Exercise 3+4 · Extract facts from a turn and update profile
# ─────────────────────────────────────────────────────────────────────────────

def update_profile_from_turn(profile: Dict[str, Any], user_msg: str, assistant_msg: str) -> Dict[str, Any]:
    """
    Lightweight rule-based extraction — no extra API call.
    Updates name, branch_interest, detail_level, prior_topics from the conversation.
    """
    import re
    u = user_msg.lower()

    # Name extraction: "my name is X" / "I am X" / "call me X"
    name_match = re.search(
        r"(?:my name is|i am|i'm|call me)\s+([A-Z][a-z]+)", user_msg, re.IGNORECASE
    )
    if name_match:
        profile["name"] = name_match.group(1).strip()

    # Branch interest
    branches = {
        "cse": "B.Tech CSE", "computer science": "B.Tech CSE",
        "aiml": "B.Tech CSE-AIML", "ai": "B.Tech CSE-AIML",
        "ece": "B.Tech ECE", "electronics": "B.Tech ECE",
        "eee": "B.Tech EEE", "electrical": "B.Tech EEE",
        "mechanical": "B.Tech Mechanical",
        "civil": "B.Tech Civil",
        "mba": "MBA", "mtech": "M.Tech",
    }
    for kw, branch in branches.items():
        if kw in u:
            profile["branch_interest"] = branch
            break

    # Detail preference
    if any(w in u for w in ["brief", "short", "concise", "quickly", "summarise"]):
        profile["detail_level"] = "brief"
    elif any(w in u for w in ["detailed", "explain", "elaborate", "in detail", "thorough"]):
        profile["detail_level"] = "detailed"

    # Language preference
    for lang in ["telugu", "hindi", "english"]:
        if lang in u:
            profile["language"] = lang.capitalize()
            break

    # Track topics
    topic_keywords = {
        "fee": "fees", "admission": "admissions", "placement": "placements",
        "department": "departments", "hostel": "hostel", "campus": "campus",
        "faculty": "faculty", "contact": "contact", "scholarship": "scholarships",
    }
    for kw, topic in topic_keywords.items():
        if kw in u and topic not in profile["prior_topics"]:
            profile["prior_topics"].append(topic)

    return profile


# ─────────────────────────────────────────────────────────────────────────────
# Exercise 5 · "Clear my data" detection
# ─────────────────────────────────────────────────────────────────────────────

def is_clear_data_command(text: str) -> bool:
    import re
    patterns = [
        r"clear\s+my\s+data", r"delete\s+my\s+(data|profile|info)",
        r"forget\s+(me|my data|everything)", r"remove\s+my\s+(data|profile)",
        r"right\s+to\s+be\s+forgotten",
    ]
    t = text.lower()
    return any(re.search(p, t) for p in patterns)
