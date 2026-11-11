import json
from pathlib import Path
from typing import List, Optional
from core.interfaces import ChatMessage
from config.settings import get_settings


class EpisodicMemory:
    """Manages short- and medium-term conversational memory with automatic window compaction."""

    def __init__(self, persistence_file: Optional[Path] = None, max_context_tokens: int = 8192):
        settings = get_settings()
        self.persistence_file = persistence_file or settings.memory_db_path
        self.max_context_tokens = max_context_tokens
        self.messages: List[ChatMessage] = []
        self._load()

    def add_message(self, role: str, content: str, tool_calls=None, tool_call_id=None) -> ChatMessage:
        msg = ChatMessage(role=role, content=content, tool_calls=tool_calls, tool_call_id=tool_call_id)
        self.messages.append(msg)
        self._compact_if_needed()
        self._save()
        return msg

    def get_messages(self) -> List[ChatMessage]:
        return list(self.messages)

    def clear(self) -> None:
        self.messages = []
        self._save()

    def estimate_tokens(self) -> int:
        """Rough token estimate (~4 chars per token)."""
        total_chars = sum(len(m.content) for m in self.messages)
        return total_chars // 4

    def _compact_if_needed(self) -> None:
        """Prunes earliest messages if approaching context limit, preserving system prompt."""
        estimated = self.estimate_tokens()
        threshold = int(self.max_context_tokens * 0.7)
        if estimated <= threshold or len(self.messages) <= 4:
            return

        # Separate system messages from conversational turns
        system_msgs = [m for m in self.messages if m.role == "system"]
        conversation = [m for m in self.messages if m.role != "system"]

        # Keep the most recent turns
        while self.estimate_tokens() > threshold and len(conversation) > 4:
            conversation.pop(0)

        # Insert a summary notice of truncated context
        summary_notice = ChatMessage(
            role="system",
            content="[Notice: Earlier turns of this conversation have been compacted to conserve memory.]"
        )
        self.messages = system_msgs + [summary_notice] + conversation

    def _save(self) -> None:
        try:
            self.persistence_file.parent.mkdir(parents=True, exist_ok=True)
            with open(self.persistence_file, "w", encoding="utf-8") as f:
                json.dump([m.model_dump() for m in self.messages], f, indent=2)
        except Exception:
            pass

    def _load(self) -> None:
        if self.persistence_file.exists():
            try:
                with open(self.persistence_file, "r", encoding="utf-8") as f:
                    data = json.load(f)
                    self.messages = [ChatMessage(**m) for m in data]
            except Exception:
                self.messages = []
