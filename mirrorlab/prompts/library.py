"""
Prompt Library Registry and Version Tracker for MIRRORLAB.

Provides centralized management, version tracking, and schema verification
for all benchmark task prompts.
"""

from __future__ import annotations

from typing import Any

from mirrorlab.prompts.templates import (
    PROMPT_OBSTRUCTION_V1,
    PROMPT_REPAIR_V1,
    PROMPT_VISIBILITY_V1,
    PromptTemplate,
)
from mirrorlab.schema.common import TaskType


class PromptRegistry:
    """
    Centralized registry of versioned prompt templates for MIRRORLAB.
    Tracks semantic versions and cryptographic SHA-256 hashes for all prompts.
    """

    def __init__(self) -> None:
        self._registry: dict[tuple[TaskType, str], PromptTemplate] = {}
        self._by_id: dict[str, PromptTemplate] = {}

        # Register standard v1.0.0 benchmark prompts
        self.register(PROMPT_VISIBILITY_V1)
        self.register(PROMPT_OBSTRUCTION_V1)
        self.register(PROMPT_REPAIR_V1)

    def register(self, template: PromptTemplate) -> None:
        """Register a new or updated prompt template."""
        key = (template.metadata.task_type, template.metadata.version)
        self._registry[key] = template
        self._by_id[template.metadata.prompt_id] = template

    def get(
        self, task_type: TaskType | str, version: str | None = None
    ) -> PromptTemplate:
        """
        Retrieve a prompt template by task type and version.
        If version is omitted, returns the latest registered version.
        """
        tt = TaskType(task_type) if isinstance(task_type, str) else task_type

        if version:
            key = (tt, version)
            if key not in self._registry:
                raise KeyError(
                    f"Prompt template for task '{tt.value}' version '{version}' not found."
                )
            return self._registry[key]

        # Find latest version for this task type
        matching = [
            t for (t_type, _), t in self._registry.items() if t_type == tt
        ]
        if not matching:
            raise KeyError(f"No prompt template found for task '{tt.value}'.")

        # Sort by semver
        matching.sort(key=lambda t: [int(x) for x in t.metadata.version.split(".")])
        return matching[-1]

    def get_by_id(self, prompt_id: str) -> PromptTemplate:
        """Retrieve a prompt template by its unique prompt_id."""
        if prompt_id not in self._by_id:
            raise KeyError(f"Prompt template with id '{prompt_id}' not found.")
        return self._by_id[prompt_id]

    def list_prompts(self) -> list[dict[str, Any]]:
        """Return a catalog of all registered prompt templates and their cryptographic hashes."""
        catalog = []
        for template in self._registry.values():
            catalog.append(
                {
                    "prompt_id": template.metadata.prompt_id,
                    "task_type": template.metadata.task_type.value,
                    "version": template.metadata.version,
                    "version_hash": template.version_hash,
                    "created_date": template.metadata.created_date,
                    "description": template.metadata.description,
                    "question": template.question,
                }
            )
        return catalog

    def validate_response(
        self,
        task_type: TaskType | str,
        response_data: dict[str, Any],
        version: str | None = None,
    ) -> tuple[bool, str | None]:
        """Validate model output dictionary against the prompt's registered JSON schema."""
        template = self.get(task_type, version)
        return template.validate_response(response_data)


# Global singleton registry instance
PROMPT_REGISTRY = PromptRegistry()


def get_prompt(
    task_type: TaskType | str, version: str | None = None
) -> PromptTemplate:
    """Convenience function to get a prompt template."""
    return PROMPT_REGISTRY.get(task_type, version)


def render_prompt(
    task_type: TaskType | str, version: str | None = None
) -> str:
    """Convenience function to render prompt text."""
    return PROMPT_REGISTRY.get(task_type, version).render()


def get_prompt_schema(
    task_type: TaskType | str, version: str | None = None
) -> dict[str, Any]:
    """Convenience function to get prompt JSON schema."""
    return PROMPT_REGISTRY.get(task_type, version).json_schema_spec


def get_prompt_example(
    task_type: TaskType | str, version: str | None = None
) -> dict[str, Any]:
    """Convenience function to get example response."""
    return PROMPT_REGISTRY.get(task_type, version).example_response


def list_all_prompts() -> list[dict[str, Any]]:
    """Convenience function to list all prompt metadata."""
    return PROMPT_REGISTRY.list_prompts()
