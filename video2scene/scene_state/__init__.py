"""Persistent scene revisions and isolated candidates for the v1 contract."""

from .repository import SQLiteSceneRepository

SceneRepository = SQLiteSceneRepository

__all__ = ["SQLiteSceneRepository", "SceneRepository"]
