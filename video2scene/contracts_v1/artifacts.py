from pathlib import Path
from urllib.parse import unquote, urlsplit
from .core import ContractError, file_hash, validate


class ArtifactResolver:
    def __init__(self, root):
        self.root = Path(root).resolve()

    def path(self, uri, scene_id=None):
        parsed = urlsplit(uri)
        key = unquote(parsed.path).lstrip("/")
        if (parsed.scheme != "artifact" or not parsed.netloc or parsed.query or parsed.fragment
                or not key or "\\" in key or ":" in key
                or any(p in (".", "..", "") for p in key.split("/"))
                or (scene_id is not None and parsed.netloc != scene_id)):
            raise ContractError("FORBIDDEN_SCOPE", "Artifact URI is outside the allowed scope")
        scene = (self.root / parsed.netloc).resolve()
        path = (scene / key).resolve()
        if not scene.is_relative_to(self.root) or not path.is_relative_to(scene):
            raise ContractError("FORBIDDEN_SCOPE", "Artifact path escapes storage")
        return path

    def resolve(self, ref, scene_id=None):
        validate(ref, "ArtifactRef")
        path = self.path(ref["uri"], scene_id)
        if not path.is_file():
            raise ContractError("NOT_FOUND", "Artifact does not exist", {"uri": ref["uri"]})
        if file_hash(path) != ref["sha256"]:
            raise ContractError("STALE_ARTIFACT", "Artifact content changed", {"uri": ref["uri"]})
        return path

