
import fnmatch
from deepagents.backends.protocol import WriteResult, ReadResult, EditResult, LsResult, GlobResult, GrepResult
from deepagents.backends.filesystem import perform_string_replacement,FilesystemBackend
from pathlib import Path

class StagingFilesystemBackend(FilesystemBackend):
    def __init__(self, root_dir, **kwargs):
        super().__init__(root_dir=root_dir, **kwargs)
        self.virtual_files = {}  # Relative path string -> content string

    def _get_rel_path(self, file_path: str) -> str:
        try:
            p = Path(file_path)
            if p.is_absolute():
                return str(p.relative_to(self.cwd))
            return str(p)
        except ValueError:
            return file_path

    def write(self, file_path: str, content: str) -> WriteResult:
        rel_path = self._get_rel_path(file_path)
        self.virtual_files[rel_path] = content
        return WriteResult(path=file_path)

    async def awrite(self, file_path: str, content: str) -> WriteResult:
        return self.write(file_path, content)

    def read(self, file_path: str, offset: int = 0, limit: int = 2000) -> ReadResult:
        rel_path = self._get_rel_path(file_path)
        if rel_path in self.virtual_files:
            content = self.virtual_files[rel_path]
            sliced_content = content[offset : offset + limit]
            return ReadResult(content=sliced_content)
        return super().read(file_path, offset=offset, limit=limit)

    async def aread(self, file_path: str, offset: int = 0, limit: int = 2000) -> ReadResult:
        rel_path = self._get_rel_path(file_path)
        if rel_path in self.virtual_files:
            content = self.virtual_files[rel_path]
            sliced_content = content[offset : offset + limit]
            return ReadResult(content=sliced_content)
        return await super().aread(file_path, offset=offset, limit=limit)

    def edit(self, file_path: str, old_string: str, new_string: str, replace_all: bool = False) -> EditResult:
        rel_path = self._get_rel_path(file_path)
        if rel_path in self.virtual_files:
            content = self.virtual_files[rel_path]
            old_string = old_string.replace("\r\n", "\n").replace("\r", "\n")
            new_string = new_string.replace("\r\n", "\n").replace("\r", "\n")
            result = perform_string_replacement(content, old_string, new_string, replace_all)
            if isinstance(result, str):
                return EditResult(error=result)
            new_content, occurrences = result
            self.virtual_files[rel_path] = new_content
            return EditResult(path=file_path, occurrences=int(occurrences))
        
        # Load from disk to cache first if file exists
        disk_read = super().read(file_path)
        if disk_read.content is not None:
            self.virtual_files[rel_path] = disk_read.content
            return self.edit(file_path, old_string, new_string, replace_all)
        return EditResult(error=f"Error: File '{file_path}' not found")

    async def aedit(self, file_path: str, old_string: str, new_string: str, replace_all: bool = False) -> EditResult:
        return self.edit(file_path, old_string, new_string, replace_all)

    def ls(self, path: str) -> LsResult:
        base_result = super().ls(path)
        rel_path = self._get_rel_path(path)
        if rel_path == "." or rel_path == "/":
            rel_path = ""
            
        entries = base_result.entries or []
        existing_paths = {entry["path"].rstrip("/") for entry in entries}

        for v_path in self.virtual_files:
            v_dir = str(Path(v_path).parent)
            if v_dir == rel_path or (rel_path == "" and v_dir == "."):
                name = Path(v_path).name
                virt_entry_path = name if rel_path == "" else f"{rel_path}/{name}"
                if virt_entry_path not in existing_paths:
                    entries.append({
                        "path": virt_entry_path,
                        "is_dir": False,
                        "size": len(self.virtual_files[v_path]),
                    })
                    existing_paths.add(virt_entry_path)
            elif v_path.startswith(rel_path + "/" if rel_path else ""):
                sub_parts = v_path[len(rel_path + "/" if rel_path else ""):].split("/")
                if len(sub_parts) > 1:
                    dir_name = sub_parts[0]
                    virt_dir_path = dir_name if rel_path == "" else f"{rel_path}/{dir_name}"
                    if virt_dir_path not in existing_paths:
                        entries.append({
                            "path": virt_dir_path + "/",
                            "is_dir": True,
                            "size": 0,
                        })
                        existing_paths.add(virt_dir_path)

        entries.sort(key=lambda x: x.get("path", ""))
        return LsResult(error=base_result.error, entries=entries)

    async def als(self, path: str) -> LsResult:
        return self.ls(path)

    def glob(self, pattern: str, path: str | None = None) -> GlobResult:
        base_result = super().glob(pattern, path)
        search_path = path or ""
        rel_search_path = self._get_rel_path(search_path)
        if rel_search_path == "." or rel_search_path == "/":
            rel_search_path = ""

        matches = base_result.matches or []
        existing_paths = {m["path"] for m in matches}

        for v_path in self.virtual_files:
            if rel_search_path == "" or v_path.startswith(rel_search_path + "/"):
                rel_v_path = v_path[len(rel_search_path + "/" if rel_search_path else ""):]
                if fnmatch.fnmatch(rel_v_path, pattern):
                    if v_path not in existing_paths:
                        matches.append({
                            "path": v_path,
                            "is_dir": False,
                            "size": len(self.virtual_files[v_path]),
                        })
                        existing_paths.add(v_path)
                        
        matches.sort(key=lambda x: x.get("path", ""))
        return GlobResult(error=base_result.error, matches=matches)

    async def aglob(self, pattern: str, path: str | None = None) -> GlobResult:
        return self.glob(pattern, path)

    def grep(self, pattern: str, path: str | None = None, glob: str | None = None) -> GrepResult:
        return super().grep(pattern, path=path, glob=glob)

    async def agrep(self, pattern: str, path: str | None = None, glob: str | None = None) -> GrepResult:
        return await super().agrep(pattern, path=path, glob=glob)
