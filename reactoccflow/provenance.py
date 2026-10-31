"""Verified ordinary-file migration and read-only source audits.

Neither reference repository contains a verified copying/provenance utility;
these standard-library helpers are project-owned infrastructure.
"""
from __future__ import annotations

import hashlib
import os
from pathlib import Path
import shutil
import stat
import tempfile


_TEXT_SUFFIXES = {'.py', '.yaml', '.yml', '.toml', '.json', '.cfg', '.ini', '.sh', '.md', '.txt'}


def sha256_file(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for chunk in iter(lambda: stream.read(8 * 1024 * 1024), b''):
            digest.update(chunk)
    return digest.hexdigest()


def _reject_links(path: Path) -> None:
    for component in [path, *path.parents]:
        if component.is_symlink():
            raise ValueError(f'Symlink paths are forbidden: {component}')


def copy_resource(source: str | Path, destination: str | Path) -> dict:
    """Copy bytes into independent storage, retaining a matching existing copy.

    Refuses link paths, inode aliases, and overwriting a nonmatching resource.
    Returns a content checksum only after independently verifying the target.
    """
    source, destination = Path(source).absolute(), Path(destination).absolute()
    _reject_links(source)
    _reject_links(destination)
    if not source.is_file():
        raise FileNotFoundError(source)
    if source == destination or (destination.exists() and os.path.samefile(source, destination)):
        raise ValueError('Source and destination must have independent storage')
    if destination.exists() and destination.stat().st_nlink != 1:
        raise ValueError(f'Hardlinked destination is forbidden: {destination}')
    source_digest = sha256_file(source)
    if destination.exists():
        if sha256_file(destination) != source_digest:
            raise FileExistsError(f'Existing resource differs: {destination}')
    else:
        destination.parent.mkdir(parents=True, exist_ok=True)
        fd, temporary = tempfile.mkstemp(prefix='.copy-', dir=destination.parent)
        os.close(fd)
        try:
            shutil.copyfile(source, temporary)
            if sha256_file(temporary) != source_digest:
                raise OSError(f'Copied content verification failed: {source}')
            # One migration writer owns a destination. Publish only a complete,
            # verified ordinary file; no source or temporary hardlinks are used.
            os.replace(temporary, destination)
        finally:
            Path(temporary).unlink(missing_ok=True)
    if sha256_file(destination) != source_digest:
        raise OSError(f'Final resource verification failed: {destination}')
    return {'source': str(source), 'destination': str(destination),
            'size': destination.stat().st_size, 'sha256': source_digest, 'verified': True}


def snapshot_tree(root: str | Path) -> dict:
    """Record source bytes, links and metadata without following directory links."""
    root = Path(root)
    entries = {}
    for directory, dirs, files in os.walk(root, followlinks=False):
        dirs[:] = sorted(d for d in dirs if d != '.git')
        base = Path(directory)
        for name in sorted(dirs + files):
            path = base / name
            info = path.lstat()
            entry = {'mode': stat.S_IFMT(info.st_mode), 'size': info.st_size,
                     'mtime_ns': info.st_mtime_ns, 'inode': info.st_ino}
            if path.is_symlink():
                entry['link'] = os.readlink(path)
            elif path.is_file() and (path.suffix in _TEXT_SUFFIXES or 'LICENSE' in path.name or path.name == '.gitignore'):
                entry['sha256'] = sha256_file(path)
            entries[str(path.relative_to(root))] = entry
    return entries


def compare_snapshot(before: dict, after: dict) -> list[str]:
    differences = []
    for path in sorted(before.keys() | after.keys()):
        if path not in before:
            differences.append(f'added: {path}')
        elif path not in after:
            differences.append(f'removed: {path}')
        elif before[path] != after[path]:
            differences.append(f'changed: {path}')
    return differences
