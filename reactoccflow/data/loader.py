"""Explicitly release persistent DataLoader workers before distributed teardown.

No matching shutdown utility was found in the reference repositories. This is
project-owned lifecycle glue for the pinned PyTorch 2.7 environment; provenance:
docs/provenance/history_input_comparison_sources_20261117.json.
"""
from __future__ import annotations


def shutdown_data_loaders(*loaders):
    """Close initialized persistent iterators; leave unstarted loaders alone."""
    for loader in loaders:
        iterator = getattr(loader, '_iterator', None)
        if iterator is not None:
            iterator._shutdown_workers()
            loader._iterator = None
