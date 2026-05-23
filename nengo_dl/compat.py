"""Compatibility utilities for nengo-dl."""


class FrozenOrderedSet:
    """An immutable, ordered collection of unique elements.

    Preserves insertion order (like ``dict`` keys in Python 3.7+) while
    also supporting ``__contains__``, ``__len__``, and hashing so it can
    be used as a dictionary key.

    Parameters
    ----------
    iterable : iterable
        Elements to include.  Duplicates are silently ignored; the first
        occurrence determines the position.
    """

    def __init__(self, iterable=()):
        seen = {}
        for item in iterable:
            if item not in seen:
                seen[item] = None
        self._items = tuple(seen)

    def __contains__(self, item):
        return item in set(self._items)

    def __iter__(self):
        return iter(self._items)

    def __len__(self):
        return len(self._items)

    def __hash__(self):
        return hash(self._items)

    def __eq__(self, other):
        if isinstance(other, FrozenOrderedSet):
            return self._items == other._items
        return NotImplemented

    def __repr__(self):
        return f"FrozenOrderedSet({list(self._items)!r})"
