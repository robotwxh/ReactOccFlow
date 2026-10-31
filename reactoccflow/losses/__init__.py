"""Independent research losses; future labels are supervision inputs only."""
from .consistency import ConsistencyLosses, decode_endpoints

__all__ = ['ConsistencyLosses', 'decode_endpoints']
