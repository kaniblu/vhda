__all__ = ["Arguments"]

from dataclasses import dataclass, fields


@dataclass
class Arguments:

    def to_json(self, max_repr=64) -> dict:
        def jsonify(val):
            if val is None:
                return None
            elif isinstance(val, dict):
                return {k: jsonify(v) for k, v in val.items()}
            elif isinstance(val, list):
                return list(map(jsonify, val))
            elif isinstance(val, (int, float, str)):
                return val
            else:
                val = repr(val)
                if len(val) > max_repr:
                    return val[:max_repr - 3] + "..."
                else:
                    return val

        return {fd.name: jsonify(getattr(self, fd.name))
                for fd in fields(self)}
