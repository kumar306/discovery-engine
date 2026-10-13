import contextvars
import json
import logging
import sys
from datetime import UTC, datetime
from typing import Any

# coroutines on one event loop share every global variable and take turns at each await
# a plain global holding the current request id would be overwritten by whichever request ran last
# so the id of one request could leak into the log lines of another request
# a ContextVar gives every task its own copy, a set in one request is invisible to the others
request_id_var: contextvars.ContextVar[str | None] = contextvars.ContextVar(
    "request_id", default=None
)


class JsonFormatter(logging.Formatter):
    def format(self, record: logging.LogRecord) -> str:
        entry: dict[str, Any] = {
            "time": datetime.fromtimestamp(record.created, UTC).isoformat(),
            "level": record.levelname,
            "logger": record.name,
            "message": record.getMessage(),
            "request_id": request_id_var.get(),
        }
        # fields passed as extra={"fields": {...}} become top level keys of the line
        entry.update(getattr(record, "fields", {}))
        if record.exc_info:
            entry["exception"] = self.formatException(record.exc_info)
        # json escapes newlines inside strings, so one record is always one physical line
        return json.dumps(entry, default=str, ensure_ascii=False)


# called once at startup by the process entry point, before the first request is served
# level comes from config so a deployment can switch on debug without a code change
def configure_logging(level: str = "INFO") -> None:
    handler = logging.StreamHandler(sys.stdout)
    handler.setFormatter(JsonFormatter())
    root = logging.getLogger()
    # replacing the handlers keeps a second call from printing every line twice
    root.handlers[:] = [handler]
    root.setLevel(level)
