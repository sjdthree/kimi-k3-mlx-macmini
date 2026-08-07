"""OpenAI-shaped error envelopes.

    {"error": {"message": ..., "type": ..., "param": ..., "code": ...}}

Two rules this module exists to enforce:

1. **Never leak internals.** No stack traces, hostnames, filesystem paths, or scheduler
   detail reaches a client. Unexpected exceptions become a generic 500 carrying only the
   request id; the detail goes to the log under that id.
2. **Never silently ignore a parameter that changes behaviour.** A request asking for
   something we cannot honour (tools, strict JSON schema, vision) gets a structured
   error naming the offending `param`, not a response that quietly ignored it.
"""
from __future__ import annotations

from fastapi import Request
from fastapi.responses import JSONResponse


class APIError(Exception):
    """An error that is safe to show a client verbatim."""

    def __init__(self, status: int, message: str, *, type: str = "invalid_request_error",
                 param: str | None = None, code: str | None = None):
        super().__init__(message)
        self.status = status
        self.message = message
        self.type = type
        self.param = param
        self.code = code

    def envelope(self) -> dict:
        return {"error": {"message": self.message, "type": self.type,
                          "param": self.param, "code": self.code}}


# ------------------------------------------------------------------ constructors

def invalid_request(message: str, param: str | None = None, code: str | None = None):
    return APIError(400, message, type="invalid_request_error", param=param, code=code)


def model_not_found(model: str):
    return APIError(404, f"The model '{model}' does not exist or you do not have access "
                         f"to it.", type="invalid_request_error", param="model",
                    code="model_not_found")


def unsupported_parameter(param: str, detail: str):
    """400 for a parameter we understand but cannot honour.

    Distinct from `unsupported_capability`: this is a *request* problem the caller can
    fix by dropping the parameter.
    """
    return APIError(400, detail, type="invalid_request_error", param=param,
                    code="unsupported_parameter")


def unsupported_capability(param: str, detail: str):
    """400 for a capability this deployment does not implement at all.

    `/v1/models` advertises the same fact in `capabilities`, so a well-behaved client can
    avoid the round trip.
    """
    return APIError(400, detail, type="invalid_request_error", param=param,
                    code="unsupported_capability")


def authentication_error(message: str = "Incorrect API key provided."):
    return APIError(401, message, type="invalid_request_error", code="invalid_api_key")


def permission_error(message: str):
    return APIError(403, message, type="permission_error", code="insufficient_quota")


def rate_limit(message: str, code: str = "rate_limit_exceeded"):
    return APIError(429, message, type="rate_limit_error", code=code)


def overloaded(message: str = "The server is currently processing another request. "
                              "This backend executes one generation at a time."):
    return APIError(429, message, type="rate_limit_error", code="server_overloaded")


def request_too_large(message: str, param: str | None = None):
    return APIError(413, message, type="invalid_request_error", param=param,
                    code="request_too_large")


def internal_error(request_id: str):
    return APIError(500, f"The server had an error processing your request. Include "
                         f"request id {request_id} if you contact support.",
                    type="api_error", code="internal_error")


# --------------------------------------------------------------------- handlers

def install(app):
    from fastapi.exceptions import RequestValidationError
    import logging

    log = logging.getLogger("air_llm.server")

    def _rid(request: Request) -> str:
        return getattr(request.state, "request_id", "-")

    @app.exception_handler(APIError)
    async def _api_error(request: Request, exc: APIError):
        log.info("api_error rid=%s status=%s code=%s param=%s",
                 _rid(request), exc.status, exc.code, exc.param)
        return JSONResponse(exc.envelope(), status_code=exc.status,
                            headers={"X-Request-ID": _rid(request)})

    @app.exception_handler(RequestValidationError)
    async def _validation(request: Request, exc: RequestValidationError):
        # Map pydantic's first error onto OpenAI's single-param shape.
        err = (exc.errors() or [{}])[0]
        loc = [p for p in err.get("loc", []) if p != "body"]
        param = ".".join(str(p) for p in loc) or None
        msg = err.get("msg", "Invalid request.")
        e = invalid_request(f"{msg}" + (f" (parameter: {param})" if param else ""),
                            param=param, code="invalid_value")
        log.info("validation_error rid=%s param=%s", _rid(request), param)
        return JSONResponse(e.envelope(), status_code=400,
                            headers={"X-Request-ID": _rid(request)})

    @app.exception_handler(Exception)
    async def _unhandled(request: Request, exc: Exception):
        rid = _rid(request)
        # Full detail to the log, nothing to the client.
        log.exception("unhandled rid=%s", rid)
        return JSONResponse(internal_error(rid).envelope(), status_code=500,
                            headers={"X-Request-ID": rid})
