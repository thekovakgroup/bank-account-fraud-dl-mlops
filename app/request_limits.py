
# ============================================================
# MILESTONE 09 - API REQUEST RESOURCE PROTECTION
# Step 09.87 - Prediction Request Body Size Limit
# ============================================================

from starlette.responses import JSONResponse
from starlette.types import (
    ASGIApp,
    Message,
    Receive,
    Scope,
    Send,
)


class PredictionBodyLimitMiddleware:
    """
    Enforce a maximum HTTP body size for POST /predict.

    Requests exceeding the limit return HTTP 413 before
    reaching prediction validation or model inference.

    The middleware never logs or persists request bodies.
    """

    MAX_BODY_BYTES = 256 * 1024

    def __init__(self, app: ASGIApp):
        self.app = app

    async def __call__(
        self,
        scope: Scope,
        receive: Receive,
        send: Send,
    ) -> None:

        # ----------------------------------------------------
        # 1. Apply the limit only to POST /predict.
        # ----------------------------------------------------

        if (
            scope["type"] != "http"
            or scope.get("path") != "/predict"
            or scope.get("method") != "POST"
        ):
            await self.app(scope, receive, send)
            return

        # ----------------------------------------------------
        # 2. Read the body in ASGI message chunks.
        # ----------------------------------------------------

        buffered_messages: list[Message] = []
        total_bytes = 0

        while True:

            message = await receive()

            if message["type"] == "http.disconnect":
                return

            if message["type"] != "http.request":
                continue

            chunk = message.get("body", b"")
            total_bytes += len(chunk)

            # ------------------------------------------------
            # 3. Reject the request as soon as it exceeds
            #    the configured limit.
            # ------------------------------------------------

            if total_bytes > self.MAX_BODY_BYTES:

                response = JSONResponse(
                    status_code=413,
                    content={
                        "detail": "Request body too large."
                    },
                )

                await response(scope, receive, send)
                return

            buffered_messages.append(message)

            if not message.get("more_body", False):
                break

        # ----------------------------------------------------
        # 4. Replay the accepted request to FastAPI.
        # ----------------------------------------------------

        message_index = 0

        async def replay_receive() -> Message:
            nonlocal message_index

            if message_index < len(buffered_messages):
                message = buffered_messages[message_index]
                message_index += 1
                return message

            return await receive()

        # ----------------------------------------------------
        # 5. Continue normal API processing.
        # ----------------------------------------------------

        await self.app(
            scope,
            replay_receive,
            send,
        )
