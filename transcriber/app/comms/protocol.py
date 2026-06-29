"""Backend control-protocol models, the Python mirror of
`src/net/control_protocol.ts`. One JSON object per message; over stdio they are
newline-delimited. Keep the two files in lockstep.
"""
from __future__ import annotations

from typing import Annotated, Literal

from pydantic import BaseModel, Field, TypeAdapter

PROTOCOL_VERSION = 1


# ---- source / result references ------------------------------------------


class PathRef(BaseModel):
    kind: Literal["path"]
    path: str


class UploadRef(BaseModel):
    kind: Literal["upload"]
    uploadId: str


SourceRef = Annotated[PathRef | UploadRef, Field(discriminator="kind")]


class UrlRef(BaseModel):
    kind: Literal["url"]
    url: str


class InlineRef(BaseModel):
    kind: Literal["inline"]
    bytesB64: str


ResultRef = Annotated[PathRef | UrlRef | InlineRef, Field(discriminator="kind")]


class Artifact(BaseModel):
    role: Literal["midi", "stem", "audio"]
    ref: ResultRef


Op = Literal["transcribe", "separate", "alignLyrics"]


# ---- client -> backend ----------------------------------------------------


class RequestArgs(BaseModel):
    audio: SourceRef
    params: dict[str, object] = Field(default_factory=dict)


class RequestMessage(BaseModel):
    v: Literal[1] = PROTOCOL_VERSION
    type: Literal["request"]
    id: str
    op: Op
    args: RequestArgs


class CancelMessage(BaseModel):
    v: Literal[1] = PROTOCOL_VERSION
    type: Literal["cancel"]
    id: str


ClientMessage = Annotated[RequestMessage | CancelMessage, Field(discriminator="type")]
CLIENT_MESSAGE_ADAPTER: TypeAdapter[ClientMessage] = TypeAdapter(ClientMessage)


# ---- backend -> client ----------------------------------------------------


class ProgressMessage(BaseModel):
    v: Literal[1] = PROTOCOL_VERSION
    type: Literal["progress"] = "progress"
    id: str
    stage: str
    frac: float = Field(ge=0.0, le=1.0)
    message: str | None = None


class LogMessage(BaseModel):
    v: Literal[1] = PROTOCOL_VERSION
    type: Literal["log"] = "log"
    id: str | None = None
    level: Literal["debug", "info", "warn", "error"]
    message: str


class ResultMessage(BaseModel):
    v: Literal[1] = PROTOCOL_VERSION
    type: Literal["result"] = "result"
    id: str
    artifacts: list[Artifact]


class ErrorMessage(BaseModel):
    v: Literal[1] = PROTOCOL_VERSION
    type: Literal["error"] = "error"
    id: str
    code: str
    message: str
    recoverable: bool


ServerMessage = Annotated[
    ProgressMessage | LogMessage | ResultMessage | ErrorMessage,
    Field(discriminator="type"),
]
SERVER_MESSAGE_ADAPTER: TypeAdapter[ServerMessage] = TypeAdapter(ServerMessage)
