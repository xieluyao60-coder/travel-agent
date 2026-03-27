from __future__ import annotations

import base64
import hashlib
import os
import time
import xml.etree.ElementTree as ET
from dataclasses import dataclass

from Crypto.Cipher import AES

from app.errors import UserInputError


PKCS7_BLOCK_SIZE = 32


@dataclass
class WeComIncomingMessage:
    to_user_name: str
    from_user_name: str
    msg_type: str
    content: str = ""
    event: str = ""
    agent_id: str = ""


class WeComCrypto:
    def __init__(self, token: str, encoding_aes_key: str, receive_id: str) -> None:
        self.token = token
        self.receive_id = receive_id
        try:
            self._aes_key = base64.b64decode(f"{encoding_aes_key}=")
        except Exception as exc:
            raise UserInputError("WECOM_ENCODING_AES_KEY 非法，无法解码") from exc
        if len(self._aes_key) != 32:
            raise UserInputError("WECOM_ENCODING_AES_KEY 长度非法")
        self._iv = self._aes_key[:16]

    def decrypt(self, encrypted: str) -> str:
        try:
            encrypted_bytes = base64.b64decode(encrypted)
        except Exception as exc:
            raise UserInputError("企业微信加密消息 Base64 解码失败") from exc

        cipher = AES.new(self._aes_key, AES.MODE_CBC, self._iv)
        raw = cipher.decrypt(encrypted_bytes)
        plain = _pkcs7_unpad(raw)
        if len(plain) < 20:
            raise UserInputError("企业微信加密消息结构非法")

        msg_len = int.from_bytes(plain[16:20], byteorder="big")
        msg_start = 20
        msg_end = msg_start + msg_len
        if msg_end > len(plain):
            raise UserInputError("企业微信消息长度字段非法")

        msg_bytes = plain[msg_start:msg_end]
        receive_id = plain[msg_end:].decode("utf-8", errors="ignore")
        if self.receive_id and receive_id != self.receive_id:
            raise UserInputError("企业微信 receive_id 校验失败，请检查 CorpID")

        try:
            return msg_bytes.decode("utf-8")
        except UnicodeDecodeError as exc:
            raise UserInputError("企业微信消息解密后 UTF-8 解码失败") from exc

    def encrypt(self, plain_text: str) -> str:
        msg_bytes = plain_text.encode("utf-8")
        random_bytes = os.urandom(16)
        msg_len = len(msg_bytes).to_bytes(4, byteorder="big")
        payload = random_bytes + msg_len + msg_bytes + self.receive_id.encode("utf-8")
        padded = _pkcs7_pad(payload)
        cipher = AES.new(self._aes_key, AES.MODE_CBC, self._iv)
        encrypted = cipher.encrypt(padded)
        return base64.b64encode(encrypted).decode("utf-8")


def build_wecom_signature(token: str, timestamp: str, nonce: str, payload: str | None = None) -> str:
    parts = [token or "", timestamp or "", nonce or ""]
    if payload is not None:
        parts.append(payload)
    parts.sort()
    joined = "".join(parts)
    return hashlib.sha1(joined.encode("utf-8")).hexdigest()


def verify_wecom_signature(
    token: str,
    timestamp: str | None,
    nonce: str | None,
    signature: str | None,
    payload: str | None = None,
) -> bool:
    if not token:
        return True
    if not signature:
        return False
    expected = build_wecom_signature(token, timestamp or "", nonce or "", payload=payload)
    return expected == signature


def parse_wecom_message(raw_body: bytes) -> WeComIncomingMessage:
    try:
        root = ET.fromstring(raw_body.decode("utf-8"))
    except Exception as exc:
        raise UserInputError("企业微信消息解析失败") from exc

    def read(tag: str) -> str:
        value = root.findtext(tag)
        return value.strip() if isinstance(value, str) else ""

    msg_type = read("MsgType") or "unknown"
    return WeComIncomingMessage(
        to_user_name=read("ToUserName"),
        from_user_name=read("FromUserName"),
        msg_type=msg_type,
        content=read("Content"),
        event=read("Event"),
        agent_id=read("AgentID"),
    )


def extract_encrypt_text(raw_body: bytes) -> str | None:
    try:
        root = ET.fromstring(raw_body.decode("utf-8"))
    except Exception as exc:
        raise UserInputError("企业微信消息解析失败") from exc
    encrypted = root.findtext("Encrypt")
    if not isinstance(encrypted, str):
        return None
    value = encrypted.strip()
    return value or None


def build_text_reply_xml(incoming: WeComIncomingMessage, content: str) -> str:
    return (
        "<xml>"
        f"<ToUserName><![CDATA[{_safe_cdata(incoming.from_user_name)}]]></ToUserName>"
        f"<FromUserName><![CDATA[{_safe_cdata(incoming.to_user_name)}]]></FromUserName>"
        f"<CreateTime>{int(time.time())}</CreateTime>"
        "<MsgType><![CDATA[text]]></MsgType>"
        f"<Content><![CDATA[{_safe_cdata(content)}]]></Content>"
        "</xml>"
    )


def build_encrypted_reply_xml(encrypt: str, signature: str, timestamp: str, nonce: str) -> str:
    return (
        "<xml>"
        f"<Encrypt><![CDATA[{_safe_cdata(encrypt)}]]></Encrypt>"
        f"<MsgSignature><![CDATA[{_safe_cdata(signature)}]]></MsgSignature>"
        f"<TimeStamp>{_safe_cdata(timestamp)}</TimeStamp>"
        f"<Nonce><![CDATA[{_safe_cdata(nonce)}]]></Nonce>"
        "</xml>"
    )


def _safe_cdata(value: str) -> str:
    return (value or "").replace("]]>", "]]]]><![CDATA[>")


def _pkcs7_pad(data: bytes) -> bytes:
    pad_len = PKCS7_BLOCK_SIZE - (len(data) % PKCS7_BLOCK_SIZE)
    if pad_len == 0:
        pad_len = PKCS7_BLOCK_SIZE
    return data + bytes([pad_len]) * pad_len


def _pkcs7_unpad(data: bytes) -> bytes:
    if not data:
        raise UserInputError("企业微信消息解密后为空")
    pad_len = data[-1]
    if pad_len < 1 or pad_len > PKCS7_BLOCK_SIZE:
        raise UserInputError("企业微信消息填充非法")
    if data[-pad_len:] != bytes([pad_len]) * pad_len:
        raise UserInputError("企业微信消息填充校验失败")
    return data[:-pad_len]
