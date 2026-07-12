#!/usr/bin/env python3
"""
Migrate JSON-file data to Neon PostgreSQL.

Reads from:
  - data/users.json
  - data/feedback.jsonl
  - data/conversations/<id>.json   (each is a conversation with messages)

Writes into the PostgreSQL database via the repository layer.

Usage:
    python scripts/migrate_json_to_db.py

Requires DATABASE_URL set in .env or environment.
"""

import asyncio
import json
import logging
import os
import sys
import uuid
from datetime import datetime, timezone
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from dotenv import load_dotenv
load_dotenv()

from sqlalchemy.ext.asyncio import AsyncSession
from db.database import init_db, get_session_factory
from db.models import User, Document, Conversation, Message, Feedback

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
logger = logging.getLogger(__name__)

DATA_DIR = Path("data")


async def migrate_users(session: AsyncSession) -> int:
    """Import users from data/users.json."""
    users_path = DATA_DIR / "users.json"
    if not users_path.exists():
        logger.info("No users.json found — skipping.")
        return 0

    with open(users_path, "r", encoding="utf-8") as f:
        raw = json.load(f)

    # Could be a dict keyed by user_id or a list
    if isinstance(raw, dict):
        users_list = list(raw.values())
    elif isinstance(raw, list):
        users_list = raw
    else:
        logger.warning("Unexpected format in users.json — skipping.")
        return 0

    count = 0
    for u in users_list:
        # Check if already exists
        existing = await session.get(User, u.get("id", ""))
        if existing:
            logger.debug("User %s already exists — skipping.", u.get("email"))
            continue

        user = User(
            id=u.get("id", ""),
            email=u.get("email", ""),
            name=u.get("name", ""),
            password_hash=u.get("password_hash"),
            google_id=u.get("google_id"),
            tier=u.get("tier", "free"),
            queries_used=u.get("queries_used", 0),
            queries_limit=u.get("queries_limit", 20),
            documents_uploaded=u.get("documents_uploaded", 0),
            conversations_count=u.get("conversations_count", 0),
            created_at=_parse_dt(u.get("created_at")),
            updated_at=_parse_dt(u.get("updated_at")) or datetime.now(timezone.utc),
        )
        session.add(user)
        count += 1

    await session.commit()
    logger.info("Migrated %d users.", count)
    return count


async def migrate_feedback(session: AsyncSession) -> int:
    """Import feedback from data/feedback.jsonl."""
    fb_path = DATA_DIR / "feedback.jsonl"
    if not fb_path.exists():
        logger.info("No feedback.jsonl found — skipping.")
        return 0

    count = 0
    with open(fb_path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            item = json.loads(line)

            # Map the feedback JSONL field names to DB model fields
            rating = item.get("rating")

            fb = Feedback(
                id=item.get("id") or str(uuid.uuid4()),
                user_id=item.get("user_id"),
                query=item.get("query_text") or item.get("query", ""),
                response=item.get("answer_text") or item.get("response"),
                rating=rating,  # now int or None
                feedback_text=item.get("comment") or item.get("feedback_text"),
                metadata_json=item.get("metadata") or item.get("metadata_json"),
                created_at=_parse_dt(item.get("timestamp") or item.get("created_at")) or datetime.now(timezone.utc),
            )
            session.add(fb)
            count += 1

    await session.commit()
    logger.info("Migrated %d feedback entries.", count)
    return count


async def migrate_conversations(session: AsyncSession) -> int:
    """Import conversations from data/conversations/<user_id>/conversations.json files."""
    conv_dir = DATA_DIR / "conversations"
    if not conv_dir.exists():
        logger.info("No conversations/ directory found — skipping.")
        return 0

    # Each subdirectory is a user_id, contains a conversations.json
    user_dirs = [d for d in conv_dir.iterdir() if d.is_dir()]
    if not user_dirs:
        logger.info("No user directories in conversations/ — skipping.")
        return 0

    count = 0
    for ud in user_dirs:
        conv_file = ud / "conversations.json"
        if not conv_file.exists():
            continue

        with open(conv_file, "r", encoding="utf-8") as f:
            data = json.load(f)

        # data is a dict keyed by conversation_id
        for conv_id, conv_data in data.items():
            existing = await session.get(Conversation, conv_id)
            if existing:
                logger.debug("Conversation %s already exists — skipping.", conv_id)
                continue

            conv = Conversation(
                id=conv_id,
                user_id=conv_data.get("user_id", ud.name),
                title=conv_data.get("title", "Migrated Conversation"),
                created_at=_parse_dt(conv_data.get("created_at")) or datetime.now(timezone.utc),
                updated_at=_parse_dt(conv_data.get("updated_at")) or datetime.now(timezone.utc),
            )
            session.add(conv)
            await session.flush()

            messages = conv_data.get("messages", [])
            for m in messages:
                msg = Message(
                    id=m.get("id", str(uuid.uuid4())),
                    conversation_id=conv_id,
                    role=m.get("role", "user"),
                    content=m.get("content", ""),
                    sources=m.get("sources"),
                    created_at=_parse_dt(m.get("timestamp") or m.get("created_at")) or datetime.now(timezone.utc),
                )
                session.add(msg)

            count += 1

    await session.commit()
    logger.info("Migrated %d conversations with messages.", count)
    return count


def _parse_dt(val) -> datetime | None:
    """Parse a datetime string or return None."""
    if not val:
        return None
    if isinstance(val, datetime):
        return val
    try:
        return datetime.fromisoformat(val.replace("Z", "+00:00"))
    except (ValueError, AttributeError):
        return None


async def main():
    logger.info("Starting JSON → PostgreSQL migration…")
    await init_db()

    factory = get_session_factory()
    async with factory() as session:
        u = await migrate_users(session)
        f = await migrate_feedback(session)
        c = await migrate_conversations(session)

    logger.info("Migration complete: users=%d feedback=%d conversations=%d", u, f, c)


if __name__ == "__main__":
    asyncio.run(main())
