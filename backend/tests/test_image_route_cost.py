import os
import tempfile
import unittest
from datetime import datetime, timedelta, timezone
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock, patch

os.environ.setdefault("SECRET_KEY", "test-secret")
os.environ.setdefault("DATABASE_URL", "postgresql+asyncpg://test:test@localhost/test")

from core import image_cache
from routers import media


class CacheHitTouchTests(unittest.IsolatedAsyncioTestCase):
    """A poster served from the cache is a read: last_accessed (it only orders
    eviction) is rewritten at most once an hour, not on every request."""

    async def hit(self, last_accessed):
        cached = SimpleNamespace(last_accessed=last_accessed)
        result = MagicMock()
        result.scalar_one_or_none.return_value = cached
        db = MagicMock()
        db.execute = AsyncMock(return_value=result)
        db.commit = AsyncMock()
        with tempfile.TemporaryDirectory() as tmp:
            poster = Path(tmp) / "image_cache" / "w185" / "poster.jpg"
            poster.parent.mkdir(parents=True)
            poster.write_bytes(b"x")
            with patch.object(image_cache.settings, "data_dir", Path(tmp)):
                served = await image_cache.download_and_cache_image(db, "w185", "/poster.jpg")
        self.assertEqual(served, str(poster))
        return db.commit.await_count, cached.last_accessed

    async def test_a_poster_touched_within_the_hour_costs_no_write(self):
        recent = datetime.now(timezone.utc) - timedelta(minutes=5)
        commits, last = await self.hit(recent)
        self.assertEqual(commits, 0)
        self.assertEqual(last, recent)

    async def test_a_stale_poster_is_touched(self):
        stale = datetime.now(timezone.utc) - timedelta(hours=2)
        commits, last = await self.hit(stale)
        self.assertEqual(commits, 1)
        self.assertGreater(last, stale)

    async def test_an_unknown_timestamp_is_touched(self):
        commits, last = await self.hit(None)
        self.assertEqual(commits, 1)
        self.assertIsNotNone(last)

    async def test_a_naive_timestamp_is_read_as_utc(self):
        naive_utc = lambda delta: (datetime.now(timezone.utc) - delta).replace(tzinfo=None)  # noqa: E731
        self.assertEqual((await self.hit(naive_utc(timedelta(minutes=5))))[0], 0)
        self.assertEqual((await self.hit(naive_utc(timedelta(hours=2))))[0], 1)


class ImageSettingsCacheTests(unittest.IsolatedAsyncioTestCase):
    """serve_image reads the two image-cache settings at most every 30 seconds,
    not once per poster."""

    def setUp(self):
        media._image_settings = None

    def tearDown(self):
        media._image_settings = None

    @staticmethod
    def db(row):
        result = MagicMock()
        result.first.return_value = row
        db = MagicMock()
        db.execute = AsyncMock(return_value=result)
        return db

    async def test_forty_posters_read_the_settings_once(self):
        db = self.db((True, 4.0))
        with patch.object(media._time, "monotonic", return_value=1000.0):
            values = [await media._cached_image_settings(db) for _ in range(40)]
        self.assertEqual(db.execute.await_count, 1)
        self.assertTrue(all(v.image_cache_enabled is True and v.image_cache_limit_gb == 4.0 for v in values))

    async def test_a_change_shows_once_the_window_has_passed(self):
        db = self.db((True, None))
        with patch.object(media._time, "monotonic", return_value=1000.0):
            self.assertTrue((await media._cached_image_settings(db)).image_cache_enabled)
        db.execute.return_value.first.return_value = (False, None)
        with patch.object(media._time, "monotonic", return_value=1029.0):
            self.assertTrue((await media._cached_image_settings(db)).image_cache_enabled)
        with patch.object(media._time, "monotonic", return_value=1031.0):
            self.assertFalse((await media._cached_image_settings(db)).image_cache_enabled)
        self.assertEqual(db.execute.await_count, 2)

    async def test_a_missing_settings_row_is_cached_as_none(self):
        db = self.db(None)
        with patch.object(media._time, "monotonic", return_value=5.0):
            self.assertIsNone(await media._cached_image_settings(db))
            self.assertIsNone(await media._cached_image_settings(db))
        self.assertEqual(db.execute.await_count, 1)

    async def test_the_cached_value_carries_what_serve_image_reads(self):
        db = self.db((True, 2.5))
        with patch.object(media._time, "monotonic", return_value=1.0):
            gs = await media._cached_image_settings(db)
        self.assertEqual((gs.image_cache_enabled, gs.image_cache_limit_gb), (True, 2.5))


if __name__ == "__main__":
    unittest.main()
