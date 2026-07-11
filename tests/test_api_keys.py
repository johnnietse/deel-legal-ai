# Tests for API Key Management
import pytest
import sys
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, patch

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from fastapi.testclient import TestClient
from sqlalchemy.ext.asyncio import AsyncSession


class TestAPIKeyGeneration:
    """Tests for API key generation in repository."""

    def test_generate_api_key_value_format(self):
        """Test that generated API key has correct format."""
        from db.repository import generate_api_key_value
        
        plaintext, key_hash = generate_api_key_value()
        
        # Check plaintext format
        assert plaintext.startswith("oj_")
        assert len(plaintext) > 10
        
        # Check hash format (SHA256 hex)
        assert len(key_hash) == 64
        assert all(c in "0123456789abcdef" for c in key_hash)
        
        # Verify hash matches plaintext
        import hashlib
        expected_hash = hashlib.sha256(plaintext.encode()).hexdigest()
        assert key_hash == expected_hash

    def test_generate_api_key_value_uniqueness(self):
        """Test that generated keys are unique."""
        from db.repository import generate_api_key_value
        
        keys = [generate_api_key_value() for _ in range(10)]
        plaintexts = [k[0] for k in keys]
        hashes = [k[1] for k in keys]
        
        assert len(set(plaintexts)) == 10
        assert len(set(hashes)) == 10


class TestAPIKeyRepository:
    """Tests for API key repository methods."""

    @pytest.mark.asyncio
    async def test_create_api_key(self):
        """Test creating an API key in the database."""
        from db.repository import create_api_key, generate_api_key_value
        from db.models import ApiKey
        
        mock_db = AsyncMock(spec=AsyncSession)
        mock_db.add = MagicMock()
        mock_db.commit = AsyncMock()
        mock_db.refresh = AsyncMock()
        
        plaintext, key_hash = generate_api_key_value()
        user_id = "test-user-id"
        name = "Test Key"
        
        result = await create_api_key(mock_db, user_id, name, key_hash)
        
        assert isinstance(result, ApiKey)
        assert result.user_id == user_id
        assert result.name == name
        assert result.key_hash == key_hash
        assert result.id is not None
        mock_db.add.assert_called_once()
        mock_db.commit.assert_awaited_once()
        mock_db.refresh.assert_awaited_once()

    @pytest.mark.asyncio
    async def test_list_api_keys(self):
        """Test listing API keys for a user."""
        from db.repository import list_api_keys
        from db.models import ApiKey
        
        mock_db = AsyncMock(spec=AsyncSession)
        mock_result = MagicMock()
        mock_scalars = MagicMock()
        mock_scalars.all.return_value = [
            ApiKey(id="key-1", user_id="user-1", name="Key 1", key_hash="hash1"),
            ApiKey(id="key-2", user_id="user-1", name="Key 2", key_hash="hash2"),
        ]
        mock_result.scalars.return_value = mock_scalars
        mock_db.execute = AsyncMock(return_value=mock_result)
        
        keys = await list_api_keys(mock_db, "user-1")
        
        assert len(keys) == 2
        assert keys[0].name == "Key 1"
        assert keys[1].name == "Key 2"

    @pytest.mark.asyncio
    async def test_delete_api_key_success(self):
        """Test deleting an existing API key."""
        from db.repository import delete_api_key
        from db.models import ApiKey
        
        mock_db = AsyncMock(spec=AsyncSession)
        mock_result = MagicMock()
        mock_result.scalar_one_or_none.return_value = ApiKey(id="key-1", user_id="user-1")
        mock_db.execute = AsyncMock(return_value=mock_result)
        mock_db.delete = AsyncMock()
        mock_db.commit = AsyncMock()
        
        result = await delete_api_key(mock_db, "key-1")
        
        assert result is True
        mock_db.delete.assert_awaited_once()
        mock_db.commit.assert_awaited_once()

    @pytest.mark.asyncio
    async def test_delete_api_key_not_found(self):
        """Test deleting a non-existent API key."""
        from db.repository import delete_api_key
        
        mock_db = AsyncMock(spec=AsyncSession)
        mock_result = MagicMock()
        mock_result.scalar_one_or_none.return_value = None
        mock_db.execute = AsyncMock(return_value=mock_result)
        
        result = await delete_api_key(mock_db, "non-existent")
        
        assert result is False
        mock_db.delete.assert_not_called()

    @pytest.mark.asyncio
    async def test_get_api_key_by_hash(self):
        """Test looking up API key by hash."""
        from db.repository import get_api_key_by_hash
        from db.models import ApiKey
        
        mock_db = AsyncMock(spec=AsyncSession)
        mock_result = MagicMock()
        mock_result.scalar_one_or_none.return_value = ApiKey(
            id="key-1", user_id="user-1", name="Test", key_hash="abc123"
        )
        mock_db.execute = AsyncMock(return_value=mock_result)
        
        result = await get_api_key_by_hash(mock_db, "abc123")
        
        assert result is not None
        assert result.id == "key-1"
        assert result.key_hash == "abc123"

    @pytest.mark.asyncio
    async def test_get_api_key_by_hash_not_found(self):
        """Test looking up non-existent API key by hash."""
        from db.repository import get_api_key_by_hash
        
        mock_db = AsyncMock(spec=AsyncSession)
        mock_result = MagicMock()
        mock_result.scalar_one_or_none.return_value = None
        mock_db.execute = AsyncMock(return_value=mock_result)
        
        result = await get_api_key_by_hash(mock_db, "nonexistent")
        
        assert result is None

    @pytest.mark.asyncio
    async def test_update_api_key_last_used(self):
        """Test updating last_used_at timestamp."""
        from db.repository import update_api_key_last_used
        
        mock_db = AsyncMock(spec=AsyncSession)
        mock_db.execute = AsyncMock()
        mock_db.commit = AsyncMock()
        
        await update_api_key_last_used(mock_db, "key-1")
        
        mock_db.execute.assert_awaited_once()
        mock_db.commit.assert_awaited_once()


class TestAPIKeyAuth:
    """Tests for API key authentication dependency."""

    @pytest.mark.asyncio
    async def test_get_api_key_user_valid(self):
        """Test successful API key authentication."""
        from api.auth import get_api_key_user, hash_api_key
        from db.models import User, ApiKey
        
        mock_db = AsyncMock(spec=AsyncSession)
        
        # Mock get_api_key_by_hash
        with patch("api.auth.get_api_key_by_hash") as mock_get_key:
            mock_get_key.return_value = ApiKey(
                id="key-1", user_id="user-1", name="Test", key_hash="hash123"
            )
            
            # Mock update_api_key_last_used
            with patch("api.auth.update_api_key_last_used") as mock_update:
                mock_update.return_value = None
                
                # Mock get_user_by_id
                with patch("api.auth.get_user_by_id") as mock_get_user:
                    mock_get_user.return_value = User(
                        id="user-1", email="test@example.com", name="Test User", tier="pro"
                    )
                    
                    result = await get_api_key_user(
                        x_api_key="oj_testkey123",
                        db=mock_db
                    )
                    
                    assert result is not None
                    assert result["user_id"] == "user-1"
                    assert result["email"] == "test@example.com"
                    assert result["tier"] == "pro"
                    assert result["auth_method"] == "api_key"
                    mock_get_key.assert_awaited_once()
                    mock_update.assert_awaited_once_with(mock_db, "key-1")
                    mock_get_user.assert_awaited_once_with(mock_db, "user-1")

    @pytest.mark.asyncio
    async def test_get_api_key_user_invalid_key(self):
        """Test API key authentication with invalid key."""
        from api.auth import get_api_key_user
        
        mock_db = AsyncMock(spec=AsyncSession)
        
        with patch("api.auth.get_api_key_by_hash") as mock_get_key:
            mock_get_key.return_value = None
            
            result = await get_api_key_user(
                x_api_key="oj_invalidkey",
                db=mock_db
            )
            
            assert result is None

    @pytest.mark.asyncio
    async def test_get_api_key_user_no_header(self):
        """Test API key authentication with no header."""
        from api.auth import get_api_key_user
        
        mock_db = AsyncMock(spec=AsyncSession)
        
        result = await get_api_key_user(
            x_api_key=None,
            db=mock_db
        )
        
        assert result is None

    @pytest.mark.asyncio
    async def test_get_api_key_user_user_not_found(self):
        """Test API key authentication when user doesn't exist."""
        from api.auth import get_api_key_user
        from db.models import ApiKey
        
        mock_db = AsyncMock(spec=AsyncSession)
        
        with patch("api.auth.get_api_key_by_hash") as mock_get_key:
            mock_get_key.return_value = ApiKey(
                id="key-1", user_id="user-1", name="Test", key_hash="hash123"
            )
            
            with patch("api.auth.update_api_key_last_used") as mock_update:
                mock_update.return_value = None
                
                with patch("api.auth.get_user_by_id") as mock_get_user:
                    mock_get_user.return_value = None
                    
                    result = await get_api_key_user(
                        x_api_key="oj_testkey123",
                        db=mock_db
                    )
                    
                    assert result is None

    def test_hash_api_key(self):
        """Test hash_api_key helper function."""
        from api.auth import hash_api_key
        import hashlib
        
        key = "oj_testkey123"
        result = hash_api_key(key)
        
        expected = hashlib.sha256(key.encode()).hexdigest()
        assert result == expected
        assert len(result) == 64


class TestAPIKeyEndpoints:
    """Integration tests for API key endpoints."""

    @pytest.fixture
    def client(self):
        """Create test client."""
        from api.main import app
        return TestClient(app)

    @pytest.fixture
    def mock_user(self):
        """Mock authenticated user."""
        return {
            "user_id": "test-user-id",
            "email": "test@example.com",
            "tier": "free",
        }

    def test_get_keys_requires_auth(self, client):
        """Test that GET /api/auth/keys requires authentication."""
        response = client.get("/api/auth/keys")
        assert response.status_code == 401

    def test_create_key_requires_auth(self, client):
        """Test that POST /api/auth/keys requires authentication."""
        response = client.post("/api/auth/keys", json={"name": "Test Key"})
        assert response.status_code == 401

    def test_revoke_key_requires_auth(self, client):
        """Test that DELETE /api/auth/keys/{id} requires authentication."""
        response = client.delete("/api/auth/keys/some-id")
        assert response.status_code == 401

    @pytest.mark.asyncio
    async def test_create_key_endpoint(self, client, mock_user):
        """Test creating an API key via endpoint."""
        from api.auth import get_current_user
        from db.database import get_db
        
        # Override dependencies
        app = client.app
        app.dependency_overrides[get_current_user] = lambda: mock_user
        
        mock_db = AsyncMock(spec=AsyncSession)
        app.dependency_overrides[get_db] = lambda: mock_db
        
        # Mock repository functions
        with patch("api.api_keys.create_api_key") as mock_create:
            from db.models import ApiKey
            from datetime import datetime, timezone
            
            mock_key = ApiKey(
                id="new-key-id",
                user_id="test-user-id",
                name="Test Key",
                key_hash="hash123",
                created_at=datetime.now(timezone.utc),
            )
            mock_create.return_value = mock_key
            
            with patch("api.api_keys.generate_api_key_value") as mock_gen:
                mock_gen.return_value = ("oj_plaintextkey123", "hash123")
                
                response = client.post("/api/auth/keys", params={"name": "Test Key"})
                
                assert response.status_code == 200
                data = response.json()
                assert data["id"] == "new-key-id"
                assert data["name"] == "Test Key"
                assert data["key"] == "oj_plaintextkey123"
                assert "created_at" in data
        
        # Clean up
        app.dependency_overrides.clear()

    @pytest.mark.asyncio
    async def test_list_keys_endpoint(self, client, mock_user):
        """Test listing API keys via endpoint."""
        from api.auth import get_current_user
        from db.database import get_db
        from db.models import ApiKey
        from datetime import datetime, timezone
        
        app = client.app
        app.dependency_overrides[get_current_user] = lambda: mock_user
        
        mock_db = AsyncMock(spec=AsyncSession)
        app.dependency_overrides[get_db] = lambda: mock_db
        
        with patch("api.api_keys.list_api_keys") as mock_list:
            mock_list.return_value = [
                ApiKey(
                    id="key-1",
                    user_id="test-user-id",
                    name="Key 1",
                    key_hash="abcdef1234567890",
                    created_at=datetime.now(timezone.utc),
                    last_used_at=datetime.now(timezone.utc),
                ),
                ApiKey(
                    id="key-2",
                    user_id="test-user-id",
                    name="Key 2",
                    key_hash="fedcba0987654321",
                    created_at=datetime.now(timezone.utc),
                    last_used_at=None,
                ),
            ]
            
            response = client.get("/api/auth/keys")
            
            assert response.status_code == 200
            data = response.json()
            assert "keys" in data
            assert len(data["keys"]) == 2
            assert data["keys"][0]["name"] == "Key 1"
            assert data["keys"][0]["key_preview"] == "abcdef123456..."
            assert data["keys"][1]["key_preview"] == "fedcba098765..."
        
        app.dependency_overrides.clear()

    @pytest.mark.asyncio
    async def test_revoke_key_endpoint(self, client, mock_user):
        """Test revoking an API key via endpoint."""
        from api.auth import get_current_user
        from db.database import get_db
        from db.models import ApiKey
        
        app = client.app
        app.dependency_overrides[get_current_user] = lambda: mock_user
        
        mock_db = AsyncMock(spec=AsyncSession)
        app.dependency_overrides[get_db] = lambda: mock_db
        
        with patch("api.api_keys.delete_api_key") as mock_delete:
            mock_delete.return_value = True
            
            with patch("api.api_keys.select") as mock_select:
                # Mock the ownership check query
                mock_result = MagicMock()
                mock_result.scalar_one_or_none.return_value = ApiKey(
                    id="key-1", user_id="test-user-id", name="Test", key_hash="hash"
                )
                mock_db.execute = AsyncMock(return_value=mock_result)
                
                response = client.delete("/api/auth/keys/key-1")
                
                assert response.status_code == 200
                data = response.json()
                assert data["message"] == "Key revoked"
                mock_delete.assert_awaited_once_with(mock_db, "key-1")
        
        app.dependency_overrides.clear()

    @pytest.mark.asyncio
    async def test_revoke_key_not_found(self, client, mock_user):
        """Test revoking a non-existent key."""
        from api.auth import get_current_user
        from db.database import get_db
        
        app = client.app
        app.dependency_overrides[get_current_user] = lambda: mock_user
        
        mock_db = AsyncMock(spec=AsyncSession)
        app.dependency_overrides[get_db] = lambda: mock_db
        
        with patch("api.api_keys.select") as mock_select:
            mock_result = MagicMock()
            mock_result.scalar_one_or_none.return_value = None
            mock_db.execute = AsyncMock(return_value=mock_result)
            
            response = client.delete("/api/auth/keys/nonexistent")
            
            assert response.status_code == 404
            assert response.json()["detail"] == "Key not found"
        
        app.dependency_overrides.clear()

    @pytest.mark.asyncio
    async def test_revoke_key_wrong_user(self, client, mock_user):
        """Test revoking a key that belongs to another user."""
        from api.auth import get_current_user
        from db.database import get_db
        from db.models import ApiKey
        
        app = client.app
        app.dependency_overrides[get_current_user] = lambda: mock_user
        
        mock_db = AsyncMock(spec=AsyncSession)
        app.dependency_overrides[get_db] = lambda: mock_db
        
        with patch("api.api_keys.select") as mock_select:
            # Key exists but belongs to different user
            mock_result = MagicMock()
            mock_result.scalar_one_or_none.return_value = None  # Ownership check fails
            mock_db.execute = AsyncMock(return_value=mock_result)
            
            response = client.delete("/api/auth/keys/key-1")
            
            assert response.status_code == 404
            assert response.json()["detail"] == "Key not found"
        
        app.dependency_overrides.clear()


if __name__ == "__main__":
    pytest.main([__file__, "-v"])