from motor.motor_asyncio import AsyncIOMotorClient
from app.core.config import settings

class DatabaseManager:
    def __init__(self):
        self.client: AsyncIOMotorClient = None
        self.db = None

    async def connect_to_mongo(self):
        """Establish connection to MongoDB."""
        self.client = AsyncIOMotorClient(settings.MONGO_URI)
        self.db = self.client[settings.MONGO_DB_NAME]
        print(f"Connected to MongoDB: {settings.MONGO_DB_NAME} ✅")

    async def close_mongo_connection(self):
        """Close connection to MongoDB."""
        if self.client:
            self.client.close()
            print("Closed MongoDB connection 🛑")

    def get_database(self):
        """Return the database instance."""
        return self.db

db_manager = DatabaseManager()

async def get_database():
    """Dependency to get the database instance."""
    return db_manager.get_database()
