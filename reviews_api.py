import os
import asyncpg
from dotenv import load_dotenv

load_dotenv()
DATABASE_URL = os.environ.get("DATABASE_URL")

async def get_product_rating(product_id: int) -> dict:
    """Return average rating and review count for a single product as async."""
    if not DATABASE_URL:
        return {"product_id": product_id, "average_rating": 0.0, "review_count": 0}

    conn = await asyncpg.connect(DATABASE_URL)
    row = await conn.fetchrow(
        "SELECT AVG(rating), COUNT(*) FROM reviews WHERE product_id = $1",
        product_id
    )
    await conn.close()

    avg = round(float(row[0]), 2) if row and row[0] is not None else 0.0
    count = row[1] if row else 0
    return {"product_id": product_id, "average_rating": avg, "review_count": count}

