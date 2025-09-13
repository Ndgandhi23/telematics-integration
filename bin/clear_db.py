#!/usr/bin/env python3
# /bin/clear_db.py
# Utility: wipes the telematics.risk_scores collection.

import os
from pymongo import MongoClient

# ---------------- Config ----------------
MONGO_URI = os.getenv("MONGO_URI", "mongodb://localhost:27017/")
DB_NAME   = os.getenv("MONGO_DB", "telematics")
COLL_NAME = os.getenv("MONGO_COLL", "risk_scores")
# ----------------------------------------

def main():
    client = MongoClient(MONGO_URI)
    col = client[DB_NAME][COLL_NAME]

    count = col.count_documents({})
    if count == 0:
        print(f"[i] Collection {DB_NAME}.{COLL_NAME} already empty.")
        return

    result = col.delete_many({})
    print(f"[✓] Cleared {result.deleted_count} documents from {DB_NAME}.{COLL_NAME}")

if __name__ == "__main__":
    main()
