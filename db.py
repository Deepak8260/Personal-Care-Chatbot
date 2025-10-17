import os
import psycopg2
from psycopg2 import OperationalError

# Optional: load from .env when present (requirements.txt includes python-dotenv)
try:
    from dotenv import load_dotenv
    load_dotenv()
except Exception:
    # If python-dotenv isn't installed or .env not present, continue — env vars may already be set.
    pass

# Read configuration from environment variables
USER = os.getenv("USER")
PASSWORD = os.getenv("PASSWORD")
HOST = os.getenv("HOST")
PORT = os.getenv("PORT")
DBNAME = os.getenv("DBNAME") 

required = {"user": USER, "password": PASSWORD, "host": HOST, "dbname": DBNAME}
missing = [k for k, v in required.items() if not v]
if missing:
    raise RuntimeError(
        "Missing required database environment variables: " + ", ".join(missing) +
        ". Please set PGUSER/PGPASSWORD/PGHOST/PGDATABASE (or USER/PASSWORD/HOST/DBNAME)"
    )

# Connect to the database (test connection)
try:
    connection = psycopg2.connect(
        user=USER,
        password=PASSWORD,
        host=HOST,
        port=PORT,
        dbname=DBNAME,
        sslmode='require'
    )
    print("Connection successful! Supabase credentials validated.")
    connection.close()

except OperationalError as e:
    # Likely a network/credential/SSL issue
    print(f"Failed to connect (OperationalError): {e}")
    raise
except Exception as e:
    # Fallback for any other exception
    print(f"Failed to connect: {e}")
    raise