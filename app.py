from fastapi import FastAPI, HTTPException, Query
from pydantic import BaseModel, validator
from openai import OpenAI
import sqlite3
import os
from dotenv import load_dotenv
from typing import List, Optional, Dict, Any
import re
import pandas as pd

app = FastAPI(
    title="Flight Delays Text-to-SQL API",
    description="An API that converts natural language queries to SQL for querying flight delays data",
    version="1.0.0",
)

load_dotenv()
openai_api_key = os.getenv("OPENAI_API_KEY")


if not openai_api_key:
    raise EnvironmentError(
        "OPENAI_API_KEY is not set. Please set it in your environment."
    )

if openai_api_key:
    os.environ["OPENAI_API_KEY"] = openai_api_key

OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
if not OPENAI_API_KEY:
    raise ValueError("OPENAI_API_KEY is not set. Please check your .env file.")

client = OpenAI(api_key=OPENAI_API_KEY)


# Sample database setup (SQLite for simplicity)
DATABASE_URL = "sqlite:///./test.db"


class QueryRequest(BaseModel):
    natural_language_query: str

    @validator("natural_language_query")
    def validate_query(cls, v):
        if len(v.strip()) < 3:
            raise ValueError("Query must be at least 3 characters long")
        return v


class QueryResponse(BaseModel):
    sql_query: str
    results: List[Dict[str, Any]]
    column_names: List[str]
    execution_time: float
    row_count: int


def validate_sql_query(sql_query: str) -> bool:
    # Basic SQL injection prevention
    dangerous_keywords = ["DROP", "DELETE", "UPDATE", "INSERT", "ALTER", "TRUNCATE"]
    normalized_query = sql_query.upper()
    return not any(keyword in normalized_query for keyword in dangerous_keywords)


from fastapi import FastAPI, HTTPException, Query, Request
from fastapi.templating import Jinja2Templates
from fastapi.staticfiles import StaticFiles
from fastapi.responses import HTMLResponse
from fastapi.middleware.cors import CORSMiddleware

# Add near the top after FastAPI initialization
templates = Jinja2Templates(
    directory="/Users/user/Desktop/NoveltyAI/cosmofoil/app/templates"
)


# Update the root endpoint
@app.get("/", response_class=HTMLResponse)
async def root(request: Request):
    return templates.TemplateResponse("base.html", {"request": request})


# Add HTMX endpoint
@app.post("/query/", response_class=HTMLResponse)
async def query(request: Request, query_request: QueryRequest):
    result = await generate_sql(query_request)
    return templates.TemplateResponse(
        "partials/results.html",
        {
            "request": request,
            "sql": result.sql_query,
            "results": result.results,
            "column_names": result.column_names,
            "execution_time": result.execution_time,
            "row_count": result.row_count,
        },
    )


@app.get("/")
async def root():
    return {
        "message": "Welcome to the Flight Delays Text-to-SQL API",
        "endpoints": {
            "/docs": "API documentation",
            "/generate-sql": "Convert natural language to SQL",
            "/execute-sql": "Execute SQL queries",
            "/schema": "Get database schema information",
        },
    }


@app.get("/schema")
async def get_schema():
    try:
        conn = sqlite3.connect(DB_PATH)
        cursor = conn.cursor()
        cursor.execute(
            "SELECT sql FROM sqlite_master WHERE type='table' AND name='flights'"
        )
        schema = cursor.fetchone()[0]
        return {"schema": schema}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to fetch schema: {str(e)}")
    finally:
        conn.close()


@app.post("/generate-sql/", response_model=QueryResponse)
async def generate_sql(query_request: QueryRequest):
    # Add state mapping dictionary
    state_mapping = {
        "ALABAMA": "AL",
        "ALASKA": "AK",
        "ARIZONA": "AZ",
        "ARKANSAS": "AR",
        "CALIFORNIA": "CA",
        "COLORADO": "CO",
        "CONNECTICUT": "CT",
        "DELAWARE": "DE",
        "FLORIDA": "FL",
        "GEORGIA": "GA",
        "HAWAII": "HI",
        "IDAHO": "ID",
        "ILLINOIS": "IL",
        "INDIANA": "IN",
        "IOWA": "IA",
        "KANSAS": "KS",
        "KENTUCKY": "KY",
        "LOUISIANA": "LA",
        "MAINE": "ME",
        "MARYLAND": "MD",
        "MASSACHUSETTS": "MA",
        "MICHIGAN": "MI",
        "MINNESOTA": "MN",
        "MISSISSIPPI": "MS",
        "MISSOURI": "MO",
        "MONTANA": "MT",
        "NEBRASKA": "NE",
        "NEVADA": "NV",
        "NEW HAMPSHIRE": "NH",
        "NEW JERSEY": "NJ",
        "NEW MEXICO": "NM",
        "NEW YORK": "NY",
        "NORTH CAROLINA": "NC",
        "NORTH DAKOTA": "ND",
        "OHIO": "OH",
        "OKLAHOMA": "OK",
        "OREGON": "OR",
        "PENNSYLVANIA": "PA",
        "RHODE ISLAND": "RI",
        "SOUTH CAROLINA": "SC",
        "SOUTH DAKOTA": "SD",
        "TENNESSEE": "TN",
        "TEXAS": "TX",
        "UTAH": "UT",
        "VERMONT": "VT",
        "VIRGINIA": "VA",
        "WASHINGTON": "WA",
        "WEST VIRGINIA": "WV",
        "WISCONSIN": "WI",
        "WYOMING": "WY",
    }

    # Modify the prompt to include state mapping
    prompt = f"""Given these flight delays database tables:
airlines (
    iata_code TEXT PRIMARY KEY,
    airline TEXT
)

airports (
    iata_code TEXT PRIMARY KEY,
    airport TEXT,
    city TEXT,
    state TEXT,
    country TEXT,
    latitude FLOAT,
    longitude FLOAT
)

flights (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    year INTEGER,
    month INTEGER,
    day INTEGER,
    airline TEXT,
    flight_number TEXT,
    tail_number TEXT,
    origin_airport TEXT,
    destination_airport TEXT,
    scheduled_departure INTEGER,
    departure_time INTEGER,
    departure_delay INTEGER,
    arrival_time INTEGER,
    arrival_delay INTEGER,
    cancelled INTEGER,
    cancellation_reason TEXT,
    air_time INTEGER,
    distance INTEGER,
    FOREIGN KEY (airline) REFERENCES airlines(iata_code),
    FOREIGN KEY (origin_airport) REFERENCES airports(iata_code),
    FOREIGN KEY (destination_airport) REFERENCES airports(iata_code)
)

Translate this query into SQL: {query_request.natural_language_query}

Important: 
1. Return only the raw SQL query on a single line without line breaks
2. Use rowid instead of id for the flights table
3. Make sure to use proper table aliases in joins
4. For state matching:
   - Use two-letter state codes: 'FL' for Florida, 'NY' for New York, 'TX' for Texas
   - State codes are in uppercase
5. Use SELECT * when retrieving all columns from a table
6. Keep the query concise and readable"""

    import time

    start_time = time.time()

    prompt = f"""Given these flight delays database tables:
    airlines (
        iata_code TEXT PRIMARY KEY,
        airline TEXT
    )
    
    airports (
        iata_code TEXT PRIMARY KEY,
        airport TEXT,
        city TEXT,
        state TEXT,
        country TEXT,
        latitude FLOAT,
        longitude FLOAT
    )
    
    flights (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        year INTEGER,
        month INTEGER,
        day INTEGER,
        airline TEXT,
        flight_number TEXT,
        tail_number TEXT,
        origin_airport TEXT,
        destination_airport TEXT,
        scheduled_departure INTEGER,
        departure_time INTEGER,
        departure_delay INTEGER,
        arrival_time INTEGER,
        arrival_delay INTEGER,
        cancelled INTEGER,
        cancellation_reason TEXT,
        air_time INTEGER,
        distance INTEGER,
        FOREIGN KEY (airline) REFERENCES airlines(iata_code),
        FOREIGN KEY (origin_airport) REFERENCES airports(iata_code),
        FOREIGN KEY (destination_airport) REFERENCES airports(iata_code)
    )

    Translate this query into SQL: {query_request.natural_language_query}
    
    Important: 
    1. Return only the raw SQL query on a single line without line breaks
    2. Use rowid instead of id for the flights table
    3. Make sure to use proper table aliases in joins
    4. For state matching:
       - Use two-letter state codes: 'FL' for Florida, 'NY' for New York, 'TX' for Texas
       - State codes are in uppercase
    5. Use simple column names in aggregations (e.g., AVG(longitude) instead of AVG(a.longitude))
    6. Keep the query concise and readable"""

    try:
        response = client.chat.completions.create(
            model="gpt-3.5-turbo",
            messages=[
                {
                    "role": "system",
                    "content": "You are a SQL expert that translates natural language into precise SQL queries. Return only raw SQL without any markdown formatting or explanation. Use proper table aliases and reference columns with their table names. When dealing with airports, prefer using IATA codes or partial airport name matches.",
                },
                {"role": "user", "content": prompt},
            ],
            max_tokens=250,  # Increased for more complex query
            temperature=0.0,
        )

        # Clean any potential markdown formatting and newlines
        sql_query = response.choices[0].message.content.strip()
        sql_query = sql_query.replace("```sql", "").replace("```", "").strip()
        sql_query = sql_query.replace("\n", " ").replace("  ", " ")  # Remove newlines and extra spaces

        if not validate_sql_query(sql_query):
            raise HTTPException(
                status_code=400, detail="Generated SQL query contains unsafe operations"
            )

        # Execute the query
        conn = sqlite3.connect("/Users/user/Desktop/NoveltyAI/cosmofoil/app/flights.db")
        cursor = conn.cursor()
        cursor.execute(sql_query)

        column_names = [description[0] for description in cursor.description]
        results = [dict(zip(column_names, row)) for row in cursor.fetchall()]

        execution_time = time.time() - start_time

        return QueryResponse(
            sql_query=sql_query,
            results=results,
            column_names=column_names,
            execution_time=round(execution_time, 3),
            row_count=len(results),
        )

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
    finally:
        if "conn" in locals():
            conn.close()


# Add at the top of the file with other constants
DB_PATH = "/Users/user/Desktop/NoveltyAI/cosmofoil/app/flights.db"
DATA_DIR = "/Users/user/Desktop/NoveltyAI/cosmofoil/app/data"


# Update the startup function
@app.on_event("startup")
async def startup():
    # Create app directory if it doesn't exist
    os.makedirs(os.path.dirname(DB_PATH), exist_ok=True)

    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()

    # Create tables for all three datasets
    cursor.execute(
        """
    CREATE TABLE IF NOT EXISTS airlines (
        iata_code TEXT PRIMARY KEY,
        airline TEXT
    )"""
    )

    cursor.execute(
        """
    CREATE TABLE IF NOT EXISTS airports (
        iata_code TEXT PRIMARY KEY,
        airport TEXT,
        city TEXT,
        state TEXT,
        country TEXT,
        latitude FLOAT,
        longitude FLOAT
    )"""
    )

    cursor.execute(
        """
    CREATE TABLE IF NOT EXISTS flights (
        id INTEGER PRIMARY KEY,
        year INTEGER,
        month INTEGER,
        day INTEGER,
        airline TEXT,
        flight_number TEXT,
        tail_number TEXT,
        origin_airport TEXT,
        destination_airport TEXT,
        scheduled_departure INTEGER,
        departure_time INTEGER,
        departure_delay INTEGER,
        arrival_time INTEGER,
        arrival_delay INTEGER,
        cancelled INTEGER,
        cancellation_reason TEXT,
        air_time INTEGER,
        distance INTEGER,
        FOREIGN KEY (airline) REFERENCES airlines(iata_code),
        FOREIGN KEY (origin_airport) REFERENCES airports(iata_code),
        FOREIGN KEY (destination_airport) REFERENCES airports(iata_code)
    )"""
    )

    try:
        # Check if data files exist
        airlines_path = os.path.join(DATA_DIR, "airlines.csv")
        airports_path = os.path.join(DATA_DIR, "airports.csv")
        flights_path = os.path.join(DATA_DIR, "flights.csv")

        if not all(
            os.path.exists(p) for p in [airlines_path, airports_path, flights_path]
        ):
            print("Error: One or more data files not found!")
            print(f"Looking in: {DATA_DIR}")
            return

        # Import data with specified dtypes
        airlines_df = pd.read_csv(airlines_path)
        airports_df = pd.read_csv(airports_path)

        # Define column types for flights data
        flights_dtypes = {
            "year": "int32",
            "month": "int32",
            "day": "int32",
            "airline": "str",
            "flight_number": "str",
            "tail_number": "str",
            "origin_airport": "str",
            "destination_airport": "str",
            "scheduled_departure": "int32",
            "departure_time": "float32",
            "departure_delay": "float32",
            "arrival_time": "float32",
            "arrival_delay": "float32",
            "cancelled": "int32",
            "cancellation_reason": "str",
            "air_time": "float32",
            "distance": "float32",
        }

        flights_df = pd.read_csv(flights_path, dtype=flights_dtypes, low_memory=False)

        # Load data with explicit index=False to prevent duplicate columns
        airlines_df.to_sql("airlines", conn, if_exists="replace", index=False)
        airports_df.to_sql("airports", conn, if_exists="replace", index=False)
        flights_df.to_sql("flights", conn, if_exists="replace", index=False)

        # Verify data
        cursor.execute("SELECT COUNT(*) FROM airlines")
        airlines_count = cursor.fetchone()[0]
        cursor.execute("SELECT COUNT(*) FROM airports")
        airports_count = cursor.fetchone()[0]
        cursor.execute("SELECT COUNT(*) FROM flights")
        flights_count = cursor.fetchone()[0]

        print("Data verification:")
        print(f"- Airlines: {airlines_count}")
        print(f"- Airports: {airports_count}")
        print(f"- Flights: {flights_count}")

        # Check some sample data
        cursor.execute("SELECT DISTINCT state FROM airports LIMIT 5")
        sample_states = cursor.fetchall()
        print(f"Sample states in airports: {[s[0] for s in sample_states]}")

    except Exception as e:
        print(f"Error during startup: {str(e)}")
        print(f"Current working directory: {os.getcwd()}")
    finally:
        conn.commit()
        conn.close()


# Add a debug endpoint
@app.get("/debug/data")
async def debug_data():
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()
    try:
        data = {}
        cursor.execute("SELECT COUNT(*), COUNT(DISTINCT state) FROM airports")
        airports_count, states_count = cursor.fetchone()
        cursor.execute("SELECT DISTINCT state FROM airports LIMIT 5")
        sample_states = [s[0] for s in cursor.fetchall()]

        data["airports_count"] = airports_count
        data["unique_states"] = states_count
        data["sample_states"] = sample_states
        return data
    finally:
        conn.close()


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="0.0.0.0", port=8000)
