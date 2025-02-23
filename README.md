# Cosmofoil Text-to-SQL Application

A natural language interface for querying airline data using FastAPI, HTMX, and OpenAI. This application allows users to query flight delays and cancellations data using plain English, which gets converted to SQL and executed against a database.

## Features

- 🔍 Natural language to SQL query conversion
- 🚀 Real-time query execution
- 💻 Interactive web interface with HTMX
- 🛡️ SQL injection prevention
- ⚠️ Comprehensive error handling
- 📊 Flight delays and cancellations data analysis

## Tech Stack

- FastAPI - Web framework
- HTMX - Dynamic UI updates
- OpenAI - Natural language processing
- SQLite - Database
- Pandas - Data processing
- Jinja2 - Template rendering

## Dataset

The application uses the 2015 Flight Delays and Cancellations dataset, which includes:
- Airlines information
- Airport details
- Flight records with delays and cancellations

## Setup

1. Clone the repository:
```bash
git clone https://github.com/yourusername/cosmofoil.git
cd cosmofoil
```

## API Endpoints

### Main Endpoints

#### `GET /`
- Description: Root endpoint, serves the main web interface
- Response: HTML page with the query interface
- Example: `http://localhost:8000/`

#### `POST /query/`
- Description: Processes natural language queries and returns results
- Request Body: `{"natural_language_query": "your query here"}`
- Response: HTML partial with query results
- Example: `http://localhost:8000/query/`

### Utility Endpoints

#### `GET /schema`
- Description: Returns the database schema information
- Response: JSON containing table schemas
- Example: `http://localhost:8000/schema`

#### `POST /generate-sql/`
- Description: Converts natural language to SQL and executes query
- Request Body: `{"natural_language_query": "your query here"}`
- Response: JSON with:
  ```json
  {
    "sql_query": "generated SQL query",
    "results": [{"column": "value"}],
    "column_names": ["column names"],
    "execution_time": 0.123,
    "row_count": 10
  }