import pandas as pd
import sqlite3

# Paths to the CSV files
flights_csv = "app/data/flights.csv"
airlines_csv = "app/data/airlines.csv"
airports_csv = "app/data/airports.csv"

# SQLite database file
db_file = "flight_delays.db"

# Read CSV files into pandas DataFrames
flights_df = pd.read_csv(flights_csv)
airlines_df = pd.read_csv(airlines_csv)
airports_df = pd.read_csv(airports_csv)

# Connect to SQLite database (or create it if it doesn't exist)
conn = sqlite3.connect(db_file)
cursor = conn.cursor()

# Import data into SQLite tables
flights_df.to_sql("flights", conn, if_exists="replace", index=False)
airlines_df.to_sql("airlines", conn, if_exists="replace", index=False)
airports_df.to_sql("airports", conn, if_exists="replace", index=False)

# Commit changes and close the connection
conn.commit()
conn.close()

print("Data imported successfully!")