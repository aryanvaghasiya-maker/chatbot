import sqlite3
# Updated import for modern LangChain standards
from langchain_core.tools import tool 
from langchain_openai import ChatOpenAI
from dotenv import load_dotenv

load_dotenv()

# Setup LLM for the query checker
llm = ChatOpenAI(
    model="gpt-5-nano",
    temperature=0,
)

@tool
def sql_db_list_tables() -> str:
    """Input is an empty string, output is a comma-separated list of tables in the database. 
    Always run this tool first to discover available tables."""
    try:
        con = sqlite3.connect("company.db")
        cursor = con.cursor()
        cursor.execute("SELECT name FROM sqlite_master WHERE type='table';")
        tables = [row[0] for row in cursor.fetchall() if not row[0].startswith("sqlite_")]
        return ", ".join(tables) if tables else "No tables found in the database."
    except Exception as e:
        return f"Error listing tables: {e}"
    finally:
        con.close()

@tool
def sql_db_schema(table_names: str) -> str:
    """Input to this tool is a comma-separated list of tables, output is the schema and sample rows.
    ONLY input table names that you verified exist using sql_db_list_tables!
    Example Input: employees, departments"""
    try:
        con = sqlite3.connect("company.db")
        cursor = con.cursor()
        
        # Verify tables exist
        cursor.execute("SELECT name FROM sqlite_master WHERE type='table';")
        valid_tables = {row[0] for row in cursor.fetchall() if not row[0].startswith("sqlite_")}
        
        results = []
        for table in table_names.split(","):
            table = table.strip()
            if table not in valid_tables:
                results.append(f"Error: table_names {{{table!r}}} not found in database.")
                continue
                
            # Fetch Schema
            cursor.execute("SELECT sql FROM sqlite_master WHERE type='table' AND name=?;", (table,))
            schema_row = cursor.fetchone()
            if schema_row:
                results.append(f"--- Schema for table: {table} ---\n{schema_row[0]}")
            
            # Fetch Sample Rows
            try:
                quoted_table = '"' + table.replace('"', '""') + '"'
                cursor.execute(f"SELECT * FROM {quoted_table} LIMIT 3;")
                rows = cursor.fetchall()
                if rows:
                    col_names = [description[0] for description in cursor.description]
                    # Format as a clean markdown table for the LLM to parse easily
                    headers = " | ".join(col_names)
                    separator = " | ".join(["---"] * len(col_names))
                    data_rows = "\n".join(" | ".join(str(x) for x in row) for row in rows)
                    
                    results.append(
                        f"/*\n3 sample rows from {table} table:\n{headers}\n{separator}\n{data_rows}\n*/"
                    )
            except Exception as e:
                results.append(f"Error fetching sample rows for {table}: {e}")
                
        return "\n\n".join(results)
    except Exception as e:
        return f"Database error: {e}"
    finally:
        con.close()

@tool
def sql_db_query(query: str) -> str:
    """Input to this tool is a detailed and correct SQL query, output is a result from the database.
    If an error is returned, rewrite the query, check the query with sql_db_query_checker, and try again."""
    try:
        con = sqlite3.connect("company.db")
        cursor = con.cursor()
        cursor.execute(query)
        res = cursor.fetchall()
        return str(res) if res else "Query executed successfully. No rows returned."
    except Exception as e:
        return f"Error: {e}"
    finally:
        con.close()

@tool
def sql_db_query_checker(query: str) -> str:
    """Use this tool to double check if your query is correct before executing it.
    Always use this tool before executing a query with sql_db_query!"""
    trigger_prompt = """{query} 
    Double check the sqlite query above for common mistakes, including:
    - Using NOT IN with NULL values
    - Using UNION when UNION ALL should have been used
    - Using BETWEEN for exclusive ranges
    - Data type mismatch in predicates
    - Properly quoting identifiers
    - Using the correct number of arguments for functions
    - Casting to the correct data type
    - Using the proper columns for joins
    If there are any of the above mistakes, rewrite the query. 
    If there are no mistakes, just reproduce the original query. 
    Output the final SQL query only. Do not include markdown formatting or backticks.
    SQL Query: """.format(query=query)
    
    response = llm.invoke(trigger_prompt)
    # FIX: Changed .text.strip() to .content.strip() for LangChain LLM compatibility
    return response.content.strip()

# Register and verify tools
tools = [sql_db_list_tables, sql_db_schema, sql_db_query, sql_db_query_checker]
for t in tools:
    print(f"{t.name}: {t.description}\n")
