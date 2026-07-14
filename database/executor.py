from sqlalchemy import text
from database.connection import engine

class SQLExecutor:

    def execute(self, sql: str):

        try:
            with engine.connect() as conn:
                result = conn.execute(text(sql))
                rows = result.fetchall()
                columns = result.keys()

                return {
                    "success": True,
                    "columns": list(columns),
                    "rows": [list(row) for row in rows],
                    "error": None,
                }

        except Exception as e:

            return {
                "success": False,
                "columns": [],
                "rows": [],
                "error": str(e),
            }


executor = SQLExecutor()