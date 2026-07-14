from sqlalchemy import inspect

from database.connection import engine


class DatabaseInspector:

    def __init__(self):
        self.inspector = inspect(engine)

    def get_tables(self):
        return self.inspector.get_table_names()

    def get_columns(self, table_name):

        columns = self.inspector.get_columns(table_name)

        return [
            {
                "name": column["name"],
                "type": str(column["type"])
            }
            for column in columns
        ]

    def get_primary_keys(self, table_name):

        pk = self.inspector.get_pk_constraint(table_name)

        return pk.get("constrained_columns", [])

    def get_foreign_keys(self, table_name):

        return self.inspector.get_foreign_keys(table_name)

    def get_schema(self):

        schema = {}

        for table in self.get_tables():

            schema[table] = {
                "columns": self.get_columns(table),
                "primary_keys": self.get_primary_keys(table),
                "foreign_keys": self.get_foreign_keys(table),
            }

        return schema


inspector = DatabaseInspector()