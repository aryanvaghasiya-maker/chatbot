from database.inspector import inspector
from database.executor import executor


print("=" * 60)
print("TABLES")
print("=" * 60)

print(inspector.get_tables())

print()

print("=" * 60)
print("SCHEMA")
print("=" * 60)

schema = inspector.get_schema()

for table, info in schema.items():

    print(f"\nTable: {table}")

    print("Columns:")

    for column in info["columns"]:
        print(f"  - {column['name']} ({column['type']})")

print()

print("=" * 60)
print("SQL TEST")
print("=" * 60)

result = executor.execute(
    """
    SELECT *
    FROM customers
    LIMIT 5;
    """
)

print(result)