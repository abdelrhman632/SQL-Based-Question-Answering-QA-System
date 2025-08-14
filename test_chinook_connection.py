from langchain.utilities import SQLDatabase
from langchain_community.utilities import SQLDatabase

db = SQLDatabase.from_uri("sqlite:///C:/Users/abdel/Documents/chinook/Chinook.db")

print(db.dialect)
print(db.get_usable_table_names())

result = db.run("SELECT * FROM Track LIMIT 10;")
print("Sample Tracks:\n", result)
