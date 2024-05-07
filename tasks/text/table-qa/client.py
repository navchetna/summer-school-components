from main import TapexQuery
import pandas as pd

data = {
    "year": [1896, 1900, 1904, 2004, 2008, 2012],
    "city": ["athens", "paris", "st. louis", "athens", "beijing", "london"]
}

table = pd.DataFrame.from_dict(data)

# Initialize TapexQuery
query_engine = TapexQuery()

# tapex accepts uncased input since it is pre-trained on the uncased corpus
query = "select year where city is equal to st. louis"
result = query_engine.query_table(table, query)

print(result[0])
