import sqlite3
import pandas as pd

# Connect to SQLite database
conn = sqlite3.connect('chat_copy.db')
conn.execute("ATTACH DATABASE '/Users/damodarpai/Desktop/chat_copy.db' AS db;")
# Load the messages (filter if necessary)
query = "SELECT text FROM db.message WHERE handle_id != 0 AND is_from_me = 1"
df = pd.read_sql_query(query, conn)

# Clean the text data
df = df.dropna(subset=['text'])
df = df[df['text'] != 'None'] 
df = df[df['text'] != 'Null']
df['text'] = df['text'].str.strip()  # Remove extra whitespace
df['text'] = df['text'].str.lower()  # Normalize case 
df.index = range(1, len(df) + 1)

print(df.iloc[-1]) 

# Save to a text file for training
with open('messages.txt', 'w') as f:
    f.write("\n".join(df['text'].tolist()))
