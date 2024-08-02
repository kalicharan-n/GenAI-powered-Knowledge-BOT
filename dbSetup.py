import sqlite3

def create_tables():
    # Connect to SQLite database
    conn = sqlite3.connect('admin_panel.db')
    cursor = conn.cursor()

    # Create FilePath table
    cursor.execute('''CREATE TABLE IF NOT EXISTS "Misc" ( "id" INTEGER, "label" TEXT NOT NULL, "value" TEXT NOT NULL, PRIMARY KEY("id") )
                    )''')

    # Create ChatHistory table
    cursor.execute('''CREATE TABLE IF NOT EXISTS ChatHistory ( id INTEGER PRIMARY KEY, time TIMESTAMP DEFAULT CURRENT_TIMESTAMP, user_id TEXT NOT NULL, user_query TEXT NOT NULL, bot_response TEXT NOT NULL, document_name TEXT )
                    )''')

    # Create Misc table
    cursor.execute('''CREATE TABLE if not exists "quiz" ( "id" INTEGER, "emp_id" TEXT NOT NULL, "question" TEXT, "options" TEXT, "correct_ans" TEXT, "user_resp" TEXT, "score" NUMERIC, "date_created" TEXT, "test_id" INTEGER, PRIMARY KEY("id") )
                    )''')


    cursor.execute("""
    INSERT INTO "main"."Misc" ("id", "label", "value") VALUES ('1', 'kb_path', 'C:\charan\kbase');
    INSERT INTO "main"."Misc" ("id", "label", "value") VALUES ('2', 'admin_pwd', '');
    """)

    cursor.execute("""
    """)
    # Commit changes and close connection
    conn.commit()
    conn.close()

if __name__ == '__main__':
    create_tables()
    print("Database and tables created successfully.")
