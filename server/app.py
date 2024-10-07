from flask import Flask, jsonify, send_file, send_from_directory
import mysql.connector
import sqlite3
import os
import traceback
from flask_cors import CORS

from langchain_community.utilities.sql_database import SQLDatabase
from langchain_experimental.sql import SQLDatabaseChain
from langchain_openai import ChatOpenAI
from langchain.prompts import PromptTemplate
import os
from flask import request

from langchain.chains.openai_tools import create_extraction_chain_pydantic
from langchain_core.pydantic_v1 import BaseModel, Field

from langchain.chains import create_sql_query_chain
from operator import itemgetter
from langchain_core.runnables import RunnablePassthrough
from typing import List
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers.openai_tools import PydanticToolsParser
from langchain_experimental.sql import SQLDatabaseSequentialChain

from dotenv import load_dotenv
import os

load_dotenv()
baseurl = os.getenv("DATABASE_URL")




# loading environment variables
load_dotenv()

# declaring flask application
app = Flask(__name__, static_folder="./dist", template_folder='templates')

CORS(app)



# app = Flask(__name__)
app = Flask(__name__, static_folder="./dist", template_folder='templates')
CORS(app)
# MySQL database configuration
HOST = os.getenv("HOST")
PORT = os.getenv("PORT")
DATABASE = os.getenv("DATABASE")
USER = os.getenv("USER")
PASSWORD = os.getenv("PASSWORD")
CHARSET = os.getenv("CHARSET")
COLLATION = os.getenv("COLLATION")


db_config = {
    'host': HOST,
    'port': 3306,
    'database': DATABASE,
    'user': USER,
    'password': PASSWORD,
    'charset': CHARSET,
    'collation': COLLATION
}

OPENAI_KEY = os.getenv("OPENAI_KEY")

openai_key = OPENAI_KEY
llm = ChatOpenAI(model_name='gpt-4', openai_api_key=openai_key, temperature=0.9)
current_path = os.path.dirname(__file__)

dburi = os.getenv("DATABASE_URL")

db = SQLDatabase.from_uri(dburi)


message = []


@app.route('/', defaults={'path': ''})
@app.route('/<path:path>')
def serve(path):
    if path != "" and os.path.exists(app.static_folder + '/' + path):
        return send_from_directory(app.static_folder, path)
    else:
        return send_from_directory('./dist', 'index.html')
    

@app.route('/produit', methods=['GET'])
def get_produit():
    try:
        mysql_connection = mysql.connector.connect(**db_config)
        mysql_cursor = mysql_connection.cursor(dictionary=True)
        mysql_cursor.execute("SELECT * FROM produit")
        rows = mysql_cursor.fetchall()
        mysql_cursor.close()
        mysql_connection.close()

        return jsonify(rows)
    except Exception as e:
        return jsonify({"error": str(e)}), 500


@app.route('/export/sqlite', methods=['GET'])
def export_sqlite():
    try:
        mysql_connection = mysql.connector.connect(**db_config)
        mysql_cursor = mysql_connection.cursor(dictionary=True)
        mysql_cursor.execute("SELECT * FROM produit")
        rows = mysql_cursor.fetchall()
        mysql_cursor.close()
        mysql_connection.close()

        temp_filename = '1.db'
        sqlite_connection = sqlite3.connect(temp_filename)
        sqlite_cursor = sqlite_connection.cursor()

        columns = rows[0].keys()
        columns_definition = ', '.join([f'"{col}" TEXT' for col in columns])
        sqlite_cursor.execute(f'CREATE TABLE IF NOT EXISTS produit ({columns_definition})')

        for row in rows:
            values = tuple(str(value) if value is not None else '' for value in row.values())
            placeholders = ', '.join(['?'] * len(values))
            sqlite_cursor.execute(f'INSERT INTO produit VALUES ({placeholders})', values)

        sqlite_connection.commit()
        sqlite_connection.close()

        return send_file(
            temp_filename,
            as_attachment=True,
            download_name='1.db', 
            mimetype='application/x-sqlite3'
        )
    except Exception as e:
        error_message = str(e)
        stack_trace = traceback.format_exc()
        print("Error:", error_message)
        print("Stack Trace:", stack_trace)
        return jsonify({"error": error_message}), 500
    finally:
        # Clean up temporary files if they exist
        if os.path.exists(temp_filename):
            try:
                os.remove(temp_filename)
            except OSError:
                pass

@app.route('/chat', methods=['POST'])

def chat():
    body = request.get_json() 
    print("response", body["query"])

    query = body['query'] + ". Here,  you are not AI model, you should be a assistant helpful bot, so you have to answer more flexibility , not robotic, like humans.If query is 'hi', 'hello' or 'hey', response as 'Hi, I am a AI assistant. How can I help you."
    print(query)
    query1 = body['query']
    PROMPT = """ 
    Given an input question, first create a syntactically correct only mysql query to run,  
    then look at the results of the query and return the answer. 

    There are some table such as Bbs_user, user, etc. so use only user table, not Bbs_user. 
 
    The question: {question}
    """
    openai_key = OPENAI_KEY

    llm = ChatOpenAI(temperature=0, openai_api_key=openai_key, model_name='gpt-3.5-turbo')

    db_chain = SQLDatabaseSequentialChain.from_llm(llm=llm, db=db, verbose=True,
                                        return_intermediate_steps=True, top_k=1)
    question = query1
    result = db_chain((PROMPT.format(question=question))) 
    print(result['result'])
 
    return {"message": result['result']}


if __name__ == '__main__':
    app.run(host='0.0.0.0', port = 5000)
