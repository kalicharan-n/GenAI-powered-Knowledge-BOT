import ast

import numpy as np
from dotenv import load_dotenv
import streamlit as st
from PyPDF2 import PdfReader
from langchain.text_splitter import CharacterTextSplitter
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain_community.llms.openai import OpenAI
from langchain_community.chat_models import ChatOpenAI
from langchain_community.vectorstores import FAISS
from langchain.chains.question_answering import load_qa_chain
from langchain_community.callbacks import get_openai_callback
from langchain.prompts import ChatPromptTemplate
import time
import shutil
from datetime import datetime
import os

from flask import Flask, render_template, request, redirect, url_for, jsonify, json
from flask_sqlalchemy import SQLAlchemy
import sqlite3

app = Flask(__name__)
load_dotenv()
# Dummy admin password (for demonstration purposes)
ADMIN_PASSWORD = "admin123"

# Dummy database of documents (for demonstration purposes)
documents = []

# Dummy data for questions (replace with actual data)
questions_by_id = {
    'user1': [
        {'id': 1, 'text': 'Question 1', 'options': ['Option A', 'Option B', 'Option C'], 'correct_answer': 'Option A'},
        {'id': 2, 'text': 'Question 2', 'options': ['Option X', 'Option Y', 'Option Z'], 'correct_answer': 'Option Y'}
    ],
    'user2': [
        {'id': 3, 'text': 'Question 3', 'options': ['Option A', 'Option B', 'Option C'], 'correct_answer': 'Option B'},
        {'id': 4, 'text': 'Question 4', 'options': ['Option X', 'Option Y', 'Option Z'], 'correct_answer': 'Option X'}
    ]
}

# Function to verify admin password
def verify_password(password):
    return password == ADMIN_PASSWORD

@app.route('/')
def home():
    return render_template('home.html')

@app.route('/index')
def index():
    return render_template('index.html')

@app.route('/evaluation')
def evaluation():
    return render_template('evaluation.html')

@app.route('/admin', methods=['GET', 'POST'])
def admin():
    if request.method == 'POST':
        password = request.form['password']
        if verify_password(password):
            return render_template('admin.html')
        else:
            return "Incorrect password. Please try again."
    return render_template('admin_login.html')

@app.route('/_admin_chat', methods=['GET','POST'])
def admin_chat():
    # Here you can process the user's message and generate a response
    # For demonstration purposes, let's echo the user's message
    connection = sqlite3.connect('admin_panel.db')
    cursor = connection.cursor()
    lbl_nm = "kb_path"
    kb_path = cursor.execute("SELECT value from misc where label= ?", (lbl_nm,), ).fetchall()[0][0]
    connection.close()
    user_question = request.form['user_message']
    embeddings = OpenAIEmbeddings()
    kbase=FAISS.load_local(kb_path,embeddings)
    docs=kbase.similarity_search(user_question)
    llm = OpenAI(model_name="gpt-3.5-turbo")
    chain = load_qa_chain(llm, chain_type="stuff")

    with get_openai_callback() as cb:
        response = chain.run(input_documents=docs, question=user_question)
    return jsonify({'bot_response':response})


@app.route('/add_document', methods=['POST'])
def add_document():
    if 'file' not in request.files:
        return 'No file part'
    pdf = request.files['file']
    connection = sqlite3.connect('admin_panel.db')
    cursor = connection.cursor()
    lbl_nm = "kb_path"
    kb_path = cursor.execute("SELECT value from misc where label= ?", (lbl_nm,), ).fetchall()[0][0]
    connection.close()
    print(pdf.filename)
    if pdf is not None:
        pdf_reader = PdfReader(pdf)
        text = ""
        for page in pdf_reader.pages:
            text += page.extract_text()

        # split into chunks
        text_splitter = CharacterTextSplitter(
            separator="\n",
            chunk_size=1000,
            chunk_overlap=200,
            length_function=len
        )
        chunks = text_splitter.split_text(text)

        # create embeddings
        embeddings = OpenAIEmbeddings()
        print(len(chunks))
        ll=len(chunks)
        current_datetime = datetime.now()
        formatted_datetime = current_datetime.strftime("%d-%m-%Y %H:%M:%S")
        new_knowledge_base = FAISS.from_texts(chunks, embeddings, metadatas=[{'source': pdf.filename,'data_added':formatted_datetime} for i in range(ll)])
        if len(os.listdir(kb_path)) == 0:
            new_knowledge_base.save_local(kb_path)
            v_dict = new_knowledge_base.docstore._dict
            for k in v_dict.keys():
                doc_name = v_dict[k].metadata['source']
                print(doc_name)
        else:
            knowledge_base=FAISS.load_local(kb_path,embeddings)
            knowledge_base.merge_from(new_knowledge_base)
            knowledge_base.save_local(kb_path)
            v_dict=knowledge_base.docstore._dict
            for k in v_dict.keys():
                doc_name = v_dict[k].metadata['source']
                print(doc_name)

        return jsonify({'message': 'Document added successfully'})
    else:
        return jsonify({'message': 'Document not found'})


@app.route('/get_documents', methods=['GET'])
def get_documents():
    connection = sqlite3.connect('admin_panel.db')
    cursor = connection.cursor()
    lbl_nm = "kb_path"
    kb_path = cursor.execute("SELECT value from misc where label= ?", (lbl_nm,), ).fetchall()[0][0]
    connection.close()
    embeddings = OpenAIEmbeddings()
    knowledge_base = FAISS.load_local(kb_path, embeddings)
    v_dict = knowledge_base.docstore._dict
    doc_name=[]
    for k in v_dict.keys():
        doc_name.append(v_dict[k].metadata['source'])
    documents=list(set(doc_name))
    #print(documents)
    return jsonify(documents)
@app.route('/remove_document', methods=['POST'])
def remove_document():
    source_to_delete = request.form['document']
    print(source_to_delete)
    connection = sqlite3.connect('admin_panel.db')
    cursor = connection.cursor()
    lbl_nm = "kb_path"
    kb_path = cursor.execute("SELECT value from misc where label= ?", (lbl_nm,), ).fetchall()[0][0]
    connection.close()
    embeddings = OpenAIEmbeddings()
    knowledge_base = FAISS.load_local(kb_path, embeddings)
    # List to store IDs of documents to delete
    documents_to_delete = []
    documents = knowledge_base.docstore._dict
    # Loop through documents and find IDs to delete
    for doc_id, doc_data in documents.items():
        if doc_data.dict()['metadata']['source'].strip() == source_to_delete:
            documents_to_delete.append(doc_id)
    print("proceding to delete")
    # Now, delete documents from the index
    if documents_to_delete:
        for d_id in documents_to_delete:
            knowledge_base.delete([d_id])

        #shutil.rmtree(kb_path)
        knowledge_base.save_local(kb_path)
        return jsonify({'message': 'Document removed successfully'})
    else:
        print("No documents found with the specified source.")
        return jsonify({'message': 'Document not found'})

@app.route('/update_document', methods=['POST'])
def update_document():
    old_document_name = request.form['old_document_name']
    new_document_name = request.form['new_document_name']
    if old_document_name in documents:
        documents.remove(old_document_name)
        documents.append(new_document_name)
    return redirect(url_for('admin'))


@app.route('/file_path', methods=['GET'])
def get_file_path():
    connection = sqlite3.connect('admin_panel.db')
    cursor = connection.cursor()
    lbl_nm="kb_path"
    file_path = cursor.execute("SELECT value from misc where label= ?", (lbl_nm,), ).fetchall()
    connection.close()
    if file_path[0]:
        return jsonify({'path': file_path[0][0]})
    else:
        return jsonify({'path': ''})

@app.route('/file_path', methods=['POST'])
def set_file_path():
    path = request.form['path']
    lbl_nm="kb_path"
    if path:
        connection = sqlite3.connect('admin_panel.db')
        cursor = connection.cursor()
        cursor.execute("update misc set value= ? where label= ?", (path,lbl_nm,), )
        cursor.execute("commit;")
        connection.close()
    print("added")
    return 'File path updated successfully'

@app.route('/chat_history', methods=['POST'])
def add_chat_history():
    user_id = request.form['user_id']
    user_query = request.form['user_query']
    bot_response = request.form['bot_response']
    document_name = request.form.get('document_name')
    # chat_entry = ChatHistory(user_id=user_id, user_query=user_query, bot_response=bot_response,
    #                          document_name=document_name)
    # db.session.add(chat_entry)
    # db.session.commit()
    return 'Chat history stored successfully'

@app.route('/generate_questions', methods=['POST'])
def generate_questions():
    user_id = request.form['user_id']
    connection = sqlite3.connect('admin_panel.db')
    cursor = connection.cursor()
    lbl_nm = "kb_path"
    kb_path = cursor.execute("SELECT value from misc where label= ?", (lbl_nm,), ).fetchall()[0][0]
    #connection.close()

    PROMPT_TEMPLATE = """
    Answer the question based only on the following context:

    {context}

    ---

    Answer the question based on the above context: {question}
    """
    user_question = 'generate 2 multiple choice questions with answer in the following format. i need the output in json format. Output Template: [{ "question": question goes here,   "choices": list all the choices(a,b,c,d) here in the list,   "answer": correct answer}]'
    test_id = datetime.now().strftime("%Y%m%d%H%M%S")
    embeddings = OpenAIEmbeddings()
    kbase = FAISS.load_local(kb_path, embeddings)
    docs = kbase.similarity_search_with_relevance_scores(user_question,k=3)
    context_text = "\n\n---\n\n".join([doc.page_content for doc, _score in docs])
    prompt_template = ChatPromptTemplate.from_template(PROMPT_TEMPLATE)
    prompt = prompt_template.format(context=context_text, question=user_question)

    print(prompt)
    llm = OpenAI(model_name="gpt-3.5-turbo")
    #chain = load_qa_chain(llm, chain_type="stuff")
    response = llm.predict(prompt)
    # with get_openai_callback() as cb:
    #     response = chain.run(input_documents=docs, question=user_question)

    print(response)

    sources = [doc.metadata.get("source", None) for doc, _score in docs]
    formatted_response = f"Response: {response}\nSources: {sources}"
    print(formatted_response)

    response = response.replace('"questions": ', '')
    questions_data = json.loads(response)
    for i, question_data in enumerate(questions_data, start=1):
        question = question_data["question"]
        choices = question_data["choices"]
        answer = question_data["answer"]
        cursor.execute("insert into quiz values(NULL,?,?,?,?,'','','',?)", (str(user_id), str(question), str(choices), str(answer), str(test_id),), )
        cursor.execute("commit;")
    connection.close()
    return jsonify("completed")

@app.route('/get_questions', methods=['GET'])
def get_questions():
    user_id =  request.args.get('user_id')
    connection = sqlite3.connect('admin_panel.db')
    cursor = connection.cursor()

    result = cursor.execute("SELECT emp_id,question,options,correct_ans,id from quiz where user_resp is not null and emp_id= ?", (user_id,), ).fetchall()
    questions_by_id = {}
    question_id_counter = 1
    print(result)
    for row in result:
        user_id = row[0]
        question_text = row[1]
        options = ast.literal_eval(row[2])  # Parse the string into a Python list
        correct_answer = row[3]
        question_id=row[4]
        # Creating the question dictionary with a unique question ID
        question = {'id': question_id, 'text': question_text, 'options': options,
                    'correct_answer': correct_answer}
        # Adding the question to the user's list of questions
        if user_id not in questions_by_id:
            questions_by_id[user_id] = []
        questions_by_id[user_id].append(question)
    connection.close()
    print(questions_by_id)
    questions = questions_by_id.get(user_id, [])
    return jsonify(questions)

@app.route('/submit_answers', methods=['POST'])
def submit_answers():
    user_id = request.form['user_id']
    user_answers = request.form.get('user_answers')
    if isinstance(user_answers, str):
        user_answers = jsonify(user_answers)
        user_answers = user_answers.get_json()  # Convert JSON string to Python dictionary
    # Verify answers
    connection = sqlite3.connect('admin_panel.db')
    cursor = connection.cursor()

    result = cursor.execute("SELECT emp_id,question,options,correct_ans,id from quiz where user_resp is not null and emp_id= ?",
                            (user_id,), ).fetchall()
    questions_by_id = {}

    for row in result:
        user_id = row[0]
        question_text = row[1]
        options = ast.literal_eval(row[2])  # Parse the string into a Python list
        correct_answer = row[3]
        question_id = row[4]
        # Creating the question dictionary with a unique question ID
        question = {'id': question_id, 'text': question_text, 'options': options,
                    'correct_answer': correct_answer}
        # Adding the question to the user's list of questions
        if user_id not in questions_by_id:
            questions_by_id[user_id] = []
        questions_by_id[user_id].append(question)

    questions = questions_by_id.get(user_id, [])
    result = {}
    user_answers = json.loads(user_answers)
    for question in questions:
        question_id = str(question['id'])
        print(user_answers[question_id]["correct"])
        print(question['correct_answer'])
        if user_answers[question_id]["selected"] == question['correct_answer']:
            result[question_id] = 'correct'
            cursor.execute("update quiz set user_resp=?, score=1 where id= ?", (user_answers[question_id]['selected'],question_id,), )
        else:
            result[question_id] = 'incorrect'
            cursor.execute("update quiz set user_resp=?, score=0 where id= ?",
                           (user_answers[question_id]['selected'], question_id,), )
    cursor.execute("commit;")
    connection.close()
    return jsonify(result)


if __name__ == '__main__':
    app.run(debug=True)
