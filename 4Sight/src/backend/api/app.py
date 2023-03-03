#%%
import csv
import re
import time
import io
import os
import random
from subprocess import PIPE, Popen
import subprocess
import time
import psycopg2
from distutils.log import debug
from fileinput import filename
from flask import Flask, abort, make_response, render_template, request, jsonify, session
from flask_sqlalchemy import SQLAlchemy
from flask_migrate import Migrate
import pandas as pd
from sqlalchemy import inspect
from werkzeug.utils import secure_filename
from flask_cors import CORS

from pipelines.pipeline import *
from pipelines.modeling import *
from pipelines.data_import import *
from pipelines.data_preparation import *

app = Flask(__name__)

app.config['SQLALCHEMY_DATABASE_URI'] = "postgresql://postgres:oui@localhost:5432/flask_db"
app.config['SQLALCHEMY_TRACK_MODIFICATIONS'] = True
app.config['SECRET_KEY'] = 'this_is_a_super_secret_key'
db = SQLAlchemy(app)
migrate = Migrate(app, db)
ALLOWED_EXTENSIONS = {'csv'}
cors = CORS(app, resources={r"/*": {"origins": "http://localhost:4200"}})
table_name = ''

global db_config
db_config = {     
    'host': 'localhost',
    'database': 'flask_db',
    'user': 'postgres',
    'password': 'oui'
}

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

def get_model(model_name):
    model = os.path.join('../models', model_name)
    print(model_name, " saved")
    return model

model = get_model("autoencoder.h5")

@app.route('/')
def index():  
    return render_template("index.html") 

# @app.route('/predict',methods=['POST']) 
# def running():
#     # receive the data
#     req = request.get_json(force=True)  
#     data = req['data']
#     # create the response as a dict
#     response = {'response': 'data received, ' + data + '!'} 
#     # put the response in json and return
#     return jsonify(response) 

# # function predict is called at each request  
# @app.route("/predict", methods=["POST"])
# def predict():
#     """Retrieves the table to determine whether or not the machines are failed
#     """
#     print("[+] request received")
#     f = request.files['file']
#     if f and allowed_file(f.filename):
#         # get the data from the request and put ir under the right format
#         req = request.get_json(force=True)
#         filepath = os.path.join('cd./uploaded_data', f.filename)
#         data = pd.read_csv(filepath)
#         #image = decode_request(req)
#         #batch = preprocess(image)
#         prediction = model.predict(data)
#         #top_label = [(i[1],str(i[2])) for i in decode_predictions(prediction)[0][0]]
#         response = {"prediction": prediction}
#         print("[+] results {}".format(response))
            
#     return jsonify(response) # return it as json



@app.route('/upload', methods=['POST'])  
def upload():  
    if request.method == 'POST':
        f = request.files['file']
        if f and allowed_file(f.filename):
            print(f.filename)
            filepath = os.path.join('uploaded_data', f.filename)
            f.save(filepath)
            with open(filepath) as file_obj:
                metadata = db.metadata
                metadata.clear()
                data = csv.reader(file_obj)
                header = next(data)

                # Dynamically create a new SQLAlchemy model class

                table_name = f"{f.filename.rsplit('.', 1)[0]}_{str(int(time.time()))}"
                

                table_dict = {
                    "__tablename__": table_name,
                }
                table_dict.update({col: db.Column(db.String(50), primary_key=True) for col in header})

                model_class = type("Data", (db.Model,), table_dict)
                # Clear the session's state
                db.session.remove()

                # Create the table
                model_class.__table__.drop(db.engine, checkfirst=True)
                model_class.__table__.create(db.engine)

                # Create instances of the new model class and add them to the database session
                for row in data:
                    instance = model_class(**dict(zip(header, row)))
                    db.session.add(instance)
                db.session.commit()

            return jsonify({'success': True, 'message': 'File uploaded successfully'})
        else: 
            return {"error": "Not a CSV file"}

@app.route('/get_data/<table_name>')
def get_data(table_name):
    # Get the table name from the session
    #table_name = session.get('table_name')
    if not table_name:
        return make_response(jsonify({'error': 'Table name not found in session' + str(table_name)}), 400)
    
   # Define the columns of the table dynamically
    metadata = db.MetaData()
    table = db.Table(table_name, metadata, autoload_with=db.engine, extend_existing=True)
    columns = [column.name for column in table.columns]

    # Query the database for the first 20 rows of the table
    query = db.session.query(*[table.columns[column] for column in columns]).limit(20)
    rows = query.all()

    # Convert the rows to a list of dictionaries
    data = []
    for row in rows:
        data.append({columns[i]: row[i] for i in range(len(row))})

    # Return the data as a JSON response
    return jsonify(data)

@app.route('/get_tables')
def get_tables():
    # Get all the table names in the database
    inspector = inspect(db.engine)
    tables = inspector.get_table_names()
    
    # Filter the table names based on the naming convention you specified
    filtered_tables = [table for table in tables if re.match(r'^[a-zA-Z]+_[0-9]{10}$', table)]
    
    # Return the filtered table names as a JSON response
    return jsonify(filtered_tables)

@app.route('/train_and_predict', methods=['POST'])
def predict():
    # Get the request data
    req_data = request.get_json()

    # Extract the table names and database configuration from the request data
    train_table_name = req_data['train_table_name']
    test_table_name = train_table_name + '_test'

    # Call the train pipeline and predict pipeline functions
    train_pipeline(train_table_name, db_config)
    result = predict_pipeline(train_table_name, test_table_name)

    # Return the prediction results as JSON
    return jsonify({'success': True, 'message': 'It Worked Queen !'})

if __name__ == '__main__':  
    app.run(debug=True)



