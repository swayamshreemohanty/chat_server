import os
import zipfile
from flask import Flask, render_template, request, jsonify
from flask_sqlalchemy import SQLAlchemy
from geopy.geocoders import Nominatim
import csv


geolocator = Nominatim(user_agent="chat-address")

app = Flask(__name__)
app.config['SQLALCHEMY_DATABASE_URI'] = 'sqlite:///customer.db'
db = SQLAlchemy(app)

#Chat bot
import nltk
from nltk.stem.lancaster import LancasterStemmer
import numpy
import tflearn
import tensorflow
import random
import json
import pickle
stemmer = LancasterStemmer()

nltk.download('punkt')

with open("intents.json") as file:
    data = json.load(file)

try:
    with open("data.pickle", "rb") as f:
        words, labels, training, output = pickle.load(f)
except:
    words = []
    labels = []
    docs_x = []
    docs_y = []

    for intent in data["intents"]:
        for pattern in intent["patterns"]:
            wrds = nltk.word_tokenize(pattern)
            words.extend(wrds)
            docs_x.append(wrds)
            docs_y.append(intent["tag"])

        if intent["tag"] not in labels:
            labels.append(intent["tag"])

    words = [stemmer.stem(w.lower()) for w in words if w != "?"]
    words = sorted(list(set(words)))

    labels = sorted(labels)

    training = []
    output = []

    out_empty = [0 for _ in range(len(labels))]

    for x, doc in enumerate(docs_x):
        bag = []

        wrds = [stemmer.stem(w.lower()) for w in doc]

        for w in words:
            if w in wrds:
                bag.append(1)
            else:
                bag.append(0)

        output_row = out_empty[:]
        output_row[labels.index(docs_y[x])] = 1

        training.append(bag)
        output.append(output_row)


    training = numpy.array(training)
    output = numpy.array(output)

    with open("data.pickle", "wb") as f:
        pickle.dump((words, labels, training, output), f)

# tensorflow.reset_default_graph()

net = tflearn.input_data(shape=[None, len(training[0])])
net = tflearn.fully_connected(net, 8)
net = tflearn.fully_connected(net, 8)
net = tflearn.fully_connected(net, len(output[0]), activation="softmax")
net = tflearn.regression(net)
model = tflearn.DNN(net)

#
model.fit(training, output, n_epoch=1000, batch_size=8, show_metric=True)
model.save("model.tflearn")

try:
    model.load("model.tflearn")
except:
    model.fit(training, output, n_epoch=1000, batch_size=8, show_metric=True)
    model.save("model.tflearn")
    
    
def bag_of_words(s, words):
    bag = [0 for _ in range(len(words))]

    s_words = nltk.word_tokenize(s)
    s_words = [stemmer.stem(word.lower()) for word in s_words]

    for se in s_words:
        for i, w in enumerate(words):
            if w == se:
                bag[i] = 1
            
    return numpy.array(bag)
#END of chat bot
    

# __tablename__ = 'customer'
class Customer(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    name = db.Column(db.String(80), nullable=False)
    email = db.Column(db.String(120), nullable=False)
    phone = db.Column(db.String(20))
    address = db.Column(db.String(120))
    city = db.Column(db.String(50))
    state = db.Column(db.String(50))
    latitude = db.Column(db.String(50))
    longitude = db.Column(db.String(50))
    zip = db.Column(db.String(10))

@app.route('/customer-details')
def customerDetails():
    return render_template('customer_form.html')

@app.route('/')
def home():
    return "Chat bot server"


def generate_id():
    # Generate a random 6-digit integer between 100000 and 999999
    new_id = random.randint(100000, 999999)

    # Return the new ID
    return new_id


@app.route('/generate_zip', methods=['GET'])
def generate_zip():
    try:
        #Check for DATA folder exist or not
        if not os.path.exists('DATA'):
            return jsonify({'response': "DATA folder not found"}),400

        # Create a ZipFile object
        with zipfile.ZipFile('data.zip', 'w') as zip_obj:
            # Iterate over all the files in the data folder and add them to the zip file
            for foldername,subfolders, filenames in os.walk('DATA'):
                for filename in filenames:
                    file_path = os.path.join(foldername, filename)
                    zip_obj.write(file_path)
        return jsonify({'response': 'Zip file created successfully!'})
        
    except Exception as e:
        return jsonify({'response': f"Error generating zip file: {e}"}),400
        
    
def generateDummyCSV():
    try:
        
        # Create Location directory if it does not exist
        if not os.path.exists('DATA'):
            os.makedirs('DATA')
            
        shipment="shipments.csv"
        optimize="optimize.csv"
        route="route.csv"
        vehicles="vehicles.csv"
        # Check if these files exists
        shipments_file_exists = os.path.isfile(f'DATA/{shipment}')
        optimize_file_exists = os.path.isfile(f'DATA/{optimize}')
        route_matrix_file_exists = os.path.isfile(f'DATA/{route}')
        vehicles_file_exists = os.path.isfile(f'DATA/{vehicles}')
        
        # Write the data to a CSV file
        if not shipments_file_exists:
            with open(f'DATA/shipments.csv', mode='w', newline='') as file:
                csv.writer(file) # create csv writer object
        if not optimize_file_exists:
            with open(f'DATA/{optimize}', mode='w', newline='') as file:
                csv.writer(file) # create csv writer object
        if not route_matrix_file_exists:
            with open(f'DATA/{route}', mode='w', newline='') as file:
                csv.writer(file) # create csv writer object
        if not vehicles_file_exists:
            with open(f'DATA/{vehicles}', mode='w', newline='') as file:
                csv.writer(file) # create csv writer object
        
                
    except Exception as e:
        raise Exception(f"Error creating files to CSV: {e}")  
def generateLocationCSV(customer):
    try:
        # Define the column headers
        fieldnames = ['City', 'Latitude', 'Longitude', 'Is Depot?', 'Earliest Departure', 'Latest Return']
        
        # Create Location directory if it does not exist
        if not os.path.exists('DATA'):
            os.makedirs('DATA')
            
        # Check if the file exists
        file_exists = os.path.isfile('DATA/location.csv')
        
        # Write the data to a CSV file
        with open('DATA/location.csv', mode='a', newline='') as file:
            writer = csv.DictWriter(file, fieldnames=fieldnames)
            # Write header only if the file is newly created
            if not file_exists:
                writer.writeheader()
            writer.writerow({'City': customer.city, 'Latitude': customer.latitude, 'Longitude': customer.longitude})
        generateDummyCSV()
    except Exception as e:
        raise Exception(f"Error writing data to CSV: {e}")  

@app.route('/submit', methods=['POST'])
def submit():
    try:
        id=generate_id()
        name = request.form['name']
        email = request.form['email']
        phone = request.form['phone']
        address = request.form['address']
        city = request.form['city']
        state = request.form['state']
        zip = request.form['zip']
        location=getLocation(address)
        
        customer = Customer(id=id,name=name, email=email, phone=phone, address=address, city=city, state=state, zip=zip,latitude=location.latitude,longitude=location.longitude)
        db.create_all()
        db.session.add(customer)
        db.session.commit()
        
        #Generate location csv file
        generateLocationCSV(customer)
        
        return 'Customer details submitted successfully'
    except Exception as e:
        print(str(e))
        return str(e)
        

@app.route("/chat", methods=["POST"])
def chat():
    # message = request.form["message"]
    message = request.json["message"]
    response = get_response(message)
    return response

def get_response(message):
    inp = message
    results = model.predict([bag_of_words(inp, words)])
    results_index = numpy.argmax(results)
    tag = labels[results_index]

    for tg in data["intents"]:
        if tg['tag'] == tag:
            responses = tg['responses']

    print(random.choice(responses))
    return jsonify({'response': random.choice(responses)})


def getLocation(addressKeyword):
    location = geolocator.geocode(addressKeyword)
    return location


if __name__ == '__main__':
    app.run(host="0.0.0.0",port=8000, debug=True)