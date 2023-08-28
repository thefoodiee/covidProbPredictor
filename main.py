from flask import Flask, render_template, request, redirect #for creating webserver
import pickle #for taking python value from dump file
import csv

app = Flask(__name__) #initalising webserver
file = open('model.pkl', 'rb') #opening the python dump file
clf = pickle.load(file) #taking the value from the dump file
file.close #closing dump file
global inputFeatures
@app.route('/home', methods=["GET","POST"])
def home():
    
    if request.method == "POST":
        
        myDict = request.form #requesting data from html form
        fever = int(myDict['fever']) #taking fever parameter from form
        age = int(myDict['age'])#taking age parameter from form
        diffBreath = int(myDict['diffBreath'])#taking diffbreath parameter from form
        bodyPain = int(myDict['bodyPain'])#taking bodypain parameter from form
        runnyNose = int(myDict['runnyNose'])#taking runnynose parameter from form
        
        inputFeatures = [fever, bodyPain, age, runnyNose, diffBreath] #putting all parameters into array
        print(inputFeatures)
        infProb = clf.predict_proba([inputFeatures])[0][1]#predicting probability using inbuilt function "predictproba"
        print(infProb)#printing probability on cosole
        return redirect(location='show',code=301)
        # return render_template('show.html', inf=round(infProb*100))#returning probability to html file
    return render_template('index.html')#loading index.html with input values
@app.route('/', methods=["GET","POST"])
def home1():
    if request.method == "POST":
        
        myDict = request.form 
        fever = int(myDict['fever']) 
        age = int(myDict['age'])
        diffBreath = int(myDict['diffBreath'])
        bodyPain = int(myDict['bodyPain'])
        runnyNose = int(myDict['runnyNose'])
        global inputFeatures
        inputFeatures = [fever, bodyPain, age, runnyNose, diffBreath] 
        
        print(inputFeatures)
        global infProb
        infProb = clf.predict_proba([inputFeatures])[0][1]
        # return infProb
        print(infProb)
        return redirect(location='show',code=301)
        # return render_template('show.html', inf=round(infProb*100))
    return render_template('index.html')


@app.route('/show', methods=["GET","POST"])
def showresult():
    return render_template('show.html', inf=round(infProb*100))


@app.route("/chat")

def chat():
    return render_template('chat.html')
if __name__ == "__main__":
    app.run(debug=True)#running the webapp
