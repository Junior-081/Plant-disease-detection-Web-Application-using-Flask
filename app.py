from flask import Flask, render_template, request, send_from_directory
import cv2
import pickle
import joblib
import numpy as np
from sklearn.svm import SVC
from keras.models import load_model
import tensorflow as tf
import tensorflow_hub as hub
from tensorflow.keras.preprocessing import image
import os

#load model
model_corn =load_model("AG_Corn_Plant_VGG19 .h5")
model_cotton =load_model("AG_COTTON_plant_VGG19.h5")
model_grape= load_model("AI_Grape.h5")
model_potato= load_model("AI_Potato_VGG19.h5")
model_tomato= load_model("AI_Tomato_model_inception.h5")
model_rice= load_model("padimobile.h5")
# model_manioc= load_model("cropnet_1.h5")
model_manioc = load_model('cropnet_1.h5', custom_objects={'KerasLayer': hub.KerasLayer})
# model_rice = tf.keras.models.load_model('my_model_paddy.h5',custom_objects={'KerasLayer':hub.KerasLayer})
# model =tf.keras.models.load_model('PlantDNet.h5',compile=False)

# model_rice = pickle.load(open('modelePaddy.pkl', 'rb'))

# with open('modelePaddy.pkl', 'rb') as testfile:
#     model_rice=pickle.load(testfile)


COUNT = 0
app = Flask(__name__)
app.config["SEND_FILE_MAX_AGE_DEFAULT"] = 1

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/leaf_detection')
def leaf_detection():
    return render_template('leaf_detection.html')

@app.route('/inputcotton')
def inputcotton():
    return render_template('prediction_cotton.html')


@app.route('/inputcorn')
def inputcorn():
    return render_template('prediction_Corn.html')

@app.route('/inputgrape')
def inputgrape():
    return render_template('prediction_Grape.html')

@app.route('/inputpotato')
def inputpotato():
    return render_template('prediction_potato.html')

@app.route('/inputtomato')
def inputtomato():
    return render_template('prediction_tomato.html')

@app.route('/input_crop_recommendation')
def input_crop_recommendation():
    return render_template('crop_recomdation.html')

@app.route('/input_riz')
def input_riz():
    return render_template('prediction_rice.html')

@app.route('/inputmanioc')
def inputmanioc():
    return render_template('prediction_manioc.html')



@app.route('/data' , methods = ['POST','GET'])
def submit():
    if request.method == 'POST':
        name = request.form['name']
        phone = int(request.form['phone'])
        email = request.form['email']
        subject =request.form['subject']
        message =request.form['message']

        print("Name Of User:",name)
        print("Phone no:",phone)
        print("Email:",email)
        print("subject:",subject)
        print("message:",message)

        return render_template('index.html')
    
    else :
        return render_template('index.html')

# --------------------------------------------------------------------------------coton--------------------------------------------
@app.route('/predictioncotton',methods = ['POST'])
def predictioncotton():
    global COUNT
    img = request.files['image']

    img.save('static/img/{}.jpg'.format(COUNT))
    img_arr = cv2.imread('static/img/{}.jpg'.format(COUNT))

    img_arr = cv2.resize(img_arr, (224, 224))
    img_arr = img_arr / 255.0
    img_arr = img_arr.reshape(1, 224, 224, 3)
    predictions = model_cotton.predict(img_arr)
    prediction=np.argmax(predictions, axis=1)
    print(prediction[0])
    #
    # x = round(prediction[0])
    # # y = round(prediction[0, 1], 2)
    # preds = np.array([x])
    COUNT += 1
    if prediction[0] == 0:
        # cv2.imwrite('static/images/{}.jpg'.format(COUNT), img)
        return render_template('Output.html', data=["diseased cotton leaf", 'green'])
    elif prediction[0] == 1:
        # cv2.imwrite('static/images/{}.jpg'.format(COUNT), img)
        return render_template('Output.html', data=["diseased cotton plant", 'red'])
    elif prediction[0] == 2:
        # cv2.imwrite('static/images/{}.jpg'.format(COUNT), img)
        return render_template('Output.html', data=["fresh cotton leaf", 'red'])
    else:
        # cv2.imwrite('static/images/{}.jpg'.format(COUNT), img)
        return render_template('Output.html', data=["fresh cotton plant", 'red'])

# --------------------------------------------------------------------------------Mais--------------------------------------------
@app.route('/predictioncorn', methods=['POST'])
def predictioncorn():
    global COUNT
    img = request.files['image']

    img.save('static/img/{}.jpg'.format(COUNT))
    img_arr = cv2.imread('static/img/{}.jpg'.format(COUNT))

    img_arr = cv2.resize(img_arr, (224, 224))
    img_arr = img_arr / 255.0
    img_arr = img_arr.reshape(1, 224, 224, 3)
    predictions = model_corn.predict(img_arr)
    prediction = np.argmax(predictions, axis=1)
    pred_probs = round(np.max(predictions)*100, 2)
    print(prediction[0])

    if pred_probs > 55:
        COUNT += 1
        if prediction[0] == 0:
            return render_template('Output.html', data=["Mildiou du maïs avec prob = " + str(pred_probs)])
        elif prediction[0] == 1:
            return render_template('Output.html', data=["Rouille du maïs avec prob = " + str(pred_probs)])
        elif prediction[0] == 2:
            return render_template('Output.html', data=["Taches grises de l'épi du maïs avec prob = " + str(pred_probs)])
        else:
            return render_template('Output.html', data=["En Bonne Santé avec prob = " + str(pred_probs)])
    else:
        return render_template('Output.html', data=["Désolé, nous n'arrivons pas à identifier cet image, prière de séléctionner l'image de Mais"])  

# --------------------------------------------------------------------------------Riz--------------------------------------------  
@app.route('/predictionriz', methods=['POST'])
def predictionriz():
    
    global COUNT
    img = request.files['image']

    img.save('static/img/{}.jpg'.format(COUNT))
    img_arr = cv2.imread('static/img/{}.jpg'.format(COUNT))

    if img_arr is None:
        print("L'image n'est pas correct")
    else:
        img_arr = cv2.resize(img_arr, (224, 224))
        img_arr = img_arr / 255.0
        img_arr = img_arr.reshape(1, 224, 224, 3)
        # with open("modelePaddy.pkl", 'rb') as file:
        #     model_rice = pickle.load(file)
        predictions = model_rice.predict(img_arr)
        prediction = np.argmax(predictions, axis=1)
        pred_probs = round(np.max(predictions)*100, 2)
        print(prediction[0])

        if pred_probs >50:
            COUNT += 1
            if prediction[0] == 0:
                return render_template('Output.html', data=["La brûlure bactérienne des feuils du riz avec probaibilté de " + str(pred_probs)])
            elif prediction[0] == 1:
                return render_template('Output.html', data=["La strie bactérienne des feuilles du riz avec probaibilté de "+ str(pred_probs)])
            elif prediction[0] == 2:
                return render_template('Output.html', data=["Brûlure bactérienne des panicules du riz avec probaibilté de "+ str(pred_probs)])
            elif prediction[0] == 3:
                return render_template('Output.html', data=["Pyriculariose du riz avec probaibilté de "+ str(pred_probs)])
            elif prediction[0] == 4:
                return render_template('Output.html', data=["Tache brune du riz avec probaibilté de "+ str(pred_probs)])
            elif prediction[0] == 5:
                return render_template('Output.html', data=["Cœur mort du riz avec probaibilté de "+ str(pred_probs)])
            elif prediction[0] == 6:
                return render_template('Output.html', data=["Mildiou du riz avec probaibilté de "+ str(pred_probs)])
            elif prediction[0] == 7:
                return render_template('Output.html', data=["Hispa du riz avec probaibilté de "+ str(pred_probs)])
            elif prediction[0] == 8:
                return render_template('Output.html', data=["Riz Normal avec probaibilté de " + str(pred_probs)])
            else:
                return render_template('Output.html', data=["Tungro du riz avec probaibilté de "+ str(pred_probs)])
        else:
            return render_template('Output.html', data=["Désolé, nous n'arrivons pas à identifier cet image, prière de séléctionner l'image du riz"])
    

# --------------------------------------------------------------------------------Manioc--------------------------------------------

disease_names = ['Cassava Bacterial Blight', 'Cassava Brown Streak Disease', 'Cassava Green Mottle', 'Cassava Mosaic Disease', 'Healthy']
uploaded_folder="static/img/"

# # function to process image and predict results
# def process_predict(image_path, model):
#     # read image
#     img = image.load_img(image_path, target_size=(224, 224))
#     # preprocess image
#     img = image.img_to_array(img)
#     # now divide image and expand dims
#     img = np.expand_dims(img, axis=0) / 255
#     # Make prediction
#     pred_probs = model.predict(img)
#     # Get name from prediction
#     pred = disease_names[np.argmax(pred_probs)]
#     pred_probs = round(np.max(pred_probs)*100, 2)
#     return pred, pred_probs

@app.route('/predictionmanioc', methods=['POST'])
def predictionmanioc():
    global COUNT
    img = request.files['image']

    img.save('static/img/{}.jpg'.format(COUNT))
    img_arr = cv2.imread('static/img/{}.jpg'.format(COUNT))

    if img_arr is None:
        print("L'image n'est pas correct")
    else:
        img_arr = cv2.resize(img_arr, (224, 224))
        img_arr = img_arr / 255.0
        img_arr = img_arr.reshape(1, 224, 224, 3)
        predictions = model_manioc.predict(img_arr)
        pred= disease_names[np.argmax(predictions)]
        pred_probs = round(np.max(predictions)*100, 2)
        print(pred[0])
       

    # if request.method == 'POST':
    #      # name inside files and in html input should match
    #     image_file = request.files['image']
    #     if image_file:
    #         # filename = image_file.filename
    #         file_path = os.path.join(uploaded_folder)
    #         image_file.save(file_path+'{}.jpg'.format(COUNT))
    #             # prediction
    #         pred, pred_proba = process_predict(file_path+'{}.jpg'.format(COUNT), model_manioc)
        if pred_probs > 45:
            COUNT += 1
            if pred == 'Cassava Brown Streak Disease':
                return render_template('Output.html', data=["Maladie de la strie brune du manioc avec probaibilté de " + str(pred_probs)])
            elif pred == 'Cassava Green Mottle':
                return render_template('Output.html', data=["Marbrure verte du manioc avec probaibilté de "+ str(pred_probs)])
            elif pred == 'Cassava Bacterial Blight':
                return render_template('Output.html', data=["Bactériose du manioc avec probaibilté de "+ str(pred_probs)])
            elif pred == 'Cassava Mosaic Disease':
                return render_template('Output.html', data=["Maladie mosaïque du manioc avec probaibilté de "+ str(pred_probs)])
            else:
                return render_template('Output.html', data=["Manioc en bonne santé avec probaibilté de "+ str(pred_probs)])
        else:
            return render_template('Output.html', data=["Désolé, nous n'arrivons pas à identifier cet image, prière de séléctionner l'image de feuille de manioc>"])

# --------------------------------------------------------------------------------Vigne--------------------------------------------

@app.route('/predictiongrape',methods = ['POST'])
def predictiongrape():
    global COUNT
    img = request.files['image']

    img.save('static/img/{}.jpg'.format(COUNT))
    img_arr = cv2.imread('static/img/{}.jpg'.format(COUNT))

    img_arr = cv2.resize(img_arr, (224, 224))
    img_arr = img_arr / 255.0
    img_arr = img_arr.reshape(1, 224, 224, 3)
    predictions = model_grape.predict(img_arr)
    prediction=np.argmax(predictions, axis=1)
    print(prediction[0])
    #
    # x = round(prediction[0])
    # # y = round(prediction[0, 1], 2)
    # preds = np.array([x])
    COUNT += 1
    if prediction[0] == 0:
        # cv2.imwrite('static/images/{}.jpg'.format(COUNT), img)
        return render_template('Output.html', data=["Grape___Black_rot'", 'green'])
    elif prediction[0] == 1:
        # cv2.imwrite('static/images/{}.jpg'.format(COUNT), img)
        return render_template('Output.html', data=["Grape___Esca_(Black_Measles)", 'red'])
    elif prediction[0] == 2:
        # cv2.imwrite('static/images/{}.jpg'.format(COUNT), img)
        return render_template('Output.html', data=["Grape___Leaf_blight_(Isariopsis_Leaf_Spot)", 'red'])
    else:
        # cv2.imwrite('static/images/{}.jpg'.format(COUNT), img)
        return render_template('Output.html', data=["Grape___healthy", 'red'])

# --------------------------------------------------------------------------------paume de terre--------------------------------------------
@app.route('/predictionpotato',methods = ['POST'])
def predictionpotato():
    global COUNT
    img = request.files['image']

    img.save('static/img/{}.jpg'.format(COUNT))
    img_arr = cv2.imread('static/img/{}.jpg'.format(COUNT))

    img_arr = cv2.resize(img_arr, (224, 224))
    img_arr = img_arr / 255.0
    img_arr = img_arr.reshape(1, 224, 224, 3)
    predictions = model_potato.predict(img_arr)
    prediction=np.argmax(predictions, axis=1)
    pred_probs = round(np.max(predictions)*100, 2)
    print(prediction[0])
    #
    # x = round(prediction[0])
    # # y = round(prediction[0, 1], 2)
    # preds = np.array([x])
    if pred_probs >50:
        COUNT += 1
        if prediction[0] == 0:

            return render_template('Output.html', data=["Brûlure précoce de la pomme de terre", 'red'])
        elif prediction[0] == 1:
            # cv2.imwrite('static/images/{}.jpg'.format(COUNT), img)
            return render_template('Output.html', data=["Mildiou de la pomme de terre", 'red'])

        else:
            # cv2.imwrite('static/images/{}.jpg'.format(COUNT), img)
            return render_template('Output.html', data=["Pommes de terre saines", 'red'])
    else:
        return render_template('Output.html', data=["Désolé, nous n'arrivons pas à identifier cet image. <br> Prière de séléctionner une image de feuille de la pomme de terre"])
# --------------------------------------------------------------------------------tomate--------------------------------------------
@app.route('/predictiontomato', methods=['POST'])
def predictiontomato():
    global COUNT
    img = request.files['image']

    img.save('static/img/{}.jpg'.format(COUNT))
    img_arr = cv2.imread('static/img/{}.jpg'.format(COUNT))

    img_arr = cv2.resize(img_arr, (224, 224))
    img_arr = img_arr / 255.0
    img_arr = img_arr.reshape(1, 224, 224, 3)
    predictions = model_tomato.predict(img_arr)
    prediction = np.argmax(predictions, axis=1)
    pred_probs = round(np.max(predictions)*100, 2)
    print(prediction[0])
    if pred_probs >45:
        COUNT += 1
        if prediction[0] == 0:
            return render_template('Output.html', data=["Tache bactérienne de tomate"])
        elif prediction[0] == 1:
            return render_template('Output.html', data=["Lueur précoce de tomate"])
        elif prediction[0] == 2:
            return render_template('Output.html', data=["Feu de paille de tomate"])
        elif prediction[0] == 3:
            return render_template('Output.html', data=["Moule à feuilles de tomate"])
        elif prediction[0] == 4:
            return render_template('Output.html', data=["Septoriose des feuilles de tomate"])
        elif prediction[0] == 5:
            return render_template('Output.html', data=[" Tétranyque à deux points Tetranychus urticae de Tomate"])
        elif prediction[0] == 6:
            return render_template('Output.html', data=["Point cible de tomate"])
        elif prediction[0] == 7:
            return render_template('Output.html', data=["Virus de la feuille jaune de la tomate"])
        elif prediction[0] == 8:
            return render_template('Output.html', data=["Virus de la mosaïque de la tomate"])
        else:
            return render_template('Output.html', data=["En bonne santé"])
    else:
        return render_template('Output.html', data=["Nous n'arrivons pas à identifier votre image"])

# --------------------------------------------------------------------------------recommandation--------------------------------------------

@app.route('/crop_recommendation' , methods = ['POST','GET'])
def crop_recommendation():
    if request.method == 'POST':
        Nitrogen = float(request.form['Nitrogen'])
        Phosphorus = float(request.form['Phosphorus'])
        Potassium = float(request.form['Potassium'])
        temperature =float(request.form['temperature'])
        humidity =float(request.form['humidity'])
        rainfall =float(request.form['rainfall'])
        ph =float(request.form['ph'])
        # State =request.form['State']
        print(Nitrogen,Phosphorus,Potassium,temperature,humidity,rainfall,ph)

        # Load the Model back from file
        with open("Crop_Recomandation_RF.pkl", 'rb') as file:
            Pickled_RF_Model = pickle.load(file)
        result = Pickled_RF_Model.predict([[Nitrogen,Phosphorus,Potassium,temperature,humidity,ph,rainfall]])
        if result[0] == 20:
            return render_template('crop_recomdation.html', data=["Riz",'green'])
        elif result[0] == 11:
            return render_template('crop_recomdation.html', data=["Maïs",'green'])
        elif result[0] == 3:
            return render_template('crop_recomdation.html', data=["Pois chiche",'green'])
        elif result[0] == 9:
            return render_template('crop_recomdation.html', data=["Haricots rouges",'green'])
        elif result[0] == 18:
            return render_template('crop_recomdation.html', data=["Pois d'Angole",'green'])
        elif result[0] == 13:
            return render_template('crop_recomdation.html', data=["Haricots mites",'green'])
        elif result[0] == 14:
            return render_template('crop_recomdation.html', data=["Haricot",'green'])
        elif result[0] == 2:
            return render_template('crop_recomdation.html', data=["Amarante",'green'])
        elif result[0] == 10:
            return render_template('crop_recomdation.html', data=["Lentille",'green'])
        elif result[0] == 19:
            return render_template('crop_recomdation.html', data=["Manioc",'green'])
        elif result[0] == 1:
            return render_template('crop_recomdation.html', data=["Banane",'green'])
        elif result[0] == 12:
            return render_template('crop_recomdation.html', data=["Mangue",'green'])
        elif result[0] == 7:
            return render_template('crop_recomdation.html', data=["Vigne",'green'])
        elif result[0] == 21:
            return render_template('crop_recomdation.html', data=["Pastèque",'green'])
        elif result[0] == 15:
            return render_template('crop_recomdation.html', data=["Melon d'amour",'green'])
        elif result[0] == 0:
            return render_template('crop_recomdation.html', data=["Pomme",'green'])
        elif result[0] == 16:
            return render_template('crop_recomdation.html', data=["Orange",'green'])
        elif result[0] == 17:
            return render_template('crop_recomdation.html', data=["Papayer",'green'])
        elif result[0] == 4:
            return render_template('crop_recomdation.html', data=["Cocotier",'green'])
        elif result[0] == 6:
            return render_template('crop_recomdation.html', data=["Coton",'green'])
        elif result[0] == 8:
            return render_template('crop_recomdation.html', data=["ute",'green'])

        else:
            return render_template('crop_recomdation.html', data=['Cafetier','green'])


    else :
        return render_template('crop_recomdation.html')


@app.route('/load_img')
def load_img():
    global COUNT
    return send_from_directory('static/img', "{}.jpg".format(COUNT-1))


if __name__ == '__main__':
    app.run(debug=True)

