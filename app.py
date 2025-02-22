from flask import Flask, request, render_template
from flask import Response
import os
from flask_cors import CORS, cross_origin
from ml_creditcard_defaulter.pipeline.stage_05_prediction import PredictionPipeline
from ml_creditcard_defaulter.config.configuration import ConfigurationManager
os.putenv('LANG', 'en_US.UTF-8')
os.putenv('LC_ALL', 'en_US.UTF-8')

app = Flask(__name__)
#dashboard.bind(app)
CORS(app)

@app.route("/", methods=['GET'])
@cross_origin()
def home():
    return render_template('index.html')

@app.route("/predict", methods=['POST'])
@cross_origin()
def predictRouteClient():
    try:
        config = ConfigurationManager()
        model_prediction_config = config.get_model_predict_config()
        print('started.....1')

        print('started.....2')
        # path = request.json['filepath']
        data_ingestion = PredictionPipeline()
        data_ingestion.main()
        print('started.....3')
        path = os.path.join(model_prediction_config.prediction_output,'prediction.csv')
        print('started.....4', path)
        return Response("Prediction File created at %s!!!" % path)
        # elif request.form is not None:
        #     path = request.form['filepath']

        #     pred_val = pred_validation(path) #object initialization

        #     pred_val.prediction_validation() #calling the prediction_validation function

        #     pred = prediction(path) #object initialization

        #     # predicting for dataset present in database
        #     path = pred.predictionFromModel()
        #     return Response("Prediction File created at %s!!!" % path)

    except ValueError:
        return Response("Error1 Occurred! %s" %ValueError)
    except KeyError:
        return Response("Error2 Occurred! %s" %KeyError)
    except Exception as e:
        return Response("Error3 Occurred! %s" %e)



@app.route("/train", methods=['POST'])
@cross_origin()
def trainRouteClient():
    try:
         os.system("python main.py")

    except ValueError:

        return Response("Error Occurred! %s" % ValueError)

    except KeyError:

        return Response("Error Occurred! %s" % KeyError)

    except Exception as e:

        return Response("Error Occurred! %s" % e)
    return Response("Training successfull!!")

port = int(os.getenv("PORT",5001))
if __name__ == "__main__":
    app.run(port=port,debug=True)
