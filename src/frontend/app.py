import sys
sys.path.append('.')
sys.path.append('./src/backend')
sys.path.append('./src')
sys.path.append('/usr/src')
sys.path.append('/usr/src/backend')
sys.path.append('/usr/')
print(sys.path)

from flask import Flask, request, Response
from src.backend.utils import write_error
from src.backend import predict
from src.backend import utils 
import pandas as pd
import json

app = Flask(__name__)

@app.route('/')
def hello_world():
    return "Hello World!!"

@app.route('/ping')
def pingme():
    try:
        return {
            "message": "App is listening!",
            'status_code':200
            }
    except Exception as e:
        write_error("frontend-app", f"Failed to /ping due to str{e}")

@app.route("/infer", methods=['POST'])
def infer():
    if request.method == "POST":
        try:
            req_data_dict = json.loads(request.data.decode("utf-8"))
            df = pd.DataFrame.from_records(req_data_dict["instances"])
            print(f"Invoked with {df.shape[0]} records")
            predictions_response = predict.infer(df)

            return Response(
                response=predictions_response,
                status=200,
                mimetype="application/json",
            )

        except Exception as e:
            write_error('frontend-app', f"Failed to make prediction because of {e}")
    else:
        write_error("frontend-app", "/infer must receive a POST request")

    return {'status_code': 500}

if __name__ == "__main__":
    app.run(host='0.0.0.0', port=8080, debug=True)