# Real-Time Pose Classifier

Trained on 200 male and 200 female poses from google search. The pre-trained model is stored in models/export.pkl. The webcam sends a screenshot every 3 seconds to the server for classification. Feel free to reuse the code for your own classifier.


### Run on localhost:
- `pip install fastai, starlette, starlette[full], uvicorn`
- install `mkcert` to generate a ssl certificate for local https (webRTC 🙄) - follow these steps - https://github.com/FiloSottile/mkcert#installation. (`brew install mkcert` for mac)
- run `mkcert -install` & `mkcert example.com "*.example.com" example.test localhost 127.0.0.1 ::1`
- run the server using `uvicorn server:app --port 5000 --ssl-keyfile=./example.com+5-key.pem --ssl-certfile=./example.com+5.pem`
- The site will run at https://localhost:5000/, 
