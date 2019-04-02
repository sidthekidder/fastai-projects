# uvicorn server:app --port 5000 --ssl-keyfile=./example.com+5-key.pem --ssl-certfile=./example.com+5.pem

from starlette.applications import Starlette
from starlette.responses import JSONResponse
from starlette.routing import Router, Mount
from starlette.staticfiles import StaticFiles
from starlette.responses import FileResponse
from binascii import a2b_base64
from io import BytesIO
import os
import uvicorn
from fastai import *
from fastai.vision import *

classes = ['maleposes', 'femaleposes']

learner = load_learner('models')

app = Router(routes=[
    Mount('/static', app=StaticFiles(directory='static')),
])
@app.route('/')
async def homepage(request):
    return FileResponse('static/index.html')


@app.route('/face', methods=["GET","POST"])
async def face(request):
    body = await request.form()
    binary_data = a2b_base64(body['imgBase64'])
    img = open_image(BytesIO(binary_data))
    _,_,losses = learner.predict(img)
    analysis = {
        "predictions": dict(sorted(
            zip(learner.data.classes, map(float, losses)),
            key=lambda p: p[1],
            reverse=True
        ))}
    return JSONResponse(analysis)


if __name__ == '__main__':
    uvicorn.run(app, host='0.0.0.0', port=8000)
