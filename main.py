from fastapi import FastAPI

app = FastAPI()

@app.get('/')
def rootGet():
    return {'message:': 'retornado da API via GET'}

# if __name__ == '__main__':
#     import uvicorn
#     uvicorn.run(app, host='0.0.0.0', port=8002)
