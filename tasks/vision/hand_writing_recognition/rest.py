from fastapi import FastAPI, UploadFile
from fastapi.responses import HTMLResponse
from hand_writing_recognition import transcribe
import uuid

app = FastAPI()


@app.post("/uploadfiles/")
async def create_upload_files(files: list[UploadFile]):
    content = await files[0].read()
    file_name = f"/tmp/{uuid.uuid4().__str__().replace('-', '_')}"
    f = open(file_name, "xb")
    f.write(content)
    print(transcribe(file_name))

@app.get("/")
async def main():
    content = """
<body>
<form action="/uploadfiles/" enctype="multipart/form-data" method="post">
<input name="files" type="file" multiple>
<input type="submit">
</form>
</body>
    """
    return HTMLResponse(content=content)
