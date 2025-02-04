import asyncio
import traceback
import click
import os
from typing import Optional
from pydantic import BaseModel, Field
from starlette.responses import HTMLResponse
from marker.config.parser import ConfigParser
from marker.output import text_from_rendered
import base64
import io
from fastapi import FastAPI, Form, File, UploadFile
from marker.converters.pdf import PdfConverter
from marker.models import create_model_dict
from marker.settings import settings

UPLOAD_DIRECTORY = "./uploads"
os.makedirs(UPLOAD_DIRECTORY, exist_ok=True)

app_data = {"models": None}

def load_models():
    if app_data["models"] is None:
        app_data["models"] = create_model_dict()

def unload_models():
    app_data["models"] = None

async def delayed_unload():
    await asyncio.sleep(60)
    unload_models()

app = FastAPI()

@app.get("/")
async def root():
    return HTMLResponse(
        """
<h1>Marker API</h1>
<ul>
    <li><a href="/docs">API Documentation</a></li>
    <li><a href="/marker">Run marker (post request only)</a></li>
</ul>
"""
    )

class CommonParams(BaseModel):
    filepath: str
    page_range: Optional[str] = None
    languages: Optional[str] = None
    force_ocr: bool = False
    paginate_output: bool = False
    output_format: str = "markdown"

async def _convert_pdf(params: CommonParams):
    assert params.output_format in ["markdown", "json", "html"], "Invalid output format"
    try:
        load_models()
        options = params.model_dump()
        config_parser = ConfigParser(options)
        config_dict = config_parser.generate_config_dict()
        config_dict["pdftext_workers"] = 1
        converter_cls = PdfConverter
        converter = converter_cls(
            config=config_dict,
            artifact_dict=app_data["models"],
            processor_list=config_parser.get_processors(),
            renderer=config_parser.get_renderer()
        )
        rendered = converter(params.filepath)
        text, _, images = text_from_rendered(rendered)
        metadata = rendered.metadata
    except Exception as e:
        traceback.print_exc()
        return {"success": False, "error": str(e)}
    
    encoded = {k: base64.b64encode(io.BytesIO(v.tobytes()).getvalue()).decode(settings.OUTPUT_ENCODING) for k, v in images.items()}
    
    asyncio.create_task(delayed_unload())
    
    return {"format": params.output_format, "output": text, "images": encoded, "metadata": metadata, "success": True}

@app.post("/marker/upload")
async def convert_pdf_upload(
    page_range: Optional[str] = Form(default=None),
    languages: Optional[str] = Form(default=None),
    force_ocr: Optional[bool] = Form(default=False),
    paginate_output: Optional[bool] = Form(default=False),
    output_format: Optional[str] = Form(default="markdown"),
    file: UploadFile = File(..., description="The PDF file to convert.", media_type="application/pdf"),
):
    upload_path = os.path.join(UPLOAD_DIRECTORY, file.filename)
    with open(upload_path, "wb+") as upload_file:
        file_contents = await file.read()
        upload_file.write(file_contents)
    
    params = CommonParams(
        filepath=upload_path,
        page_range=page_range,
        languages=languages,
        force_ocr=force_ocr,
        paginate_output=paginate_output,
        output_format=output_format,
    )
    results = await _convert_pdf(params)
    os.remove(upload_path)
    return results

@click.command()
@click.option("--port", type=int, default=8000, help="Port to run the server on")
@click.option("--host", type=str, default="127.0.0.1", help="Host to run the server on")
def server_cli(port: int, host: str):
    import uvicorn
    uvicorn.run(app, host=host, port=port)
