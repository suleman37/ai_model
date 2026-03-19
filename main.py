import os

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.routing import APIRoute

from FASTAPI import app as fastapi_module
from fastapi_module_2 import app as fast_module_2
from API_updated.API import app as api_updated


MODULES = [
    {
        "tag": "FASTAPI",
        "prefix": "/fastapi",
        "app": fastapi_module.app,
        "module": fastapi_module,
        "description": "Base ear segmentation and measurement APIs.",
    },
    {
        "tag": "fast_module_2",
        "prefix": "/fast_module_2",
        "app": fast_module_2.app,
        "module": fast_module_2,
        "description": "Landmark mapping and mirror-measure workflow APIs.",
    },
    {
        "tag": "API_updated",
        "prefix": "/api_updated",
        "app": api_updated.app,
        "module": api_updated,
        "description": "Updated landmark validation and live frame guidance APIs.",
    },
]


app = FastAPI(
    title="AI Ear Unified API",
    description="Single Swagger UI for FASTAPI, fast_module_2, and API_updated modules.",
    version="1.0.0",
    openapi_tags=[
        {
            "name": module["tag"],
            "description": module["description"],
        }
        for module in MODULES
    ],
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
    allow_credentials=False,
)


def include_prefixed_routes(source_app: FastAPI, prefix: str, tag: str) -> None:
    for route in source_app.routes:
        if not isinstance(route, APIRoute) or not route.include_in_schema:
            continue

        app.add_api_route(
            path=f"{prefix}{route.path}",
            endpoint=route.endpoint,
            methods=route.methods,
            tags=[tag],
            name=route.name,
            summary=route.summary,
            description=route.description,
            response_description=route.response_description,
            deprecated=route.deprecated,
            operation_id=route.operation_id,
            include_in_schema=True,
            responses=route.responses,
        )


for module in MODULES:
    include_prefixed_routes(module["app"], module["prefix"], module["tag"])


def get_module_status(module: dict) -> dict:
    module_ref = module["module"]
    return {
        "prefix": module["prefix"],
        "model_loaded": getattr(module_ref, "model", None) is not None,
        "loaded_model_path": getattr(module_ref, "loaded_model_path", None),
        "model_error": getattr(module_ref, "model_load_error", None),
    }


@app.get("/", tags=["FASTAPI"])
async def root():
    return {
        "message": "AI Ear Unified API",
        "status": "running",
        "docs": "/docs",
        "modules": [
            {
                "name": module["tag"],
                **get_module_status(module),
            }
            for module in MODULES
        ],
    }


@app.get("/health", tags=["FASTAPI"])
async def health():
    return {
        "status": "healthy",
        "modules": {
            module["tag"]: get_module_status(module)
            for module in MODULES
        },
    }


if __name__ == "__main__":
    import uvicorn

    uvicorn.run("main:app", host="0.0.0.0", port=int(os.getenv("PORT", "8001")))
