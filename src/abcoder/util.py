import os
import json
from typing import Optional, List, Dict


async def get_figure(request):
    from starlette.responses import FileResponse, JSONResponse

    figure_name = request.path_params["figure_name"]
    figure_path = f"./figures/{figure_name}"

    if not os.path.isfile(figure_path):
        return JSONResponse({"error": "figure not found"})
    return FileResponse(figure_path)


def add_figure_route(server):
    from starlette.routing import Route

    server._additional_http_routes = [
        Route("/figures/{figure_name}", endpoint=get_figure)
    ]


def get_flat_dir_structure(
    path: str, ignore_hidden: bool = True, file_types: Optional[List[str]] = None
) -> Dict[str, Dict[str, List[str]]]:
    structure = {}
    for root, dirs, files in os.walk(path):
        rel_root = os.path.relpath(root, path)
        if rel_root == ".":
            rel_root = ""
        # Filter hidden files and directories
        dirs[:] = [d for d in dirs if not (ignore_hidden and d.startswith("."))]
        if file_types:
            files = [f for f in files if any(f.endswith(ext) for ext in file_types)]
        files = [f for f in files if not (ignore_hidden and f.startswith("."))]
        structure[rel_root] = {"dirs": dirs, "files": files}
    return structure


def get_path_info(
    path: str, ignore_hidden: bool = True, file_types: Optional[List[str]] = None
) -> str:
    if not os.path.exists(path):
        return json.dumps({"error": "Path does not exist"}, ensure_ascii=False)
    if os.path.isfile(path):
        if file_types and not any(path.endswith(ext) for ext in file_types):
            return json.dumps({"error": "File type does not match"}, ensure_ascii=False)
        return json.dumps({"type": "file", "path": path}, ensure_ascii=False)
    elif os.path.isdir(path):
        structure = get_flat_dir_structure(path, ignore_hidden, file_types)
        return json.dumps(
            {"type": "directory", "path": path, "structure": structure},
            ensure_ascii=False,
            indent=2,
        )
    else:
        return json.dumps({"error": "Unknown type"}, ensure_ascii=False)
