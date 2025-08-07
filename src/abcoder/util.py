import os


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
