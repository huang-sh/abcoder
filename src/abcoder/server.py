from fastmcp import FastMCP
from typing import Any, AsyncIterator
from contextlib import asynccontextmanager
from pydantic import Field
from fastmcp.server.dependencies import get_context
from .backend import NotebookManager
from .util import add_figure_route, get_path_info


def get_nbm():
    ctx = get_context()
    nbm = ctx.request_context.lifespan_context
    return nbm


nbm = NotebookManager()


@asynccontextmanager
async def nb_lifespan(server: FastMCP) -> AsyncIterator[Any]:
    """Context manager for AdataState lifecycle."""
    yield nbm


nb_mcp = FastMCP("Notebook-Server", lifespan=nb_lifespan)
add_figure_route(nb_mcp)


@nb_mcp.tool(tags={"nb"})
def create_notebook(
    nbid: str = Field(description="The notebook id to create."),
    path: str | None = Field(default=None, description="The notebook file path."),
    kernel: str = Field(
        default="python3", description="The kernel to use for the notebook."
    ),
):
    """Create a notebook."""
    if path is None:
        import os

        path = f"{os.getcwd()}/ab_notebook.ipynb"
    nbm = get_nbm()
    nbm.create_notebook(nbid, path, kernel)
    return f"Notebook {nbid} created at {path} with kernel {kernel}."


@nb_mcp.tool(tags={"nb"})
def list_notebooks():
    """List all notebooks."""
    nbm = get_nbm()
    return {"active_notebook": nbm.active_nbid, "all_notebooks": nbm.list_notebook()}


@nb_mcp.tool(tags={"nb"})
def switch_active_notebook(
    nbid: str = Field(description="The notebook id to switch to."),
):
    """Switch to a notebook."""
    nbm = get_nbm()
    nbm.switch_notebook(nbid)
    return f"You have switched to notebook {nbid}."


@nb_mcp.tool(tags={"nb"})
def kill_notebook(
    nbid: str | None = Field(
        description="The notebook id to shutdown. Default None is the active notebook."
    ),
):
    """kill/shutdown a notebook."""
    nbm = get_nbm()
    return nbm.shutdown_notebook(nbid)


@nb_mcp.tool(tags={"nb"})
def single_step_execute(
    code: str = Field(description="The code to execute a single step operation."),
    backup_var: list[str] | None = Field(
        description="The variable name to backup before execution in code (e.g., anndata object).",
        default=None,
    ),
    show_var: str | None = Field(
        description="the anndata object variable name to print anndata if you want to see anndata in Output Area after execution.",
        default=None,
    ),
):
    """Execute a single step operation in the Jupyter kernel."""
    if backup_var:
        for var in backup_var:
            if var not in code:
                return f"Variable {var} not found in the code."
    if bool(show_var):
        if show_var not in code:
            return f"Variable {show_var} not found in the code."
        code = f"{code}\n{show_var}"
    nbm = get_nbm()
    jce = nbm.active_notebook
    res = jce.execute(code, backup_var=backup_var)
    res["notebook_id"] = nbm.active_nbid
    return res


@nb_mcp.tool(tags={"nb"})
def multi_step_execute(
    code: str = Field(description="The code to execute multiple steps of operations."),
    backup_var: list[str] | None = Field(
        description="The variable name to backup before execution in code (e.g., anndata object).",
        default=None,
    ),
    show_var: str | None = Field(
        description="the anndata object variable name to print anndata if you want to see anndata in Output Area after execution.",
        default=None,
    ),
):
    """Execute multiple steps of operations in the Jupyter kernel."""
    if backup_var:
        for var in backup_var:
            if var not in code:
                return f"Variable {var} not found in the code."
    if show_var:
        if show_var not in code:
            return f"Variable {show_var} not found in the code."
        code = f"{code}\n{show_var}"
    nbm = get_nbm()
    jce = nbm.active_notebook
    res = jce.execute(
        code, add_cell=True, backup_var=backup_var
    )  # Add cell for multi-step operations
    res["notebook_id"] = nbm.active_nbid
    return res


@nb_mcp.tool(tags={"nb"})
def query_api_doc(
    code: str = Field(description="The full name of the API or function to query. "),
):
    """query API or function doc by .__doc__ . import library first,like input code 'import scanpy as sc\nsc.pp.pca.__doc__'"""
    nbm = get_nbm()
    jce = nbm.active_notebook
    res = jce.execute(code, add_cell=False)
    res["notebook_id"] = nbm.active_nbid
    return res


@nb_mcp.tool(tags={"nb"})
def get_path_structure(
    path: str = Field(description="The path to get the structure of"),
) -> str:
    """get the directory structure of a path"""
    return get_path_info(path)
