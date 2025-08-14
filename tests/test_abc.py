import pytest
from fastmcp import Client
import os


@pytest.mark.asyncio
async def test_notebook(mcp):
    async with Client(mcp) as client:
        # Test create_notebook
        result = await client.call_tool(
            "create_notebook", {"nbid": "test", "kernel": "python3"}
        )
        assert "test" in result.content[0].text

        # Test switch_active_notebook
        result = await client.call_tool("switch_active_notebook", {"nbid": "test"})
        assert "switched to notebook test" in result.content[0].text

        # Test single_step_execute (mock code)
        result = await client.call_tool(
            "single_step_execute",
            {"code": "print('hello')", "backup_var": None, "show_var": None},
        )
        assert "hello" in result.content[0].text

        # Test single_step_execute show_var
        result = await client.call_tool(
            "single_step_execute",
            {"code": "hello = 'hello2'", "backup_var": None, "show_var": "hello"},
        )
        assert "hello2" in result.content[0].text

        # Test single_step_execute show_var
        result = await client.call_tool(
            "single_step_execute",
            {"code": "hello = 'hello3'\nprint(hello)", "backup_var": ["hello"]},
        )
        assert "hello3" in result.content[0].text

        # Test multi_step_execute (mock code)
        result = await client.call_tool(
            "multi_step_execute",
            {"code": "a = 123\nprint(a)", "backup_var": None, "show_var": None},
        )
        assert "123" in result.content[0].text

        # Test query_api_doc (mock code)
        result = await client.call_tool(
            "query_api_doc", {"code": "import math\nmath.sqrt.__doc__"}
        )
        assert "square root" in result.content[0].text

        # Test list_notebooks
        result = await client.call_tool("list_notebooks")
        assert "test" in result.content[0].text

        # Test single_step_execute generate image
        result = await client.call_tool(
            "single_step_execute",
            {
                "code": "import matplotlib.pyplot as plt\nplt.plot([1,2,3],[4,5,6])\nplt.show()\n",
                "backup_var": None,
            },
        )
        assert ".png" in result.content[0].text

        result = await client.call_tool(
            "get_path_structure", {"path": str(os.getcwd())}
        )
        assert "tests" in result.content[0].text

        # Test file path
        result = await client.call_tool("get_path_structure", {"path": str(__file__)})
        assert "test_abc.py" in result.content[0].text

        # Test shutdown_notebook
        result = await client.call_tool("kill_notebook", {"nbid": "test"})
        assert "Notebook test shutdown" in result.content[0].text
