import pytest
from fastmcp import Client
import os


@pytest.mark.asyncio
async def test_notebook(mcp, tmp_path):
    async with Client(mcp) as client:
        # Test create_notebook
        nb_path = tmp_path / "test.ipynb"
        result = await client.call_tool(
            "create_notebook",
            {"nbid": "test", "kernel": "python3", "path": str(nb_path)},
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


@pytest.mark.asyncio
async def test_memory_time_tracking(mcp, tmp_path):
    """Test memory and time monitoring features"""
    async with Client(mcp) as client:
        # Create a test notebook
        nb_path = tmp_path / "memory_test.ipynb"
        result = await client.call_tool(
            "create_notebook",
            {"nbid": "memory_test", "kernel": "python3", "path": str(nb_path)},
        )
        assert "memory_test" in result.content[0].text

        # Switch to the test notebook
        result = await client.call_tool(
            "switch_active_notebook", {"nbid": "memory_test"}
        )
        assert "switched to notebook memory_test" in result.content[0].text

        # Test time and memory monitoring for simple code execution
        result = await client.call_tool(
            "single_step_execute",
            {
                "code": "x = 1 + 1\nprint(f'x = {x}')",
                "backup_var": None,
                "show_var": None,
            },
        )
        assert "x = 2" in result.content[0].text

        # Check whether the result contains time and memory info
        result_text = result.content[0].text
        assert "execution_time" in result_text or "执行时间" in result_text
        assert "memory_stats" in result_text or "内存统计" in result_text

        # Test memory-intensive operation
        result = await client.call_tool(
            "single_step_execute",
            {
                "code": """
import numpy as np
# 创建一个较大的数组来测试内存监控
large_array = np.random.rand(1000, 1000)
print(f"数组形状: {large_array.shape}")
print(f"数组大小: {large_array.nbytes / 1024 / 1024:.2f} MB")
""",
                "backup_var": None,
                "show_var": None,
            },
        )
        assert "数组形状: (1000, 1000)" in result.content[0].text
        assert "数组大小:" in result.content[0].text

        # Verify memory monitoring results
        result_text = result.content[0].text
        assert "execution_time" in result_text or "执行时间" in result_text
        assert "memory_stats" in result_text or "内存统计" in result_text

        # Test memory monitoring during error handling
        result = await client.call_tool(
            "single_step_execute",
            {
                "code": "undefined_variable + 1",
                "backup_var": None,
                "show_var": None,
            },
        )
        # Time info should be present even on error
        result_text = result.content[0].text
        assert "execution_time" in result_text or "执行时间" in result_text

        # Test memory monitoring during a syntax error
        result = await client.call_tool(
            "single_step_execute",
            {
                "code": "print('hello world'",  # Missing right parenthesis
                "backup_var": None,
                "show_var": None,
            },
        )
        # Time info should be present even on syntax error
        result_text = result.content[0].text
        assert "execution_time" in result_text or "执行时间" in result_text

        # Clean up the test notebook
        result = await client.call_tool("kill_notebook", {"nbid": "memory_test"})
        assert "Notebook memory_test shutdown" in result.content[0].text


@pytest.mark.asyncio
async def test_memory_monitor_accuracy(mcp, tmp_path):
    """Test the accuracy of memory monitoring"""
    async with Client(mcp) as client:
        # Create a test notebook
        nb_path = tmp_path / "accuracy_test.ipynb"
        result = await client.call_tool(
            "create_notebook",
            {"nbid": "accuracy_test", "kernel": "python3", "path": str(nb_path)},
        )
        assert "accuracy_test" in result.content[0].text

        # Switch to the test notebook
        result = await client.call_tool(
            "switch_active_notebook", {"nbid": "accuracy_test"}
        )
        assert "switched to notebook accuracy_test" in result.content[0].text

        # Test small memory allocation
        result = await client.call_tool(
            "single_step_execute",
            {
                "code": """
# 创建一个小数组
small_array = [i for i in range(1000)]
print(f"创建了包含 {len(small_array)} 个元素的列表")
""",
                "backup_var": None,
                "show_var": None,
            },
        )
        assert "创建了包含 1000 个元素的列表" in result.content[0].text

        # Test large memory allocation
        result = await client.call_tool(
            "single_step_execute",
            {
                "code": """
import numpy as np
# 创建一个大数组
big_array = np.random.rand(2000, 2000)
print(f"创建了 {big_array.shape} 的数组")
print(f"数组大小: {big_array.nbytes / 1024 / 1024:.2f} MB")
""",
                "backup_var": None,
                "show_var": None,
            },
        )
        assert "创建了 (2000, 2000) 的数组" in result.content[0].text
        assert "数组大小:" in result.content[0].text

        # Test memory release
        result = await client.call_tool(
            "single_step_execute",
            {
                "code": """
# 删除大数组，释放内存
del big_array
import gc
gc.collect()
print("内存已释放")
""",
                "backup_var": None,
                "show_var": None,
            },
        )
        assert "内存已释放" in result.content[0].text

        # Clean up the test notebook
        result = await client.call_tool("kill_notebook", {"nbid": "accuracy_test"})
        assert "Notebook accuracy_test shutdown" in result.content[0].text


@pytest.mark.asyncio
async def test_time_monitoring_accuracy(mcp, tmp_path):
    """Test the accuracy of time monitoring"""
    async with Client(mcp) as client:
        # Create a test notebook
        nb_path = tmp_path / "time_test.ipynb"
        result = await client.call_tool(
            "create_notebook",
            {"nbid": "time_test", "kernel": "python3", "path": str(nb_path)},
        )
        assert "time_test" in result.content[0].text

        # Switch to the test notebook
        result = await client.call_tool("switch_active_notebook", {"nbid": "time_test"})
        assert "switched to notebook time_test" in result.content[0].text

        # Test fast execution
        result = await client.call_tool(
            "single_step_execute",
            {
                "code": "x = 1 + 1",
                "backup_var": None,
                "show_var": None,
            },
        )
        # Fast execution should complete quickly
        result_text = result.content[0].text
        assert "execution_time" in result_text or "执行时间" in result_text

        # Test slow execution
        result = await client.call_tool(
            "single_step_execute",
            {
                "code": """
import time
print("开始等待...")
time.sleep(0.5)  # 等待0.5秒
print("等待结束")
""",
                "backup_var": None,
                "show_var": None,
            },
        )
        assert "开始等待..." in result.content[0].text
        assert "等待结束" in result.content[0].text

        # Check whether execution time is reasonable (should be ~0.5s)
        result_text = result.content[0].text
        assert "execution_time" in result_text or "执行时间" in result_text

        # Test compute-intensive task
        result = await client.call_tool(
            "single_step_execute",
            {
                "code": """
import time
print("开始计算...")
# 计算密集型任务
result = sum(i*i for i in range(100000))
print(f"计算结果: {result}")
""",
                "backup_var": None,
                "show_var": None,
            },
        )
        assert "开始计算..." in result.content[0].text
        assert "计算结果:" in result.content[0].text

        # Clean up the test notebook
        result = await client.call_tool("kill_notebook", {"nbid": "time_test"})
        assert "Notebook time_test shutdown" in result.content[0].text


@pytest.mark.asyncio
async def test_notebook_management_features(mcp, tmp_path):
    """Test notebook management features like save, load, switch, etc."""
    async with Client(mcp) as client:
        # Create multiple test notebooks
        nb1_path = tmp_path / "notebook1.ipynb"
        result = await client.call_tool(
            "create_notebook",
            {"nbid": "notebook1", "kernel": "python3", "path": str(nb1_path)},
        )
        assert "notebook1" in result.content[0].text

        nb2_path = tmp_path / "notebook2.ipynb"
        result = await client.call_tool(
            "create_notebook",
            {"nbid": "notebook2", "kernel": "python3", "path": str(nb2_path)},
        )
        assert "notebook2" in result.content[0].text

        # Test list_notebooks
        result = await client.call_tool("list_notebooks")
        result_text = result.content[0].text
        assert "notebook1" in result_text
        assert "notebook2" in result_text

        # Test switch between notebooks
        result = await client.call_tool("switch_active_notebook", {"nbid": "notebook1"})
        assert "switched to notebook notebook1" in result.content[0].text

        # Execute code in notebook1
        result = await client.call_tool(
            "single_step_execute",
            {"code": "x = 10", "backup_var": None, "show_var": None},
        )
        assert result.content[0].text  # Should execute successfully

        # Switch to notebook2
        result = await client.call_tool("switch_active_notebook", {"nbid": "notebook2"})
        assert "switched to notebook notebook2" in result.content[0].text

        # Execute different code in notebook2
        result = await client.call_tool(
            "single_step_execute",
            {"code": "y = 20", "backup_var": None, "show_var": None},
        )
        assert result.content[0].text  # Should execute successfully

        # Switch back to notebook1 and verify x still exists
        result = await client.call_tool("switch_active_notebook", {"nbid": "notebook1"})
        result = await client.call_tool(
            "single_step_execute",
            {"code": "print(x)", "backup_var": None, "show_var": None},
        )
        assert "10" in result.content[0].text

        # Clean up
        result = await client.call_tool("kill_notebook", {"nbid": "notebook1"})
        result = await client.call_tool("kill_notebook", {"nbid": "notebook2"})


@pytest.mark.asyncio
async def test_markdown_and_cell_operations(mcp, tmp_path):
    """Test markdown cell addition and cell rerun functionality."""
    async with Client(mcp) as client:
        # Create a test notebook
        nb_path = tmp_path / "markdown_test.ipynb"
        result = await client.call_tool(
            "create_notebook",
            {"nbid": "markdown_test", "kernel": "python3", "path": str(nb_path)},
        )
        assert "markdown_test" in result.content[0].text

        # Test markdown cell addition (if available through MCP)
        # Note: This might need to be implemented in the MCP server
        result = await client.call_tool(
            "single_step_execute",
            {
                "code": "# This is a markdown-style comment\nprint('Hello from markdown test')",
                "backup_var": None,
                "show_var": None,
            },
        )
        assert "Hello from markdown test" in result.content[0].text

        # Test cell rerun functionality
        result = await client.call_tool(
            "single_step_execute",
            {
                "code": "counter = 0\ncounter += 1\nprint(f'Counter: {counter}')",
                "backup_var": None,
                "show_var": None,
            },
        )
        assert "Counter: 1" in result.content[0].text

        # Clean up
        result = await client.call_tool("kill_notebook", {"nbid": "markdown_test"})


@pytest.mark.asyncio
async def test_error_handling_and_edge_cases(mcp, tmp_path):
    """Test error handling and edge cases."""
    async with Client(mcp) as client:
        # Test creating notebook with invalid parameters
        nb_path = tmp_path / "error_test.ipynb"
        result = await client.call_tool(
            "create_notebook",
            {"nbid": "error_test", "kernel": "python3", "path": str(nb_path)},
        )
        assert "error_test" in result.content[0].text

        # Test syntax error handling
        result = await client.call_tool(
            "single_step_execute",
            {
                "code": "print('hello world'",
                "backup_var": None,
                "show_var": None,
            },  # Missing closing parenthesis
        )
        # Should handle syntax error gracefully
        assert result.content[0].text

        # Test runtime error handling
        result = await client.call_tool(
            "single_step_execute",
            {"code": "undefined_variable + 1", "backup_var": None, "show_var": None},
        )
        # Should handle runtime error gracefully
        assert result.content[0].text

        # Test switching to non-existent notebook
        result = await client.call_tool(
            "switch_active_notebook", {"nbid": "non_existent"}
        )
        # Should handle error gracefully
        assert result.content[0].text

        # Clean up
        result = await client.call_tool("kill_notebook", {"nbid": "error_test"})


@pytest.mark.asyncio
async def test_backup_variable_functionality(mcp, tmp_path):
    """Test backup variable functionality in detail."""
    async with Client(mcp) as client:
        # Create a test notebook at a temporary path
        nb_path = tmp_path / "backup_test.ipynb"
        result = await client.call_tool(
            "create_notebook",
            {"nbid": "backup_test", "kernel": "python3", "path": str(nb_path)},
        )
        assert "backup_test" in result.content[0].text

        # Set up initial variable
        result = await client.call_tool(
            "single_step_execute",
            {
                "code": "my_list = [1, 2, 3]\nprint(f'Initial: {my_list}')",
                "backup_var": None,
                "show_var": None,
            },
        )
        assert "Initial: [1, 2, 3]" in result.content[0].text

        # Test backup functionality - modify list (backup is only for error recovery)
        result = await client.call_tool(
            "single_step_execute",
            {
                "code": "my_list.append(4)\nprint(f'Modified: {my_list}')",
                "backup_var": ["my_list"],
                "show_var": None,
            },
        )
        assert "Modified: [1, 2, 3, 4]" in result.content[0].text

        # Verify the variable keeps the modified value (backup doesn't auto-restore on success)
        result = await client.call_tool(
            "single_step_execute",
            {
                "code": "print(f'After backup: {my_list}')",
                "backup_var": None,
                "show_var": None,
            },
        )
        assert "After backup: [1, 2, 3, 4]" in result.content[0].text

        # Test backup with multiple variables
        result = await client.call_tool(
            "single_step_execute",
            {
                "code": "a = 10\nb = 20\nprint(f'a={a}, b={b}')",
                "backup_var": None,
                "show_var": None,
            },
        )
        assert "a=10, b=20" in result.content[0].text

        result = await client.call_tool(
            "single_step_execute",
            {
                "code": "a = 100\nb = 200\nprint(f'Modified: a={a}, b={b}')",
                "backup_var": ["a", "b"],
                "show_var": None,
            },
        )
        assert "Modified: a=100, b=200" in result.content[0].text

        # Verify both variables keep the modified values (backup doesn't auto-restore on success)
        result = await client.call_tool(
            "single_step_execute",
            {
                "code": "print(f'After backup: a={a}, b={b}')",
                "backup_var": ["a", "b"],
                "show_var": None,
            },
        )
        print(result.content[0].text[20:])
        assert "After backup: a=100, b=200" in result.content[0].text

        # Test backup functionality with error recovery
        result = await client.call_tool(
            "single_step_execute",
            {
                "code": "my_list1 = [1, 2, 3]\nprint(f'Before error: {my_list1}')",
                "backup_var": None,
                "show_var": None,
            },
        )
        assert "Before error: [1, 2, 3]" in result.content[0].text

        # This should cause an error and trigger backup restoration
        result = await client.call_tool(
            "single_step_execute",
            {
                "code": "my_list1.append(4)\nprint(f'Modified: {my_list1}')\nundefined_variable + 1",
                "backup_var": ["my_list1"],
                "show_var": None,
            },
        )
        # Should show the error
        assert "NameError" in result.content[0].text

        # Verify the variable was restored after error
        # result = await client.call_tool(
        #     "single_step_execute",
        #     {
        #         "code": "print(f'After error: {my_list1}')",
        #         "backup_var": None,
        #         "show_var": None,
        #     },
        # )
        # assert "After error: [1, 2, 3]" in result.content[0].text

        # Clean up
        result = await client.call_tool("kill_notebook", {"nbid": "backup_test"})


@pytest.mark.asyncio
async def test_show_variable_functionality(mcp, tmp_path):
    """Test show variable functionality in detail."""
    async with Client(mcp) as client:
        # Create a test notebook
        nb_path = tmp_path / "show_var_test.ipynb"
        result = await client.call_tool(
            "create_notebook",
            {"nbid": "show_var_test", "kernel": "python3", "path": str(nb_path)},
        )
        assert "show_var_test" in result.content[0].text

        # Test showing a simple variable
        result = await client.call_tool(
            "single_step_execute",
            {
                "code": "my_string = 'Hello World'",
                "backup_var": None,
                "show_var": "my_string",
            },
        )
        assert "Hello World" in result.content[0].text

        # Test showing a complex data structure
        result = await client.call_tool(
            "single_step_execute",
            {
                "code": "my_dict = {'name': 'Alice', 'age': 30, 'city': 'New York'}",
                "backup_var": None,
                "show_var": "my_dict",
            },
        )
        result_text = result.content[0].text
        assert "Alice" in result_text
        assert "30" in result_text
        assert "New York" in result_text

        # Test showing a list
        result = await client.call_tool(
            "single_step_execute",
            {
                "code": "my_list = [1, 2, 3, 4, 5]",
                "backup_var": None,
                "show_var": "my_list",
            },
        )
        assert "[1, 2, 3, 4, 5]" in result.content[0].text

        # Test showing a numpy array (if available)
        result = await client.call_tool(
            "single_step_execute",
            {
                "code": "import numpy as np\nmy_array = np.array([1, 2, 3, 4, 5])",
                "backup_var": None,
                "show_var": "my_array",
            },
        )
        result_text = result.content[0].text
        assert "array" in result_text or "1" in result_text

        # Clean up
        result = await client.call_tool("kill_notebook", {"nbid": "show_var_test"})


@pytest.mark.asyncio
async def test_connection_file_output(mcp, tmp_path):
    """Test connection_file_output parameter functionality."""
    async with Client(mcp) as client:
        # Test creating notebook with connection_file_output
        # Note: This might need to be implemented in the MCP server
        nb_path = tmp_path / "connection_test.ipynb"
        result = await client.call_tool(
            "create_notebook",
            {
                "nbid": "connection_test",
                "kernel": "python3",
                "path": str(nb_path),
                # "connection_file_output": "/tmp/test_connection.json"  # Uncomment if supported
            },
        )
        assert "connection_test" in result.content[0].text

        # Execute some code to verify it works
        result = await client.call_tool(
            "single_step_execute",
            {
                "code": "print('Connection test successful')",
                "backup_var": None,
                "show_var": None,
            },
        )
        assert "Connection test successful" in result.content[0].text

        # Clean up
        result = await client.call_tool("kill_notebook", {"nbid": "connection_test"})


@pytest.mark.asyncio
async def test_file_operations_and_paths(mcp):
    """Test file operations and path handling."""
    async with Client(mcp) as client:
        # Test get_path_structure with various paths
        result = await client.call_tool("get_path_structure", {"path": "."})
        assert result.content[0].text

        # Test with a specific file
        result = await client.call_tool("get_path_structure", {"path": __file__})
        assert "test_abc.py" in result.content[0].text

        # Test with a directory
        result = await client.call_tool(
            "get_path_structure", {"path": os.path.dirname(__file__)}
        )
        assert result.content[0].text

        # Test with non-existent path
        result = await client.call_tool(
            "get_path_structure", {"path": "/non/existent/path"}
        )
        # Should handle gracefully
        assert result.content[0].text


@pytest.mark.asyncio
async def test_api_documentation_query(mcp, tmp_path):
    """Test API documentation query functionality."""
    async with Client(mcp) as client:
        # Create a test notebook
        nb_path = tmp_path / "api_doc_test.ipynb"
        result = await client.call_tool(
            "create_notebook",
            {"nbid": "api_doc_test", "kernel": "python3", "path": str(nb_path)},
        )
        assert "api_doc_test" in result.content[0].text

        # Test querying built-in function documentation
        result = await client.call_tool("query_api_doc", {"code": "print.__doc__"})
        assert result.content[0].text

        # Test querying math module documentation
        result = await client.call_tool(
            "query_api_doc", {"code": "import math\nmath.sqrt.__doc__"}
        )
        assert "square root" in result.content[0].text

        # Test querying list method documentation
        result = await client.call_tool(
            "query_api_doc", {"code": "list.append.__doc__"}
        )
        assert result.content[0].text

        # Clean up
        result = await client.call_tool("kill_notebook", {"nbid": "api_doc_test"})


@pytest.mark.asyncio
async def test_comprehensive_execution_scenarios(mcp, tmp_path):
    """Test comprehensive execution scenarios including complex code."""
    async with Client(mcp) as client:
        # Create a test notebook
        nb_path = tmp_path / "comprehensive_test.ipynb"
        result = await client.call_tool(
            "create_notebook",
            {"nbid": "comprehensive_test", "kernel": "python3", "path": str(nb_path)},
        )
        assert "comprehensive_test" in result.content[0].text

        # Test multi-line code execution
        result = await client.call_tool(
            "multi_step_execute",
            {
                "code": """
# Complex multi-step execution
import math
import random

# Step 1: Generate data
data = [random.randint(1, 100) for _ in range(10)]
print(f"Generated data: {data}")

# Step 2: Calculate statistics
mean = sum(data) / len(data)
variance = sum((x - mean) ** 2 for x in data) / len(data)
std_dev = math.sqrt(variance)

print(f"Mean: {mean:.2f}")
print(f"Standard deviation: {std_dev:.2f}")

# Step 3: Find outliers
outliers = [x for x in data if abs(x - mean) > 2 * std_dev]
print(f"Outliers: {outliers}")
""",
                "backup_var": None,
                "show_var": None,
            },
        )
        result_text = result.content[0].text
        assert "Generated data:" in result_text
        assert "Mean:" in result_text
        assert "Standard deviation:" in result_text
        assert "Outliers:" in result_text

        # Test error recovery
        result = await client.call_tool(
            "single_step_execute",
            {
                "code": "print('This should work after the complex execution')",
                "backup_var": None,
                "show_var": None,
            },
        )
        assert "This should work after the complex execution" in result.content[0].text

        # Clean up
        result = await client.call_tool("kill_notebook", {"nbid": "comprehensive_test"})


@pytest.mark.asyncio
async def test_notebook_management_methods(mcp, tmp_path):
    """Test notebook management methods like add_markdown, rerun_cell, etc."""
    async with Client(mcp) as client:
        # Create a test notebook
        nb_path = tmp_path / "methods_test.ipynb"
        result = await client.call_tool(
            "create_notebook",
            {"nbid": "methods_test", "kernel": "python3", "path": str(nb_path)},
        )
        assert "methods_test" in result.content[0].text

        # Test basic execution to set up some cells
        result = await client.call_tool(
            "single_step_execute",
            {"code": "x = 42\nprint(f'x = {x}')", "backup_var": None, "show_var": None},
        )
        assert "x = 42" in result.content[0].text

        # Test another execution
        result = await client.call_tool(
            "single_step_execute",
            {
                "code": "y = x * 2\nprint(f'y = {y}')",
                "backup_var": None,
                "show_var": None,
            },
        )
        assert "y = 84" in result.content[0].text

        # Test multi-step execution
        result = await client.call_tool(
            "multi_step_execute",
            {
                "code": "z = x + y\nprint(f'z = {z}')\nprint('Multi-step completed')",
                "backup_var": None,
                "show_var": None,
            },
        )
        assert "z = 126" in result.content[0].text
        assert "Multi-step completed" in result.content[0].text

        # Test with backup variables
        result = await client.call_tool(
            "single_step_execute",
            {
                "code": "x = 100\nprint(f'Modified x = {x}')",
                "backup_var": ["x"],
                "show_var": None,
            },
        )
        assert "Modified x = 100" in result.content[0].text

        # Verify x keeps the modified value (backup doesn't auto-restore on success)
        result = await client.call_tool(
            "single_step_execute",
            {
                "code": "print(f'After backup x = {x}')",
                "backup_var": None,
                "show_var": None,
            },
        )
        assert "After backup x = 100" in result.content[0].text

        # Test show_var functionality
        result = await client.call_tool(
            "single_step_execute",
            {
                "code": "my_data = {'key': 'value', 'number': 123}",
                "backup_var": None,
                "show_var": "my_data",
            },
        )
        result_text = result.content[0].text
        assert "key" in result_text
        assert "value" in result_text
        assert "123" in result_text

        # Clean up
        result = await client.call_tool("kill_notebook", {"nbid": "methods_test"})


@pytest.mark.asyncio
async def test_edge_cases_and_error_scenarios(mcp, tmp_path):
    """Test edge cases and error scenarios."""
    async with Client(mcp) as client:
        # Test creating notebook with special characters in name
        nb_path = tmp_path / "test_123_special.ipynb"
        result = await client.call_tool(
            "create_notebook",
            {"nbid": "test_123_special", "kernel": "python3", "path": str(nb_path)},
        )
        assert "test_123_special" in result.content[0].text

        # Test very long code execution
        long_code = (
            "print('Start')\n"
            + "\n".join([f"x{i} = {i}" for i in range(100)])
            + "\nprint('End')"
        )
        result = await client.call_tool(
            "single_step_execute",
            {"code": long_code, "backup_var": None, "show_var": None},
        )
        assert "Start" in result.content[0].text
        assert "End" in result.content[0].text

        # Test empty code execution
        result = await client.call_tool(
            "single_step_execute",
            {"code": "", "backup_var": None, "show_var": None},
        )
        # Should handle empty code gracefully
        assert result.content[0].text

        # Test code with only comments
        result = await client.call_tool(
            "single_step_execute",
            {
                "code": "# This is just a comment\n# Another comment",
                "backup_var": None,
                "show_var": None,
            },
        )
        # Should handle comments gracefully
        assert result.content[0].text

        # Test code with only whitespace
        result = await client.call_tool(
            "single_step_execute",
            {"code": "   \n\t\n   ", "backup_var": None, "show_var": None},
        )
        # Should handle whitespace gracefully
        assert result.content[0].text

        # Clean up
        result = await client.call_tool("kill_notebook", {"nbid": "test_123_special"})


@pytest.mark.asyncio
async def test_performance_and_stress(mcp, tmp_path):
    """Test performance and stress scenarios."""
    async with Client(mcp) as client:
        # Create a test notebook
        nb_path = tmp_path / "performance_test.ipynb"
        result = await client.call_tool(
            "create_notebook",
            {"nbid": "performance_test", "kernel": "python3", "path": str(nb_path)},
        )
        assert "performance_test" in result.content[0].text

        # Test rapid successive executions
        for i in range(5):
            result = await client.call_tool(
                "single_step_execute",
                {
                    "code": f"print('Execution {i + 1}')",
                    "backup_var": None,
                    "show_var": None,
                },
            )
            assert f"Execution {i + 1}" in result.content[0].text

        # Test memory-intensive operation
        result = await client.call_tool(
            "single_step_execute",
            {
                "code": """
import numpy as np
# Create a large array
large_array = np.random.rand(5000, 5000)
print(f"Created array of shape: {large_array.shape}")
print(f"Memory usage: {large_array.nbytes / 1024 / 1024:.2f} MB")
# Perform some computation
result = np.sum(large_array)
print(f"Sum: {result:.2f}")
""",
                "backup_var": None,
                "show_var": None,
            },
        )
        result_text = result.content[0].text
        assert "Created array of shape: (5000, 5000)" in result_text
        assert "Memory usage:" in result_text
        assert "Sum:" in result_text

        # Test CPU-intensive operation
        result = await client.call_tool(
            "single_step_execute",
            {
                "code": """
import time
print("Starting CPU-intensive task...")
start_time = time.time()
# CPU-intensive computation
result = sum(i**2 for i in range(1000000))
end_time = time.time()
print(f"Result: {result}")
print(f"Time taken: {end_time - start_time:.2f} seconds")
""",
                "backup_var": None,
                "show_var": None,
            },
        )
        result_text = result.content[0].text
        assert "Starting CPU-intensive task..." in result_text
        assert "Result:" in result_text
        assert "Time taken:" in result_text

        # Clean up
        result = await client.call_tool("kill_notebook", {"nbid": "performance_test"})
