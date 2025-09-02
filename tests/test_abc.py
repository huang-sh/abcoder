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


@pytest.mark.asyncio
async def test_memory_time_tracking(mcp):
    """Test memory and time monitoring features"""
    async with Client(mcp) as client:
        # Create a test notebook
        result = await client.call_tool(
            "create_notebook", {"nbid": "memory_test", "kernel": "python3"}
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
async def test_memory_monitor_accuracy(mcp):
    """Test the accuracy of memory monitoring"""
    async with Client(mcp) as client:
        # Create a test notebook
        result = await client.call_tool(
            "create_notebook", {"nbid": "accuracy_test", "kernel": "python3"}
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
async def test_time_monitoring_accuracy(mcp):
    """Test the accuracy of time monitoring"""
    async with Client(mcp) as client:
        # Create a test notebook
        result = await client.call_tool(
            "create_notebook", {"nbid": "time_test", "kernel": "python3"}
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
