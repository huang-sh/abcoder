import pytest
from abcoder.backend import JupyterClientExecutor


@pytest.mark.asyncio
async def test_jupyterclientexecutor_backup_restore():
    # 1. 启动 JupyterClientExecutor
    executor = JupyterClientExecutor(kernel_name="python3")

    # 2. 初始化变量
    code_init = "my_list = [1, 2, 3]\nprint(f'Initial: {my_list}')"
    result = executor.execute(code_init)
    assert "Initial: [1, 2, 3]" in result["result"]

    # 3. 正常 append
    code_append = "my_list.append(4)\nprint(f'Modified: {my_list}')"
    result = executor.execute(code_append, backup_var=["my_list"])
    assert "Modified: [1, 2, 3, 4]" in result["result"]

    # 4. 触发错误，测试恢复
    code_error = "my_list.append(5)\nprint(f'Error: {my_list}')\nraise Exception('trigger error')"
    result = executor.execute(code_error, backup_var=["my_list"])
    assert "trigger error" in str(result["error"])
    # 恢复后 my_list 应该回到 [1, 2, 3, 4]
    code_check = "print(f'After error: {my_list}')"
    result = executor.execute(code_check)
    assert "After error: [1, 2, 3, 4]" in result["result"]

    # 5. 彻底关闭 kernel
    executor.shutdown()
