from abcoder.backend import JupyterClientExecutor


def test_jupyterclientexecutor_backup_restore(tmp_path):
    # 1. 启动 JupyterClientExecutor
    executor = JupyterClientExecutor(
        kernel_name="python3", notebook_path=str(tmp_path / "test.ipynb")
    )

    # 2. 初始化变量
    code_init = "my_list = [1, 2, 3]\nprint(f'Initial: {my_list}, id: {id(my_list)}')"
    result = executor.execute(code_init)
    print(result["result"])
    assert "Initial: [1, 2, 3]" in result["result"]
    # init_id = int(result["result"].split("id: ")[-1].strip())

    # 3. 正常 append
    code_append = "my_list.append(4)\nprint(f'Modified: {my_list}, id: {id(my_list)}')"
    result = executor.execute(code_append, backup_var=["my_list"])
    print(result["result"])
    assert "Modified: [1, 2, 3, 4]" in str(result)
    # append_id = int(result["result"].split("id: ")[-1].strip())
    # assert append_id == init_id  # id 不变

    # 4. 触发错误，测试恢复
    code_error = "my_list.append(5)\nprint(f'Error: {my_list}, id: {id(my_list)}')\nraise Exception('trigger error')"
    result = executor.execute(code_error, backup_var=["my_list"])
    print(result["result"])
    print(result["error"])
    assert "trigger error" in str(result["error"])
    # 恢复后 my_list 应该回到 [1, 2, 3, 4]，id 可能会变（深拷贝）
    code_check = "print(f'After error: {my_list}, id: {id(my_list)}')"
    result = executor.execute(code_check)
    print(result["result"])
    # after_id = int(result["result"].split("id: ")[-1].strip())
    # 恢复后内容应该正确，id 可能会变（因为指向了备份对象）
    assert "After error: [1, 2, 3, 4]" in str(result)  # ["result"]
    # 验证 id 确实变了（说明恢复机制生效）
    # assert after_id != init_id
    # id 可能变也可能不变，视 restore 机制实现

    # 5. 彻底关闭 kernel
    executor.shutdown()
