import jupyter_client
import json
from datetime import datetime
import os
import re
from typing import Optional, Dict, Any, List


class JupyterClientExecutor:
    def __init__(
        self, kernel_name: str = "python", notebook_path: Optional[str] = None
    ):
        kernel_manager, kernel_client = jupyter_client.manager.start_new_kernel(
            kernel_name=kernel_name
        )
        self.ksm = jupyter_client.kernelspec.KernelSpecManager()
        self.spec = self.ksm.get_kernel_spec(kernel_name)
        self.kernel_client = kernel_client
        self.kernel_manager = kernel_manager
        self.cells: List[Dict[str, Any]] = []
        self.notebook_path = notebook_path
        if notebook_path:
            self.save_notebook()

    def _strip_ansi_codes(self, text: str) -> str:
        """
        Remove ANSI color codes from text.

        Args:
            text (str): Text containing ANSI color codes

        Returns:
            str: Text with ANSI color codes removed
        """
        ansi_escape = re.compile(r"\x1B(?:[@-Z\\-_]|\[[0-?]*[ -/]*[@-~])")
        return ansi_escape.sub("", text)

    def add_markdown(self, markdown_text: str) -> None:
        """
        Add a markdown cell to the notebook.

        Args:
            markdown_text (str): The markdown text to add
        """
        cell = {"cell_type": "markdown", "metadata": {}, "source": markdown_text}
        self.cells.append(cell)

    def _clear_messages(self) -> None:
        """Clear all pending messages from the kernel."""
        if self.kernel_client is None:
            return
        while True:
            try:
                self.kernel_client.get_iopub_msg(timeout=0.1)
            except Exception:
                break

    def execute(
        self, code: str, add_cell: bool = True, backup_var: Optional[str] = None
    ) -> Dict[str, Any]:
        """
        Execute code in the Jupyter kernel and return the results.

        Args:
            code (str): The code to execute
            add_cell (bool): Whether to add the executed cell to the notebook. Default is True.

        Returns:
            dict: A dictionary containing execution results with the following keys:
                - stdout: Standard output text
                - stderr: Standard error text
                - result: Execution result (if any)
                - display_data: Display data (if any)
                - error: Error information (if any)
                - success: Boolean indicating if execution was successful
        """
        if self.kernel_client is None:
            return {
                "result": "",
                "display_data": "",
                "error": "Kernel client is not available",
                "success": False,
            }

        # First check if the code is valid
        if backup_var:
            backup_code = code.replace(backup_var, f"{backup_var}_backup")
            backup_code = f"{backup_var}_backup = {backup_var}.copy()\n{backup_code}\n{backup_var} = {backup_var}_backup"
        else:
            backup_code = code

        try:
            # Try to compile the code to check syntax
            if self.spec.language == "python":
                if not backup_code.startswith(("!", "%")):
                    compile(backup_code, "<string>", "exec")
        except SyntaxError as e:
            result = {
                "result": "",
                "display_data": "",
                "error": {
                    "error_type": "SyntaxError",
                    "error_info": str(e),
                    "error_traceback": [f"SyntaxError: {str(e)}"],
                    "error_code": code,
                },
                "success": False,
            }
            return result

        # Create a new cell with the code
        cell = {
            "cell_type": "code",
            "execution_count": None,
            "metadata": {},
            "source": code,
            "outputs": [],
        }

        # Initialize result dictionary
        result = {"result": "", "display_data": "", "error": "", "success": True}

        msg_id = self.kernel_client.execute(code)
        while True:
            try:
                msg = self.kernel_client.get_iopub_msg()

                content = msg["content"]

                # Only process messages for our current execution
                if msg.get("parent_header", {}).get("msg_id") != msg_id:
                    print("Debug: Skipping message with different msg_id")
                    continue

                # Skip messages we don't need to process
                if msg["msg_type"] in ["execute_input", "clear_output", "busy"]:
                    continue

                if msg["msg_type"] == "stream":
                    # Handle stdout and stderr
                    text = content["text"]
                    if content.get("name") == "stderr":
                        result["error"] += text
                    else:
                        result["result"] += text
                    cell["outputs"].append(
                        {
                            "name": content.get("name", "stdout"),
                            "output_type": "stream",
                            "text": text,
                        }
                    )
                elif msg["msg_type"] == "execute_result":
                    # Handle execution results
                    if "text/plain" in content["data"]:
                        result_text = content["data"]["text/plain"]
                        result["result"] = result_text
                        cell["outputs"].append(
                            {
                                "output_type": "execute_result",
                                "data": content["data"],
                                "metadata": content.get("metadata", {}),
                                "execution_count": content.get("execution_count", None),
                            }
                        )
                elif msg["msg_type"] == "display_data":
                    # Handle display data (images, HTML, etc.)
                    if "image/png" in content["data"]:
                        result["display_data"] = {
                            "image/png": "success to create figure"
                        }
                        cell["outputs"].append(
                            {
                                "output_type": "display_data",
                                "data": content["data"],
                                "metadata": content.get("metadata", {}),
                            }
                        )
                elif msg["msg_type"] == "error":
                    error_msg = "\n".join(content["traceback"])

                    clean_traceback = [
                        self._strip_ansi_codes(line) for line in content["traceback"]
                    ]
                    result["error"] = {
                        "error_type": content.get("ename", "Error"),
                        "error_info": content.get("evalue", ""),
                        "error_traceback": clean_traceback,
                    }
                    result["success"] = False
                    self._clear_messages()
                    break
                elif msg["msg_type"] == "status":
                    if content["execution_state"] == "idle":
                        self._clear_messages()
                        break
                    continue

            except KeyboardInterrupt:
                print("Interrupted by user.")
                error_msg = "KeyboardInterrupt: Interrupted by user"
                result["error"] = {
                    "error_type": "KeyboardInterrupt",
                    "error_info": "Interrupted by user",
                    "error_traceback": [error_msg],
                }
                result["success"] = False
                cell["outputs"].append(
                    {
                        "output_type": "error",
                        "ename": "KeyboardInterrupt",
                        "evalue": "Interrupted by user",
                        "traceback": [error_msg],
                    }
                )
                self._clear_messages()
                break
            except Exception as e:
                error_msg = f"Unexpected error: {str(e)}"
                result["error"] = {
                    "error_type": type(e).__name__,
                    "error_info": str(e),
                    "error_traceback": [error_msg],
                }
                result["success"] = False
                cell["outputs"].append(
                    {
                        "output_type": "error",
                        "ename": type(e).__name__,
                        "evalue": str(e),
                        "traceback": [error_msg],
                    }
                )
                self._clear_messages()
                break

        # Only add the cell to our list of cells if execution was successful and add_cell is True
        if result["success"] and add_cell:
            cell["source"] = code
            self.cells.append(cell)

            # Auto-save if notebook_path is set
            if self.notebook_path:
                self.save_notebook()
        if result["display_data"]:
            result["display_data"]["image/png"] = "success to create image"
        return result

    def rerun_cell(self, cell_index: int) -> None:
        """
        重新运行指定的 cell。

        Args:
            cell_index (int): 要重新运行的 cell 的索引（从0开始）
        """
        if not 0 <= cell_index < len(self.cells):
            print(f"错误：cell 索引 {cell_index} 超出范围")
            return

        cell = self.cells[cell_index]
        if cell["cell_type"] != "code":
            print(f"错误：cell {cell_index} 不是代码 cell")
            return

        # 清除之前的输出
        cell["outputs"] = []

        # 重新执行代码
        self.execute(cell["source"])

        # 添加说明文档
        self.add_markdown(f"重新运行 cell {cell_index}")

    def save_notebook(self, filename: Optional[str] = None) -> Optional[str]:
        """
        Save the current notebook state to a file.

        Args:
            filename (str, optional): Name of the notebook file. If None, uses the notebook_path
                                     set during initialization or generates a timestamp-based name.

        Returns:
            str: Path to the saved notebook file
        """
        if filename is None:
            if self.notebook_path:
                filename = self.notebook_path
            else:
                timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                filename = f"notebook_{timestamp}.ipynb"

        # Create directory if it doesn't exist
        os.makedirs(os.path.dirname(os.path.abspath(filename)), exist_ok=True)

        notebook = {
            "cells": self.cells,
            "metadata": {
                "kernelspec": {
                    "display_name": "Python 3",
                    "language": "python",
                    "name": "python3",
                },
                "language_info": {
                    "codemirror_mode": {"name": "ipython", "version": 3},
                    "file_extension": ".py",
                    "mimetype": "text/x-python",
                    "name": "python",
                    "nbconvert_exporter": "python",
                    "pygments_lexer": "ipython3",
                    "version": "3.8.0",
                },
            },
            "nbformat": 4,
            "nbformat_minor": 4,
        }

        try:
            with open(filename, "w", encoding="utf-8") as f:
                json.dump(notebook, f, indent=1)
            print(f"Notebook saved to: {filename}")
            return filename
        except Exception as e:
            print(f"Error saving notebook: {str(e)}")
            return None

    def shutdown(self) -> None:
        """Shut down the kernel and the client."""
        if self.kernel_client:
            self.kernel_client.stop_channels()
            self.kernel_client = None
        if self.kernel_manager:
            self.kernel_manager.shutdown_kernel()
            self.kernel_manager = None


class NotebookManager:
    def __init__(self):
        self.notebook = {}
        self.active_nbid = None

    def create_notebook(self, nbid, path, kernel="python"):
        self.notebook[nbid] = JupyterClientExecutor(kernel, path)
        self.active_nbid = nbid

    def delete_notebook(self, nbid):
        self.notebook[nbid].shutdown()
        del self.notebook[nbid]

    def switch_notebook(self, nbid):
        self.active_nbid = nbid

    @property
    def active_notebook(self):
        if not self.notebook:
            raise ValueError("No notebook created.")
        if self.active_nbid not in self.notebook:
            raise ValueError(f"Notebook {self.active_nbid} not found.")
        return self.notebook[self.active_nbid]
