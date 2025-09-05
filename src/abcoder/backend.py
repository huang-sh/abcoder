import jupyter_client
import json
from datetime import datetime
import os
import re
import time
import psutil
import threading
from typing import Optional, Dict, Any, List
from pathlib import Path
from contextlib import contextmanager
from copy import deepcopy
from queue import Empty
from uuid import uuid4
from textwrap import indent


class MemoryMonitor:
    """Monitor Jupyter kernel memory usage."""

    def __init__(self, kernel_manager=None):
        self.kernel_manager = kernel_manager
        self.kernel_process = None
        self.initial_memory = 0.0
        self.peak_memory = 0.0
        self.monitoring = False
        self.monitor_thread = None
        self._lock = threading.Lock()

    def _get_kernel_process(self):
        """Get the kernel process object."""
        if self.kernel_process is None and self.kernel_manager is not None:
            try:
                # Method 1: try to get kernel process from kernel_manager
                if hasattr(self.kernel_manager, "kernel") and hasattr(
                    self.kernel_manager.kernel, "pid"
                ):
                    pid = self.kernel_manager.kernel.pid
                    self.kernel_process = psutil.Process(pid)
                elif hasattr(self.kernel_manager, "pid"):
                    pid = self.kernel_manager.pid
                    self.kernel_process = psutil.Process(pid)
                else:
                    # Method 2: find kernel process by matching the connection file
                    connection_file = getattr(
                        self.kernel_manager, "connection_file", None
                    )
                    if connection_file:
                        self.kernel_process = self._find_kernel_by_connection_file(
                            connection_file
                        )

                    # Method 3: fallback to the latest ipykernel process by start time
                    if self.kernel_process is None:
                        self.kernel_process = self._find_latest_ipykernel_process()

            except (psutil.NoSuchProcess, psutil.AccessDenied, AttributeError):
                self.kernel_process = None
        return self.kernel_process

    def _find_kernel_by_connection_file(self, connection_file):
        """Find kernel process via the connection file path."""
        try:
            import psutil

            for proc in psutil.process_iter(["pid", "cmdline"]):
                try:
                    cmdline = proc.info["cmdline"]
                    if cmdline and any("ipykernel" in str(arg) for arg in cmdline):
                        # Check if the connection file path is present in cmdline
                        if connection_file in " ".join(cmdline):
                            return psutil.Process(proc.info["pid"])
                except (psutil.NoSuchProcess, psutil.AccessDenied):
                    continue
        except Exception:
            pass
        return None

    def _find_latest_ipykernel_process(self):
        """Find the most recently started ipykernel process."""
        try:
            import psutil

            latest_proc = None
            latest_time = 0

            for proc in psutil.process_iter(["pid", "cmdline", "create_time"]):
                try:
                    cmdline = proc.info["cmdline"]
                    if cmdline and any("ipykernel" in str(arg) for arg in cmdline):
                        create_time = proc.info["create_time"]
                        if create_time > latest_time:
                            latest_time = create_time
                            latest_proc = psutil.Process(proc.info["pid"])
                except (psutil.NoSuchProcess, psutil.AccessDenied):
                    continue

            return latest_proc
        except Exception:
            return None

    def get_memory_usage(self) -> float:
        """Get Jupyter kernel RSS memory usage in MB."""
        kernel_process = self._get_kernel_process()
        if kernel_process is not None:
            try:
                memory_info = kernel_process.memory_info()
                return memory_info.rss / 1024 / 1024  # convert to MB
            except (psutil.NoSuchProcess, psutil.AccessDenied):
                # Kernel process might have exited; reset reference
                self.kernel_process = None
                return 0.0
        else:
            # If kernel process cannot be obtained, return 0
            return 0.0

    def start_monitoring(self):
        """Start monitoring memory usage."""
        self.initial_memory = self.get_memory_usage()
        self.peak_memory = self.initial_memory
        self.monitoring = True
        self.monitor_thread = threading.Thread(target=self._monitor_loop, daemon=True)
        self.monitor_thread.start()

    def stop_monitoring(self) -> Dict[str, float]:
        """Stop monitoring and return memory statistics."""
        self.monitoring = False
        if self.monitor_thread:
            self.monitor_thread.join(timeout=1.0)

        final_memory = self.get_memory_usage()
        memory_delta = final_memory - self.initial_memory

        return {
            "initial_memory_mb": self.initial_memory,
            "final_memory_mb": final_memory,
            "peak_memory_mb": self.peak_memory,
            "memory_delta_mb": memory_delta,
        }

    def _monitor_loop(self):
        """Monitoring loop."""
        while self.monitoring:
            try:
                current_memory = self.get_memory_usage()
                with self._lock:
                    if current_memory > self.peak_memory:
                        self.peak_memory = current_memory
                time.sleep(0.1)  # check every 100ms
            except Exception:
                # Log error and continue monitoring
                # print(f"Memory monitoring error: {e}")  # debug suppressed
                time.sleep(0.1)


class JupyterClientExecutor:
    def __init__(
        self,
        kernel_name: str = "python",
        notebook_path: Optional[str] = None,
        connection_file: Optional[str] = None,
        runtime_dir: Optional[str] = None,
        auto_save_overwrite: bool = False,
        connection_file_output: Optional[str] = None,
    ):
        # If a connection_file is provided, attach to an existing kernel; otherwise start a new one
        if connection_file:
            # Connect to an existing kernel via connection file
            km = jupyter_client.KernelManager(connection_file=connection_file)
            try:
                km.load_connection_file(connection_file)
            except Exception:
                # If the path is not absolute, let jupyter_client resolve it
                from jupyter_client import find_connection_file

                resolved = find_connection_file(connection_file)
                km.connection_file = resolved
                km.load_connection_file(resolved)
            kc = km.client()
            try:
                kc.load_connection_file(km.connection_file)
            except Exception:
                pass
            try:
                kc.start_channels()
            except Exception:
                pass
            self.kernel_manager = km
            self.kernel_client = kc
            # Best effort: set a minimal spec-like object for language checks
            try:
                self.ksm = jupyter_client.kernelspec.KernelSpecManager()
                self.spec = self.ksm.get_kernel_spec(kernel_name)
            except Exception:

                class _Spec:
                    language = "python"

                self.spec = _Spec()
            self._owns_kernel = False
        else:
            if connection_file_output:
                # Start a new kernel and write connection file to the exact specified path
                os.makedirs(
                    os.path.dirname(os.path.abspath(connection_file_output)),
                    exist_ok=True,
                )
                km = jupyter_client.KernelManager()
                km.kernel_name = kernel_name
                km.connection_file = connection_file_output
                km.start_kernel()
                kc = km.client()
                try:
                    kc.load_connection_file(km.connection_file)
                except Exception:
                    pass
                kc.start_channels()
                kernel_manager, kernel_client = km, kc
            elif runtime_dir:
                # Start a new kernel and force connection file under runtime_dir
                os.makedirs(runtime_dir, exist_ok=True)
                km = jupyter_client.KernelManager()
                km.kernel_name = kernel_name
                km.connection_file = os.path.join(
                    runtime_dir, f"kernel-{uuid4().hex}.json"
                )
                km.start_kernel()
                kc = km.client()
                try:
                    kc.load_connection_file(km.connection_file)
                except Exception:
                    pass
                kc.start_channels()
                kernel_manager, kernel_client = km, kc
            else:
                # Default behavior
                kernel_manager, kernel_client = jupyter_client.manager.start_new_kernel(
                    kernel_name=kernel_name
                )
            self.ksm = jupyter_client.kernelspec.KernelSpecManager()
            self.spec = self.ksm.get_kernel_spec(kernel_name)
            self.kernel_client = kernel_client
            self.kernel_manager = kernel_manager
            self._owns_kernel = True
        self.cells: List[Dict[str, Any]] = []
        self.notebook_path = notebook_path
        self.memory_monitor = MemoryMonitor(self.kernel_manager)
        # Controls whether auto-save during execute() should overwrite existing files
        self.auto_save_overwrite = auto_save_overwrite
        # If a notebook file exists and overwrite mode is enabled, load existing cells
        if notebook_path and Path(notebook_path).exists() and auto_save_overwrite:
            try:
                self.load_notebook(notebook_path)
            except Exception:
                # Ignore load errors and start fresh
                pass

    @classmethod
    def from_connection_file(
        cls,
        connection_file: str,
        notebook_path: Optional[str] = None,
        kernel_name: str = "python",
    ) -> "JupyterClientExecutor":
        return cls(
            kernel_name=kernel_name,
            notebook_path=notebook_path,
            connection_file=connection_file,
        )

    @property
    def connection_file(self) -> Optional[str]:
        """Return the path to the kernel connection file if available."""
        if getattr(self, "kernel_manager", None) is None:
            return None
        return getattr(self.kernel_manager, "connection_file", None)

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

    def generate_atomic_execute_code(self, code: str, backup_var) -> str:
        """
        Generate atomic code for the given variables.
        """
        backup_dict = "{" + ", ".join([f"'{v}': {v}" for v in backup_var]) + "}"
        code_literal = repr(code)
        code = indent(code, " " * 4)
        def_inner_func = ("def inner_func({args}):\n{code}\n").format(
            args=", ".join(backup_var), code=code
        )

        run_inner_func = "inner_func({args})".format(
            args=", ".join([f"box['{v}']" for v in backup_var])
        )
        atomic_code_lines = [
            "from abcoder.atomic import atomic_objects",
            f"{def_inner_func}",
            f"_objects = {backup_dict}",
            "_scalar_types = (int, float, str)",
            "_scalar_names = [n for n, v in _objects.items() if isinstance(v, _scalar_types)]",
            "_commit_map = {n: (lambda dst, src: None) for n in _scalar_names}",
            "with atomic_objects(_objects, commit_map=_commit_map) as box:",
            "    if _scalar_names:",
            "        _locals = {n: box[n] for n in _objects.keys()}",
            f"        _code = compile({code_literal}, '<string>', 'exec')",
            "        exec(_code, globals(), _locals)",
            "        for _name in _scalar_names:",
            "            box[_name] = _locals.get(_name, box[_name])",
            "        for _name in _scalar_names:",
            "            globals()[_name] = box[_name]",
            "    else:",
            f"        {run_inner_func}",
            "",
        ]
        atomic_code = "\n".join(atomic_code_lines)
        return atomic_code

    def execute(
        self,
        code: str,
        add_cell: bool = True,
        backup_var: List[str] = None,
    ) -> Dict[str, Any]:
        """
        Execute code in the Jupyter kernel and return the results.

        Args:
            code (str): The code to execute
            add_cell (bool): Whether to add the executed cell to the notebook. Default is True.
            backup_var (List[str]): List of variables to backup before execution

        Returns:
            dict: A dictionary containing execution results with the following keys:
                - stdout: Standard output text
                - stderr: Standard error text
                - result: Execution result (if any)
                - display_data: Display data (if any)
                - error: Error information (if any)
                - success: Boolean indicating if execution was successful
                - execution_time: Execution time in seconds
                - memory_stats: Memory usage statistics
        """
        if self.kernel_client is None:
            return {
                "result": "",
                "display_data": "",
                "error": "Kernel client is not available",
                "success": False,
                "execution_time": 0.0,
                "memory_stats": {},
            }

        # Start timer and begin memory monitoring
        start_time = time.time()
        self.memory_monitor.start_monitoring()

        if backup_var:
            backup_code = self.generate_atomic_execute_code(code, backup_var)
        else:
            backup_code = code

        try:
            # Try to compile the code to check syntax
            if self.spec.language == "python":
                if not backup_code.startswith(("!", "%")):
                    compile(backup_code, "<string>", "exec")
        except SyntaxError as e:
            # Stop memory monitoring
            memory_stats = self.memory_monitor.stop_monitoring()
            execution_time = time.time() - start_time

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
                "execution_time": execution_time,
                "memory_stats": memory_stats,
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
        result = {
            "result": "",
            "display_data": "",
            "error": "",
            "success": True,
            "execution_time": 0.0,
            "memory_stats": {},
        }

        msg_id = self.kernel_client.execute(backup_code)
        # Watchdog to avoid hanging indefinitely waiting for kernel messages
        max_wait_seconds = 30.0
        deadline = time.time() + max_wait_seconds
        while True:
            try:
                msg = self.kernel_client.get_iopub_msg(timeout=1.0)

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
                    figdir = Path("./figures")
                    # Handle display data (images, HTML, etc.)
                    if "image/png" in content["data"]:
                        import base64

                        figdir.mkdir(exist_ok=True)
                        figpath = (
                            figdir / f"{datetime.now().strftime('%Y%m%d_%H%M%S')}.png"
                        )
                        with open(figpath, "wb") as f:
                            f.write(base64.b64decode(content["data"]["image/png"]))
                        result["display_data"] = {"image/png": figpath}
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

            except Empty:
                # Timed out waiting for a message; check global deadline
                if time.time() > deadline:
                    error_msg = (
                        "TimeoutError: Execution timed out waiting for kernel messages"
                    )
                    result["error"] = {
                        "error_type": "TimeoutError",
                        "error_info": "Execution exceeded maximum wait time",
                        "error_traceback": [error_msg],
                    }
                    result["success"] = False
                    self._clear_messages()
                    break
                continue
            except KeyboardInterrupt:
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

        # Stop memory monitoring and compute execution time
        memory_stats = self.memory_monitor.stop_monitoring()
        execution_time = time.time() - start_time

        # Attach execution time and memory stats to result
        result["execution_time"] = execution_time
        result["memory_stats"] = memory_stats

        # Only add the cell to our list of cells if execution was successful and add_cell is True
        if result["success"] and add_cell:
            cell["source"] = code
            self.cells.append(cell)

            # Auto-save if notebook_path is set
            self.save_notebook()
        return result

    def restore_backup_var(self, backup_var: List[str]) -> None:
        pass

    def rerun_cell(self, cell_index: int) -> None:
        """
        Re-run the specified cell.

        Args:
            cell_index (int): Index of the cell to re-run (0-based)
        """
        if not 0 <= cell_index < len(self.cells):
            print(f"Error: cell index {cell_index} out of range")
            return

        cell = self.cells[cell_index]
        if cell["cell_type"] != "code":
            print(f"Error: cell {cell_index} is not a code cell")
            return

        # Clear previous outputs
        cell["outputs"] = []

        # Re-execute the code
        self.execute(cell["source"])

        # Add documentation
        self.add_markdown(f"Re-run cell {cell_index}")

    def save_notebook(self, filename: Optional[str] = None) -> Optional[str]:
        """
        Save the current notebook state to a file.

        Args:
            filename (str, optional): Name of the notebook file. If None, uses the notebook_path
                                     set during initialization or generates a timestamp-based name.
            overwrite (bool): If False and the file exists, append a timestamp to avoid overwriting.

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

    def load_notebook(self, filename: Optional[str] = None) -> None:
        """Load notebook cells from a .ipynb file into memory (does not execute)."""
        if filename is None:
            filename = self.notebook_path
        if not filename:
            return
        with open(filename, "r", encoding="utf-8") as f:
            data = json.load(f)
        cells = data.get("cells", [])
        # Only accept list of dicts to avoid malformed inputs
        if isinstance(cells, list):
            self.cells = cells

    def get_all_code(self) -> str:
        """Get all code from notebook cells as a single string."""
        code_lines = []
        for cell in self.cells:
            if cell.get("cell_type") == "code":
                source = cell.get("source", "")
                if isinstance(source, list):
                    source = "".join(source)
                if source.strip():
                    code_lines.append(source)
        return "\n\n".join(code_lines)

    def export_code_to_file(self, filename: str) -> str:
        """Export all code cells to a Python file."""
        code = self.get_all_code()
        os.makedirs(os.path.dirname(os.path.abspath(filename)), exist_ok=True)
        with open(filename, "w", encoding="utf-8") as f:
            f.write(code)
        return filename

    def shutdown(self) -> None:
        """Shut down the kernel and the client."""
        if self.kernel_client:
            self.kernel_client.stop_channels()
            self.kernel_client = None
        if self.kernel_manager:
            # Only shut down kernels that we started ourselves
            try:
                if getattr(self, "_owns_kernel", False):
                    self.kernel_manager.shutdown_kernel()
            finally:
                self.kernel_manager = None

    def kill_kernel(self) -> None:
        """Forcefully terminate the kernel process and close channels."""
        # Try to stop channels first
        if getattr(self, "kernel_client", None):
            try:
                self.kernel_client.stop_channels()
            except Exception:
                pass
            finally:
                self.kernel_client = None
        # Then force shutdown the kernel (now=True attempts immediate kill)
        if getattr(self, "kernel_manager", None):
            try:
                try:
                    self.kernel_manager.shutdown_kernel(now=True)
                except TypeError:
                    # Older jupyter_client may not support 'now'; fallback
                    self.kernel_manager.shutdown_kernel()
            except Exception:
                pass
            finally:
                self.kernel_manager = None


class NotebookManager:
    def __init__(self):
        self.notebook = {}
        self.active_nbid = None

    def create_notebook(
        self,
        nbid,
        path=None,
        kernel="python3",
        connection_file=None,
        runtime_dir=None,
        auto_save_overwrite: bool = False,
        connection_file_output: Optional[str] = None,
    ):
        """Create a notebook. If connection_file is provided, attach to an existing kernel; otherwise start a new one.

        If starting a new kernel, runtime_dir can be used to control where the connection file is written.
        """
        self.notebook[nbid] = JupyterClientExecutor(
            kernel_name=kernel,
            notebook_path=path,
            connection_file=connection_file,
            runtime_dir=runtime_dir,
            auto_save_overwrite=auto_save_overwrite,
            connection_file_output=connection_file_output,
        )
        self.active_nbid = nbid

    def shutdown_notebook(self, nbid=None):
        if nbid is None:
            nbid = self.active_nbid
        if nbid not in self.notebook:
            raise ValueError(
                f"Notebook {nbid} not found. Available notebooks: {self.list_notebook()}"
            )
        self.notebook[nbid].shutdown()
        del self.notebook[nbid]
        return f"Notebook {nbid} shutdown."

    def save_notebook(self, nbid=None, filename=None, overwrite: bool = False):
        """Save the specified (or active) notebook to disk with safe-save support."""
        if nbid is None:
            nbid = self.active_nbid
        if nbid not in self.notebook:
            raise ValueError(
                f"Notebook {nbid} not found. Available notebooks: {self.list_notebook()}"
            )
        return self.notebook[nbid].save_notebook(filename=filename, overwrite=overwrite)

    def close_connection(self, nbid=None):
        """Close client channels for the notebook without terminating the kernel."""
        if nbid is None:
            nbid = self.active_nbid
        if nbid not in self.notebook:
            raise ValueError(
                f"Notebook {nbid} not found. Available notebooks: {self.list_notebook()}"
            )
        executor = self.notebook[nbid]
        if getattr(executor, "kernel_client", None):
            try:
                executor.kernel_client.stop_channels()
            finally:
                executor.kernel_client = None
        return f"Notebook {nbid} connection closed."

    def kill_kernel(self, nbid=None):
        """Forcefully terminate the kernel for the given notebook and drop the handle."""
        if nbid is None:
            nbid = self.active_nbid
        if nbid not in self.notebook:
            raise ValueError(
                f"Notebook {nbid} not found. Available notebooks: {self.list_notebook()}"
            )
        self.notebook[nbid].kill_kernel()
        # Keep the entry but it's effectively disconnected; caller may choose to delete
        return f"Notebook {nbid} kernel killed."

    def switch_notebook(self, nbid):
        self.active_nbid = nbid

    def list_notebook(self):
        return list(self.notebook.keys())

    @property
    def active_notebook(self):
        if not self.notebook:
            raise ValueError("No notebook created.")
        if self.active_nbid not in self.notebook:
            raise ValueError(f"Notebook {self.active_nbid} not found.")
        return self.notebook[self.active_nbid]


@contextmanager
def atomic_adata(adata):
    # 1) Prohibit transactions on views to avoid "lost" write-backs
    if getattr(adata, "is_view", False):
        raise ValueError(
            "adata is a view. Please use adata = adata.copy() first before atomic_adata."
        )

    # 2) Work on a complete copy (deep copy semantics handled by AnnData's own copy implementation)
    work = adata.copy()

    # 3) Mutable container: allows rebinding within the with block (e.g., A = A[:, genes])
    box = {"A": work}

    try:
        # Make all modifications to box["A"] (the working copy) within the with block
        yield box
    except Exception:
        # Error: don't commit, original adata remains unchanged
        raise
    else:
        A = box["A"]
        adata._init_as_actual(
            X=A.X if A.X is not None else None,
            obs=A.obs.copy(deep=True) if A.obs is not None else None,
            var=A.var.copy(deep=True) if A.var is not None else None,
            uns=deepcopy(A.uns) if A.uns is not None else None,
            obsm=A.obsm.copy() if A.obsm is not None else None,
            varm=A.varm.copy() if A.varm is not None else None,
            layers=A.layers.copy() if A.layers is not None else None,
            raw=A.raw.copy() if A.raw is not None else None,
        )
