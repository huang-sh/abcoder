# abcoder

Agentic backend coder - A Jupyter notebook manager with MCP (Model Context Protocol) integration for AI-assisted code execution and bioinformatics workflows.

**Supports multiple Jupyter kernels (Python, R, etc.) for parallel notebook management.**

## 🪩 What can it do?

- **Jupyter Notebook Management**: Create, switch between, and manage multiple Jupyter notebooks
- **Multi-kernel Support**: Manage and run code in multiple Jupyter kernels (e.g., Python, R) simultaneously
- **Code Execution**: Execute single-step or multi-step code in Jupyter kernels
- **Variable Backup**: Safely backup variables before code execution to prevent data loss
- **API Documentation**: Query function and API documentation directly from the kernel
- **Output Handling**: Capture and display execution results, errors, and generated figures
- **Bioinformatics Integration**: Designed for bioinformatics workflows with support for common libraries like scanpy, pandas, numpy, etc.

## ❓ Who is this for?

- **Bioinformaticians** who want AI assistance in their Jupyter workflows
- **AI developers** building agents that need to execute code in Jupyter environments
- **Researchers** who want to integrate AI tools with their computational notebooks
- **Anyone** who wants to use natural language to control Jupyter notebook execution

## 🌐 Where to use it?

You can use abcoder in most AI clients, plugins, or agent frameworks that support the MCP:

- **AI clients**: Cherry Studio, Claude Desktop, etc.
- **Plugins**: Cline, etc.
- **Agent frameworks**: Agno, etc.

## 🎬 Demo

A demo showing AI-assisted bioinformatics analysis in a Jupyter notebook using natural language commands through abcoder.

[![abcoder Demo](https://img.youtube.com/vi/3jtXIeapslI/0.jpg)](https://youtu.be/3jtXIeapslI)

**Click the image above to watch the demo video**

## 📚 Documentation

For complete documentation, visit: https://github.com/huang-sh/abcoder

## 🏎️ Quickstart

### Install

Install from PyPI:
```bash
pip install abcoder
```

Test the installation:
```bash
abcoder run
```

### Configuration

#### Run abcoder locally

First, check the installation path:
```bash
which abcoder
# Example output: /home/user/bin/abcoder
```

Configure your MCP client:
```json
{
  "mcpServers": {
    "abcoder": {
      "command": "/home/user/bin/abcoder",
      "args": ["run"]
    }
  }
}
```

#### Run abcoder remotely

Start the server on your remote machine:
```bash
abcoder run --transport shttp --port 8000
```

Configure your local MCP client:
```json
{
  "mcpServers": {
    "abcoder": {
      "url": "http://localhost:8000/mcp"
    }
  }
}
```

## 🛠️ Available Tools

### Notebook Management
- `create_notebook`: Create a new Jupyter notebook with specified ID and path
- `switch_active_notebook`: Switch between different notebooks

### Code Execution
- `single_step_execute`: Execute a single code block
- `multi_step_execute`: Execute multiple code steps with cell addition
- `query_api_doc`: Query function documentation from the kernel

### Features
- **Variable Backup**: Automatically backup variables before execution
- **Error Handling**: Comprehensive error capture and reporting
- **Output Display**: Support for text, images, and other display data
- **Kernel Management**: Automatic kernel lifecycle management


## 🤝 Contributing

If you have any questions, welcome to submit an issue, or contact me(hsh-me@outlook.com). Contributions to the code are also welcome!