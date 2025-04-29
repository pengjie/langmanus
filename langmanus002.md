# 手撕LangManus [工具篇]

### 5. src.tools

`decorators.py`

- 该文件定义了与log相关的装饰器
- `log_io`记录函数的输入和输出
```Python
logger.debug(f"Tool {func_name} called with parameters: {params}")
logger.debug(f"Tool {func_name} returned: {result}")
```
- `LoggedToolMinxin`
  - `_log_operation` 记录工具调用操作
```Python
logger.debug(f"Tool {tool_name}.{method_name} called with parameters: {params}")
```
- `_run`: 重写`run`方法，添加日志调用功能
```Python
# 记录方法调用日志
self._log_operation("_run", *args, **kwargs)
# 调用父类的_run方法
result = super()._run(*args, **kwargs)
# 记录方法返回结果日志
logger.debug(f"Tool {self.__class__.__name__.replace('Logged', '')} returned: {result}")
```
- `create_logged_tool`: 基于`LoggedToolMixin`创建具有log功能的工具类型
```Python
class LoggedTool(LoggedToolMixin, base_tool_class):
    pass
# Set a more descriptive name for the class
LoggedTool.__name__ = f"Logged{base_tool_class.__name__}"
return LoggedTool
```

`search.py`
- 调用`TavilySearchResults`定义检索增强工具`tavily_tool`
```Python
from langchain_community.tools.tavily_search import TavilySearchResults

# Initialize Tavily search tool with logging
LoggedTavilySearchResults = create_logged_tool(TavilySearchResults)
tavily_tool = LoggedTavilySearch(name="tavily_search",max_results=TAVILY_MAX_RESULTS)
```

`crawl.py`
- 调用`src.crawler`定义网站爬取工具`crawl_tool`,根据url返回爬取的文章
```Python
@tool
@log_io
def crawl_tool(url: Annotated[str, "The url to crawl."]) -> HumanMessage:
    try:
        crawler = Crawler()
        article = crawler.crawl(url)
        return {"role": "user", "content": article.to_message()}
    except BaseException as e:
        error_msg = f"Failed to crawl. Error: {repr(e)}"
```

`python_repl.py`
-  调用`PythonREPL`定义Python代码执行工具`python_repl_tool`
```Python
from langchain_experimental.untilities import PythonREPL

@tool
@log_io
def python_repl_tool(code: Annotated[str, "The python code to execute to do further analysis or calculation."]):
    """Use this to execute python code and do data ananlysie or calculation. If you want to see the output of a value,
    you should print it out with `print(...)`. This is visible to the user."""
    logger.info(f"Executing code: {code}")
    try:
        result = PythonREPL().run(code)
        logger.info(f"Code execution successful,Result: {result}")
    except BaseException as e:
        error_msg = f"Failed to execute code. Error: {repr(e)}"
        logger.error(error_msg)
        re
    result_str = f"Successfully executed:\n```python\n{code}\n```\nStdout: {result}"
    return result_str
```

`bash_tool.py`
- 调用`subprocess`定义命令行执行工具`bash_tool`
```Python
import subprocess

@tool
@log_io
def bash_tool(cmd: Annotated[str, "The bash command to execute."]):
    """Use this to execute bash command and do necessary operations."""
    logger.info(f"Executing command: {cmd}")
    try:
        # Execute the command and capture the output
        result = subprocess.run(cmd, shell=True,check=True, capture_output=True)
        # Return stdout as the result
        return result.stdout.decode("utf-8")
    except subprocess.CalledProcessError as e:
        # If command fails, return stderr as the result
        error_message = f"Command failed with exit code {e.returncode}.\nStdout: {e.stdout}\nStderr: {e.stderr}"
        logger.error(error_message)
        return error_message
    except Exception as e:
        # Catch any other exceptions
        error_message = f"Error executing command: {str(e)}"
        logger.error(error_message)
        return error_message
```

`file_management.py`
- 调用`WriteFileTool`定义文件写入工具`write_file_tool`，当前版本暂未被agent调用
```Python
from langchain_community.tools.file_management import WriteFileTool

# Initialize file management tool with logging
LoggedWriteFile = create_logged_tool(WriteFileTool)
write_file_tool = LoggedWriteFile()
```

`browser.py`
- 调用`browser_use`定义浏览器操作工具`browser_tool`
```Python
from browser_use import AgentHistoryList, Browser, BrowserConfig
from browser_use import Agent as BrowserAgent
```
- 通过`Browser`创建浏览器实例
```Python
# 如果配置了Chrome实例路径，则创建浏览器实例
if CHROME_INSTANCE_PATH:
    expected_browser = Browser(config=BrowserConfig(chrome_instance_path=CHROME_INSTANCE_PATH))
```
- 定义输入指令的格式
```Python
class BrowserUseInput(BaseModel):
    """Input for WriteFileTool"""
    instructions: str = Field(..., description="The instructions to execute in the browser.")
```
- `BrowserTool`的同步运行方法
```Python
def _run(self, instruction: str) -> str:
    # 创建BrowserAgent实例
    self.agent = BrowserAgent(task=instruction, llm=vl_llm, browser=expected_browser)
    try:
        # 创建新的事件循环（因为这是同步方法）
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        try:
            # 在同步环境中允许异步任务
            result = loop.run_until_complete(self.agent.run())
            # 返回格式化后的结果
            return str(result) if not isinstance(result, AgentHistoryList) else result.final_result
        finally:
            # 关闭事件循环
            loop.close()
    except Exception as e:
        return f"Error executing browser task: {str(e)}"
```
- `BrowserTool`的异步运行方法
```Python
async def _arun(self, instruction: str) -> str:
    self.agent = BrowserAgent(task=instruction, llm=vl_llm)
    try:
        # 直接等待异步任务完成
        result = await self.agent.arun()
        # 返回格式化后的结果
        return str(result) if not isinstance(result, AgentHistoryList) else result.final_result
    except Exception as e:
        return f"Error executing browser task: {str(e)}"
```

### 6. src.crawler

`crawler.py`
- 定义`crawl`方法，输入URL，用于`JinaClient`解析为html，用`ReadabilityExtractor`提取文本（Article类型）
```Python
class Crawler:
    def crawl(self, url: str) -> Article:
        jina_client = JinaClient()
        html = jina_client.crawl(url, return_format="html")
        extractor = ReadabilityExtractor()
        article = extractor.extract(html)
        article.url = url
        logger.info(f"Crawled {url} successfully.")
        return article
```

`jina_client.py`
- 通过`requests`方式请求jina解析网页
```Python
class JinaClient:
    def crawl(self, url: str, return_format: str = "html") -> str:
        headers = {
            "Content-Type": "application/json",
            "X-Return-Format": return_format,
        }
        if os.getenv("JINA_API_KEY"):
            headers["Authorization"] = f"Bearer {os.getenv('JINA_API_KEY')}"
        else:
            logger.warning("JINA_API_KEY not set, using default API key.")
        data = {"url": url}
        response = requests.post("https://r.jina.ai/", headers=headers, json=data)
        return response.text
```

`article.py`

- 初始化：包含titile和html_content
```Python
def __init__(self, title: str, html_content: str):
    self.title = title
    self.html_content = html_content
```
- 方法`to_markdown`：使用`markdownify`库将html_content转换为markdown
```Python
def to_markdown(self, including_title: bool = True) -> str:
    markdown = ""
    if including_title:
        markdown += f"# {self.title}\n\n"
    markdown += markdownify(self.html_content)
    return markdown
```
- 方法`to_message`
```Python
def to_message(self) -> list[dict]:
    # 定义匹配Markdown图片的正则表达式
    image_pattern = r"!\[.*?\]\((.*?)\)"  
    
    # 初始化空内容列表
    content: list[dict[str, str]] = []
    
    # 将Markdown按图片分割成多个部分
    parts = re.split(image_pattern, self.to_markdown())

    # 遍历分割后的各部分
    for i, part in enumerate(parts):
        if i % 2 == 1:  # 奇数索引是图片URL
            # 拼接完整图片URL
            image_url = urljoin(self.url, part.strip())  
            # 添加图片消息
            content.append({"type": "image_url", "image_url": {"url": image_url}})
        else:  # 偶数索引是文本内容
            # 添加文本消息
            content.append({"type": "text", "text": part.strip()})

    return content
```

`readability_extractor.py`
- 使用`simple_json_from_html_string`将Jina返回的html转化为json
    - JSON包含字段：`title`:文章标题，`byline`:作者信息，`date`:发布日期，`content`:文章内容(HTML格式)，`plain_content`:纯文本内容，`plain_text`:分块的文本内容
```Python
from readability import simple_json_from_html_string

class ReadabilityExtractor:
    def extract(self, html: str) -> Article:
        article = simple_json_from_html_string(html, use_readability=True)
        return Article(
            title = article["title"], 
            html_content = article["content"])
```

### 7. server.py
- 通过uvicorn启动FastAPI服务
```Python
uvicorn.run("src.api.app:app", host="0.0.0.0", port=8000, reload=True,log_level="info")
```

### 8. src.api.app
- 构建FastAPI服务
```Python
# 创建FastAPI实例
app = FastAPI(
    title="LangManus API",
    description="基于LangGraph的代理工作流API", 
    version="0.1.0",
)

# 添加CORSMiddleware 跨域支持
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"], # 允许的所有来源
    allow_credentials=True,
    allow_methods=["*"], # 允许的所有HTTP方法
    allow_headers=["*"], # 允许的所有头
)
```
- 初始化工作流
```Python
graph = build_graph()
```
- 定义各种请求类型
  - `ContentItem`存储图片内容
  - `ChatMessage`为常规的{role, content}形式的聊天信息，可包含`ContentItem`
  - `ChatRequest`为聊天请求，包含`ChatMessage`
```Python
class ContentItem(BaseModel):
    """内容项数据模型，支持多种类型内容"""
    type: str = Field(..., description="内容类型(text，image等)")
    text: str = Field(..., description="文本内容")
    image_url: str = Field(None, description="图片URL")

class ChatMessage(BaseModel):
    """聊天消息数据模型"""
    role: str = Field(..., description="消息发送者角色(user，assistant等)")
    content: Union[str, List[ContentItem]] = Field(..., description="消息内容，可以是字符串或或内容项列表")

class ChatRequest(BaseModel):
    """聊天请求数据模型"""
    messages: List[ChatMessage] = Field(..., description="聊天对话历史")
    debug: Option[bool] = Field(False, description="是否开启调试日志")
    deep_thinking_mode: Option[bool] = Field(False, description="是否开启深度思考模式")
    search_before_planning: Option[bool] = Field(False, description="是否在规划之前搜索")   
```
- 提供流式聊天API端点，服务放在`/api/chat/stream`
```Python
@app.post("/api/chat/stream")
async def chat_stream(request: ChatRequest, req: Request):
```
- 转换消息格式为工作流所需格式
```Python
messages = []
for msg in request.messages:
    message_dict = {"role": msg.role}
    if isinstance(msg.content, str):
        # 如果content是字符串，直接取出
        message_dict["content"] = msg.content
    else:
        # 如果content是ContentItem的列表，则逐个解析
        message_dict["content"] = []
        for item in msg.content:
            if item.type == "text" and item.text:
                content_items.append({"type": "text", "text": item.text})
            elif item.type == "image" and item.image_url:
                content_items.append({"type": "image", "image_url": item.image_url})
        message_dict["content"] = content_items
```
- 定义事件生成器
  - 异步调用`src.service.workflow_service.run_agent_workflow`，注意与`src.workflow.run_agent_workflow`区分
```Python 
async def event_generator():
    try:
        async for event in run_agent_workflow(
            messages,
            request.debug,
            request.deep_thinking_mode,
            request.search_before_planning,
        ):
        if await req.is_disconnected():
            logger.info("Client disconnected, stopping workflow")
            break
        yield {
            "event": event["event"],
            "data": json.dumps(event["data"], ensure_ascii=False),
        }
    except asyncio.CancelledError:
        logger.info("Stream processing cancelled")
        raise
```
- 返回发送事件(Server-Sent Events, SSE)的响应类
```Python 
from sse_starlette.sse import EventSourceResponse

return EventSourceResponse(
    event_generator(),
    media_type="text/event-stream",
    sep="\n",
)
```

### 9. src.service.workflow_service.run_agent_workflow
- 定义支持流式响应的LLM代理列表，注意`supervisor`只调度
```Python
streaming_llm_agents = [*TEAM_MEMBERS, "planner", "coordinator"]
```
- 监听事件流
```Python
async def run_agent_workflow(
    user_input_messages: list,
    debug: bool = False,
    deep_thinking_mode: bool = False,
    search_before_planning: bool = False,
)
```
- 下面根据不同情况进行处理
  - 工作流开始
```Python
if kind == "on_chain_start" and name in streaming_llm_agents:
    # 处理链开始事件
    if name == "planner":
        yield {
            "event": "start_of_workflow",
            "data": {"workflow_id": workflow_id, "input": user_input_messages},
        }
    ydata = {
        "event": "start_of_agent",
        "data": {
            "agent_name": name,
            "agent_id": f"{workflow_id}_{name}_{langgraph_step}",
        },
    }
```  
  - 工作流结束
```Python
elif kind == "on_chain_end" and name in streaming_llm_agents:
    # 处理链结束事件
    ydata = {
        "event": "end_of_agent",
        "data": {
            "agent_name": name,
            "agent_id": f"{workflow_id}_{name}_{langgraph_step}",
        },
    }
```  
  - 聊天模型开始和结束
```Python
elif kind == "on_chat_model_start" and node in streaming_llm_agents:
    # 处理聊天模型开始事件
    ydata = {
        "event": "start_of_llm",
        "data": {"agent_name": node},
    }
elif kind == "on_chat_model_end" and node in streaming_llm_agents:
    # 处理聊天模型结束事件
    ydata = {
        "event": "end_of_llm",
        "data": {"agent_name": node},
    }
```  
  - 流式聊天
```Python
elif kind == "on_chat_model_stream" and node in streaming_llm_agents:
    # 处理聊天模型流式输出
    content = data["chunk"].content
    if content is None or content == "":
        if not data["chunk"].additional_kwargs.get("reasoning_content"):
            # 内容为空且思考过程为空，直接跳过
            continue
        # 内容为空但存在思考过程，存储思考过程
        ydata = {
            "event": "message",
            "data": {
                "message_id": data["chunk"].id,
                "delta": {
                    "reasoning_content": (
                        data["chunk"].additional_kwargs["reasoning_content"]
                    )
                },
            },
        }
    else:
        # 处理coordinator缓存历史对话的情况
        if node == "coordinator":
            if len(coordinator_cache) < MAX_CACHE_SIZE:
                coordinator_cache.append(content)
                # coordinator级别的用户输入都缓存到cached_content中
                cached_content = "".join(coordinator_cache)
                # handoff直接跳到结束
                if cached_content.startswith("handoff"):
                    is_handoff_case = True
                    continue
                # 小于MAX_CACHE_SIZE的时候处在流式输出，说明用户的是简单任务，也结束
                if len(coordinator_cache) < MAX_CACHE_SIZE:
                    continue
                ydata = {
                    "event": "message",
                    "data": {
                        "message_id": data["chunk"].id,
                        "delta": {"content": cached_content},
                    },
                }
            elif not is_handoff_case:
                # For other agents, send the message directly
                ydata = {
                    "event": "message",
                    "data": {
                         "message_id": data["chunk"].id,
                         "delta": {"content": content},
                    },
                }
        else:
            # For other agents, send the message directly
            ydata = {
                "event": "message",
                "data": {
                     "message_id": data["chunk"].id,
                       "delta": {"content": content},
                 },
            }
```
  - 工具调用
```Python
elif kind == "on_tool_start" and node in TEAM_MEMBERS:
    # 处理工具调用开始事件
    ydata = {
        "event": "tool_call",
        "data": {
            "tool_call_id": f"{workflow_id}_{node}_{name}_{run_id}",
            "tool_name": name,
            "tool_input": data.get("input"),
        },
    }
elif kind == "on_tool_end" and node in TEAM_MEMBERS:
    # 处理工具调用结束事件
    ydata = {
        "event": "tool_call_result",
        "data": {
            "tool_call_id": f"{workflow_id}_{node}_{name}_{run_id}",
            "tool_name": name,
            "tool_result": data["output"].content if data.get("output") else "",
        },
    }
```  