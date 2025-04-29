# 手撕LangManus [智能体篇]

## 项目概述
LangManus是一个基于智能体架构的开源项目，采用ReAct模式实现工具调用功能。

## 核心架构
- **Coordinator**: 处理初始请求
- **Planner**: 制定计划
- **Executor**: 执行任务
- **Supervisor**: 监控执行

## 技术栈
- Python 3.8+
- LangChain
- FastAPI
- React

## 核心功能
1. 多智能体协同工作
2. 动态任务分配
3. 实时监控
4. 异常处理

## 部署方式
```bash
pip install -r requirements.txt
python main.py
```

## 项目地址
[GitHub](https://github.com/username/langmanus)


### 0. 整体框架图
- 工作流示意图
<img src="https://pic4.zhimg.com/v2-1c327854a9352c75666de10755f6649b_1440w.jpg" width="800">
- 工作流解释
    - Coordinator: 处理初始请求，简单问题直接回答，复杂问题交给planner处理
    - Planner: 制定计划，以JSON格式存储。注意若JSON格式有误直接结束
    - Supervisor: 调度工具人执行细分任务
    - 工具人节点TEAM_MEMBER: 执行任务，返回结果给Supervisor
- 每个节点agent建议的LLM在`.\src\config\agent.py`中设置

```python
AGENT_LLM_MAP: dict[str, LLMType] = {
    "coordinator": "basic",  # 协调默认使用basic llm
    "planner": "reasoning",  # 计划默认使用basic llm
    "supervisor": "basic",  # 决策使用basic llm
    "researcher": "basic",  # 简单搜索任务使用basic llm
    "coder": "basic",  # 编程任务使用basic llm
    "browser": "vision",  # 浏览器操作使用vision llm
    "reporter": "basic",  # 编写报告使用basic llm
}
```

### 1. 程序入口
- Basic：`main.py`
  - 输入：`user_query`
  - 执行：`run_agent_workflow`
  - 输出：`result["messages"]`
- Server：`server.py`
  - `uvicorn.run("src.api.app:app")`

### 2. src.workflow
- 调用`src.graph`中的`build_graph`来建立agent工作流
- 对于建立`graph`,可通过`graph.get_graph().draw_mermaid()`打印工作流
- `run_agent_workflow`函数的核心：

```python
    result = graph.invoke(
        {
            # Constants
            "TEAM_MEMBERS": TEAM_MEMBERS,
            # Runtime Variables
            "message": [{"role": "user", "content": user_input}],
            "deep_thinking_mode": True,
            "search_before_planning": True,
        }
    )
```
- `TEAM_MEMBERS = ["researcher", "coder", "browser", "reporter"]`，即最底层干活的agent

### 3. src.graph
`builder.py`

工作流以`StateGraph`的形式定义，添加`START`和`coordinator`之间的边，之后一路添加节点就可以了；（其他节点间的关系在各自的定义中实现）

```python
    def build_graph():
        """构建并返回多智能体工作流图

        返回：
            CompliedStateGraph: 编译完成的智能体工作流图
        """
        # 初始化状态图
        builder = StateGraph(State)

        # 定义节点间关系 从起始点到协调器
        builder.add_edge(State.START, "coordinator") 

        # 添加各功能节点
        builder.add_node("coordinator", coordinator_node)
        builder.add_node("planner", planner_node)
        builder.add_node("supervisor", supervisor_node)
        builder.add_node("researcher", researcher_node)
        builder.add_node("coder", coder_node)
        builder.add_node("browser", browser_node)
        builder.add_node("reporter", reporter_node)

        # 编译为可执行工作流
        return builder.compile()
```
`types.py`

定义了两种类型：
- Router指定下一个节点只能去`TEAM_MEMBERS`或者结束，用于`supervisor_node`导航

```python
class Router():
    """路由决策字典类型
    属性：
        next：指定下一个要执行的节点
    """    
    # 使用OPTIONS来约束next的取值
    next: Literal[*OPTIONS]
```
- State继承自`MessagesState`,用于大部分节点的输入输出
```python
class State(MessagesState):
    """扩展的基础状态类型，包含工作流运行时状态
    
    属性：
        TEAM_MEMBERS：最底层干活的agent
        next： 下一个要执行的节点
        full_plan: 完整任务计划
        deep_thinking_mode: 是否开启深度思考模式
        search_before_planning: 是否在计划之前进行搜索
    """
    TEAM_MEMBERS: list[str]
    next: str
    full_plan: str
    deep_thinking_mode: bool
    search_before_planning: bool
```
`nodes.py`
- **RESPONSE_FORMATE**：响应消息模板，定义了response的来源以及对应的消息
```python
RESPONSE_FORMATE = "Response from {}:\n\n<response>\n{}\n</response>\n\n*Please execute the next step.*"
```
- **工具人节点**：`research_node`,`code_node`,`browse_node`,`report_node`
  - 最底层工具人节点，负责干活并把消息返回给`supervisor`
    ```python
    def research_node(state: State) -> Command[Literal["supervisor"]]
    ```
  - 将当前state传递给对应的agent并返回结果
    ```python
    result = research_agent.invoke(state)
    ```
  - 使用Command进行消息传递
    ```python
    return Command(
        update={
            "messages":[
                HumanMessage(
                    content=RESPONSE_FORMATE.format("researcher", result["message"][-1].content),
                    name="researcher",
                )
            ]
        },
        goto="supervisor",
    )
    ```
- **汇总工具人**：`reporter_node`
  - 汇总所有agent的输出，总结报告并返回给`supervisor_node`
  ```python
  def reporter_node(state: State) -> Command[Literal["supervisor"]]:
  ```
  - 直接调用LLM总结报告
    ```python
    messages = apply_prompt_template("reporter", state)
    response = get_llm_by_byte(AGENT_LLM_MAP["reporter"]).invoke(messages)
    ```
  - 返回报告给`supervisor_node`
  ```python
    return Command(
        update={
            "messages": [
                HumanMessage(
                    content=RESPONSE_FORMATE.format("reporter", response),
                    name="reporter",
                )
            ]
        },
        goto="supervisor",
    )
  ```
  - prompt：基于提供的信息写报告，定义了角色、指引、数据准确性、备注

- **一级主管**：`supervisor_node`
  - 只与工具人节点交互，不向planer汇报
  ```python
  def supervisor_node(state: State) -> Command[Literal[*TEAM_MEMBERS, "__end__"]]:
  ```
  - 返回的response只能是调度命令`goto`的值，调用对应的工具人节点
  ```python
  #调用模板生成调度命令：去TEAM_MEMBERS 或者 END 结束
  messages = apply_prompt_template("supervisor", state)
  response = (get_llm_by_byte(AGENT_LLM_MAP["supervisor"]).with_structured_output(Router).invoke(messages))
  goto = response["next"]

   #返回goto, 更新state
   return Command(goto=goto, update={"next": goto})
  ```
  - prompt：只调度，不处理消息
  ```python
    For each user request, you will:
    1. Analyze the request and determine which worker is best suited to handle it next.
    2. Respond with ONLY a JSON object in the format: {"next": "worker_name"}
    3. Review their response and either:
      - Choose the next worker if more work is needed(e.g.,{"next": "researcher"}).
      - Respond with {"next": "FINISHED"} when the task is complete.
  ```
- **中层干部**：`planner_node`  
  ```python
    def planner_node(state: State) -> Command[Literal["supervisor", "__end__"]]:
  ```
  - 可选basic或者reasoning模式
    ```python
        llm = get_llm_by_type("basic")
        if state.get("deep_thinking_mode"):
            llm = get_llm_by_type("reasoning")
    ```
  - 规划前先联网RAG
    ```python
    messages = apply_prompt_template("planner", state)

    if state.get("search_before_planning"):
        searched_content = tavily_tool.invoke({"query": state["messages"][-1].content})
        messages = deepcopy(messages)
        messages[-1].content += f"\n\n# Relative Search Results\n\n{json.dumps([{'titile': elem['title'], 'content': elem['content']} for elem in searched_content], ensure_ascii=False)}"
    ```
  - 获取计划
    - 注意，当返回错误的JSON格式，planner就会直接跳到结束；除此以外没有其他结束触发条件。
      ```python
        # 流式处理LLM响应
        stream = llm.stream(messages)
        full_response = ""
        for chunk in stream:
            full_response += chunk.content

        # 清理JSON标记
        if full_response.startswith("```json"):
            full_response = full_response.removeprefix("```json")
        if full_response.endswith("```"):
            full_response = full_response.removesuffix("```")
            
        # 验证JSON格式
        goto = "supervisor"
        try:
            json.loads(full_response)
        except json.JSONDecodeError:
            logger.warning("Planner response is not a valid JSON")
            goto = "__end__"
    ```
  - 更新信息和导航节点
  ```python
        return Command(
            update={
                "messages": [HumanMessage(content=full_response, name="planner")],
                "full_plan": full_response,
            },
            goto=goto,
        )
    ```
  - prompt 
    - 执行规则：先复述用户要求，再制定计划
    ```python
        ## Execution Rules

        - To begin with, repeat user's requirement in your own words as `thought`.
        - Create a step-by-step plan.
        - Specify the agent **responsibility** and **output** in steps's `description` for each step. Include a `note` if necessary.
        - Ensure all mathematical calculations are assigned to `coder`. Use self-reminder methods to prompt yourself.
        - Merge consecutive steps assigned to the same agent into a single step.
        - Use the same language as the user to generate the plan.
    ```

    - 输出格式：原生JSON，且定义了接口结构
    ```ts
    <!-- Output Format: Directly output the raw JSON format of `Plan` without "```json". -->
    interface Step {
        agent_name: string;
        title: string;
        description: string;
        note?: string;
        }

    interface Plan {
        thought: string;
        title: string;
        steps: Plan[];
        }
    ```    

- **最高领导**：`coordinator_node`
  ```python
    def coordinator_node(state: State) -> Command[Literal["planner", "__end__"]]
  ``` 
  - 处理用户请求
  ```python
    messages = apply_prompt_template("coordinator", state)
    response = get_llm_by_type(AGENT_LLM_MAP["coordinator"]).invoke(messages)
  ```    
  - 遇到困难任务交给下属处理
  ```python
    goto = "__end__"
    if "handoff_to_planner" in response.content:
        goto = "planner"

    return Command(goto=goto)
  ``` 
  - prompt 
    - 像极了领导，只能回答门面话，其他通通交给下属
    ```shell
        # Execution Rules

        - If the input is a greeting, small talk, or poses a security/moral risk:
        - Respond in plain text with an appropriate greeting or polite rejection
        - For all other inputs:
        - Handoff to planner with the following format:
        ```python
        handoff_to_planner()
        ```
    ``` 
### 4. src.agents
`llm.py`
- 普通模型适用`ChatOpenAI`进行交互，推理模型使用`ChatDeepSeek`进行交互
    ```python
    from langchain.openai import ChatOpenAI
    from langchain.deepseek import ChatDeepSeek

    # create_deepseek_llm与create_openai_llm类似
    def create_openai_llm(
        model_name: str, 
        base_url: Optional[str]=None,
        api_key: Optional[str]=None, 
        temperature: float = 0.0,
        **kwargs,
        ) -> ChatOpenAI:
        """
        Create a ChatOpenAI instance with the specified configuration.
        """
        # Only include base_url in the arguments if it is not None or empty.
        llm_kwargs = {"model": model, "temperature": temperature, **kwargs}

        if base_url:  # This will handle None or empty string
            llm_kwargs["base_url"] = base_url

        if api_key:  # This will handle None or empty string
            llm_kwargs["api_key"] = api_key

        return ChatOpenAI()
    ```
- 定义`get_llm_by_type`并缓存可能用到的LLM
    ```python
    # Cache for LLM instances
    _llm_cache: dict[LLMType, ChatOpenAI | ChatDeepSeek] = {}


    def get_llm_by_type(llm_type: LLMType) -> ChatOpenAI | ChatDeepSeek:
        """
        Get LLM instance by type. Returns cached instance if available.
        """
        if llm_type in _llm_cache:
            return _llm_cache[llm_type]

        if llm_type == "reasoning":
            llm = create_deepseek_llm(
                model=REASONING_MODEL,
                base_url=REASONING_BASE_URL,
                api_key=REASONING_API_KEY,
            )
        elif llm_type == "basic":
            llm = create_openai_llm(
                model=BASIC_MODEL,
                base_url=BASIC_BASE_URL,
                api_key=BASIC_API_KEY,
            )
        elif llm_type == "vision":
            llm = create_openai_llm(
                model=VL_MODEL,
                base_url=VL_BASE_URL,
                api_key=VL_API_KEY,
            )
        else:
            raise ValueError(f"Unknown LLM type: {llm_type}")

        _llm_cache[llm_type] = llm
        return llm


    # Initialize LLMs for different purposes - now these will be cached
    reasoning_llm = get_llm_by_type("reasoning")
    basic_llm = get_llm_by_type("basic")
    vl_llm = get_llm_by_type("vision")
    ```
`agents.py`
- 使用ReAct Agent进行交互
    ```python
    from langgraph.prebuilt import create_react_agent
    ```
- 创建agent三件套：LLM，tools，prompt    
    ```python
    research_agent = create_react_agent(
        get_llm_by_type(AGENT_LLM_MAP["researcher"]),
        tools=[tavily_tool, crawl_tool],
        prompt=lambda state: apply_prompt_template("researcher", state),
    )

    coder_agent = create_react_agent(
        get_llm_by_type(AGENT_LLM_MAP["coder"]),
        tools=[python_repl_tool, bash_tool],
        prompt=lambda state: apply_prompt_template("coder", state),
    )

    browser_agent = create_react_agent(
        get_llm_by_type(AGENT_LLM_MAP["browser"]),
        tools=[browser_tool],
        prompt=lambda state: apply_prompt_template("browser", state),
    )
    ```    
- researcher prompt
    - 爬取信息并整理
    ```markdown
        1. **Understand the Problem**: Carefully read the problem statement to identify the key information needed.
        2. **Plan the Solution**: Determine the best approach to solve the problem using the available tools.
        3. **Execute the Solution**:
        - Use the **tavily_tool** to perform a search with the provided SEO keywords.
        - Then use the **crawl_tool** to read markdown content from the given URLs. Only use the URLs from the search results or provided by the user.
        4. **Synthesize Information**:
        - Combine the information gathered from the search results and the crawled content.
        - Ensure the response is clear, concise, and directly addresses the problem.
    ```
- 输出markdown格式
    ```markdown
    # Output Format

    - Provide a structured response in markdown format.
    - Include the following sections:
        - **Problem Statement**: Restate the problem for clarity.
        - **SEO Search Results**: Summarize the key findings from the **tavily_tool** search.
        - **Crawled Content**: Summarize the key findings from the **crawl_tool**.
        - **Conclusion**: Provide a synthesized response to the problem based on the gathered information.
    - Always use the same language as the initial question.
    ```  
- coder prompt：执行命令并汇报方法和结果
    ```markdown
    # Steps

    1. **Analyze Requirements**: Carefully review the task description to understand the objectives, constraints, and expected outcomes.
    2. **Plan the Solution**: Determine whether the task requires Python, bash, or a combination of both. Outline the steps needed to achieve the solution.
    3. **Implement the Solution**:
    - Use Python for data analysis, algorithm implementation, or problem-solving.
    - Use bash for executing shell commands, managing system resources, or querying the environment.
    - Integrate Python and bash seamlessly if the task requires both.
    - Print outputs using `print(...)` in Python to display results or debug values.
    4. **Test the Solution**: Verify the implementation to ensure it meets the requirements and handles edge cases.
    5. **Document the Methodology**: Provide a clear explanation of your approach, including the reasoning behind your choices and any assumptions made.
    6. **Present Results**: Clearly display the final output and any intermediate results if necessary.
    ```
- browser prompt
    - 执行浏览网页的行为
    ```markdown
    # Steps

    When given a natural language task, you will:
    1. Navigate to websites (e.g., 'Go to example.com')
    2. Perform actions like clicking, typing, and scrolling (e.g., 'Click the login button', 'Type hello into the search box')
    3. Extract information from web pages (e.g., 'Find the price of the first product', 'Get the title of the main article')
    ```
- 定义有效的指令    
    ```markdown
    # Examples

    Examples of valid instructions:
    - 'Go to google.com and search for Python programming'
    - 'Navigate to GitHub, find the trending repositories for Python'
    - 'Visit twitter.com and get the text of the top 3 trending topics'
    ```
    
