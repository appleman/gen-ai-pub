import os
import logging
from dotenv import load_dotenv
import chainlit as cl
from crewai import Agent, Task, Crew, LLM

load_dotenv()
os.environ["CREWAI_DISABLE_TELEMETRY"] = "true"

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


from langfuse import get_client
 
langfuse = get_client()
 
# Verify connection
if langfuse.auth_check():
    print("Langfuse client is authenticated and ready!")
else:
    print("Authentication failed. Please check your credentials and host.")
 

from openinference.instrumentation.crewai import CrewAIInstrumentor
from openinference.instrumentation.litellm import LiteLLMInstrumentor
 
CrewAIInstrumentor().instrument(skip_dep_check=True)
LiteLLMInstrumentor().instrument()



##################################
# 🧠 LLM Setup
##################################
def get_llm():
    model_id = os.getenv("BEDROCK_MODEL_ID")
    region = os.getenv("AWS_DEFAULT_REGION")
    return LLM(
        model=f"bedrock/{model_id}",
        temperature=0.2,
        max_tokens=512,
        aws_region_name=region
    )

llm = get_llm()

##################################
# 🤖 Pair Programmer Agent
##################################
pair_programmer = Agent(
    role='Pair Programming Buddy',
    goal='Assist with coding, explanations, and advice while remembering context.',
    backstory="A senior dev buddy who helps you write and explain code.",
    verbose=True,
    allow_delegation=False,
    llm=llm,
    max_iter=3
)

##################################
# 📦 Memory Settings
##################################
MAX_MEMORY = 50   # Total stored messages (user + assistant)
TRIM_AT = 0.8     # 80% threshold
KEEP_LAST = 10    # Keep last 10 messages if trimming

##################################
# 🔗 Chainlit Integration
##################################
@cl.on_chat_start
async def on_chat_start():
    session_id = cl.user_session.get("id")
    logger.info(f"🚀 New Chainlit session started: {session_id}")

    # ✅ Start empty memory for session
    cl.user_session.set("memory", [])

    await cl.Message(content="👋 Hey! I'm your Pair Programming Buddy. I’ll remember our conversation!").send()

@cl.on_message
async def on_message(message: cl.Message):
    session_id = cl.user_session.get("id")
    memory = cl.user_session.get("memory")

    logger.info(f"📥 [Session {session_id}] User said: {message.content}")

    # ✅ Add user message to memory
    memory.append({"role": "user", "content": message.content})

    # 🔄 MEMORY MANAGEMENT
    if len(memory) >= MAX_MEMORY * TRIM_AT:
        logger.warning(f"⚠️ Memory reached {len(memory)} messages (~80%). Trimming...")
        memory[:] = memory[-KEEP_LAST:]  # Keep only last 10

    # ✅ Build context string
    full_context = "\n".join([f"{m['role'].capitalize()}: {m['content']}" for m in memory])

    # ✅ Create task
    task = Task(
        description=f"Conversation so far:\n{full_context}\n\nRespond to the latest user query.",
        expected_output="A coding buddy response with context.",
        agent=pair_programmer
    )

    crew = Crew(agents=[pair_programmer], tasks=[task], verbose=True)
    with langfuse.start_as_current_span(name="crewai-index-trace") as span:
    # Pass additional attributes to the span
        result = crew.kickoff()
        span.update_trace(
            input=task.description,
            output=str(result),
            user_id=cl.user_session.get("user"),
            session_id=session_id
        )

    # ✅ Store assistant response in memory
    memory.append({"role": "assistant", "content": result})

    # ✅ Send response back to UI
    await cl.Message(content=f"💻 {result}").send()
