from typing import List, Dict, Any, NamedTuple, Optional
from pydantic import BaseModel
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import PydanticOutputParser, JsonOutputParser
from langchain.agents import create_tool_calling_agent, AgentExecutor
from langchain_openai import ChatOpenAI
from langchain.tools import tool
from loguru import logger
from dotenv import load_dotenv
import os
import tempfile
import json
from pathlib import Path
from datetime import datetime
import pandas as pd
import chainlit as cl
from models.filters import Filters, DeconstructedUserQueries
from models.eligibility import CohortProcessor
import traceback
from langchain.memory import ConversationBufferMemory
from langchain_core.messages import HumanMessage, AIMessage

class ToolDetail(NamedTuple):
    tool_name: str
    tool_input: Any
    tool_output: Any

class LegalAssistantTools:
    def __init__(self, llm: ChatOpenAI):
        self.llm = llm
        self._setup_tools()
        self.columns_context: List[str] = []

    def _setup_tools(self):
        """Setup all tools and their associated prompts"""
        self.filters_parser = PydanticOutputParser(pydantic_object=Filters)
        self.deconstruction_parser = PydanticOutputParser(pydantic_object=DeconstructedUserQueries)
        
        self.filters_prompt = ChatPromptTemplate.from_messages([
            (
                "system",
                """You are a legal expert who understands the logical conditions provided in the user prompt. 
                You must determine the parameters of interest and whether these are demographics and offenses related. 
                Answer the user query as JSON. Wrap the output in `json` tags\n{format_instructions}
                Follow the instructions in the final answer:
                    --- Use default values for ALL fields depending on the data type instead of NULL when no value is found.
                    --- To denote conditional operators for numerical values use >, <, ==, !=, >=, <=  
                    --- To denote conditional operators for text or string values in a table, use 'include', 'exclude'
                    --- To denote conditional operators for text or string values compared to another text or string value, us 'exact'
                    --- Categorize offenses type as 'controlling', 'current','prior' or 'unknown'
                    --- Use the list of fields or variables provided in {columns_context} to map with the fields with suffix "_column". Do not introduce any new column or variable names.
                 Consider the conversation history for context: {chat_history}
                 """
            ),
            ("human", "{query}")
        ]).partial(format_instructions=self.filters_parser.get_format_instructions())

        self.classification_prompt = ChatPromptTemplate.from_messages([
            (
                "system",
                """You are a helpful assistant with good understanding of human language, specifically conditional statements. 
                Given a statement in the query, figure out if it is a cohort statement or a general dialogue.
                Consider the conversation history for context: {chat_history}
                Always provide the ouput as a valid JSON object.
                """
            ),
            ("human", "{query}")
        ])

        self.available_tools = [
            tool(self.generate_pydantic_conditions),
            tool(self.do_user_query_classification),
            tool(self.generate_decontructed_subqueries),
            tool(self.explain_pydantic_output)
        ]

    async def generate_pydantic_conditions(self, query: str, chat_history: str = "") -> Dict:
        """Generate Pydantic conditions from user query representing a legal cohort statement"""
        
        logger.debug(f"generate_pydantic_conditions column context {self.columns_context}")
        chain = self.filters_prompt | self.llm | self.filters_parser
        conditions = await chain.ainvoke({
            'query': query, 
            'columns_context': self.columns_context,
            'chat_history': chat_history
        })
        return conditions.model_dump()

    async def do_user_query_classification(self, query: str, chat_history: str = "") -> str:
        """Classify user query as general dialogue or cohort statement"""
        chain = self.classification_prompt | self.llm | JsonOutputParser()
        return await chain.ainvoke({
            "query": query,
            "chat_history": chat_history
        })

    async def generate_decontructed_subqueries(self, query: str, chat_history: str = "") -> DeconstructedUserQueries:
        """Deconstruct user query into sub-queries"""
        deconstruction_prompt = ChatPromptTemplate.from_messages([
            (
                "system",
                """You are a linguistic assistant in legal domain, with expertise in sentence deconstruction. 
                Please break down the user provided query into simple statements based on following categories.
                    --- Demographics, containing details about individuals age, gender, ethnicity, sentenced years etc.
                    --- Offenses, containing mentions of current or prior offences, types of offenses, offense tables etc.
                Consider the conversation history for context: {chat_history}
                Answer the user query as JSON. Wrap the output in `json` tags\n{format_instructions}
                """
            ),
            ("human", "{query}")
        ]).partial(format_instructions=self.deconstruction_parser.get_format_instructions())

        chain = deconstruction_prompt | self.llm | self.deconstruction_parser
        return await chain.ainvoke({
            "query": query,
            "chat_history": chat_history
        })

    async def explain_pydantic_output(self, query: str, json_output: str, chat_history: str = "") -> str:
        """Explain the details of a JSON output"""
        explanation_prompt = ChatPromptTemplate.from_messages([
            (
                "system",
                """ You are an expert in understanding JSON structure. Your job is explain the details of a JSON as {{json_output}} provided in context with respect to the user's original query.
                Please keep note of following considerations in your response -
                1. Keep your output explaination limited to the information provided in the JSON 
                2. While explaining the attibutes, choose the attirbutes with suffix '_column' as opposed to their counterparts.
                3. In your explanation, stick to the values mapped to the attributes of the JSON, DO NOT introduce any artificial value that is not in the JSON.
              Consider the conversation history for context: {chat_history}
                """
            ),
            ("human", "{query}")
        ])
        
        chain = explanation_prompt | self.llm | JsonOutputParser()
        return await chain.ainvoke({
            "query": query, 
            "json_output": json_output,
            "chat_history": chat_history
        })

    def get_tools(self) -> List:
        """Return list of available tools"""
        return self.available_tools


class LegalAssistant:
    def __init__(self):
        load_dotenv()
        os.environ["OPENAI_API_KEY"] = os.getenv('OPENAI_API_KEY')
        self.llm = ChatOpenAI(model_name="gpt-4")
        
        # Initialize tools and data
        self.tools_manager = LegalAssistantTools(self.llm)
        self.tools = self.tools_manager.get_tools()
        self.llm_with_tools = self.llm.bind_tools(self.tools)
        
        # Initialize state
        self.data_df: Optional[pd.DataFrame] = None
        self.columns_context: List[str] = []

        self.columns_context = self.load_columns_context()
         # Initialize memory
        self.memory = ConversationBufferMemory(
            return_messages=True,
            output_key="output",
            input_key="query"
        )
        
        # Create agent and chain
        self.agent = self._create_agent()
        self.chain = self._create_chain()


    def _create_agent(self):
        qa_prompt = ChatPromptTemplate.from_messages([
            (
                "system",
                """You are a specialized legal assistant that analyzes conditional statements and data queries. Your primary role is to interpret and explain conditional logic in the legal context.

                First, use the {{do_user_query_classification}} tool to determine if the input is a conditional statement that is population related or if it is a general dialogue.

                For general dialogues:
                1. Politely indicate that you specialize in data queries and conditional statements analysis
                2. Provide this example: "Show me all individuals who were sentenced to over 20 years for PC666 (petty theft) at the age of 14 or 15"
                3. Request a conditional statement reformulation 

                For cohort statements:
                1. Confirm understanding of the cohort criteria
                2. Using the {{generate_pydantic_conditions}} tool, map the fields in the context to the right values and analyze the JSON output
                3. Provide a focused explanation that:
                - Lists only the relevant conditions found in the JSON
                - Prioritizes fields ending with '_column' when available
                - Maps directly to the values in the JSON structure
                - Relates each point back to the user's original query
                - Omits any field descriptions that are not relevant to the query

                Consider the conversation history for context: {chat_history}
                Keep your explanation concise and query-relevant, avoiding generic structural descriptions of the JSON."""
            ),
            ("human", "{query}"),
            ("placeholder", "{agent_scratchpad}")
        ])
        return create_tool_calling_agent(self.llm_with_tools, self.tools, qa_prompt)

    def _create_chain(self):
        return AgentExecutor.from_agent_and_tools(
            agent=self.agent,
            tools=self.tools,
            verbose=True,
            return_intermediate_steps=True,
            handle_parsing_errors=True
        )
    
    def get_chat_history(self) -> str:
        """Get formatted chat history from memory"""
        messages = self.memory.chat_memory.messages
        formatted_history = []
        
        for message in messages:
            if isinstance(message, HumanMessage):
                formatted_history.append(f"Human: {message.content}")
            elif isinstance(message, AIMessage):
                formatted_history.append(f"Assistant: {message.content}")
                
        return "\n".join(formatted_history)

    def load_columns_context(self) -> None:
        """Load column context from demographics data"""
        try:
            if self.data_df is not None:
                self.columns_context = list(self.data_df.columns)
                self.tools_manager.columns_context = self.columns_context
                logger.info(f"Loaded {len(self.columns_context)} columns from input demographics data")
            else:
                print("Input dataframe is not loaded yet to assign a columns context. A default context will be used instead")
                demographics_df = pd.read_csv("./data/demographics.csv")
                self.columns_context = list(demographics_df.columns)
                self.tools_manager.columns_context = self.columns_context
                logger.info(f"Loaded {len(self.columns_context)} columns from demographics data")
        except Exception as e:
            logger.error(f"Error loading columns context: {str(e)}. No column context will be available.")
            self.columns_context = []

    def load_data(self, file_path: str) -> str:
        """Load data from uploaded file"""
        try:
            self.data_df = pd.read_csv(file_path)
            # new
            self.columns_context = list(self.data_df.columns)
            self.tools_manager.columns_context = self.columns_context
            #
            logger.info(f"Loaded data with columns: {self.data_df.columns}")
            logger.info(f"Updated : {self.data_df.columns}")
            return f"Data loaded successfully with {self.data_df.shape[0]} rows"
        except Exception as e:
            logger.error(f"Error loading data: {str(e)}")
            return "Error loading data file"

    async def process_cohort(self, json_output: str) -> Dict:
        """Process the loaded data and pydantic output to generate the output"""
        try:
            processing_msg = "Processing the data provded uing the filters from the JSON output. This may take sometime please wait."
            msg = cl.Message(content="")
            for char in processing_msg:
                await msg.stream_token(token=char)
            await msg.send()
            
            print(json_output)
            print(type(json_output))
            
            # The processing is done here
            self.cohort_processor = CohortProcessor(
                 self.data_df, 
                 json_output
            )

            self.filtered_df = self.cohort_processor.generate_cohort()
            
            with tempfile.NamedTemporaryFile(mode='w', delete=False, suffix='.csv') as tmp_file:
                self.filtered_df.to_csv(tmp_file.name, index=False)
                tmp_path = Path(tmp_file.name)
    
            elements = [
                    cl.File(
                        name=f"redo_io_filtered_output_data.csv",
                        path=str(tmp_path),
                        display="inline"
                    )
                ]
    
            await cl.Message(
                    content=f"Filtered data ready with {self.filtered_df.shape[0]} rows",
                    elements=elements
                ).send()
        except Exception as e:
            logger.error(f"Cohort processing failed {e} {traceback.print_exc()}")
            await cl.Message(
                    content=f"Sorry, something went wrong with processing the data using above JSON output",
            ).send()

    
    async def process_message(self, message_content: str) -> Dict:
        """Process incoming message and return response"""

        chat_history = self.get_chat_history()
            
        return await self.chain.ainvoke({
            "query": message_content,
            "columns_context": self.columns_context,
            "chat_history": chat_history
        })

    async def handle_pydantic_output(self, tools_involved: Dict[str, ToolDetail]):
        """Handle Pydantic output and create file if needed"""
        
        if 'generate_pydantic_conditions' not in tools_involved:
            logger.error("No output found for the tool generate_pydantic_conditions")

        logger.debug("Processing Pydantic Output")
        tool_output = tools_involved['generate_pydantic_conditions'].tool_output
        self.current_json_output = tool_output
        
        with tempfile.NamedTemporaryFile(mode='w', delete=False, suffix='.json') as tmp_file:
            json.dump(tool_output, tmp_file, indent=2)
            tmp_path = Path(tmp_file.name)

        elements = [
            cl.File(
                name=f"redo_io_cohort_{datetime.now().isoformat()}.json",
                path=str(tmp_path),
                display="inline"
            )
        ]
        
        await cl.Message(
            content="Here is the JSON representation of the cohort that will be applied to the data files provided.",
            elements=elements
        ).send()


# Chainlit handlers
@cl.on_chat_start
async def on_chat_start():
    """Initialize the chat session"""
    legal_assistant = LegalAssistant()
    cl.user_session.set("assistant", legal_assistant)
    
    welcome_msg = "Hello! I'm your legal assistant from Redo.io, specializing in cohort analysis. Please share your data file(s) first and then let me know your cohort criteria."
    msg = cl.Message(content="")

    for char in welcome_msg:
        await msg.stream_token(token=char)
    await msg.send()

    await ask_for_file()

async def ask_for_file():
    """Request file upload from user"""
    files = None
    while files is None:
        files = await cl.AskFileMessage(
            content="Please upload the data file here.", 
            accept=["text/csv"],
            max_size_mb=25,
        ).send()

    legal_assistant = cl.user_session.get("assistant")
    response = legal_assistant.load_data(files[0].path)
    legal_assistant.load_columns_context()
    await cl.Message(content=response).send()

@cl.on_message
async def on_message(message: cl.Message):
    """Handle incoming messages"""
    try:
        legal_assistant = cl.user_session.get("assistant")
        await cl.Message(content="Processing...").send()
        
        msg = cl.Message(content="")
        response = await legal_assistant.process_message(message.content)

        tools_involved = {
            step[0].tool: ToolDetail(
                tool_name=step[0].tool,
                tool_input=step[0].tool_input,
                tool_output=step[1]
            )
            for step in response['intermediate_steps']
        }

        for char in response.get('output'):
            await msg.stream_token(token=char)
        
        pydantic_output_as_json = await legal_assistant.handle_pydantic_output(tools_involved)
        await legal_assistant.process_cohort(legal_assistant.current_json_output)
        await msg.send()

    except Exception as e:
        logger.error(f"Error processing message: {str(e)}")
        await cl.Message(
            content="I encountered an error while processing your request. Please try rephrasing your query."
        ).send()