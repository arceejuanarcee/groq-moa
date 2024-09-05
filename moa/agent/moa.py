import time
from functools import wraps
from typing import Generator, Dict, Optional, Literal, TypedDict, List, Any, Union
from pydantic import BaseModel, Field

from langchain_groq import ChatGroq
from langchain_openai import ChatOpenAI
from langchain_anthropic import ChatAnthropic
from langchain.memory import ConversationBufferMemory
from langchain.prompts import ChatPromptTemplate, HumanMessagePromptTemplate, SystemMessagePromptTemplate
from langchain_core.messages import BaseMessage, HumanMessage, SystemMessage, AIMessage, AIMessageChunk
from langchain_core.runnables import RunnablePassthrough, RunnableLambda, RunnableSerializable
from langchain_core.output_parsers import StrOutputParser
from langchain.text_splitter import TokenTextSplitter
import tiktoken

from .prompts import SYSTEM_PROMPT, REFERENCE_SYSTEM_PROMPT

# Define the ResponseChunk TypedDict
class ResponseChunk(TypedDict):
    delta: str
    response_type: Literal['intermediate', 'output', 'error']
    metadata: Dict[str, Any]

# Define the MOAgentConfig model
class MOAgentConfig(BaseModel):
    main_model: Optional[str] = None
    system_prompt: Optional[str] = None
    cycles: int = Field(...)
    layer_agent_config: Optional[Dict[str, Any]] = None
    reference_system_prompt: Optional[str] = None
    max_tokens: Optional[int] = None

    class Config:
        extra = "allow"

valid_model_names = Literal[
    'llama-3.1-70b-versatile',
    'gemma2-9b-it',
    'gpt-4',  # Ensure this is "gpt-4", not "gpt-4o"
    'gpt-3.5-turbo',
    'claude-3-sonnet-20240229'
]

def retry_on_error(max_retries=3, delay=2):
    def decorator(func):
        @wraps(func)
        def wrapper(*args, **kwargs):
            retries = 0
            while retries < max_retries:
                try:
                    return func(*args, **kwargs)
                except Exception as e:
                    retries += 1
                    if retries == max_retries:
                        raise e
                    time.sleep(delay)
            return func(*args, **kwargs)
        return wrapper
    return decorator

class MOAgent:
    def __init__(
        self,
        main_agent: RunnableSerializable[Dict, str],
        layer_agent: RunnableSerializable[Dict, Dict],
        reference_system_prompt: Optional[str] = None,
        cycles: Optional[int] = None,
        chat_memory: Optional[ConversationBufferMemory] = None,
        transcript: str = "",
        max_tokens: int = 4000,
        overlap: int = 200
    ) -> None:
        self.reference_system_prompt = reference_system_prompt or REFERENCE_SYSTEM_PROMPT
        self.main_agent = main_agent
        self.layer_agent = layer_agent
        self.cycles = cycles or 1
        self.chat_memory = chat_memory or ConversationBufferMemory(
            memory_key="messages",
            return_messages=True
        )
        self.transcript = transcript
        self.max_tokens = max_tokens
        self.overlap = overlap
        self.tokenizer = tiktoken.encoding_for_model("gpt-3.5-turbo")
        self.text_splitter = TokenTextSplitter(chunk_size=self.max_tokens, chunk_overlap=self.overlap)

    @classmethod
    def from_config(cls, main_model, cycles, temperature, system_prompt, reference_system_prompt, layer_agent_config, transcript="", **kwargs):
        main_agent = cls._create_agent_from_system_prompt(
            system_prompt=system_prompt,
            model_name=main_model,
            temperature=temperature,
            **{k: v for k, v in kwargs.items() if k not in ['max_tokens', 'overlap']}
        )

        kwargs_for_layer = {k: v for k, v in kwargs.items() if k not in ['transcript', 'max_tokens', 'overlap']}
        layer_agent = cls._configure_layer_agent(layer_agent_config, **kwargs_for_layer)

        return cls(
            main_agent=main_agent,
            layer_agent=layer_agent,
            reference_system_prompt=reference_system_prompt,
            cycles=cycles,
            transcript=transcript,
            max_tokens=kwargs.get('max_tokens', 4000),
            overlap=kwargs.get('overlap', 200)
        )

    def process_large_input(self, text: str, process_func) -> str:
        chunks = self.text_splitter.split_text(text)
        results = []
        for chunk in chunks:
            result = process_func(chunk)
            results.append(result)
        return " ".join(results)

    def _process_layer(self, layer_agent, input_text, helper_response):
        def process_chunk(chunk):
            llm_inp = {
                'input': chunk,
                'helper_response': helper_response,
                'transcript': self.transcript[:1000]  # Limit transcript size
            }
            layer_output = self._invoke_layer_agent(llm_inp)
            return layer_output['formatted_response']

        return self.process_large_input(input_text, process_chunk)

    @retry_on_error(max_retries=3, delay=2)
    def _invoke_layer_agent(self, llm_inp):
        return self.layer_agent.invoke(llm_inp)

    @retry_on_error(max_retries=3, delay=2)
    def _stream_main_agent(self, llm_inp):
        for chunk in self.main_agent.stream(llm_inp):
            if isinstance(chunk, AIMessageChunk):
                yield chunk.content
            else:
                yield str(chunk)

    def chat(
        self, 
        input: str,
        messages: Optional[List[BaseMessage]] = None,
        cycles: Optional[int] = None,
        save: bool = True,
        output_format: Literal['string', 'json'] = 'string'
    ) -> Generator[Union[str, ResponseChunk], None, None]:
        cycles = cycles or self.cycles
        helper_response = ""
        
        for cyc in range(cycles):
            try:
                layer_output = self._process_layer(self.layer_agent, input, helper_response)
                helper_response = layer_output

                if output_format == 'json':
                    yield ResponseChunk(
                        delta=helper_response,
                        response_type='intermediate',
                        metadata={'layer': cyc + 1}
                    )
            except Exception as e:
                yield ResponseChunk(
                    delta=f"Error in layer {cyc + 1}: {str(e)}",
                    response_type='error',
                    metadata={'layer': cyc + 1}
                )

        try:
            def process_main_chunk(chunk):
                llm_inp = {
                    'input': chunk,
                    'helper_response': helper_response[:1000],  # Limit helper_response size
                    'transcript': self.transcript[:1000]  # Limit transcript size
                }
                result = self._stream_main_agent(llm_inp)
                return ''.join(str(item) for item in result)

            main_output = self.process_large_input(input, process_main_chunk)
            if output_format == 'json':
                yield ResponseChunk(
                    delta=main_output,
                    response_type='output',
                    metadata={}
                )
            else:
                yield main_output

            if save:
                self.chat_memory.save_context({'input': input}, {'output': main_output})
        except Exception as e:
            yield ResponseChunk(
                delta=f"Error in main agent: {str(e)}",
                response_type='error',
                metadata={}
            )

    @staticmethod
    def _create_agent_from_system_prompt(
        system_prompt: str = SYSTEM_PROMPT,
        model_name: str = "gpt-4",
        groq_api_key: Optional[str] = None,
        openai_api_key: Optional[str] = None,
        anthropic_api_key: Optional[str] = None,
        **llm_kwargs
    ) -> RunnableSerializable[Dict, str]:
        if model_name in ['llama-3.1-70b-versatile', 'gemma2-9b-it']:
            llm = ChatGroq(model_name=model_name, groq_api_key=groq_api_key, **llm_kwargs)
        elif model_name in ['gpt-4', 'gpt-3.5-turbo']:
            llm = ChatOpenAI(model_name=model_name, openai_api_key=openai_api_key, **llm_kwargs)
        elif model_name == 'claude-3-sonnet-20240229':
            llm = ChatAnthropic(model=model_name, anthropic_api_key=anthropic_api_key, **llm_kwargs)
        else:
            raise ValueError(f"Unsupported model: {model_name}")

        prompt = ChatPromptTemplate.from_messages([
            SystemMessagePromptTemplate.from_template(system_prompt),
            HumanMessagePromptTemplate.from_template("Transcript: {transcript}\n\nInput: {input}\n\nHelper Response: {helper_response}")
        ])

        return RunnablePassthrough() | prompt | llm | RunnableLambda(MOAgent.concat_response)

    @staticmethod
    def concat_response(
        inputs: Union[Dict[str, str], AIMessage, AIMessageChunk],
        reference_system_prompt: Optional[str] = None
    ):
        reference_system_prompt = reference_system_prompt or REFERENCE_SYSTEM_PROMPT

        if isinstance(inputs, (AIMessage, AIMessageChunk)):
            return {
                'formatted_response': reference_system_prompt.format(helper_response=inputs.content),
                'responses': [inputs.content]
            }

        responses = ""
        res_list = []
        for i, out in enumerate(inputs.values()):
            if isinstance(out, (AIMessage, AIMessageChunk)):
                content = out.content
            else:
                content = str(out)
            responses += f"{i+1}. {content}\n"
            res_list.append(content)

        formatted_prompt = reference_system_prompt.format(helper_response=responses)
        return {
            'formatted_response': formatted_prompt,
            'responses': res_list
        }

    @staticmethod
    def _configure_layer_agent(layer_agent_config: Optional[Dict] = None, **kwargs) -> RunnableSerializable[Dict, Dict]:
        if not layer_agent_config:
            layer_agent_config = {
                'layer_agent_1': {'system_prompt': SYSTEM_PROMPT, 'model_name': 'llama-3.1-70b-versatile'},
                'layer_agent_2': {'system_prompt': SYSTEM_PROMPT, 'model_name': 'gemma2-9b-it'},
                'layer_agent_3': {'system_prompt': SYSTEM_PROMPT, 'model_name': 'gpt-4'},
                'layer_agent_4': {'system_prompt': SYSTEM_PROMPT, 'model_name': 'gpt-3.5-turbo'},
                'layer_agent_5': {'system_prompt': SYSTEM_PROMPT, 'model_name': 'claude-3-sonnet-20240229'}
            }

        layer_agents = {}
        for name, config in layer_agent_config.items():
            layer_config = {**config, **kwargs}
            layer_agents[name] = MOAgent._create_agent_from_system_prompt(**layer_config)

        return RunnablePassthrough() | layer_agents | RunnableLambda(MOAgent.concat_response)
