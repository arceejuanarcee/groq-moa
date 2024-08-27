from langchain.memory import ConversationBufferMemory
from langchain_core.runnables import RunnableSerializable, RunnablePassthrough, RunnableLambda
from langchain.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.output_parsers import StrOutputParser
from langchain_groq import ChatGroq

class MOAgent:
    def __init__(
        self,
        main_agent: RunnableSerializable,
        layer_agent: RunnableSerializable,
        reference_system_prompt=None,
        cycles=None,
        chat_memory=None
    ):
        self.reference_system_prompt = reference_system_prompt or "Your default reference system prompt here"
        self.main_agent = main_agent
        self.layer_agent = layer_agent
        self.cycles = cycles or 1
        self.chat_memory = chat_memory or ConversationBufferMemory(
            memory_key="messages",
            return_messages=True
        )

    @staticmethod
    def concat_response(inputs, reference_system_prompt=None):
        reference_system_prompt = reference_system_prompt or "Your default reference system prompt here"

        responses = ""
        res_list = []
        for i, out in enumerate(inputs.values()):
            responses += f"{i}. {out}\n"
            res_list.append(out)

        formatted_prompt = reference_system_prompt.format(responses=responses)
        return {
            'formatted_response': formatted_prompt,
            'responses': res_list
        }

    @classmethod
    def from_config(cls, main_model='llama-3.1-405b', system_prompt=None, cycles=1, layer_agent_config=None, reference_system_prompt=None, **main_model_kwargs):
        reference_system_prompt = reference_system_prompt or "Your default reference system prompt here"
        system_prompt = system_prompt or "Your default system prompt here"
        layer_agent = cls._configure_layer_agent(layer_agent_config)
        main_agent = cls._create_agent_from_system_prompt(
            system_prompt=system_prompt,
            model_name=main_model,
            **main_model_kwargs
        )
        return cls(
            main_agent=main_agent,
            layer_agent=layer_agent,
            reference_system_prompt=reference_system_prompt,
            cycles=cycles
        )

    @staticmethod
    def _configure_layer_agent(layer_agent_config=None) -> RunnableSerializable:
        if not layer_agent_config:
            layer_agent_config = {
                'layer_agent_1': {'system_prompt': "Your default system prompt here", 'model_name': 'llama-3.1-405b'},
                'layer_agent_2': {'system_prompt': "Your default system prompt here", 'model_name': 'gemma2-9b-it'},
                'layer_agent_3': {'system_prompt': "Your default system prompt here", 'model_name': 'llama-3.1-70b-versatile'}
            }

        parallel_chain_map = {}
        for key, value in layer_agent_config.items():
            chain = MOAgent._create_agent_from_system_prompt(
                system_prompt=value.pop("system_prompt", "Your default system prompt here"),
                model_name=value.pop("model_name", 'llama-3.1-405b'),
                **value
            )
            parallel_chain_map[key] = RunnablePassthrough() | chain

        chain = parallel_chain_map | RunnableLambda(MOAgent.concat_response)
        return chain

    @staticmethod
    def _create_agent_from_system_prompt(system_prompt: str = "Your default system prompt here", model_name: str = "llama-3.1-405b", **llm_kwargs) -> RunnableSerializable:
        prompt = ChatPromptTemplate.from_messages([
            ("system", system_prompt),
            MessagesPlaceholder(variable_name="messages", optional=True),
            ("human", "{input}")
        ])

        assert 'helper_response' in system_prompt, "system_prompt must contain 'helper_response'"

        llm = ChatGroq(model=model_name, **llm_kwargs)

        chain = prompt | llm | StrOutputParser()
        return chain

    def chat(self, input, transcript=None, brief=None, brief_and_transcript=None, content_strategy=None, research_report=None, RESEARCH_FINDINGS=None, story_strategy=None, composed_content=None, output=None, messages=None, cycles=None, save=True, output_format='string'):
        cycles = cycles or self.cycles
        llm_inp = {
            'input': input,
            'transcript': transcript,
            'brief': brief,
            'brief_and_transcript': brief_and_transcript,
            'content_strategy': content_strategy,
            'research_report': research_report,
            'RESEARCH_FINDINGS': RESEARCH_FINDINGS,
            'story_strategy': story_strategy,
            'composed_content': composed_content,
            'output': output,
            'messages': messages or self.chat_memory.load_memory_variables({})['messages'],
            'helper_response': ""
        }
        for cyc in range(cycles):
            layer_output = self.layer_agent.invoke(llm_inp)
            l_frm_resp = layer_output['formatted_response']
            l_resps = layer_output['responses']

            # Update variables for the next cycle
            if 'brief_and_transcript' in l_resps:
                brief_and_transcript = l_resps['brief_and_transcript']
            if 'brief' in l_resps:
                brief = l_resps['brief']
            if 'content_strategy' in l_resps:
                content_strategy = l_resps['content_strategy']
            if 'research_report' in l_resps:
                research_report = l_resps['research_report']
            if 'RESEARCH_FINDINGS' in l_resps:
                RESEARCH_FINDINGS = l_resps['RESEARCH_FINDINGS']
            if 'story_strategy' in l_resps:
                story_strategy = l_resps['story_strategy']
            if 'composed_content' in l_resps:
                composed_content = l_resps['composed_content']
            if 'output' in l_resps:
                output = l_resps['output']

            llm_inp = {
                'input': input,
                'transcript': transcript,
                'brief': brief,
                'brief_and_transcript': brief_and_transcript,
                'content_strategy': content_strategy,
                'research_report': research_report,
                'RESEARCH_FINDINGS': RESEARCH_FINDINGS,
                'story_strategy': story_strategy,
                'composed_content': composed_content,
                'output': output,
                'messages': self.chat_memory.load_memory_variables({})['messages'],
                'helper_response': l_frm_resp
            }

            if output_format == 'json':
                for l_out in l_resps:
                    yield {
                        'delta': l_out,
                        'response_type': 'intermediate',
                        'metadata': {'layer': cyc + 1}
                    }

        stream = self.main_agent.stream(llm_inp)
        response = ""
        for chunk in stream:
            if output_format == 'json':
                yield {
                    'delta': chunk,
                    'response_type': 'output',
                    'metadata': {}
                }
            else:
                yield chunk
            response += chunk

        if save:
            self.chat_memory.save_context({'input': input}, {'output': response})
