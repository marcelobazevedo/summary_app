from langchain.llms import OpenAI
from langchain.prompts.chat import (ChatPromptTemplate, SystemMessagePromptTemplate, HumanMessagePromptTemplate)

from langchain_openai import ChatOpenAI
from langchain.chains import LLMChain, SequentialChain

import logging

logging.basicConfig(level=logging.INFO)


class SummaryTemplate:
    def __init__(self):
        self.system_template = """
        Você é um assistente de inteligência artificial especializado em resumir textos longos em resumos concisos e informativos. 
        Seu objetivo é ajudar os leitores a compreender rapidamente os pontos principais de qualquer artigo.
        Os textos devem ser resumidos, no entanto, não devem perder a sua essência, não devendo ser omitido fatos, nomes, datas e locais que
        fazem com que o texto tenha a sua relevância.
        Por favor, leia o artigo abaixo com atenção e forneça um resumo claro e abrangente que capture as ideias e pontos principais. O resumo deve ser bem estruturado e fácil de entender, destacando as informações mais importantes sem perder a essência do conteúdo original.

        """
        self.human_template = """
        ####{request}####
        """
        self.system_message_prompt = SystemMessagePromptTemplate.from_template(
            self.system_template)
        self.human_message_prompt = HumanMessagePromptTemplate.from_template(
            self.human_template, input_variables=["request"]
        )
        self.chat_prompt = ChatPromptTemplate.from_messages(
            [self.system_message_prompt, self.human_message_prompt]
        )

class Agent:
    def __init__(
            self,
            open_ai_api_key,
            model='gpt-4-turbo',
            temperature = 0,
            verbose = True
    ):
        self.logger = logging.getLogger(__name__)
        if verbose:
            self.logger.setLevel(logging.INFO)
        self.logger.setLevel(logging.INFO)

        self._openai_key = open_ai_api_key
        self.chat_model = ChatOpenAI(model=model,
                                     temperature=temperature,
                                     openai_api_key=self._openai_key)
        self.verbose = verbose
        

    def get_summary(self, request):
        summary_template = SummaryTemplate()
           
        summary_agent = LLMChain(
            llm=self.chat_model,
            prompt=summary_template.chat_prompt,
            verbose=self.verbose,
            output_key='agent_summary'
        )
        
        overall_chain = SequentialChain(
            chains=[summary_agent],
            input_variables=["request"],
            output_variables=["agent_summary"],
            verbose=self.verbose
        )

        return overall_chain(
            {"request": request},
            return_only_outputs=True
        )