from langchain.prompts import PromptTemplate
from langchain.chains.question_answering import load_qa_chain
from langchain.prompts import (
    SystemMessagePromptTemplate,
    HumanMessagePromptTemplate,
    ChatPromptTemplate,
    MessagesPlaceholder
)
from langchain.chains import ConversationChain

class Chain:
    def __init__(self, llm, history=None):
        self.llm = llm
        # self.chain = self.get_conversational_chain()
        if history is not None:
            self.history = history

    def run_conversational_chain(self, context, question):

        prompt_template = f"""
        You are an investment banker. 
        Answer the question as detailed as possible from the provided context and make sure to provide all the details.
        Try to answer only from the context as long as possible. 
        Each part of the context has source along with it. Cite that source whenever you use that particular part in the final answer.
        If the answer is not in provided context, try to give a generic answer to the question as an investment banker, don't provide wrong answers.\n\n
        Context:\n {context}\n\n
        Question: \n{question}\n

        Answer:
        """
        ans = self.llm.invoke(prompt_template).content

        return ans
    
    def get_chain_with_history(self):
        system_msg_template = SystemMessagePromptTemplate.from_template(template="""Answer the question as truthfully as possible using the provided context, 
        and if the answer is not contained within the text below, say 'I don't know'""")
        human_msg_template = HumanMessagePromptTemplate.from_template(template="{input}")
        prompt_template = ChatPromptTemplate.from_messages([system_msg_template, MessagesPlaceholder(variable_name="history"), human_msg_template])
        conversation = ConversationChain(memory=self.history, prompt=prompt_template, llm=self.llm, verbose=True)
        return conversation