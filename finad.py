from datetime import datetime, timedelta
import yfinance as yf
from langchain.prompts import MessagesPlaceholder, ChatPromptTemplate, HumanMessagePromptTemplate, PromptTemplate, AIMessagePromptTemplate
from pydantic import BaseModel, Field
from langchain.tools import BaseTool
from typing import Optional, Type
from typing import List
from langchain.agents.output_parsers import OpenAIFunctionsAgentOutputParser
from langchain.agents import AgentExecutor, Tool
from langchain.agents.format_scratchpad import format_to_openai_function_messages
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.messages import AIMessage, HumanMessage
from langchain.memory import ConversationBufferWindowMemory
from langchain_google_vertexai import ChatVertexAI
from embeddings import Embeddings
from chain import Chain
from llm import LLM
from retriever import Retriever
import streamlit as st

emb = Embeddings("hf", "all-MiniLM-L6-v2")
llm = LLM('gemini').get_llm()
ch = Chain(llm, ConversationBufferWindowMemory(k=0,return_messages=True))
ret = Retriever('pinecone', 'pineconedb', emb.embedding, 'ensemble', 5)

def investment_banker(query):
    context = ret.get_context(query)
    response = ch.run_conversational_chain(context, query)
    return response

def get_stock_price(symbol):
    ticker = yf.Ticker(symbol)
    todays_data = ticker.history(period='1d')
    price = round(todays_data['Close'][0], 2)
    currency = ticker.info['currency']
    return price, currency

def get_stock_data_yahoo(ticker):
    stock = yf.Ticker(ticker)
    data = stock.history(period="1y")
    return data

def get_company_profile_yahoo(ticker):
    stock = yf.Ticker(ticker)
    info = stock.info
    profile = {
        "name": info.get("shortName"),
        "sector": info.get("sector"),
        "industry": info.get("industry"),
        "marketCap": info.get("marketCap"),
        "website": info.get("website"),
        "description": info.get("longBusinessSummary"),
    }
    return profile

def get_company_news_yahoo(ticker):
    stock = yf.Ticker(ticker)
    news = stock.news
    return news

def get_price_change_percent(symbol, days_ago):
    ticker = yf.Ticker(symbol)
    end_date = datetime.now()
    start_date = end_date - timedelta(days=days_ago)

    # Convert dates to string format that yfinance can accept
    start_date = start_date.strftime('%Y-%m-%d')
    end_date = end_date.strftime('%Y-%m-%d')

    historical_data = ticker.history(start=start_date, end=end_date)

    old_price = historical_data['Close'].iloc[0]
    new_price = historical_data['Close'].iloc[-1]

    percent_change = ((new_price - old_price) / old_price) * 100
    return round(percent_change, 2)

def calculate_performance(symbol, days_ago):
    ticker = yf.Ticker(symbol)
    end_date = datetime.now()
    start_date = end_date - timedelta(days=days_ago)
    start_date = start_date.strftime('%Y-%m-%d')
    end_date = end_date.strftime('%Y-%m-%d')
    historical_data = ticker.history(start=start_date, end=end_date)
    old_price = historical_data['Close'].iloc[0]
    new_price = historical_data['Close'].iloc[-1]
    percent_change = ((new_price - old_price) / old_price) * 100
    return round(percent_change, 2)

def get_best_performing(stocks, days_ago):
    best_stock = None
    best_performance = None
    for stock in stocks:
        try:
            performance = calculate_performance(stock, days_ago)
            if best_performance is None or performance > best_performance:
                best_stock = stock
                best_performance = performance
        except Exception as e:
            print(f"Could not calculate performance for {stock}: {e}")
    return best_stock, best_performance

class StockPriceCheckInput(BaseModel):
    """Input for Stock price check."""

    stockticker: str = Field(..., description="Ticker symbol for stock or index")

class StockPriceTool(BaseTool):
    name = "get_stock_ticker_price"
    description = "Useful for when you need to find out the price of the stock today. You should input the stock ticker used on the yfinance API"

    def _run(self, stockticker: str):
        # print("i'm running")
        price_response, currency = get_stock_price(stockticker)

        return f"{currency} {price_response}"

    def _arun(self, stockticker: str):
        raise NotImplementedError("This tool does not support async")

    args_schema: Optional[Type[BaseModel]] = StockPriceCheckInput

class PrevYearStockTool(BaseTool):
    name = "get_past_year_stock_data"
    description = "Useful for when you need to find out the past 1 year performance of a stock. You should input the stock ticker used on the yfinance API"

    def _run(self, stockticker: str):
        price_response = get_stock_data_yahoo(stockticker)
        return price_response

    def _arun(self, stockticker: str):
        raise NotImplementedError("This tool does not support async")

    args_schema: Optional[Type[BaseModel]] = StockPriceCheckInput
    
class StockNewsTool(BaseTool):
    name = "get_news_about_stock"
    description = "Useful for when you need recent news related to a stock. You should input the stock ticker used on the yfinance API"

    def _run(self, stockticker: str):
        price_response = get_company_news_yahoo(stockticker)
        return price_response

    def _arun(self, stockticker: str):
        raise NotImplementedError("This tool does not support async")

    args_schema: Optional[Type[BaseModel]] = StockPriceCheckInput

class StockProfileTool(BaseTool):
    name = "get_profile_of_stock"
    description = "Useful for when you need details or profile of a stock. You should input the stock ticker used on the yfinance API"

    def _run(self, stockticker: str):
        price_response = get_company_profile_yahoo(stockticker)
        return price_response

    def _arun(self, stockticker: str):
        raise NotImplementedError("This tool does not support async")

    args_schema: Optional[Type[BaseModel]] = StockPriceCheckInput

class StockChangePercentageCheckInput(BaseModel):
    """Input for Stock ticker check. for percentage check"""

    stockticker: str = Field(..., description="Ticker symbol for stock or index")
    days_ago: int = Field(..., description="Int number of days to look back")

class StockPercentageChangeTool(BaseTool):
    name = "get_price_change_percent"
    description = "Useful for when you need to find out the percentage change in a stock's value. You should input the stock ticker used on the yfinance API and also input the number of days to check the change over"

    def _run(self, stockticker: str, days_ago: int):
        price_change_response = get_price_change_percent(stockticker, days_ago)

        return price_change_response

    def _arun(self, stockticker: str, days_ago: int):
        raise NotImplementedError("This tool does not support async")

    args_schema: Optional[Type[BaseModel]] = StockChangePercentageCheckInput

class StockBestPerformingInput(BaseModel):
    """Input for Stock ticker check. for percentage check"""

    stocktickers: List[str] = Field(..., description="Ticker symbols for stocks or indices")
    days_ago: int = Field(..., description="Int number of days to look back")

class StockGetBestPerformingTool(BaseTool):
    name = "get_best_performing"
    description = "Useful for when you need to the performance of multiple stocks over a period. You should input a list of stock tickers used on the yfinance API and also input the number of days to check the change over"

    def _run(self, stocktickers: List[str], days_ago: int):
        price_change_response = get_best_performing(stocktickers, days_ago)

        return price_change_response

    def _arun(self, stockticker: List[str], days_ago: int):
        raise NotImplementedError("This tool does not support async")

    args_schema: Optional[Type[BaseModel]] = StockBestPerformingInput

class InvestmentBankerInput(BaseModel):
    """Input for Investment Banker."""

    query: str = Field(..., description="User question as it is.")

class InvestmentBankerTool(BaseTool):
    name = "get_info_as_from_an_investment_banker"
    description = "Useful when you need to find answers to generic or senario based finance questions. It may or may not be specific to one company."

    def _run(self, query: str):
        response = investment_banker(query)
        return response

    def _arun(self, stockticker: str):
        raise NotImplementedError("This tool does not support async")

    args_schema: Optional[Type[BaseModel]] = InvestmentBankerInput

tools = [InvestmentBankerTool(),StockPriceTool(),StockPercentageChangeTool(), 
         StockGetBestPerformingTool(), PrevYearStockTool(), StockNewsTool(), 
         StockProfileTool()]

prompt = ChatPromptTemplate(
    input_variables= [
        "agent_scratchpad",
        "chat_history",
        "input"
    ],
    messages=[
        HumanMessagePromptTemplate(
            prompt=PromptTemplate(
                input_variables=[],
                template=(
                    """Answer the questions as best you can as a financial consultant. 
                    If you are not sure say 'I am not sure.'
                    If Sources are present in the agent_scratchpad, return it as citation in the final answer."""
                ),
            ),
        ),
        AIMessagePromptTemplate(
            prompt=PromptTemplate(
                input_variables=[],
                template=("Understood!")
            )
        ),
        MessagesPlaceholder(variable_name='chat_history'),
        HumanMessagePromptTemplate(
            prompt=PromptTemplate(
                input_variables=["input"],
                template="{input}"
            ),
        ),
        MessagesPlaceholder(variable_name='agent_scratchpad'),
    ],
)

# llm = ChatGoogleGenerativeAI(temperature=0, model='gemini-pro')
llm_with_tools = llm.bind(functions = tools)

agent = (
    {
        "input": lambda x:x["input"],
        "chat_history": lambda x: x["chat_history"],
        "agent_scratchpad": lambda x: format_to_openai_function_messages(
            x["intermediate_steps"]
        ),
    }
    | prompt
    | llm_with_tools
    | OpenAIFunctionsAgentOutputParser()
)

memory=ConversationBufferWindowMemory(k=2, memory_key='chat_history', output_key='output', return_messages=True)
agent_executor = AgentExecutor(agent=agent, tools=tools, verbose=True, memory=memory)
cnt = 0

st.set_page_config(page_title="Financial Adviser Assistant")
st.header("Financial Adviser Assistant")

# Create a form for input to handle Enter key submission
with st.form(key='finance_form'):
    input_text = st.text_input("How may I assist you with your financial inquiries today?", key="input")
    form_submit = st.form_submit_button("Submit")
if form_submit:
    try:
        ans = agent_executor.invoke({'input': f"{input_text}"})
    except:
        ans = {}
        ans['output'] = 'Could not get the info based on my knowledge. Try again with more info.'
        memory.clear()
        cnt=0
    st.write(f"Response: {ans['output']}")
    cnt+=1
    if cnt==3:
        memory.clear()
        cnt=0