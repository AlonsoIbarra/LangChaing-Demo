import streamlit as st
from langchain.llms import OpenAI
from langchain.prompts import PromptTemplate
from langchain.chains import LLMChain, SequentialChain
from langchain.memory import ConversationBufferMemory
from langchain.utilities import WikipediaAPIWrapper

# promp templates
title_template = PromptTemplate(
    input_variables=['topic'],
    template = 'write me a youtube video title aboput {topic}'
)

script_template = PromptTemplate(
    input_variables=['title','wikipedia_research'],
    template = 'write me a youtube video script base on this title: {title} while leaveraging this wilipedia research : {wikipedia_research}'
)

# app framework
st.title('Youtube GPT creator')
prompt = st.text_input('Plug in your topic text here!')

# memory
title_memory = ConversationBufferMemory(
    input_key='topic',
    memory_key='chat_history'
)
script_memory = ConversationBufferMemory(
    input_key='title',
    memory_key='chat_history'
)


# llms
llm = OpenAI(temperature=0.9)
title_chain = LLMChain(
    llm=llm,
    prompt=title_template,
    verbose=True,
    output_key='title',
    memory=title_memory
)
script_chain = LLMChain(
    llm=llm,
    prompt=script_template,
    verbose=True,
    output_key='script',
    memory=script_memory
)

wikipedia = WikipediaAPIWrapper()

if prompt:
    title = title_chain.run(prompt)
    wiki_research = wikipedia.run(prompt)
    script = script_chain.run(title=title, wikipedia_research=wiki_research)

    st.write(title)
    st.write(script)

    with st.expander('Title History'):
        st.info(title_memory.buffer)
    with st.expander('Script History'):
        st.info(script_memory.buffer)
    with st.expander('Wikipedia research'):
        st.info(wiki_research)