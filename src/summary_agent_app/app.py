import streamlit as st
from agent import Agent

agent = Agent("sk-None-P0kU2nBBpoRr0ypZHx4xT3BlbkFJKFKbu6YjHU2iHHhV9Kd1")


st.set_page_config(page_title="Sumarizador de Artigos", layout="centered")

st.title('Sumarizador de Artigos')


# Coluna para entrada de texto
request = st.text_area("Insira o texto do seu artigo abaixo e obtenha um resumo conciso do conteúdo.", height=300)

button = st.button("Gerar resumo...")

# Contêiner para o resumo
box = st.container()
with box:
    container = st.empty()
    container.header("Resumo")

if button and request:
    with st.spinner('Gerando resumo...'):
        summary = agent.get_summary(request)
        try:
            container.write(summary["agent_summary"])
        except KeyError:
            container.write("Não foi possível resumir o artigo.")


    
