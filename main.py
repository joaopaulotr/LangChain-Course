import os

from dotenv import load_dotenv
from langchain_core.prompts import PromptTemplate
from langchain_openai import ChatOpenAI
from langchain_ollama import ChatOllama

# Variáveis de ambiente
load_dotenv()

def main():
    print("Hello from Langchain course!")
    information = """
    Nascido em Londrina, interior do Paraná, na juventude (1973 a 1975) experimentou a vida monástica em um convento da Ordem dos Carmelitas Descalços, mas abandonou a perspectiva de ser monge para seguir a carreira acadêmica. Concluiu sua graduação em 1975 na Faculdade de Filosofia Nossa Senhora Medianeira. Em 1989, concluiu seu mestrado em Educação pela Pontifícia Universidade Católica de São Paulo (PUC-SP), sob a orientação do Prof. Dr. Moacir Gadotti.

    Em 1997, conclui seu Doutorado em Educação, pela PUC-SP, sob orientação do Prof. Dr. Paulo Freire (que morrera na semana anterior à data original da defesa da tese, remarcada para 40 dias depois), "A Escola e o Conhecimento (reflexão sobre fundamentos epistemológicos e políticos dessa relação)", que viraria livro, publicado no ano seguinte (Ed. Cortez).

    É professor titular aposentado da PUC-SP, na qual atuou por 35 anos (de 1977 até 2012), no Departamento de Teologia e Ciências da Religião e Pós-Graduação (Mestrado e Doutorado) em Educação: Currículo, além de professor convidado da Fundação Dom Cabral, desde 1997, e no GVPec da Fundação Getúlio Vargas (FGV-SP), entre 1998 e 2010.

    Apresentou por 11 anos o programa Diálogos Impertinentes na TV PUC (1995–2006).

    Ocupou o cargo de Secretário Municipal de Educação da cidade de São Paulo (1991–1992), durante a administração de Luiza Erundina,[2][3] e foi membro-conselheiro do Conselho Técnico Científico da Educação Básica da CAPES/MEC (2008–2010).[3]

    Em 31 de março de 2012 foi conferido, pela Câmara Municipal de Londrina, seu título de Cidadão Benemérito de Londrina.[4]

    No dia 19 de outubro de 2015, recebeu o título de Cidadão Paulistano outorgado pela Câmara Municipal de São Paulo.[5]

    Em 2017, Cortella foi um dos dez finalistas do Prêmio Darcy Ribeiro de Educação. O Prêmio foi criado para contemplar pessoas ou entidades cujos trabalhos ou ações mereceram destaque especial na defesa e na promoção da educação brasileira.[6][7]

    No início de 2018, adentrou ao mundo das redes sociais, tendo atualmente mais de 22 milhões de inscrições somadas em todos os seus canais oficiais.[8][9][10][11][12]

    No dia 03 de dezembro de 2018 recebeu o título de Cidadão Benemérito do Paraná outorgado pela Assembleia Legislativa do Estado do Paraná.[13]

    Em 1º de agosto de 2022, a Câmara Municipal de Porto Alegre entregou o título de Cidadão de Porto Alegre, ao filósofo e professor Mario Sergio Cortella.[14]

    Na data de 16 de outubro de 2024, recebe a mais alta honraria da Assembleia Legislativa do Espírito Santo, o título de Cidadão Espírito-Santense.[15]

    No dia 18 de outubro de 2025, o Prof. Cortella recebeu a Medalha Cívico-Cultural D.Pedro II do Instituto Histórico e Geográfico de São Paulo, em Sessão Solene comemorativa ao Dia do Professor. [16]
   """
    
    summary_template = """
    Resuma o seguinte texto: {information}, 
    com um pequeno sumario e 2 fatos interessantes sobre ele.
    """
    
    summary_prompt_template = PromptTemplate(
        input_variables=["information"], template=summary_template
        )

    #llm = ChatOllama(model="gemma3:270m", temperature=0) # 0 Menos criativos 1 mais criativos
    llm = ChatOpenAI(model="gpt-5", temperature=0) # 0 Menos criativos 1 mais criativos
    chain = summary_prompt_template | llm # Criando uma corrente usando um prompt template e um modelo de linguagem
    
    response = chain.invoke(input={"information":information}) # Passando a informação para o prompt template e executando a corrente
    print(response.content)


if __name__ == "__main__":
    main()
