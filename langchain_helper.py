from langchain_community.llms import OpenAI
from langchain.prompts import PromptTemplate
from langchain.chains import LLMChain
from dotenv import load_dotenv
from langchain.agents import load_tools
from langchain.agents import initialize_agent
from langchain.agents import AgentType

load_dotenv()

def generate_pet_name(animal_type, pet_color):
    llm=OpenAI(temperature=0.7)

    prompt_template_name = PromptTemplate(
        input_variables=['animal_type', "pet_color"],
        template="I have a {animal_type} pet and I want a cool name for it. It is {pet_color} SUggest five cool names for it"
    )
    name_chain = LLMChain(llm=llm, prompt=prompt_template_name, output_key="pet_name")

    response = name_chain({'animal_type': animal_type, "pet_color": pet_color})
    return response

def langchain_agent():
    llm = OpenAI(temperature=0.5)

    tools = load_tools(["wikipedia", "llm-math"], llm=llm)

    agent = initialize_agent(tools, llm, agent = AgentType.ZERO_SHOT_REACT_DESCRIPTION, verbose =True)

    results = agent.run(
        "What is the average age of a dog? Multiply the age by 6"
    )

    print(results)

if __name__ == "__main__":
    langchain_agent()
    # print(generate_pet_name('dragon', 'red'))