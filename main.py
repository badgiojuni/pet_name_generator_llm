import langchain_helper as lch
import streamlit as st

st.title("Pets name generator")

user_animal_type = st.sidebar.selectbox("What is your pet?", ("Cat", "Dog", "Cow", "Hamster", 'Dragon', "Unicorn"))

if user_animal_type =="Cat":
    user_pet_color= st.sidebar.text_area(label="What color is your cat?", max_chars=15)

if user_animal_type =="Dog":
    user_pet_color= st.sidebar.text_area(label="What color is your dog?", max_chars=15)

if user_animal_type =="Cow":
    user_pet_color= st.sidebar.text_area(label="What color is your cow?", max_chars=15)

if user_animal_type =="Unicorn":
    user_pet_color= st.sidebar.text_area(label="What color is your Unicorn?", max_chars=15)

if user_animal_type =="Dragon":
    user_pet_color= st.sidebar.text_area(label="What color is your Dragon?", max_chars=15)

if user_animal_type =="Hamster":
    user_pet_color= st.sidebar.text_area(label="What color is your Hamster?", max_chars=15)

if user_pet_color:
    response = lch.generate_pet_name(user_animal_type, user_pet_color)
    st.text(response['pet_name'])