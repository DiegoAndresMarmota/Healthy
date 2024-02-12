import re
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfVectorizer, HashingVectorizer

data = [
    "Frente a la sospecha de un IAM", 
    "es fundamental realizar la confirmación diagnóstica precoz", 
    "ya que la evolución del daño es rápidamente progresiva.", 
    "La necrosis miocárdica se inicia a los 20 a 30 min de la oclusión de la arteria coronaria en la región subendocárdica",
    "y se extiende en forma progresiva en sentido externo hacia la zona subepicárdica.",
    "Así, en un período de 3 hrs, la necrosis compromete al 75 por ciento de la pared del miocardio",
    "y se completa después de las primeras 6 hrs de evolución.",
    "En este contexto, el pronóstico precoz del paciente es determinante,",
    "de manera que el inicio rápido del tratamiento permitirá salvar mayor cantidad de miocardio viable.",
    "Hasta el momento, el análisis del contenido plasmático de Troponina y Creatina-quinasa (CK y CK-MB)",
    "es el mejor y más usado indicador bioquímico de IAM.",
    "La elevación en los niveles de estas proteínas indica que ha ocurrido un IAM, ",
    "mientras que un posterior descenso es indicativo que el IAM ha terminado.",
    "Estos marcadores son liberados durante la destrucción del cardiomiocito",
    "y no se elevan antes de tres o cuatro horas de iniciado el infarto (por ello no permiten su detección precoz)",
    "y, además, no son totalmente exclusivos de este cuadro."
]


# The code is using the CountVectorizer class from the sklearn library to transform the given data
# into a matrix of token counts.

count_vectorizer = CountVectorizer()
count_vectorizer.fit(data)

transformed_data = count_vectorizer.transform(data)
transformed_data.todense()

count_vectorizer.vocabulary_
count_vectorizer.inverse_transform(transformed_data)


# The code is creating a modified version of the CountVectorizer class from the sklearn library.
modified_count_vectorizer = CountVectorizer(
    binary=True,
    max_features=10,
    max_df=1,
    min_df=1,
)
modified_count_vectorizer.fit(data)
modified_count_vectorizer.transform(data).todense()


def selected_word_one(text):
    """
    The function `selected_word_one` takes a text as input and returns a list of words that have two or
    more characters.
    
    :param text: The input text from which you want to extract the selected word
    :return: a list of words that have at least two characters and are separated by a space.
    """
    captcha_word = re.findall(r'\b\w{2, }\b', text)
    return captcha_word


def selected_word_two(text):
    """
    The function `selected_word_two` takes in a text and tries to find specific words in it, and then
    uses a CountVectorizer to transform the data.
    
    :param text: The parameter "text" is a string that represents the text you want to search for in the
    given data
    :return: two values: "find_it_word" and "other_chance".
    """
    try:
        find_it_word = [[ word for word in list if word in text.split()] for text in data.split(', ')]
        other_chance = CountVectorizer(tokenizer=selected_word_one), selected_word_two.fit_transform(data)
    except:
        pass
    finally:
        return find_it_word, other_chance