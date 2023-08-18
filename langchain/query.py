
import  os
from langchain.chains import VectorDBQA
from langchain.vectorstores import Chroma
from langchain.chains import RetrievalQA
from langchain.llms import OpenAI
# import openai embeddings
import openai
import time
from langchain.embeddings import OpenAIEmbeddings
import pandas as pd

persist_directory =  'db'
embedding = OpenAIEmbeddings()

# Now we can load the persisted database from disk, and use it as normal. 
vectordb = Chroma(persist_directory=persist_directory, embedding_function=embedding)

# retriever = vectordb.as_retriever()

# qa = RetrievalQA.from_chain_type(llm=OpenAI(), chain_type="stuff", retriever=retriever)

def paraphrase_ruling(text):
    prompt = """Fasse das folgende Gerichtsurteil zusammen. Pass auf dass du keine Details verlierst. Fokussiere dich
                auf Fakten die spezifisch sind wie das Datum, Geldbeträge, involvierte Personen, Orte, Kosten und am
                wichtigsten das gefällte Urteil. Hier ist das Urteil:""" + text
    response = openai.ChatCompletion.create(
                model="gpt-3.5-turbo-16k-0613", # use gpt-3.5-turbo-0613 for less strict rate limits
                messages=[
                    { "role": "user", "content": prompt },
                ],
                # temperature=0.5,
                max_tokens=2000,
                top_p=0.5,
                n=1,
                frequency_penalty=0,
                presence_penalty=1,
                stop=["\n"]
            )
    return response.choices[0]["message"]["content"]

# load the rulings
rulings = pd.read_csv('data/manually_reidentified_updated.csv')
# load the gound truth
gt = pd.read_csv("data/manually_reidentified_gt.csv", header=None, names=["file_number", "name"])
gts = {}
for index, row in gt.iterrows():
    gts[row["file_number"]] = row["name"]

# for every ruling do a prediction
for index, ruling in rulings.iterrows():
    print(ruling["file_number"])
    # prompt the qa system with the ruling
    ruling_text = ruling["full_text"] # [:10000]
    
    # paraphrase the rulings text to make it fit into the token limit
    paraphrased_ruling = paraphrase_ruling(ruling_text)

    # get the top 5 documents
    documents = vectordb.similarity_search(ruling_text, k=5)

    input = """Wer ist die Person die im folgenden Gerichtsurteil als <mask> bezeichnet wird?
               Die nötigen Informationen findest du in den angehängten Dokumenten.
               Gib den gesamten Namen an, also mit Vor- und Nachname wenn möglich, sonst das was du weisst.\n"""
    input + "Das Urteil:\n\n" + paraphrased_ruling + "\n\n"
    input += "Und die Texte in denen sich die Antwort befindet:\n\n"
    for document in documents:
        input += document.page_content + "\n\n"

    response = openai.ChatCompletion.create(
                model="gpt-4-0613", # use gpt-3.5-turbo-0613 for less strict rate limits
                messages=[
                    { "role": "user", "content": input },
                ],
                # temperature=0.5,
                max_tokens=200,
                top_p=0.1,
                n=5,
                frequency_penalty=0,
                presence_penalty=1,
                stop=["\n"]
            )

    # print the results
    print(f"Prediction: {response.choices[0]['message']['content']} | Target: {gts[ruling['file_number']]}")
    time.sleep(65)
