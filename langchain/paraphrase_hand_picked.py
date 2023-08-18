
import openai
import pandas as pd
from tqdm.auto import tqdm

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
ruling_texts = []
# for every ruling do a prediction
for index, ruling in tqdm(rulings.iterrows(), total=len(rulings)):
    ruling_text = ruling["full_text"]
    # paraphrase the rulings text to make it fit into the token limit
    paraphrased_ruling = paraphrase_ruling(ruling_text)

    # save paraphrased version to rulings as well
    ruling_texts.append(paraphrased_ruling)

# append the paraphrased rulings
rulings['masked_text_paraphrased'] = ruling_texts
# rename full text
rulings.rename(columns={"full_text": "masked_text_original", "decision_id": "id" }, inplace=True)

rulings.to_csv('data/manually_reidentified_updated_with_paraphrased.csv', index=False)