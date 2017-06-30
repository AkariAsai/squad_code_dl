from pycorenlp import StanfordCoreNLP
import pandas as pd
import re
import json

def split_paragraph_into_sentences(paragraph):
    sentenceEnders = re.compile('[.!?]')
    sentenceList = sentenceEnders.split(paragraph)
    return [sentence for sentence in sentenceList if len(sentence) > 0]

def main():
    nlp = StanfordCoreNLP('http://localhost:9000')
    df_dev = pd.read_csv("train_v1.csv")
    df_tokens = pd.DataFrame()
    index  = 0
    corenlp_json = {}
    total_num = set(df_dev["id"].values)
    for question_id in df_dev["id"].values:
        paragraph = df_dev.loc[df_dev["id"] == question_id, "context"].iloc[0]
        sentences = split_paragraph_into_sentences(paragraph)

        question_json_result = []
        for sentence in sentences:
            output = nlp.annotate(sentence, properties={'annotators': 'tokenize, pos, lemma, ner','outputFormat': 'json'})
            if len(output["sentences"]) > 0:
                question_json_result.append(output["sentences"][0]["tokens"])
        corenlp_json[question_id] = question_json_result
        print(index)
        index+=1
        if index % 10001 == 0:
            with open("corenlp_paragraph_to" + str(index) + ".json", 'w+') as fp:
                json.dump(corenlp_json, fp)
                corenlp_json = {}

if __name__ == '__main__':
    main()
