from pycorenlp import StanfordCoreNLP
import pandas as pd
import re
import json

def main():
    nlp = StanfordCoreNLP('http://localhost:9000')
    df_dev = pd.read_csv("train_v1.csv")
    df_tokens = pd.DataFrame()
    for index, row in df_dev.iterrows():
        if index == 76027:
            df_tokens_paragraph = pd.DataFrame([row["id"], "gabage"], columns=["question_id", "token"])
        else:
            print(index)
            text = (row["question"])
            output = nlp.annotate(text, properties={'annotators': 'tokenize, ssplit, pos, lemma, ner, depparse','outputFormat': 'json'})
            tokens = [output['sentences'][0]['tokens'][i]["word"] for i in range(len(output['sentences'][0]['tokens']))]
            question_id = row["id"]
            df_tokens_paragraph = pd.DataFrame([[question_id, token] for token in tokens], columns=["question_id", "token"])
            df_tokens = df_tokens.append(df_tokens_paragraph, ignore_index=True)

            if index % 10001 == 0:
                df_tokens.to_csv("df_question_tokens_to" + str(index) + ".csv")
                df_tokens = pd.DataFrame()

def split_paragraph_into_sentences(paragraph):
    sentenceEnders = re.compile('[.!?]')
    sentenceList = sentenceEnders.split(paragraph)
    return [sentence for sentence in sentenceList if len(sentence) > 0]

def experiment():
    nlp = StanfordCoreNLP('http://localhost:9000')
    df_dev = pd.read_csv("train_v1.csv")
    corenlp_json = {}
    df_tokens = pd.DataFrame()
    index  = 0
    for question_id in set(df_dev["id"].values):
        paragraph = df_dev.loc[df_dev["id"] == question_id, "context"].iloc[0]
        sentences = split_paragraph_into_sentences(paragraph)

        for sentence in sentences:
            output = nlp.annotate(sentence, properties={'annotators': 'tokenize, pos, lemma, ner','outputFormat': 'json'})
            if question_id not in corenlp_json:
                corenlp_json[question_id] =  output["sentences"][0]["tokens"]
            else:
                corenlp_json[question_id].append(output["sentences"][0]["tokens"])
        with open("corenlp_paragraph.json", 'a') as fp:
            json.dump(corenlp_json, fp)
        print(index)
        index+=1
    #tokens = [output['sentences'][0]['tokens'][i]["word"] for i in range(len(output['sentences'][0]['tokens']))]
if __name__ == '__main__':
    # main()
    experiment()
