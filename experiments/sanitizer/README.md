# Sensitive Dataset
We developed the Sensitive Dataset (including *SensiMarked* and *SensiReplaced*) by sanitizing the sensitive entities or contexts in the [CNN-DailyMail News Text Summarization dataset](https://www.kaggle.com/datasets/gowrishankarp/newspaper-text-summarization-cnn-dailymail).

***SensiMarked***: use spaCy's NER (Named Entity Recognition), DEP (Dependency Parsing) and POS (Part of Speech) [pipelines](https://spacy.io/usage/linguistic-features).

***SensiReplaced***: use the responses from GPT-4 API to fine-tune LM, and further use the fine-tuned LM to automatically generate the sanitized text.

**Prompt & training template**:
```text
Replace given words in the text with other random words.
Text: %s
Given words: %s
Replaced text:
```

**NER types**:

|      Type       | Description                                          |
|:---------------:|:-----------------------------------------------------|
|   `<PERSON>`    | People, including fictional                          |
|     `<GPE>`     | Countries, cities, states                            |
|     `<LOC>`     | Non-GPE locations                                    |
|    `<DATE>`     | Absolute/relative dates or periods                   |
|  `<QUANTITY>`   | Measurements (weight/distance...)                    |
|    `<TIME>`     | Times smaller than a day                             |
|   `<PERCENT>`   | Percentage                                           |
|     `<ORG>`     | Companies, institutions, etc.                        |
|    `<NORP>`     | Nationalities or religious                           |
|    `<MONEY>`    | Monetary values, including unit                      |
|     `<LAW>`     | Named documents made into laws                       |
| `<WORK_OF_ART>` | Titles of books, songs, etc.                         |

**DEP types**: 

|      Type        | Description                                          |
|:----------------:|:-----------------------------------------------------|
|     `<ROOT>`     | The root of the dependency relation                  |
|      `<OBJ>`     | Object                                               |
|     `<SUBJ>`     | Subject                                              |

**POS types**: 

|      Type        | Description                                          |
|:----------------:|:-----------------------------------------------------|
|     `<VERB>`     | Verb                                                 |
|     `<PRON>`     | Pronoun                                              |
|     `<PROPN>`    | Proper Noun                                          |


### Environment Setup

```shell
pip install ltp
pip install langdetect
pip install spacy
python -m spacy download en_core_web_trf
python -m spacy download zh_core_web_trf
```