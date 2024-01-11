# coding=utf-8

import re

import jsonlines
from datasets import load_dataset
import spacy

# Load the spaCy English model
nlp = spacy.load("en_core_web_sm")


def paragraph_splitter(paragraph):
    # Process the paragraph with spaCy
    doc = nlp(paragraph)

    # Extract sentences
    sentences = [sent.text for sent in doc.sents]

    return sentences


def backward(sentence):
    # Split paragraph into sentences
    sen_split = paragraph_splitter(sentence)

    if len(sen_split) > 1:
        # Reverse the order of list elements
        sen_split.reverse()

        # Changing the alphabet letters to lowercase in the first sentence
        sen_split[0] = sen_split[0].lower()

        sentence = " ".join(sen_split)
        sentence = "Given the following statements, " + sentence

    return sentence


# Load the dataset
dataset = load_dataset("shuyuej/GSM8K-Consistency")
dataset = dataset["train"]

data = []
num = 0
# Define a regular expression pattern
# pattern = re.compile(r'\n####(.*?)\nThe answer is: ', re.DOTALL)
for example in dataset:
    # if 'SV' not in example["type"] and 'FOBAR' not in example["type"]:
    original_question = example['original_question']
    paraphrased_question = example['paraphrased_question']
    answer_detail = example['answer_detail']
    backward_question = backward(sentence=original_question)

    data.append({"original_question": original_question,
                 "paraphrased_question": paraphrased_question,
                 "backward_question": backward_question,
                 "answer_detail": answer_detail})
    num += 1
    print("Successfully processed", num, "samples")

# data = []
# num = 0
# # Define a regular expression pattern
# pattern = re.compile(r'\n####(.*?)\nThe answer is: ', re.DOTALL)
# for example in dataset:
#     # if 'SV' not in example["type"] and 'FOBAR' not in example["type"]:
#     original_question = example['original_question']
#     paraphrased_question = example['query']
#     answer_detail = example['response']
#
#     if 'SV' in example["type"] or 'FOBAR' in example["type"]:
#         original_question = paraphrased_question
#
#         # Split paragraph into sentences
#         sen_split = paragraph_splitter(example['original_question'])
#         if len(sen_split) > 1:
#             sen_split = sen_split[:-1]
#             paraphrased_question = " ".join(sen_split)
#         else:
#             paraphrased_question = original_question
#
#     # Use the pattern to find the information
#     match = re.search(pattern, answer_detail)
#     if match:
#         info = match.group(1).strip()
#         answer_detail = answer_detail.replace('\n#### ' + info, '')
#
#     backward_question = backward(sentence=original_question)
#
#     data.append({"original_question": original_question,
#                  "paraphrased_question": paraphrased_question,
#                  "backward_question": backward_question,
#                  "answer_detail": answer_detail})
#     num += 1
#     print("Successfully processed", num, "samples")

# Save the modified data to a jsonl file
output_file = 'temporary_consistency_data.jsonl'
with jsonlines.open(output_file, 'w') as writer:
    writer.write_all(data)

print(f"Modified data saved to {output_file}", "the number of data is", num)
