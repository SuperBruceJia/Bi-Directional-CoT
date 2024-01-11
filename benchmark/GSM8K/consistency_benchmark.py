# coding=utf-8

import re
import jsonlines
import datasets
from datasets import load_dataset
import matplotlib.pyplot as plt
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

        # if len(sen_split) > 2:
        #     # Store the first sentence separately
        #     first_sen_split = sen_split[0]
        #
        #     # Shuffle the remaining sentences
        #     shuffled_sen_split = sen_split[1:]
        #     random.shuffle(shuffled_sen_split)
        #
        #     # Combine the first sentence and shuffled sentences
        #     sen_split = [first_sen_split] + shuffled_sen_split

        # Changing the alphabet letters to lowercase in the first sentence
        sen_split[0] = sen_split[0].lower()

        sentence = " ".join(sen_split)
        sentence = "Given the following statements, " + sentence

    return sentence


# Load the dataset
MetaMathQA = load_dataset("meta-math/MetaMathQA")
MetaMathQA = MetaMathQA["train"]

data = []
originals = []
ids = []
# Define a regular expression pattern
pattern = re.compile(r'\n####(.*?)\nThe answer is: ', re.DOTALL)
for example in MetaMathQA:
    type = example['type']
    # {'GSM_Rephrased', 'GSM_AnsAug', 'MATH_SV', 'GSM_FOBAR', 'MATH_FOBAR', 'MATH_AnsAug', 'GSM_SV', 'MATH_Rephrased'}
    if type == "GSM_Rephrased" or type == "GSM_AnsAug" or type == "MATH_AnsAug" or type == "MATH_Rephrased":
        ori_question = example['original_question']
        if ori_question not in originals:
            originals.append(ori_question)

        ques_id = originals.index(ori_question)
        ids.append(ques_id)
        question = example['query']
        answer = example['response']

        # Use the pattern to find the information
        match = re.search(pattern, answer)
        if match:
            info = match.group(1).strip()
            answer = answer.replace('\n#### ' + info, '')

        if type == "GSM_AnsAug" or type == "MATH_AnsAug":
            question = backward(question)

        data.append(
            {
                "id": ques_id,
                "original_question": ori_question,
                "question": question,
                "answer": answer
            }
        )

print(datasets.Dataset.from_list(data))

# Rank the lines by "id"
data = sorted(data, key=lambda x: x["id"])

# # Create a histogram
# plt.hist(ids, bins=max(ids) + 1, edgecolor='black')
#
# # Add labels and title
# plt.xlabel('Value')
# plt.ylabel('Frequency')
# plt.title('Histogram of List a')
#
# # Show the plot
# plt.show()

# Save the modified data to a jsonl file
output_file = 'mathdata_consistency.jsonl'
with jsonlines.open(output_file, 'w') as writer:
    writer.write_all(data)

print(f"Modified data saved to {output_file}")
