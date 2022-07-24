# ARC Direct Answer Questions

Release Notes: This version fixes some content and formatting issues with answers from the original release.

A dataset of 2,985 grade-school level, direct-answer science questions derived from the ARC multiple-choice question set released for the [AI2 Reasoning Challenge in 2018](https://allenai.org/data/arc). The ARC Easy and ARC Challenge set questions in the original dataset were combined and then filtered/modified by the following process:

1. Turking: Each of the multiple-choice questions was presented as a direct answer question to five crowdsourced workers to gather additional answers.

2. Heuristic filtering: The questions were filtered based on the following heuristic filters:
  * Questions having a threshold number of turker answers, as a proxy for concreteness of the question. 
  * Questions having at least two turker-provided answers with word overlap, as a measure of confidence in the correctness of the answers, and also straightforwardness of the question.
  * Other heuristics to identify questions that only make sense as multiple-choice questions, such as, questions starting with the phrase "Which of the following".

3. Further manual vetting: We had volunteers in house do another pass of vetting where they:
  * Marked highly open-ended questions with too many answer choices, such as "Name an insect", or otherwise invalid questions, for removal. These are filtered out.
  * Removed some of the bad answers gathered from turking.
  * Reworded questions to make them more suited to direct answer question format, for e.g., a question such as "What element is contained in table salt?" which would make sense as a multiple-choice question, needs be reworded to something like "Name an element present in table salt".
  * Added any additional answers to the questions they could think of that were not present in the turker provided answers.


The dataset consists of 2,985 questions in JSONL format, with the following split:

Train: 1,250
Dev: 338
Test: 1,397


## JSONL Structure

```
{"question_id": "ARCEZ_Mercury_7221148", "tag": "EASY-TRAIN", "question": "A baby kit fox grows to become an adult with a mass of over 3.5 kg. What factor will have the greatest influence on this kit fox's survival?", "answers": ["food availability", "larger predators prevalence", "the availability of food", "the population of predator in the area", "food sources", "habitat", "availability of food", "amount of predators around", "how smart the fox is"]}
```
question_id - a unique identifier for the question; this consists of the original question id from the ARC multiple-choice set, prefixed with 'ARCEZ_' or 'ARCCH_' for ARC Easy and ARC Challenge sets, respectively.
question
tag - the tag as present in the original ARC multiple-choice set. Contains the split.
question - direct-answer question, verbatim or modified via turking/vetting process.
answers - the answer list obtained via turking/vetting process, and the original gold answer.
