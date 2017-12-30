# keras_mn
Memory networks are used for automatic question answering.

The model is trained using story, question, answer triplets and tested using story, question to predict the answer.

## Requirements
* nltk==3.2.3
* matplotlib==2.0.2
* Keras==2.0.6
* numpy==1.12.1

## Train/Test data
bAbI Tasks Data 1-20 (v1.2) https://research.fb.com/downloads/babi/

## Test result
The model is tested against all 10,000 samples in test data. The first 10 results show that only one mis-prediction happens.

| Story | Question | Answer | Prediction |
| ------------- | ------------- | ------------- | ------------- |
| john travelled to the hallway . mary journeyed to the bathroom . | where is john ? | hallway | hallway |
| daniel went back to the bathroom . john moved to the bedroom . | where is mary ? | bathroom | bedroom |
| john went to the hallway . sandra journeyed to the kitchen . | where is sandra ? | kitchen | kitchen |
| sandra travelled to the hallway . john went to the garden . | where is sandra ? | hallway | hallway |
| sandra went back to the bathroom . sandra moved to the kitchen . | where is sandra ? | kitchen | kitchen |
| sandra travelled to the kitchen . sandra travelled to the hallway . | where is sandra ? | hallway | hallway |
| mary went to the bathroom . sandra moved to the garden . | where is sandra ? | garden | garden |
| sandra travelled to the office . daniel journeyed to the hallway . | where is daniel ? | hallway | hallway |
| daniel journeyed to the office . john moved to the hallway . | where is sandra ? | office | office |
| john travelled to the bathroom . john journeyed to the office . | where is daniel ? | office | kitchen |

## Model plot
<p>
  <img src="https://github.com/shuuchen/keras_mn/blob/master/model.png" />
</p>

## License
Released under the MIT license. See LICENSE for details.
