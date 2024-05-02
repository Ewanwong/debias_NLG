# debias_NLG

This is the implementation of "A Parameter-Efficient Multi-Objective Approach to Mitigate Stereotypical Bias in Language Models". We have incorporated multiple probability alignment objectives to achieve comprehensive bias mitigations while maintaining language abilities of generative language models.


Training data and bias word lists are already in `data/`, run `main.py` to train debiasing prefix using the same hyperparameters as in the paper. 



Evaluation is conducted using codes from [Meade et al.](https://github.com/McGill-NLP/bias-bench) and [Xie and Lukasiewicz](https://github.com/x-zb/pedb).
