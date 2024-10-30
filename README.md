**Contextual Real-time Intelligent Typo Identifier & Corrector (CRITIC)**

A context-aware spell-check ML System that provides appropriate feedback to the user in real-time after they make an error.

_Problem_

Typos are a common issue since we don't always type what we intend to. Both humans and LLMs are good at understanding context. We need a way to leverage LLMs to automate spell correcting with real-time feedback, low resource utilization, and offline inference.

_Solution_

A language model can give probabilities for different sequences of bytes or tokens, which we can interpret as the probability that a user meant to type something. Using knowledge of how keyboards work and how people make typos, we can also calculate a probability that the user meant to type something, conditioned on what they actually typed.

The combination of these two sources of information should, in theory, give very accurate estimations of what someone wanted to type. Autocorrecting to that predicted actual text can prevent the need for traditional spell checking and avoid missed typos.

_Evaluation_

We will evaluate performance measuring latency of time taken from input to result, optimal resource utilization by monitoring CPU/GPU/memory usage, and correctness of the predicted vs. intended outcome. 

_Feedback_

- Running on CPU vs GPU: GPU inference is clearly preferable for efficient and fast deep learning inference, but they are less common and present challenges for deployment. There are many kinds of consumer GPU. A WebGPU interface would support most devices, but we may need to explicitly target CPU inference instead. That will require different optimizations from a performance perspective.
- Language and input heterogeneity: one student brought up the issue of different languages, and with it different keyboard models. Supporting many languages will be challenging, given that we're limited by the LLMs we can use or train and we can't use a really big multilingual LLM and meet our performance requirements. Keyboard modeling is much easier to do, and if that isn't possible to detect automatically we can allow the user to specify it.

_References_

https://link.springer.com/article/10.1007/s10462-019-09787-4 - discusses how an LLM is used to spell check
https://arxiv.org/abs/2401.13660 - MambaByte, a byte-level LLM we intend to apply for spell checking
https://github.com/nicholas-miklaucic/aispell - Prior work by Nicholas Miklaucic applying a token-level language model

_Risk_

A possible risk of using LLMs to spell check is overcorrection/context misinterpretation. If an LLM does not understand the context such as in a technical field of medicine or engineering, it can change what the user intended to spell. This can be mitigated by incorporating supervised learning where the LLM has oversight and its spell checks can be flagged for review.

CSCE 585 Project: Trevor La, Nicholas Miklaucic, and Kevin Francis
