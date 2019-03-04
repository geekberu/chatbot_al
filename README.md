# AL/RL based Chatbot
An implementation of Chatbot using Adversarial Learning and Reinforcement Learning
## References
- **Paper**:
[Adversarial Learning for Neural Dialogue Generation](https://arxiv.org/pdf/1701.06547.pdf "Adversarial Learning for Neural Dialogue Generation")
- **Initial implementation**:
Many thanks to [@liuyuemaicha](https://github.com/liuyuemaicha/Adversarial-Learning-for-Neural-Dialogue-Generation-in-Tensorflow "@liuyuemaicha")

## Dependancy
- Python 3.6
- TensorFlow 1.12.0
- nltk
- tqdm
- numpy

Note: You might also need to download additional data to make nltk work
```
python -m nltk.downloader punkt
```

## Corpus
- [Cornell Movie Dialogs Corpus](http://www.cs.cornell.edu/~cristian/Cornell_Movie-Dialogs_Corpus.html "Cornell Movie Dialogs Corpus") (default). Already included when cloning the repository.
- [OpenSubtitles](http://opus.nlpl.eu/OpenSubtitles.php "OpenSubtitles"). Follow the [instructions](corpus/data/opensubs "instructions") and use the argument `--corpus opensubs`.
- Supreme Court Conversation Data. Follow the [instructions](corpus/data/scotus "instructions") and use the argument `--corpus scotus`.
- [Ubuntu Dialogue Corpus](https://arxiv.org/abs/1506.08909 "Ubuntu Dialogue Corpus"). Follow the [instructions](corpus/data/ubuntu "instructions") and use the argument `--corpus ubuntu`.

Many thanks to @Conchylicultor for providing above corpus information.

## Training
`python main.py`

**Follow below steps:**
1. Pre train the Generator and get the GEN_0 model.
2. GEN model test
3. Pre train the Discriminator and get the DISC_0 model
4. Train the GEN model and DISC model using AL/RL
5. GEN model test

## Test 
`python main.py --test 1`

Results

	Q: Hi.
	A: Hey.
	Q: How are you..
	A: Fine.
	Q: What is your name
	A: Harold ...
	Q: You are so nice
	A: I am?
	Q: Wow
	A: What do you mean?
	Q: That's great
	A: That's right.
	Q: Cool
	A: Yeah?
	Q: My name is Amigo
	A: I know.
	Q: What is the first letter of the alphabet ?
	A: What?
	Q: See you later
	A: Yeah.
	Q: Tell me something ...
	A: I'm not saying ...
	Q: You are so cute !!!!!!!!!!
	A: Wonderful!
	Q: Can we make this quick?
	A: Sure.
	Q: Where are you from
	A: Here.
	Q: Merry christmas !
	A: Merry christmas.
	Q: How old are you
	A: Twenty-eight.