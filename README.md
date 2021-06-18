<h2 align="center">Hi 👋, I'm Sheikh Md. Faysal</h2>
<h4 align="center">Machine Learning Engineer</h4>

<p align="left"> <img src="https://komarev.com/ghpvc/?username=skfaysal&label=Profile%20views&color=0e75b6&style=flat" alt="skfaysal" /> </p>

- 📫 How to reach me **skmdfaysal@gmail.com**

<h3 align="left">Connect with me:</h3>
<p align="left">
<a href="https://linkedin.com/in/md faysal" target="blank"><img align="center" src="https://raw.githubusercontent.com/rahuldkjain/github-profile-readme-generator/master/src/images/icons/Social/linked-in-alt.svg" alt="md faysal" height="20" width="30" /></a>
<a href="https://kaggle.com/faysal" target="blank"><img align="center" src="https://raw.githubusercontent.com/rahuldkjain/github-profile-readme-generator/master/src/images/icons/Social/kaggle.svg" alt="faysal" height="20" width="30" /></a>
<a href="https://www.hackerrank.com/md faysal" target="blank"><img align="center" src="https://raw.githubusercontent.com/rahuldkjain/github-profile-readme-generator/master/src/images/icons/Social/hackerrank.svg" alt="md faysal" height="20" width="30" /></a>
</p>

<h3 align="left">Languages and Tools:</h3>
<p align="left"> <a href="https://www.docker.com/" target="_blank"> <img src="https://raw.githubusercontent.com/devicons/devicon/master/icons/docker/docker-original-wordmark.svg" alt="docker" width="40" height="40"/> </a> <a href="https://flask.palletsprojects.com/" target="_blank"> <img src="https://www.vectorlogo.zone/logos/pocoo_flask/pocoo_flask-icon.svg" alt="flask" width="40" height="40"/> </a> <a href="https://git-scm.com/" target="_blank"> <img src="https://www.vectorlogo.zone/logos/git-scm/git-scm-icon.svg" alt="git" width="40" height="40"/> </a> <a href="https://www.linux.org/" target="_blank"> <img src="https://raw.githubusercontent.com/devicons/devicon/master/icons/linux/linux-original.svg" alt="linux" width="40" height="40"/> </a> <a href="https://www.python.org" target="_blank"> <img src="https://raw.githubusercontent.com/devicons/devicon/master/icons/python/python-original.svg" alt="python" width="40" height="40"/> </a> <a href="https://scikit-learn.org/" target="_blank"> <img src="https://upload.wikimedia.org/wikipedia/commons/0/05/Scikit_learn_logo_small.svg" alt="scikit_learn" width="40" height="40"/> </a> <a href="https://www.tensorflow.org" target="_blank"> <img src="https://www.vectorlogo.zone/logos/tensorflow/tensorflow-icon.svg" alt="tensorflow" width="40" height="40"/> </a> </p>

## Project Title

<h3 align="center">Model Testing and Monitoring pipeline</h3>

## Project Overview
Monitoring models after deployed into production is one of the most major task when anyone wants to serve a machine learning model to the world. The way model performed in test data doesnt mean it will perform the same for new user data. Data drift and concept drift may occur and decrease the model performence in production.So it's necessary to monitor continuously and create metrics for error analysis then take action accordingly to model.
**N.B: All the models and weights here are kept dummy as it's sensitive and not shareable**.

## Run Locally

Clone the project

```bash
  git clone https://github.com/skfaysal/Model-testing-and-monitoring-pipeline.git
```

Go to the project directory

```bash
  cd Model-testing-and-monitoring-pipeline
```

Create virtual environment using environement.yml

```bash
  conda env create -f environment.yml
```

Activate environment

```bash
  conda activate heat_map
```
For Training Model. We will pass parameters using CLI
```bash
 python3 TestModel_cli.py --drmodel models/b5_newpreprocessed_full_fold4.h5
--lfmodel models/model_binary_right_leaft_retina.h5
--imgdata eyepacs_train --savepath output/
```
For generating confusion matrix and save missclassified images
```bash
  cd confusionMatrix
  python3 main.py
```
