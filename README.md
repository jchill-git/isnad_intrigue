# isnadNameDisambiguation
Contributors: [Joe Hilleary](https://github.com/jchill-git) and [Kyle Sayers](https://github.com/kylesayrs) \
Tufts CS 142: Network Science Final Project

## About Our Project
The goal was to develop a graph-based approach to the problem of disambiguating reference names in isnads (a form of citation chain used in the historic Islamic scholarly tradition). Our work built on that of [Ryan Muther](https://github.com/mutherr) as published in ["From Networks to Named Entities and Back Again: Exploring Classical Arabic Isnad Networks"](https://jhnr.uni.lu/index.php/jhnr/article/view/135). While Muther relied on contextual BERT embeddings in a pure NLP approach, we leveraged social features from a network model of the citations chains within the text of interest.  View our [slides](https://github.com/jchill-git/isnad_intrigue/blob/main/Isnad%20Project%20Presentation.pdf) for a full project overivew.

The disambiguated names and their embeddings can be found in the nameData folder.

The detected communities are found in the communities folder.

The code for evaluating detected communities can be found in the evaluation folder.

Scripts for creating name embedding datasets, constructing mention mebedding networks (which are not themselves included due to file size limitations), and training antecedent classifiers can be found in the scripts folder.
