# HANDYMAN CHATBOT

Handy is a chatbot for Handyman Prosthetic, an arm prosthetics company, trained using deep learning on text data, with a limited response set.

This is a maven project where we use deeplearnign4j and StanfordCoreNLP libraries to implement the chatbot.

## Prerequisites
- JDK 1.8 or later (1.8 used in this project)
- Apache Maven
- IntelliJ IDEA/ Eclipse (IntelliJ used in this project)
- Git

## Maven Libraries
- apache.cassandra.all (4.0.5 used)
- deeplearning4j.core (0.9.1 used)
- nd4j.native.platform (0.9.1 used)
- deeplearning4j.vertx (1.0.0-M2.1 used)
- edu.stanford.nlp.corenlp (4.4.0 used)
- opencsv (4.1 used)

## About Deeplearning4j
As the name suggests us, the library is written in Java and is compatible with Java Virtual Machine language, such as Kotlin, Scala etc. It is an open-source distributed deep learning library which has an advantage of the latest distributed computing frameworks such as Apache Spark and Hadoop. Some of its features are:

- Commercial-grade and open-source
- Brings AI to business environments
- Detailed API doc
- Sample projects in multiple languages
- Integrated with Hadoop and Apache Spark

## Further Reading and References
- https://medium.com/datactw/deep-learning-for-java-dl4j-getting-started-tutorial-2259c76c0a7c
- https://deeplearning4j.konduit.ai/multi-project/tutorials/beginners 
- https://www.youtube.com/c/Deeplearning4jSkymind 
- https://stanfordnlp.github.io/CoreNLP/
- https://stackoverflow.com/questions/1578062/lemmatization-java


## Demo conversation
Here's a real sample conversation with Handy:

```
Handy:Hi! I am Handy, AI chatbot of Handyman Prosthetics. How may I help you?
User:tell me about handyman
Handy:Handyman Prosthetics create hand prosthesis and exoskeleton support for people who lost their arms or the ability to control it.
Handy:Our perosonalized solutions enable them to carry out simple day to day tasks without having to rely on others. 
User:show all your products
Handy:Here's our complete catalogue:
Handy:MARK I: For arm amputees. Our simple and light weight model of arm prosthesis. Suitable for simple tasks.
Handy:MARK II: For arm amputees. An advanced and dextrous arm. Involves softroboric for best grip and individual finger control.
Handy:EXO 0: For hand injury rehabilitation and hand paralysis. Comfortable, light-weight and simple.
Handy:EXO I: For hand injury rehabilitation and hand paralysis. Dextrous movements and powerful gripping action.
User:solutions for hand amputees
Handy:Here are our arm prosthesis solutions:
Handy:MARK I: For arm amputees. Our simple and light weight model of arm prosthesis. Suitable for simple tasks.
Handy:MARK II: For arm amputees. An advanced and dextrous arm. Involves softroboric for best grip and individual finger control.
User:video demo of mark ii
Handy:Sure! Here's a video demo: 
Handy:https://youtube.com/playlist?list=PLsE-jsMAf6MONnhrq0_Z4-rIY8xSVyArC
User:i would like to buy mark ii
Handy:Prosthesis is a very personalized solution, differening from user to user. 
Handy:Kindly provide your details here and we will contact you:
Handy:https://forms.gle/C6Em8mSkH5zgbS576
User:ok thank you
Handy:Thank you for contacting us. Have a nice day!
```
