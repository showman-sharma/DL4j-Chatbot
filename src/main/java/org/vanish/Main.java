package org.vanish;


import com.opencsv.CSVWriter;
import org.apache.log4j.BasicConfigurator;
import org.datavec.api.records.reader.RecordReader;
import org.datavec.api.records.reader.impl.csv.CSVRecordReader;
import org.datavec.api.split.FileSplit;
import org.deeplearning4j.datasets.datavec.RecordReaderDataSetIterator;
import org.deeplearning4j.eval.Evaluation;
import org.deeplearning4j.models.embeddings.loader.WordVectorSerializer;
import org.deeplearning4j.models.word2vec.Word2Vec;
import org.deeplearning4j.nn.conf.MultiLayerConfiguration;
import org.deeplearning4j.nn.conf.NeuralNetConfiguration;
import org.deeplearning4j.nn.conf.layers.DenseLayer;
import org.deeplearning4j.nn.conf.layers.OutputLayer;
import org.deeplearning4j.nn.multilayer.MultiLayerNetwork;
import org.deeplearning4j.nn.weights.WeightInit;
import org.deeplearning4j.optimize.listeners.PerformanceListener;
import org.deeplearning4j.text.sentenceiterator.LineSentenceIterator;
import org.deeplearning4j.text.sentenceiterator.SentenceIterator;
import org.deeplearning4j.text.sentenceiterator.SentencePreProcessor;
import org.deeplearning4j.text.tokenization.tokenizer.preprocessor.CommonPreprocessor;
import org.deeplearning4j.text.tokenization.tokenizerfactory.DefaultTokenizerFactory;
import org.deeplearning4j.text.tokenization.tokenizerfactory.TokenizerFactory;
import org.deeplearning4j.util.ModelSerializer;
import org.nd4j.linalg.activations.Activation;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.dataset.DataSet;
import org.nd4j.linalg.dataset.SplitTestAndTrain;
import org.nd4j.linalg.dataset.api.iterator.DataSetIterator;
import org.nd4j.linalg.factory.Nd4j;
import org.nd4j.linalg.io.ClassPathResource;
import org.nd4j.linalg.learning.config.Adam;
import org.nd4j.linalg.lossfunctions.LossFunctions;

import java.io.*;
import java.util.*;


public class Main {
    public static ArrayList<String> stopWords;
    public static StanfordLemmatizer slem = new StanfordLemmatizer();

    public static int FEATURES_COUNT = 73;
    public static int CLASSES_COUNT = 7;
    public static int NUM_CHATS;

    public static Dictionary reply2Code = new Hashtable();
    public static String[] code2Reply = {"company_info",
            "all_products",
            "prosthesis_products",
            "exoskeleton_product",
            "demo_request",
            "buy_request",
            "end_convo"};
    public static String normalizer_path = "D:\\Chatbot\\DL4J_Chatbot\\src\\main\\resources\\chatbot_model";
    public static String path = "D:\\Chatbot\\DL4J_Chatbot\\src\\main\\resources\\chats.txt";
    public static String chat_path = "D:\\Chatbot\\DL4J_Chatbot\\src\\main\\resources\\chats.csv";
    public static String vec_path = "D:\\Chatbot\\DL4J_Chatbot\\src\\main\\resources\\chatVecs.csv";
    public static String chatbot_model = "D:\\Chatbot\\DL4J_Chatbot\\src\\main\\resources\\chatbot_model";

    public static String replies_path = "D:\\Chatbot\\DL4J_Chatbot\\src\\main\\resources\\bot_replies\\";
    public static Word2Vec trainingVec(String path) throws FileNotFoundException {
        SentenceIterator iter = new LineSentenceIterator(new File(path));
//        SentenceIterator stopWords = new LineSentenceIterator((new File()));


        iter.setPreProcessor(new SentencePreProcessor() {
            @Override
            public String preProcess(String sentence) {
                List<String> tokens = slem.lemmatize(sentence);

                sentence = String.join(" ",tokens);
                return sentence.toLowerCase();
            }
        });



        // Split on white spaces in the line to get words


        //Building model
        TokenizerFactory t = new DefaultTokenizerFactory();
        t.setTokenPreProcessor(new CommonPreprocessor());
        Word2Vec vec = new Word2Vec.Builder()
                .minWordFrequency(2)
                .layerSize(FEATURES_COUNT)
                .seed(42)
                .windowSize(5)
                .iterate(iter)
                .tokenizerFactory(t)
                .stopWords(stopWords)
                .build();

        //("Fitting Word2Vec model....");
        vec.fit();



        // Write word vectors
        try {
            WordVectorSerializer.writeWordVectors(vec, "vec.txt");
        } catch (IOException e) {
            throw new RuntimeException(e);
        }
        return vec;
    }

//    public static ArrayList tokenizeSentence(String sentence){
//        List<String> tokens = slem.lemmatize(sentence);
//        return (ArrayList) tokens;
//    }

    public static INDArray sentenceVector(String sentence, Word2Vec vec){
//        System.out.println(sentence);
//        ArrayList words = tokenizeSentence(sentence);
        List<String> words = slem.lemmatize(sentence);
//        System.out.println(words);
        INDArray sentVect = Nd4j.zeros(1,FEATURES_COUNT);
        for(int i = 0;i< words.size();i++){
//            System.out.println(words.get(i));
            String word = (String)words.get(i);
//            System.out.println(word);
            INDArray e = vec.getWordVectorMatrix(word);
//            System.out.println(words.get(i)+":"+e);
            if(e!=null){
                sentVect = sentVect.add(e);
            }

        }
//        System.out.println("\n");
        return sentVect;
    }

    public static void generateWordVecsFile(String csvFile,String modelFile,Word2Vec vec) {
        try {
            FileWriter outputfile = new FileWriter(modelFile);
            // create CSVWriter object filewriter object as parameter
            CSVWriter writer = new CSVWriter(outputfile);

            File file = new File(csvFile);
            FileReader fr = new FileReader(file);
            BufferedReader br = new BufferedReader(fr);
            String line = "";
            String[] tempArr;
            NUM_CHATS = 0;
            while((line = br.readLine()) != null) {
                tempArr = line.split(",");
                String sentence = (tempArr[0]);
                String out = tempArr[1];
                INDArray sentVec = sentenceVector(sentence,vec);
                int n = sentVec.shape()[1];
                boolean check_ok = true;
                for(int i = 0;i<n;i++){
                    if(sentVec.getDouble(i)==0d){
                        check_ok = false;
                        break;
                    }
                }
                if(check_ok){
                    String[] data = new String[n+1];
                    for(int i=0;i<n;i++){
                        data[i] = String.valueOf(sentVec.getDouble(i));
                    }
                    data[n] = (String) reply2Code.get(out);
                    writer.writeNext(data);
                    NUM_CHATS+=1;
                }
            }
            br.close();
        } catch(IOException ioe) {
            ioe.printStackTrace();
        }
    }

    public static MultiLayerNetwork loadData(String chatVecPath){
        try(RecordReader recordReader = new CSVRecordReader(0,',')) {
            recordReader.initialize(new FileSplit(
                    new ClassPathResource("chatVecs.csv").getFile()
            ));



            //we’ll iterate over the dataset
            DataSetIterator iterator = new RecordReaderDataSetIterator(recordReader,NUM_CHATS-2,FEATURES_COUNT,CLASSES_COUNT );
            DataSet allData = iterator.next();
            allData.shuffle(0);

//            //we’ll normalize the data (fit-transform)
//            DataNormalization normalizer = new NormalizerStandardize();
//            normalizer.fit(allData);
//            normalizer.transform(allData);
//
//            //we'll save the normalizer
//            NormalizerSerializer saver = NormalizerSerializer.getDefault();
//            File normalsFile = new File(normalizer_path);
//            saver.write(normalizer,normalsFile);

            //we split the data
            SplitTestAndTrain testAndTrain = allData.splitTestAndTrain(0.65);
            DataSet trainingData = testAndTrain.getTrain();
            DataSet testingData = testAndTrain.getTest();



            //Running training and testing
            System.out.println("\n\n\nInitiating Deep Learning");
            MultiLayerNetwork model = chatbotNetwork(trainingData, testingData);
            return model;


        } catch (Exception e) {
            Thread.dumpStack();
            new Exception("Stack trace").printStackTrace();
            System.out.println("Error: " + e.getLocalizedMessage());
        }

        return null;
    }

    private static MultiLayerNetwork chatbotNetwork(DataSet trainingData, DataSet testData) {
        System.out.println("\n\n\nModel Configuration");

        int layerSize = 32;
        MultiLayerConfiguration configuration = new NeuralNetConfiguration.Builder()
//                .seed(0)
                .iterations(2500)
                .activation(Activation.RELU)
                .weightInit(WeightInit.XAVIER)
                .updater(new Adam())
                .l2(0.001)
                .list()
                .layer(0, new DenseLayer.Builder().nIn(FEATURES_COUNT).nOut(layerSize).build())
                .layer(1, new DenseLayer.Builder().nIn(layerSize).nOut(layerSize).build())
                .layer(2, new DenseLayer.Builder().nIn(layerSize).nOut(layerSize).build())
                .layer(3, new OutputLayer.Builder(
                        LossFunctions.LossFunction.NEGATIVELOGLIKELIHOOD).activation(Activation.SOFTMAX)
                        .nIn(layerSize).nOut(CLASSES_COUNT).build())
                .backprop(true).pretrain(false)
                .build();

        System.out.println("\n\n\nModel Training");
        MultiLayerNetwork model = new MultiLayerNetwork(configuration);
        model.setListeners(new PerformanceListener(1,true));
        model.init();
        model.fit(trainingData);

        //Evaluating over training data
        System.out.println("\n\nEvaluating over training data");
        INDArray output2 = model.output(trainingData.getFeatureMatrix());
        Evaluation eval2 = new Evaluation(CLASSES_COUNT);
        eval2.eval(trainingData.getLabels(), output2);
        System.out.printf(eval2.stats());

        //Evaluating over test data
        System.out.println("\n\nEvaluating over test data");
        INDArray output = model.output(testData.getFeatureMatrix());
        Evaluation eval = new Evaluation(CLASSES_COUNT);
        eval.eval(testData.getLabels(), output);
        System.out.printf(eval.stats());

        return model;

    }

    public static void SetUp(){
        Scanner s = null;
        try {
            s = new Scanner(new File("D:\\Chatbot\\DL4J_Chatbot\\src\\main\\resources\\stopwords.txt"));
        } catch (FileNotFoundException e) {
            throw new RuntimeException(e);
        }
        stopWords = new ArrayList<String>();
        while (s.hasNext()){
            stopWords.add(s.next().toLowerCase());
        }
        s.close();

        BasicConfigurator.configure();


        try {
            Word2Vec vec = trainingVec(path);
            generateWordVecsFile(chat_path,vec_path,vec);

            MultiLayerNetwork model = loadData(vec_path);
            ModelSerializer.writeModel(model, chatbot_model, true);
        } catch (Exception e) {
            throw new RuntimeException(e);
        }



    }


    public static void main(String[] args) {


        for(int i = 0;i<code2Reply.length;i++){
            reply2Code.put(code2Reply[i],String.valueOf(i));
        }
        System.out.println(reply2Code);

        SetUp(); //Uncomment to train everything again

        try {
            BasicConfigurator.configure();
            Word2Vec word2Vec = WordVectorSerializer.readWord2VecModel("vec.txt");

            MultiLayerNetwork model = ModelSerializer.restoreMultiLayerNetwork(chatbot_model);

//            String input_sentence;
//            Scanner sc = new Scanner(System.in);
//            INDArray input_vector;
//            INDArray output;

//            System.out.println("\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n");
//            BufferedReader br = new BufferedReader(new FileReader(replies_path+"start_convo.txt"));
//            String line;
//            while ((line = br.readLine()) != null) {
//                System.out.println("Handy:"+line);
//            }
//
//            String reply;
//            boolean convo = true;
//            while (convo){
//                System.out.print("User:");
//                input_sentence = sc.nextLine();
//
//                input_vector = sentenceVector(input_sentence,word2Vec);
////            normalizer.transform(input_vector);
//
//                long startTime = System.nanoTime();
//                output = model.output(input_vector);
//                long endTime = System.nanoTime();
//                long duration = (endTime - startTime)/1000000;
//
//                reply = code2Reply[(int) Double.parseDouble(output.argMax().toString())];
//
//                System.out.println("***"+duration+" ms ***");
//                br = new BufferedReader(new FileReader(replies_path+reply+".txt"));
//                while ((line = br.readLine()) != null) {
//                    System.out.println("Handy:"+line);
//                }
//                if(reply=="end_convo"){
//                    convo=false;
//                }
//
//
//            }

        } catch (Exception e) {
            throw new RuntimeException(e);
        }


    }
}