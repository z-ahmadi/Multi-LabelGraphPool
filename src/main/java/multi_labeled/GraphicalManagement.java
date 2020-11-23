package multi_labeled;


import java.io.BufferedReader;
import java.io.BufferedWriter;
import java.io.File;
import java.io.FileNotFoundException;
import java.io.FileReader;
import java.io.IOException;
import java.io.PrintWriter;
import java.text.DecimalFormat;
import java.util.ArrayList;
import java.util.Iterator;
import java.util.StringTokenizer;

import mulan.data.MultiLabelInstances;
import mulan.evaluation.measure.AveragePrecision;
import mulan.evaluation.measure.Coverage;
import mulan.evaluation.measure.ExampleBasedAccuracy;
import mulan.evaluation.measure.ExampleBasedFMeasure;
import mulan.evaluation.measure.ExampleBasedPrecision;
import mulan.evaluation.measure.ExampleBasedRecall;
import mulan.evaluation.measure.ExampleBasedSpecificity;
import mulan.evaluation.measure.HammingLoss;
import mulan.evaluation.measure.MacroFMeasure;
import mulan.evaluation.measure.MacroPrecision;
import mulan.evaluation.measure.MacroRecall;
import mulan.evaluation.measure.MacroSpecificity;
import mulan.evaluation.measure.Measure;
import mulan.evaluation.measure.MicroAUC;
import mulan.evaluation.measure.MicroFMeasure;
import mulan.evaluation.measure.MicroPrecision;
import mulan.evaluation.measure.MicroRecall;
import mulan.evaluation.measure.MicroSpecificity;
import mulan.evaluation.measure.SubsetAccuracy;
import single_labeled.Measurements;
import weka.classifiers.UpdateableClassifier;
import weka.classifiers.bayes.NaiveBayesUpdateable;
import weka.classifiers.evaluation.Evaluation;
import weka.core.Instances;

public class GraphicalManagement {
	public int maxConcepts;
	static ArrayList<Double> confidenceValues = new ArrayList<Double>();
	
	/**
	 * all the changes for extending the model to the ensemble RACE is kept in Graph and Vertex classes
	 * 
	 * @param dataset: dataset path
	 * 		  windowSize: batch size
	 * 		  outputPath: the path to the directory of output
	 * 		  epsilon: noise removal threshold
	 * 		  MajorityVote: classification mode: weighted vote or single current concept 
	 * 		  label: index of class set {first, last}
	 * 		  similarityThresh: similarity threshold (in comparison of two decoding matrices) 
	 * 		  performanceThresh: threshold for the performance drop
	 * 		  statType: cosine/absolute/{KLdiv/Euclidean}
	 * 		  assignType: minDistance/random/ordered (with the same initialization ordered makes more sense and looks valid)
	 * 		  mergeType: updateLearner/ignoreRest
	 * @throws FileNotFoundException
	 * @throws IOException
	 */
	
	
	public static void main(String[] args) throws Exception {		
		
		String datasetPath = "data/testPYP_0.2_0.8_0.5_n3000_t20_d0.75-shuffled", 
				datasetName = "testPYP_0.2_0.8_0.5_n3000_t20_d0.75-shuffled" , algorithm = "RACE",
				outPath = "results/", run = "1"; //outpath includes / at the end
		int NumHidden = -1;
		int windowSize = -1;
		String hiddenL = "log";
		boolean first = false;
		boolean MajorityVote = false;
		double epsilon = Math.pow(10,-3);
		double simThresh = 0.001, perfThresh = 0.8;
		String statType = "cosine";
		String assignType = "minDistance";
		String mergeType = "updateLearner";
		int iter = 3;
		int model = 5; //used in OECC & Ensemble of RACE
		
		//========= the parameters I dont give the option to be set in input ==========
		UpdateableClassifier uclassifier = new NaiveBayesUpdateable();
		//new NaiveBayesMultinomialUpdateable(), new HoeffdingTree(), new SGD()
		String activationFunc = "HardLim";
		double hardLimThresh = 0;
		double testThresh = 0/*-1, -0.5, 0, 0.5, 1*/;
		double delta = 0.1; 
		int folds = 3;
		//==============================================================================
		
		for(int i=0; i<args.length; i++){
			if(args[i].equals("--dataset")){
				datasetPath = args[++i];
				StringTokenizer stg = new StringTokenizer(datasetPath, "/");
				while(stg.hasMoreTokens()){
					datasetName = stg.nextToken();
				}
			}
			if(args[i].equals("--method"))
				algorithm = args[++i];
			if(args[i].equals("--compress"))
				hiddenL = args[++i];
			if(args[i].equals("--hiddenNeuron"))
				NumHidden = new Integer(args[++i]);
			if(args[i].equals("--windowSize"))
				windowSize = new Integer(args[++i]);
			if(args[i].equals("--outputPath"))
				outPath = args[++i];
			if(args[i].equals("--run"))
				run = args[++i];
			if(args[i].equals("--label"))
				first = new Boolean(args[++i]);
			if(args[i].equals("--iter"))
				iter = new Integer(args[++i]);
			if(args[i].equals("--model"))
				model = new Integer(args[++i]);
			if(args[i].equals("--epsilon"))
				epsilon = new Double(args[++i]);
			if(args[i].equals("--MajorityVote")) //for classification based on neighbor vertices or current one 
				MajorityVote = new Boolean(args[++i]);
			if(args[i].equals("--similarityThresh"))
				simThresh = new Double(args[++i]);
			if(args[i].equals("--performanceThresh"))
					perfThresh = new Double(args[++i]);
			if(args[i].equals("--statType"))
				statType = args[++i];
			if(args[i].equals("--assignType"))
				assignType = args[++i];
			if(args[i].equals("--mergeType"))
				mergeType = args[++i];
				
		}		
		
//		Instances dataSet = new Instances(new FileReader(args[0]));
		MultiLabelInstances dataset = new MultiLabelInstances(datasetPath+".arff", datasetPath+".xml");
		if(datasetPath.contains("nus-wide")){
			Instances inst = new Instances(new FileReader(datasetPath+".arff"));
			inst.deleteAttributeAt(0);
			dataset = new MultiLabelInstances(inst, datasetPath+".xml");
		}
		
		if(NumHidden == -1){		//if NumHidden is not set as the input parameter
			//previous method: the multiplicatives of 5 with minimum of 10
			if(hiddenL.equals("linear")){
				double q = (double)dataset.getNumLabels()/10;
				int q5 = (int) Math.ceil(q/5);
				if(q5*5 > 10)
					NumHidden = q5*5;
				else 
					NumHidden = 10;				
			}else if(hiddenL.equals("log"))	
				NumHidden = (int)Math.ceil(Math.log((double)dataset.getNumLabels())/Math.log(2));
		}
		
		if(windowSize == -1){		//if windowSize is not set as the input parameter
			if(datasetPath.contains("CAL500"))
				windowSize = 50;
			else if(dataset.getNumInstances() <= 5000)
				windowSize = 100;
			else
				windowSize = 500;			
		}						
		
			
//===============================================================================================				
		
		String classifierName = uclassifier.getClass().getName();
		classifierName = classifierName.substring(classifierName.lastIndexOf(".")+1);
		//String outputPath = outPath+datasetName+"_"+windowSize+"_"+classifierName+"/"+NumHidden+"/"+algorithm+"/";
		String outputPath = outPath+datasetName+"_"+windowSize+"_"+classifierName+"_"+model+"_"+iter+"_"+perfThresh+"_"+simThresh+"/"+NumHidden+"/"+algorithm+"/";
		
		System.out.println(outputPath);
		
		File f = new File(outputPath+"ChangeLogs/");
		f.mkdirs();
		f = new File(outputPath+"detailedMeasures/");
		f.mkdirs();
		
		PrintWriter pwDet = new PrintWriter(new File(outputPath+"/detailedMeasures/"+run+".txt"));
		PrintWriter pwChg = new PrintWriter(new File(outputPath+"/ChangeLogs/"+run+".txt"));
//		PrintWriter pwSum = new PrintWriter(new File(outputPath+"/measureSummery.txt"));
		
		int N0 = windowSize;
		
		ArrayList<ArrayList<Double>[]> allmeasures = new ArrayList<ArrayList<Double>[]>(); //an array list of size 1
		ArrayList<Long> times = new ArrayList<Long>(); 

		OnlineLabelReduction OLR = new OnlineLabelReduction(); //moved to here to use its variables later in printing 
		if(dataset.getNumLabels() > NumHidden){
			String newXmlPath = outPath+"reducedXML/Reduced_"+datasetName+"_"+NumHidden+".xml";
			f = new File(outPath+"reducedXML/");
			if(!f.exists()) 
				f.mkdirs(); 
			long olrTime = OLR.runOnDataset(MajorityVote, first, true, dataset, datasetPath+".xml", newXmlPath, outputPath+"misclass.txt", outputPath+"snapshot.txt", pwDet, pwChg, NumHidden, 
											windowSize, N0, uclassifier, iter, model, activationFunc, false, true, hardLimThresh,testThresh,simThresh,epsilon, statType, 
											assignType, mergeType, perfThresh);
			allmeasures.add(OLR.pool.outputMeasures);
			times.add(olrTime);						
		}
		
		System.out.println("pool size = "+OLR.pool.vertexes.size());
		
		//===============================================================================================	
		//write in output file
		String outputPath1 = outputPath+"time/";
		f = new File(outputPath1);
		f.mkdirs();
		PrintWriter pw = new PrintWriter(new File(outputPath1+run+".txt"));
		for(int i=0; i<times.size();i++)
			if(algorithm.equals("offlineCompare") || algorithm.equals("PLST") || algorithm.equals("repeat-RACE"))
				pw.print((double)times.get(i)/1000+" , ");
			else
				pw.print((double)times.get(i)/1000+" ");
		pw.close();
		
		int measureLength = (dataset.getNumInstances()-N0)/windowSize;
		if(algorithm.equals("repeat-RACE"))
			measureLength *= iter;
		
		for(int i=0; i<OLR.m.measures.size(); i++){ //over all measures
			outputPath1 = outputPath+OLR.m.measures.get(i).getName()+"/";
			f = new File(outputPath1);
			f.mkdirs();
			pw = new PrintWriter(new File(outputPath1+run+".txt"));

			for(int m=0; m<measureLength;m++){ //over number of windows
				for(int l=0; l<allmeasures.size();l++){ //over all algorithms
					if(algorithm.equals("offlineCompare") || algorithm.equals("PLST") || algorithm.equals("repeat-RACE") ){
						if((allmeasures.get(l))[i].size() > m)	//added for the case of offline RACE
							pw.print((allmeasures.get(l))[i].get(m)+" , ");
						else
							pw.print(" , ");
					}else{
						pw.print((allmeasures.get(l))[i].get(m)+" ");
					}
				}
				pw.println();
			}
			pw.close();		
		}

		//make the averages of all measures (the folder is outputPath)
		System.out.println(outputPath);
		if(outputPath.endsWith("/"))
			outputPath = outputPath.substring(0, outputPath.length()-1);
		if(algorithm.equals("repeat-RACE")){
			//do nothing!
		}else if(algorithm.equals("offlineCompare") || algorithm.equals("PLST"))
			new AverageOfMeasures().calculateAvgOfOneMethod(outputPath,true);
		else
			new AverageOfMeasures().calculateAvgOfOneMethod(outputPath,false);
		
		pwDet.close();
		pwChg.close();
	}
	
	
	public static int indexOf(Vertex v, ArrayList<Vertex> av){
		int indx = -1;
		for(int i=0; i<av.size();i++){
			if(av.get(i).equals(v)){
				indx = i;
				break;
			}
		}
		
		return indx;
	}
	
	
	/**
	 * write the transpose of a into string (for Matlab use)
	 * @param a
	 * @return
	 */
	public static String arrayToString(double[][] a){
		String ret = "";
		for(int j=0; j<a[0].length; j++){
			for(int i=0; i<a.length; i++){
				ret = ret.concat(a[i][j]+",");
			}
			ret = ret.substring(0, ret.length()-1).concat("\n");
		}
		return ret;
	}

}
