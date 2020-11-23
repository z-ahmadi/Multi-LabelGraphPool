package multi_labeled;
import java.io.File;
import java.io.FileNotFoundException;
import java.io.PrintWriter;
import java.util.ArrayList;
import java.util.Iterator;

import weka.classifiers.UpdateableClassifier;
import weka.core.Instances;
import mulan.classifier.MultiLabelLearner;
import mulan.data.InvalidDataFormatException;
import mulan.data.MultiLabelInstances;
import mulan.evaluation.Evaluation;
import mulan.evaluation.Evaluator;
import mulan.evaluation.measure.Measure;
import multi_labeled.RACE.LRUpdateable;


public class OnlineLabelReduction {
	
	static String[][] betaStr; 
	
//	static ArrayList<Measure> measures = new ArrayList<Measure>();
	static Measurements m;
	static ArrayList<MultiLabelInstances> batchesOfMultiLabelData; 
	static Graph pool; 
	
	//keep the snapshot of pool in every batch for later presentations and plotting!
	static ArrayList<ArrayList<double[][][]>> CRinHistory = new ArrayList<ArrayList<double[][][]>>(); 
		
	public static LRUpdateable LR;

	static double[][][] init_IW; 
	static double[][] init_bias;
	
	public OnlineLabelReduction(){
		
	}
	
	public static void makeBetaString(double[][] curB){
		for(int i=0; i<curB.length; i++){
			for(int j=0; j<curB[i].length; j++){
				betaStr[i][j] = betaStr[i][j]+","+String.format( "%.2f", curB[i][j]);
			}
		}
	}
	
	public static long runOnDataset(boolean clsMode, boolean first, boolean gram, MultiLabelInstances dataset, String xmlPath, String newXmlPath, String threshPath, String snapshop,
									PrintWriter pwDet, PrintWriter pwChg, int NumHiddenNeuron, int window, int N0, UpdateableClassifier baseClassifier, int repIter, 
									int ensembleSize, String activationFun, boolean adaptive, boolean shallowAE, double Hthresh, double Tthresh, double sim, double epsilon, String statType, 
									String assignType, String mergeType, double perfTresh) throws Exception{
				
		makeReducedDataset(dataset, xmlPath, newXmlPath, NumHiddenNeuron, N0, window);
		

		m = new Measurements(batchesOfMultiLabelData.get(0).getNumLabels());
		pool = new Graph(ensembleSize, sim, m, clsMode, Tthresh, pwDet, pwChg);	
		
		long t1 = System.currentTimeMillis();
		
		init_IW = new double[ensembleSize][][];
		init_bias = new double[ensembleSize][];
		for(int i=0; i<ensembleSize; i++) {
			LR = new LRUpdateable(first, gram, baseClassifier, m.measures.size(), dataset.getNumLabels(), NumHiddenNeuron, xmlPath, newXmlPath, activationFun, Hthresh, Tthresh,sim);
			init_IW[i] = LR.getIW();
			init_bias[i] = LR.getBias();			
		}
		
		///////////////
		PrintWriter pw = new PrintWriter(new File(threshPath));
		PrintWriter pwsp = new PrintWriter(new File(snapshop));
		boolean drop = false; 
		//////////////
		
		for(int i=0; i<batchesOfMultiLabelData.size(); i++){
			if(i>=1){
				System.out.println("#batch = "+i+" #size = "+batchesOfMultiLabelData.get(i).getNumInstances()+" #labels = "+batchesOfMultiLabelData.get(i).getNumLabels());
				System.out.println("======================================");
//				Evaluator evaluator = new Evaluator();
//				Evaluation eval = evaluator.evaluate((MultiLabelLearner) LR, batchesOfMultiLabelData.get(i), measures);
				pool.EvaluationOfBatch(batchesOfMultiLabelData.get(i));
				//System.out.println(pool.outputMeasures[4].get(pool.outputMeasures[4].size()-1)); //get example-based accuracy of the new batch 
				if(pool.outputMeasures[4].get(pool.outputMeasures[4].size()-1)/pool.outputMeasuresMax[4] < perfTresh) {
					drop = true; 
					System.out.println("!!!!! Performance drop !!!!! "+ pool.outputMeasuresMax[4]+" , "+pool.outputMeasures[4].get(pool.outputMeasures[4].size()-1));
				}
			}
			 
			//add the new data to the graph, set the edges,...			
			pool.addVertex(batchesOfMultiLabelData.get(i), i, epsilon, statType, assignType, mergeType, sim, pw, first, init_IW, init_bias, baseClassifier, repIter, 
					m.measures.size(), dataset.getNumLabels(), NumHiddenNeuron, xmlPath, newXmlPath, activationFun, Hthresh, Tthresh, adaptive, shallowAE, drop);
						
			//add the snapshot of concept representatives to arraylist	
			ArrayList<double[][][]> curConceptRepr = new ArrayList<double[][][]>();
			Iterator<Vertex> iter = pool.vertexes.iterator();
			int countConcepts = 0; 
			pwsp.println("#batch = "+i);
			while(iter.hasNext()){
				pwsp.println("#concept = "+countConcepts);
				Vertex vx = iter.next();
				vx.printVertex(pwsp);
				multiLabelConceptualRepr[] cpr = vx.getCV(); //each concept is an ensemble of RACE
				double[][][] curDM = new double[cpr.length][][];
				for(int cp=0; cp<cpr.length; cp++) {
					curDM[cp] = cpr[cp].getDecodeMatrix();
				}
				curConceptRepr.add(curDM);
				countConcepts++;
			}
			CRinHistory.add(curConceptRepr);
			
			pwDet.write("-------------------\n");
			
//			LR.updateClassifierForMLBatch(batchesOfMultiLabelData.get(i), adaptive, shallowAE);
		}
		
		//////////////////
//		pw.println("\n\nmisclassification of each label: ");
//		for(int i=0; i<FNInLabels.length; i++)
//			pw.println("label #"+i+" : FN = "+FNInLabels[i]+" FP = "+FPInLabels[i]+" TP = "+TPInLabels[i]+" TN = "+TNInLabels[i]);
		pw.close();
		pwsp.close();
		/////////////////
		
		long t2 = System.currentTimeMillis();
		
		return (t2-t1);
	}
	
/*	
	public static long runOnTrainDataset(MultiLabelInstances train, MultiLabelInstances test, String xmlPath, String newXmlPath, 
			String threshPath, int NumHiddenNeuron,	ArrayList<Measure> measures, int window, int N0, UpdateableClassifier classifier, 
			String activationFun, boolean adaptive, boolean shallowAE, double Hthresh, double Tthresh, double d, boolean first) throws Exception{

		makeReducedDataset(train, xmlPath, newXmlPath, NumHiddenNeuron, N0, window);
		
		long t1 = System.currentTimeMillis();
		
		LR = new LRUpdateable(first, IW, bias, classifier, measures.size(), train.getNumLabels(), NumHiddenNeuron, xmlPath, newXmlPath, activationFun, Hthresh, Tthresh,d);
		
		LR.build(batchesOfMultiLabelData.get(0));
		
		double[][] doubleBeta = LR.curBeta;
		betaStr = new String[doubleBeta.length][doubleBeta[0].length];
		makeBetaString(doubleBeta);
		
		PrintWriter pw = new PrintWriter(new File(threshPath));
		
		for(int i=1; i<batchesOfMultiLabelData.size(); i++){	
		System.out.println("#batch = "+i+" #size = "+batchesOfMultiLabelData.get(i).getNumInstances()+" #labels = "+batchesOfMultiLabelData.get(i).getNumLabels());
		Evaluator evaluator = new Evaluator();
		Evaluation eval = evaluator.evaluate((MultiLabelLearner) LR, test, measures);
		
		LR.keepAllMeasures(eval.getMeasures());
		
		LR.updateClassifierForMLBatch(batchesOfMultiLabelData.get(i), adaptive, shallowAE);
		doubleBeta = LR.curBeta;
		makeBetaString(doubleBeta);
		}
		
		long t2 = System.currentTimeMillis();
		
		return (t2-t1);
}
	
	
	public static long runIterative(int runs, MultiLabelInstances train, MultiLabelInstances test, String xmlPath, String newXmlPath, 
			String threshPath, int NumHiddenNeuron,	ArrayList<Measure> measures, int window, int N0, UpdateableClassifier classifier, 
			String activationFun, boolean adaptive, boolean shallowAE, double Hthresh, double Tthresh, double d, boolean first) throws Exception{

		makeReducedDataset(train, xmlPath, newXmlPath, NumHiddenNeuron, N0, window);
		
		long t1 = System.currentTimeMillis();
		
		LR = new LRUpdateable(first, IW, bias, classifier, measures.size(), train.getNumLabels(), NumHiddenNeuron, xmlPath, newXmlPath, activationFun, Hthresh, Tthresh,d);
		
		LR.build(batchesOfMultiLabelData.get(0));
		
		double[][] doubleBeta = LR.curBeta;
		betaStr = new String[doubleBeta.length][doubleBeta[0].length];
		makeBetaString(doubleBeta);
		
		PrintWriter pw = new PrintWriter(new File(threshPath));
		
//		for(int iter=1; iter<=runs; iter++){
//			System.out.println("========= iter = "+iter+" =========");
			for(int i=1; i<batchesOfMultiLabelData.size(); i++){
				for(int iter=1; iter<=runs; iter++){
					System.out.println("#batch = "+i+" #iter = "+iter+" #size = "+batchesOfMultiLabelData.get(i).getNumInstances()+" #labels = "+batchesOfMultiLabelData.get(i).getNumLabels());
					Evaluator evaluator = new Evaluator();
					Evaluation eval = evaluator.evaluate((MultiLabelLearner) LR, test, measures);
					
					LR.keepAllMeasures(eval.getMeasures());
					
					LR.updateClassifierForMLBatch(batchesOfMultiLabelData.get(i), adaptive, shallowAE);
					doubleBeta = LR.curBeta;
					makeBetaString(doubleBeta);
				}
			}
//		}
				
		long t2 = System.currentTimeMillis();
		
		return (t2-t1);
}
*/	
	
	public static void makeReducedDataset(MultiLabelInstances dataset, String xmlPath, String newXmlPath, int hiddenNeurons, int N0, int window) throws Exception{
		
		//new xml description 
		File f = new File(newXmlPath);
		if(!f.exists())
			makeXmlDescription(newXmlPath, hiddenNeurons);
		
		//make dataset windowed
		batchesOfMultiLabelData = MakeDataWindowed(dataset, xmlPath, N0, window);

	}


	public static void makeXmlDescription(String xmlPath, int hiddenNeurons) throws FileNotFoundException{
		PrintWriter pw = new PrintWriter(new File(xmlPath));
		pw.println("<?xml version=\"1.0\" encoding=\"utf-8\"?> \n <labels xmlns=\"http://mulan.sourceforge.net/labels\">");
		
		for(int i=0; i<hiddenNeurons; i++){
			pw.println("<label name=\"hiddenLabel_"+i+"\"></label>");
		}
		
		pw.print("</labels>");
		pw.close();
	}


	//make one MultiLabelInstances dataset windowed by size length
	public static ArrayList<MultiLabelInstances> MakeDataWindowed(MultiLabelInstances multiInst, String xmlPath, int N0, int length) throws InvalidDataFormatException{
		ArrayList<MultiLabelInstances> array = new ArrayList<>();
		Instances inst = multiInst.getDataSet();
		
		array.add(new MultiLabelInstances(getWindow(inst, 0, N0), xmlPath));
		
		for(int ins=N0; ins<inst.numInstances();ins = ins+length){
			if(ins+length<inst.numInstances()){
				array.add(new MultiLabelInstances(getWindow(inst, ins, ins+length), xmlPath));
			}else{
				array.add(new MultiLabelInstances(getWindow(inst, ins, inst.numInstances()), xmlPath));
			}
		}
		
		return array;
	}
	
	//make one Instances dataset windowed by size length
		public static ArrayList<Instances> MakeDataWindowed(Instances inst, int N0, int length){
			ArrayList<Instances> array = new ArrayList<>();
			
			array.add(getWindow(inst, 0, N0));
			
			for(int ins=N0; ins<inst.numInstances();ins = ins+length){
				if(ins+length<inst.numInstances()){
					array.add(getWindow(inst, ins, ins+length));
				}else{
					array.add(getWindow(inst, ins, inst.numInstances()));
				}
			}
			
			return array;
		}
		
		
		public static Instances getWindow(Instances inst, int start, int end){
			Instances window = new Instances(inst);
			window.delete();
			
			for(int i=start; i < end; i++)
				window.add(inst.instance(i));
			
			return window;
		}

}
