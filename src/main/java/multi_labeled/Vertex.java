package multi_labeled;


import java.io.PrintWriter;
import java.util.ArrayList;

import mulan.classifier.MultiLabelLearner;
import mulan.classifier.MultiLabelOutput;
import mulan.data.MultiLabelInstances;
import multi_labeled.RACE.LRUpdateable;
import weka.classifiers.UpdateableClassifier;
import weka.core.Instance;

/**
 * @author zahra
 *
 */

public class Vertex {
	
	int timeStamp; //first time the vertex is added to the pool. min(t1, t2) in the merge case.
	//!!!! it would be interesting to keep the time stamps that a vertex is used and the performance of the corresponding batch on that 
	multiLabelConceptualRepr[] CV;
	UpdateableClassifier[] classifier;
	//	Double classifierWeights;
	public ArrayList<Vertex> neighbors, inNeighborsOf; 
	public ArrayList<Double> TransitionWeights;
	ArrayList<Integer> TransitionNumbers;
	public boolean updated = false;
	int numberOfInstances;
	int repeatIter; 
	
	
	public Vertex(MultiLabelInstances inst, PrintWriter pw, double epsilon, boolean first, double[][][] iw, double[][] b, UpdateableClassifier classifier, int MeasureLength, 
				  int inputN, int hiddenN, String xmlOriginal, String xmlReduced, String ActivationFunc, double Hthresh, double Tthresh, int iterRep, boolean adaptive, boolean shallowAE){
		this(new multiLabelConceptualRepr[iw.length], 
				makeClassifier(inst, first, iw.length, iw, b, classifier, MeasureLength, inputN, hiddenN, xmlOriginal, xmlReduced, ActivationFunc, Hthresh, 
						Tthresh, iterRep, adaptive, shallowAE), iterRep);
		for(int i=0; i<CV.length; i++) {
			CV[i] = new multiLabelConceptualRepr(inst, pw, epsilon, hiddenN, inputN);
		}
	}
	
	public Vertex(multiLabelConceptualRepr[] cr, UpdateableClassifier[] updateableClassifiers, int iterRep){
		CV = cr;
		classifier = updateableClassifiers;
		neighbors = new ArrayList<Vertex>();
		inNeighborsOf = new ArrayList<Vertex>();
		TransitionWeights = new ArrayList<Double>();
		TransitionNumbers = new ArrayList<Integer>();
		repeatIter = iterRep;
	}
	

	public int getTimeStamp() {
		return timeStamp;
	}

	public void setTimeStamp(int timeStamp) {
		this.timeStamp = timeStamp;
	}

	public multiLabelConceptualRepr[] getCV() {
		return CV;
	}

	public ArrayList<Vertex> getNeighbors() {
		return neighbors;
	}
	
	public Vertex getOneNeighbor(int i) {
		return neighbors.get(i);
	}

	public void addNeighbor(Vertex neighbors, Double weight, Integer num) {
		this.neighbors.add(neighbors);
		neighbors.inNeighborsOf.add(this);
		TransitionWeights.add(weight);
		TransitionNumbers.add(num);
	}
	
	public void removeVertexFromNeighor(int indx){
		this.neighbors.remove(indx);
		TransitionNumbers.remove(indx);
		TransitionWeights.remove(indx);
	}
	
	public int IsInNeighborhood(Vertex v){
		int Ind = -1;
		
		for(int i=0; i<neighbors.size(); i++){
			if(neighbors.get(i).equals(v)){
				Ind = i;
				break;
			}
		}
		
		return Ind;
	}
	
	public int IsInIsNeighborsOf(Vertex v){
		int Ind = -1;
		
		for(int i=0; i<inNeighborsOf.size(); i++){
			if(inNeighborsOf.get(i).equals(v)){
				Ind = i;
				break;
			}
		}
		
		return Ind;
	}
	
	public void updateTransitionWeights(){
		double sum = 0;
		for(int i=0; i<TransitionNumbers.size(); i++)
			sum += TransitionNumbers.get(i);
		for(int i=0; i<TransitionNumbers.size(); i++)
			updateOneTransitionWeight(i, (double)TransitionNumbers.get(i)/sum);
	}

	public ArrayList<Double> getTransitionWeights() {
		return TransitionWeights;
	}
	
	public ArrayList<Integer> getTransitionNumbers() {
		return TransitionNumbers;
	}

	public void updateOneTransitionWeight(int i, Double transitionWeight) {
		TransitionWeights.set(i, transitionWeight);
	}
	
	public void updateOneTransitionNumber(int i, Integer transitionNumber) {
		TransitionNumbers.set(i, transitionNumber);
	}

	public UpdateableClassifier[] getClassifier() {
		return classifier;
	}

//	public void setClassifier(MultiLabelInstances inst, boolean first, int enmbl, double[][][] iw, double[][] b, UpdateableClassifier classifier, int MeasureLength, 
//			int inputN, int hiddenN, String xmlOriginal, String xmlReduced, String ActivationFunc, double Hthresh, double Tthresh) {
//			classifier = makeClassifier(inst, first, enmbl, iw, b, classifier, MeasureLength, inputN, hiddenN, xmlOriginal, xmlReduced, ActivationFunc, Hthresh, Tthresh);
//	}


	public void updateClassifier(MultiLabelInstances ins, boolean adaptive, boolean shallowAE){
		try {
			for(int i=0; i<classifier.length; i++)
				for(int it=0; it<repeatIter; it++)
					((LRUpdateable)classifier[i]).updateClassifierForMLBatch(ins, adaptive, shallowAE);
		} catch (Exception e) { 
			System.err.println("problem in updating classifier!");
			e.printStackTrace();
		}
	}
	
	
	// update the corresponding classifier in the ensemble using the mapped indices 
	public void updateClassifier(Vertex v, int i, int[] mappedInds){
//		for(int i=0; i<classifier.length; i++) {
			try {
				((LRUpdateable)(this.classifier[i])).mergeClassifier((LRUpdateable)v.classifier[i], mappedInds);
			} catch (Exception e) {
				System.err.println("problem in updating classifier!");
				e.printStackTrace();
			}
//		}
		
	}
	
	//we need to pass the iteration to this method separately as it is the first line of the constructor. 
	static UpdateableClassifier[] makeClassifier(MultiLabelInstances inss, boolean first, int enmbl, double[][][] iw, double[][] b, UpdateableClassifier classifier, int MeasureLength, 
												int inputN, int hiddenN, String xmlOriginal, String xmlReduced, String ActivationFunc, double Hthresh, double Tthresh, int repIter
												, boolean adaptive, boolean shallowAE){
		LRUpdateable[] lru = new LRUpdateable[enmbl];
		for(int i=0; i<enmbl; i++) {
			lru[i] = new LRUpdateable(first, iw[i], b[i], classifier, MeasureLength, inputN, hiddenN, xmlOriginal, xmlReduced, ActivationFunc, Hthresh, Tthresh);
			try {
				lru[i].build(inss);
				for(int iter=1; iter<repIter; iter++)
					lru[i].updateClassifierForMLBatch(inss, adaptive, shallowAE);  
			} catch (Exception e) {
				e.printStackTrace();
			}
		}
		
		return (UpdateableClassifier[]) lru;
	}
	
	/**
	 * make voted majority prediction 
	 * @param instance
	 * @return
	 */
	public MultiLabelOutput makePrediction(Instance labelsMissing) {
		MultiLabelOutput[] output = new MultiLabelOutput[classifier.length];
		
		for(int i=0; i<classifier.length; i++){
			try {
				 output[i] = ((MultiLabelLearner) classifier[i]).makePrediction(labelsMissing);
			} catch (Exception e) {
				System.err.println("error in prediction of learner "+i);
				e.printStackTrace();
			}
		}
		
		//merge the output of ensemble
		boolean[] bipartition = new boolean[output[0].getBipartition().length]; //with the size of labels 
		double[] Confidences = new double[output[0].getBipartition().length];
		for(int l=0; l<bipartition.length; l++){
			int countBP = 0; 
			double conf = 0;
			for(int m=0; m<classifier.length; m++){
				if(output[m].getBipartition()[l]) //if the prediction of learner m is 
					countBP++; 
				conf += output[m].getConfidences()[l];
			}
			Confidences[l] = conf/classifier.length; //average over the ensemble 
			if(countBP > Math.floor((double)classifier.length/2))
				bipartition[l] = true;
		}
		
		return new MultiLabelOutput(bipartition, Confidences);
		
	}
	
	
	//print decoding matrix of CV
	public void printVertex(PrintWriter pw){
//		pw.println("decoding matrix (Hidden x Labels) :");
		for(int m=0; m<CV.length; m++) {
			pw.println("#model = "+m);
			double[][] decode = CV[m].getDecodeMatrix();
			for(int i=0; i<decode.length; i++){
				for(int c=0; c<decode[i].length; c++)
					pw.print(decode[i][c]+" ");
				pw.println();
			}
		}
		
	}

}
