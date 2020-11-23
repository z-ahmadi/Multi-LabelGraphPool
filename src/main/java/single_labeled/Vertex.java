package single_labeled;


import java.io.PrintWriter;
import java.util.ArrayList;

import weka.classifiers.Classifier;
import weka.classifiers.UpdateableClassifier;
import weka.classifiers.bayes.NaiveBayesUpdateable;
import weka.core.Instances;

public class Vertex {
	
	int timeStamp; //first time the vertex is added to the pool. min(t1, t2) in the merge case.
	nonIIDConceptualRepr CV;
	Classifier classifier;
	//	Double classifierWeights;
	public ArrayList<Vertex> neighbors, inNeighborsOf; 
	public ArrayList<Double> TransitionWeights;
	ArrayList<Integer> TransitionNumbers;
	public boolean updated = false;
	int numberOfInstances;
	
	
	public Vertex(Instances inst, ArrayList<Integer> numericIndeces, PrintWriter pw, double epsilon, String cvectorType){
		this(new nonIIDConceptualRepr(inst, numericIndeces, pw, epsilon, cvectorType), makeClassifier(inst));
	}
	
	public Vertex(nonIIDConceptualRepr cr, Classifier c){
		CV = cr;
		classifier = c;
		neighbors = new ArrayList<Vertex>();
		inNeighborsOf = new ArrayList<Vertex>();
		TransitionWeights = new ArrayList<Double>();
		TransitionNumbers = new ArrayList<Integer>();
	}
	

	public int getTimeStamp() {
		return timeStamp;
	}

	public void setTimeStamp(int timeStamp) {
		this.timeStamp = timeStamp;
	}

	public nonIIDConceptualRepr getCV() {
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

	public Classifier getClassifier() {
		return classifier;
	}

	public void setClassifier(Instances inst) {
			classifier = makeClassifier(inst);
	}
	
	//check whether it is correct!!!!!!
	public void updateClassifier(Instances ins){
		try {
//			classifier.buildClassifier(ins);
			for(int i=0; i<ins.numInstances();i++)
				((UpdateableClassifier)classifier).updateClassifier(ins.instance(i));
		} catch (Exception e) {
			System.err.println("problem in updating classifier!");
			e.printStackTrace();
		}
	}
	
	
	public void updateClassifier(Vertex v){
		try {
			((NaiveBayesUpdateable)(this.classifier)).mergeWithNaive((NaiveBayesUpdateable)v.classifier);
		} catch (Exception e) {
			System.err.println("problem in updating classifier!");
			e.printStackTrace();
		}
	}
	
	
	static Classifier makeClassifier(Instances inss){//TODO: maybe a better weak leaner
		NaiveBayesUpdateable nb=new NaiveBayesUpdateable();
		try {
			nb.buildClassifier(inss);
		} catch (Exception e) {
			e.printStackTrace();
		}
		
		return (Classifier) nb; //////////////////////// I am not sure it works well!
	}
	
	//print mean and covariance of the vertex's CV
	public void printVertex(PrintWriter pw){
		pw.println("mean vector (features - classes) :");
		double[][] meanVector = CV.getMeanVector();
		for(int i=0; i<meanVector.length; i++){
			for(int c=0; c<meanVector[i].length; c++)
				pw.print(meanVector[i][c]+" ");
			pw.println();
		}
		
		double[][][] covarianceMatrix = CV.getCovarianceMatrix();
		for(int i=0; i<covarianceMatrix.length; i++){ //loop on classes
			pw.println("Covariance matrix of class #"+ i);
			for(int r=0; r<covarianceMatrix[i].length; r++){
				for(int c=0; c<covarianceMatrix[i].length; c++){
					pw.print(covarianceMatrix[i][r][c] + " ");
//					pw.print(String.format( "%.5f", CV.covarianceMatrix[i][r][c]) + " ");
				}
				pw.println();
			}
			
		}
	}

}
