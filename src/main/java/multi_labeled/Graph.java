package multi_labeled;


import java.io.PrintWriter;
import java.util.ArrayList;
import java.util.List;

import mulan.classifier.InvalidDataException;
import mulan.classifier.ModelInitializationException;
import mulan.classifier.MultiLabelLearner;
import mulan.classifier.MultiLabelLearnerBase;
import mulan.classifier.MultiLabelOutput;
import mulan.core.ArgumentNullException;
import mulan.data.MultiLabelInstances;
import mulan.evaluation.Evaluation;
import mulan.evaluation.Evaluator;
import mulan.evaluation.measure.Measure;
import multi_labeled.RACE.LRUpdateable;
import multi_labeled.RACE.myMultiLabelLearnerInterface;
import single_labeled.covarianceObject;

import weka.classifiers.AbstractClassifier;
import weka.classifiers.Classifier;
import weka.classifiers.UpdateableClassifier;
import weka.core.Instance;
import weka.core.Instances;
import weka.core.Utils;

public class Graph implements myMultiLabelLearnerInterface {
	
	public ArrayList<Vertex> vertexes;
	int maxClusterCount;
	Vertex currentStateT; 
//	double[][] chiTable;
	double[] curMinDecode, curMaxDecode, everMinDecode, everMaxDecode; //beta extreme values 
	double[] similarityThreshold; 
	double simThresh;
	ArrayList<dissimilarityObject[]> currentDissObj; 
	PrintWriter pw, chngw;
	int conceptsMade; 
	int ensembleSize;

	boolean vote = false;
	public Measurements mObj;
	public static ArrayList<Double>[] outputMeasures;
	public static double[] outputMeasuresMax;
	int[] FNInLabels, FPInLabels, TPInLabels, TNInLabels;
	double testThresh; 

	public Graph(int es, double simThresh, Measurements m, boolean mv, double tThresh, PrintWriter p, PrintWriter ch) {
		this.ensembleSize = es;
		this.curMinDecode = new double[ensembleSize];
		this.curMaxDecode = new double[ensembleSize];
		this.everMinDecode = new double[ensembleSize];
		this.everMaxDecode = new double[ensembleSize];
		for(int i=0; i< ensembleSize; i++) {
			this.curMinDecode[i] = Double.MAX_VALUE;
			this.curMaxDecode[i] = Double.MIN_VALUE;
			this.everMinDecode[i] = Double.MAX_VALUE;
			this.everMaxDecode[i] = Double.MIN_VALUE;
		}
		this.similarityThreshold = new double[ensembleSize];
		
		pw = p;
		chngw = ch;
		vertexes= new ArrayList<Vertex>();
		this.simThresh = simThresh;
		this.testThresh = tThresh;
		conceptsMade = 0;
		this.vote = mv;
		this.mObj = m;
		outputMeasuresMax = new double[m.measures.size()];
		outputMeasures = new ArrayList[m.measures.size()];
		for(int i=0; i<m.measures.size(); i++)
			outputMeasures[i] = new ArrayList<Double>();
		FNInLabels = new int[m.numLabels];
		FPInLabels = new int[m.numLabels];
		TPInLabels = new int[m.numLabels];
		TNInLabels = new int[m.numLabels];
	}
	
	@Override
	public void keepAllMeasures(List<Measure> m) {
		for(int i=0; i<m.size(); i++){
			outputMeasures[i].add(m.get(i).getValue());
			if(m.get(i).getValue() > outputMeasuresMax[i]) //!!! consider that for measures with having smaller values this condition should be reverse! but at the moment I only want to check Accuracy or F-measure
				outputMeasuresMax[i] = m.get(i).getValue();
		}
	}
	
	
	// find the similar vertices in the pool to the new one. For that, it checks the similarity of each learner in the ensemble 
	// and if the majority of them vote for being the same, it assigns the vertex as the similar.  
	ArrayList<Integer> indexOfSimilarCVinPool(Vertex newv, String statType, String assignType, double epsilon){
		ArrayList<Integer> indexs = new ArrayList<Integer>();
		
		//if(mode.equals("statistical")){
		currentDissObj = new ArrayList<dissimilarityObject[]>();
		
		for(int i=0; i<vertexes.size(); i++){
			pw.write("compare with vertex "+i+"\n");
			System.out.println("compare with vertex "+i);
			dissimilarityObject[] cobj = new dissimilarityObject[ensembleSize]; //dummy initialization, not important!
			
			for(int k=0; k<ensembleSize; k++)
				cobj[k] = new dissimilarityObject();
			

			if(statType.equals("cosine")){
				for(int e=0; e<ensembleSize; e++)
					cobj[e] = newv.CV[e].CosineSimilarity(vertexes.get(i).CV[e].getDecodeMatrix(), assignType, epsilon);
			
			}else if(statType.equals("absolute")){
				for(int e=0; e<ensembleSize; e++)
					cobj[e] = newv.CV[e].AbsoluteSimilarity(vertexes.get(i).CV[e].getDecodeMatrix(), epsilon);
			}else{ // other similarity measures 
				//TODO implement KL-divergence or PMI 
			}
			
			boolean[] different = new boolean[ensembleSize];
			int countSame = 0;
			for(int e=0; e<ensembleSize; e++){
				System.out.println("vertex "+i+" ensemble = "+e+": th="+String.format("%.05f", simThresh*similarityThreshold[e]));
				different[e] = cobj[e].driftDetection(simThresh*similarityThreshold[e]); 
				if(!different[e]){
					countSame ++;
				}
			}
			if(countSame > Math.floor((double)ensembleSize/2)){
				currentDissObj.add(cobj);
				indexs.add(i);
			}
		}	
		
		return indexs;
	}
		
	
	//have not coded yet!
	//first find all similar vertexes to the current batch, add the batch to a copy of them
	//then have a pairwise test to see if these new copies are similar or not 
	public void greedyMergeButNotAll(Vertex newV, ArrayList<Integer> SimilarSortedIndx, String cvType){
		
	}
	
	
	
	//have not coded optimally! if have time problems can be improved!
	//merges all the vertexes similar to new vertex, even if they are not similar to eachother and to the merged concept!
	public void greedyMergeAll(Vertex newV, ArrayList<Integer> SimilarSortedIndx, String statType, String assignType, String mergeType, double epsilon){
		int mergedCInd = SimilarSortedIndx.get(0);
		Vertex simV = vertexes.get(mergedCInd);
		
		//change reference of neighbours to the merged one for all similar vertexes  
		for(int i=1; i<SimilarSortedIndx.size(); i++){
			Vertex v = vertexes.get(SimilarSortedIndx.get(i));
			//other vertexes refer to simV
			ArrayList<Vertex> neighborTo = v.inNeighborsOf;
			for(int j=0; j<neighborTo.size(); j++){
				Vertex neighV = neighborTo.get(j);
				int indx = neighV.IsInNeighborhood(v);
				int indSV = neighV.IsInNeighborhood(simV);
				if(indSV != -1){
					int upTr = neighV.TransitionNumbers.get(indSV)+ neighV.TransitionNumbers.get(indx);
					neighV.TransitionNumbers.set(indSV, upTr);
					neighV.updateTransitionWeights();  ////may be not optimal way of coding!
					neighV.removeVertexFromNeighor(indx);
				}else{
					neighV.neighbors.set(indx, simV);
					simV.inNeighborsOf.add(neighV);
				}
			}
			
			//update neighbors of simV
			ArrayList<Vertex> neighbors = v.neighbors;
			for(int j=0; j<neighbors.size(); j++){
				Vertex neighV = neighbors.get(j);
				int indSV = simV.IsInNeighborhood(neighV);
				if(indSV != -1){
					int upTr = simV.TransitionNumbers.get(indSV)+ v.TransitionNumbers.get(j);
					simV.updateOneTransitionNumber(indSV, upTr);
				}else{
					simV.addNeighbor(neighV, v.TransitionWeights.get(j), v.TransitionNumbers.get(j));
				}
				//remove v from isNeighbourOf of neighV
				int indxV = neighV.IsInIsNeighborsOf(v);
				neighV.inNeighborsOf.remove(indxV);
			}
			
		}
		simV.updateTransitionWeights();

		//update vertex (CV and Learner)
		if(mergeType.equals("updateLearner")){
			vertexes.set(mergedCInd, updateVertex(simV, newV, 0, statType, assignType, epsilon));
			for(int i=1; i<SimilarSortedIndx.size(); i++){
				vertexes.set(mergedCInd, updateVertex(simV, vertexes.get(SimilarSortedIndx.get(i)), i, statType, assignType, epsilon));
			}	
		}else if(mergeType.equals("ignoreRest")){
			vertexes.set(mergedCInd, newV);
		}
		
		//removes the merged vertexes
		for(int i=SimilarSortedIndx.size()-1; i>0; i--){
			vertexes.remove((int)SimilarSortedIndx.get(i));
		}
		
		currentStateT = simV;
	}
	
	
	public Vertex weightAdjustment(Vertex v){
		if(v.neighbors.size() > 1){
			v.updateTransitionWeights();		
		}
		return v;
	}
	
	
	public Vertex uptadeWeightOfVertexLinks(Vertex v, Vertex newV){
		v.addNeighbor(newV, new Double(1), new Integer(1));
		//adjust weights of current state
		return weightAdjustment(v);
		
	}

	//update vertex when we want to merge two vertexes in the pool 
	public Vertex updateVertex(Vertex availcv, Vertex newcv, int indx,String statType, String assignType, double epsilon){
		//TODO: shall we combine decoding matrix of two vertices? 
//		availcv.CV.updateDecodingMatrix(newcv.CV);

		dissimilarityObject[] cosObj = new dissimilarityObject[ensembleSize]; //dummy initialization, not important!
		
		if(statType.equals("cosine")){
			for(int e=0; e<ensembleSize; e++)
				cosObj[e] = availcv.CV[e].CosineSimilarity(newcv.CV[e].getDecodeMatrix(), assignType, epsilon);
		
		}else if(statType.equals("absolute")){
			for(int e=0; e<ensembleSize; e++)
				cosObj[e] = availcv.CV[e].AbsoluteSimilarity(newcv.CV[e].getDecodeMatrix(), epsilon);
		}else{ // other similarity measures 
			//TODO implement KL-divergence or PMI 
		}
		
		boolean[] different = new boolean[ensembleSize];
		int countSame = 0;
		for(int e=0; e<ensembleSize; e++){
			different[e] = cosObj[e].driftDetection(simThresh*similarityThreshold[e]);
			if(!different[e]){
				countSame ++;
			}
		}
		if(countSame <= Math.floor((double)ensembleSize/2)){
			pw.write("!!! concept "+indx+" is different from the current merged concept\n");
		}
		
		for(int k=0; k<ensembleSize; k++) {
			availcv.CV[k].updateNumberOfInstances(newcv.CV[k]); 
			availcv.CV[k].updateNumberOfOnes(newcv.CV[k]);
			
			availcv.updateClassifier(newcv, k, cosObj[k].pseudoLabelsMapping);
		}
		
		availcv.updated = true;
		
		availcv.setTimeStamp(Math.min(availcv.getTimeStamp(), newcv.getTimeStamp()));
		
		return availcv;
	}
	
	//update vertex for recent batch as the instances are available 
	public Vertex updateVertex(Vertex availcv, Vertex newcv ,int ind, MultiLabelInstances batch, boolean adaptive, boolean shallowAE){
		
		for(int k=0; k<ensembleSize; k++) {
			availcv.CV[k].updateNumberOfInstances(newcv.CV[k]);
			availcv.CV[k].updateNumberOfOnes(newcv.CV[k]);
		}

		availcv.updateClassifier(batch, adaptive, shallowAE);
		for(int k=0; k<ensembleSize; k++) 
			availcv.CV[k].setDecodeMatrix(((LRUpdateable)availcv.classifier[k]).getCurBeta());
		poolSimilarityThreshUpdate(availcv);
		availcv.updated = true;
		
		availcv.setTimeStamp(Math.min(availcv.getTimeStamp(), newcv.getTimeStamp()));
		
		return availcv;
	}
	
	
	public void poolSimilarityThreshUpdate(Vertex v) {
		for(int i=0; i<ensembleSize; i++) {
			if(curMinDecode[i] > v.CV[i].getCurMin())
				curMinDecode[i] = v.CV[i].getCurMin();
			if(curMaxDecode[i] < v.CV[i].getCurMax())
				curMaxDecode[i] = v.CV[i].getCurMax();
			if(everMinDecode[i] > v.CV[i].getEverMin())
				everMinDecode[i] = v.CV[i].getEverMin();
			if(everMaxDecode[i] < v.CV[i].getEverMax())
				everMaxDecode[i] = v.CV[i].getEverMax();
			
			similarityThreshold[i] = curMaxDecode[i] - curMinDecode[i]; //make the threshold based on the current state of the pool 
		}
	}
	
	
	//String mode: Yang or statistical
	public void addVertex(MultiLabelInstances batch , int currentT, double epsilon, String stat, String assignType, String mergeType, double simThresh, PrintWriter pw, 
			boolean first, double[][][] iw, double[][] b, UpdateableClassifier classifier, int repIter, int MeasureLength, int inputN, int hiddenN, String xmlOriginal, 
			String xmlReduced, String ActivationFunc, double Hthresh, double Tthresh, boolean adaptive, boolean shallowAE, boolean drop){
		
		Vertex v = new Vertex(batch, pw, epsilon, first, iw, b, classifier, MeasureLength, inputN, hiddenN, xmlOriginal, xmlReduced, ActivationFunc, Hthresh, 
				Tthresh, repIter, adaptive, shallowAE);
		v.setTimeStamp(currentT);
		for(int k=0; k<ensembleSize; k++)
			v.CV[k].setDecodeMatrix(((LRUpdateable[])v.getClassifier())[k].getCurBeta()); //before it was not the normal version 
		poolSimilarityThreshUpdate(v);
		
		if (vertexes.size()==0){
			vertexes.add(v);
			currentStateT = v;
			conceptsMade++;
			chngw.println("<change  omega=\""+v.timeStamp+"\"  timestamp=\""+currentT*batch.getNumInstances()+"\"  poolSize=\""+vertexes.size()+"\" />");
		}else{
			ArrayList<Integer> similarConceptIndeces;
			if(drop) //check for the decoding differences only if there is a huge drop in the performance 
				similarConceptIndeces = indexOfSimilarCVinPool(v, stat, assignType, epsilon);
			else {
				similarConceptIndeces = new ArrayList<Integer>();
				similarConceptIndeces.add(vertexes.indexOf(currentStateT));
			}
			pw.write("number of similar concepts to current batch = "+ similarConceptIndeces.size()+"\n");
			
			if(similarConceptIndeces.size() == 0){	
				pw.write("a new vertex is added to the concept!\n");
				chngw.println("<change  omega=\""+v.timeStamp+"\"  timestamp=\""+currentT*batch.getNumInstances()+"\"  poolSize=\""+(vertexes.size()+1)+"\" />");
				conceptsMade++;
				
				vertexes.add(v);
				uptadeWeightOfVertexLinks(currentStateT, v);				
				currentStateT = v;
				
			}else if(similarConceptIndeces.size() == 1){
				Vertex simV = vertexes.get(similarConceptIndeces.get(0));
				if(!simV.equals(currentStateT))		//check if it is a recurring concept 
					chngw.println("<*change  omega=\""+simV.timeStamp+"\"  timestamp=\""+currentT*batch.getNumInstances()+"\"  poolSize=\""+vertexes.size()+"\" />");
//				else
//					chngw.println("<***change  omega=\""+simV.timeStamp+"\"  timestamp=\""+currentT*batch.numInstances()+"\" />");
					
				
				//update vertex (CV and Learner)
				vertexes.set(similarConceptIndeces.get(0), updateVertex(simV, v, 0, batch, adaptive, shallowAE));
				//adjust weights
				int indx = currentStateT.IsInNeighborhood(simV);
				if(indx == -1){
					pw.write("similar vertex should be added to current state neighbourhood!\n");
					currentStateT = uptadeWeightOfVertexLinks(currentStateT, simV);
					
				}else{ //similar vertex is in the neighbours of current state
					pw.write("update similar vertex weight in current state neighbourhood!\n");
					currentStateT.TransitionNumbers.set(indx, currentStateT.TransitionNumbers.get(indx)+1);
					currentStateT = weightAdjustment(currentStateT);
				}
				
				currentStateT = simV;
				
			}else{ //more than one similar vertexes
				chngw.print("<**merge  omega=\"");
				for(int i=0; i<similarConceptIndeces.size(); i++){
					chngw.print(vertexes.get(similarConceptIndeces.get(i)).timeStamp+",");
				}
				chngw.println("\"  timestamp=\""+currentT*batch.getNumInstances()+"\"  poolSize=\""+(vertexes.size()-similarConceptIndeces.size()+1)+"\" />");
				System.out.println("\"  timestamp=\""+currentT*batch.getNumInstances()+"\"  poolSize=\""+vertexes.size()+"\" />");
				
				greedyMergeAll(v, similarConceptIndeces, stat, assignType, mergeType, epsilon);
				
				
//				for(int i=0; i<similarConceptIndeces.size(); i++){
//					Vertex simV = vertexes.get(similarConceptIndeces.get(i));
//					pw.write(simV.updated+" ");
//					System.out.print(simV.updated+" ");
//					vertexes.set(similarConceptIndeces.get(i), updateVertex(simV, v, i, batch, numericInd));
//					int[] sortInd = sortLambda(pw);
//					greedyMerge(sortInd);
//					
//				}
//				System.out.println();
			}
		}
		
	}

	

	//new
	@Override
	public double[][] makePredictionForBatch(Instances instances) throws InvalidDataException, ModelInitializationException, Exception {
		double[][] output = new double[instances.numInstances()][];
		
		for(int i=0; i<instances.numInstances(); i++){
			MultiLabelOutput instPreds = makePrediction(instances.get(i));
			boolean[] bipar = instPreds.getBipartition();
			output[i] = new double[bipar.length];
			for(int j=0; j<bipar.length; j++)
				if(bipar[j])
					output[i][j] = 1;
				else 
					output[i][j] = 0;			
		}
		
		return output;
	}
	

	//new
	public MultiLabelOutput makePrediction(Instance instance) throws Exception, InvalidDataException, ModelInitializationException {
		MultiLabelOutput mlo = new MultiLabelOutput(new boolean[0]), tmlo; //dummy initialization as it was necessary for return!
		if(vote){ //majority vote
			ArrayList<Vertex> neighbors = currentStateT.neighbors;
			ArrayList<Double> weights = currentStateT.TransitionWeights;
			double[] conf, finalConf = new double[mObj.numLabels]; 
			
			if(neighbors.size() == 0){ //the current state does not have any neighbor, so we use itself
				try {
					mlo =  currentStateT.makePrediction(instance); 
				} catch (Exception e) {
					System.err.println("current state classifier could not classify instance ");
					e.printStackTrace();
				}
			}else{
				for(int i=0; i<neighbors.size(); i++){
					try {
						tmlo = neighbors.get(i).makePrediction(instance);
						conf = tmlo.getConfidences();
						for(int j=0; j<conf.length; j++)
							finalConf[j] += conf[j]*weights.get(i);
					} catch (Exception e) {
						System.err.println("current state classifier could not classify instance ");
						e.printStackTrace();
					}
				}
				//make new output based on the weighted voting results
				//confidence is the distribution of class 1 
				boolean[] bipartition = new boolean[mObj.numLabels];
				for(int k=0; k < mObj.numLabels; k++){
			        bipartition[k] = (finalConf[k]  >= testThresh) ? true : false;
				}
				mlo = new MultiLabelOutput(bipartition, finalConf);
				
			}
		}else{ //get the distribution from current state
			mlo = currentStateT.makePrediction(instance);
		}
		
		
        return mlo;
    }
	
	
	//new
	// updates the outputMeasures for every new batch it receives 
	public void EvaluationOfBatch(MultiLabelInstances batch){
		myEvaluator evaluator = new myEvaluator();
		Evaluation eval;
		try {
			if(vote){
				eval = evaluator.evaluate((MultiLabelLearner) this, batch, mObj.measures);
			}else{ //if(mode.equals("current"))
				eval = evaluator.evaluate((MultiLabelLearner[]) currentStateT.classifier, batch, mObj.measures);  // TODO: choose the most likely neighbor not  
			}
			
			///////////////
			Instances temp = batch.getDataSet();
			double[][] labels = myMultiLabelLearnerInterface.getWindowMatrix(myMultiLabelLearnerInterface.extractBatchLabels(batch), false);
			double[][] preds = makePredictionForBatch(temp);
			int countFP = 0, countFN = 0, countTP = 0, countTN = 0;
			
			for(int mk=0; mk<temp.numInstances();mk++){
				for(int mj=0; mj<labels[mk].length;mj++){
					if(labels[mk][mj] != preds[mk][mj]){
						if(labels[mk][mj] == 1){
							FNInLabels[mj]++;
							countFN++;
						}else{
							FPInLabels[mj]++;
							countFP++;
						}
					}else{
						if(labels[mk][mj] == 1){
							TPInLabels[mj]++;
							countTP++;
						}else{
							TNInLabels[mj]++;
							countTN++;
						}
					}
				}
			}
			pw.println("FN = "+countFN+" FP = "+countFP+" TP = "+countTP+" TN = "+countTN);
			//////////////
			System.out.println("ExampleBasedAccuracy = "+mObj.measures.get(4).getValue());
			
			keepAllMeasures(mObj.measures);
//			System.out.println("keep");
			
		} catch (Exception e) {
			System.err.println("Error in Evaluation!");
			e.printStackTrace();
		}
	}
	
	
	public void print(PrintWriter pw){
		for(int i=0; i<vertexes.size(); i++){
			pw.println("=========== vertex # "+i+" ===========");
			vertexes.get(i).printVertex(pw);
		}
	}
	
	
	//Yang's scoring system for comparison of two concepts
	//returns a score in [-1,1]
	public double conceptualEquivalence(Classifier cur, Classifier c, Instances batch) throws Exception{
		double ce = 0;
		for(int i=0; i<batch.size(); i++){
			int score;
			double curPred = cur.classifyInstance(batch.instance(i));
			double cPred = c.classifyInstance(batch.instance(i));
			if(curPred != cPred)
				score = -1;
//			else if(curPred )
//				score = 0;
			else
				score = 1;
			
			ce += score;	
		}
		
		return ce/(batch.size());
	}


	@Override
	public void updateClassifier(Instance instance) throws Exception {
		// TODO Auto-generated method stub
		
	}

	@Override
	public void build(MultiLabelInstances trainingSet) throws Exception {
		// TODO Auto-generated method stub
		
	}

	@Override
	public void updateClassifierBatch(Instances instances) throws Exception {
		// TODO Auto-generated method stub
		
	}
	
	//new
	@Override
	public ArrayList<Double>[] measureGetter() {
		return outputMeasures;
	}

}
