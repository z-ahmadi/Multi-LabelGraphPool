package single_labeled;


import java.io.PrintWriter;
import java.util.ArrayList;

import weka.classifiers.AbstractClassifier;
import weka.classifiers.Classifier;
import weka.classifiers.evaluation.Evaluation;
import weka.core.Instance;
import weka.core.Instances;
import weka.core.Utils;

public class Graph extends AbstractClassifier {
	
	public ArrayList<Vertex> vertexes;
	int maxClusterCount;
	Vertex currentStateT; 
//	double[][] chiTable; 
	double chiIndex;
	ArrayList<covarianceObject> currentCovObj; 
	PrintWriter pw, chngw;
	int conceptsMade; 

	public Graph(double chiIndx, PrintWriter p, PrintWriter ch) {
		pw = p;
		chngw = ch;
		vertexes= new ArrayList<Vertex>();
		chiIndex = chiIndx;
		conceptsMade = 0;
	}
	
	ArrayList<Integer> indexOfSimilarCVinPool(Vertex newv, String mode, String statType, Instances currentBatch, double threshold, String cvType){
		ArrayList<Integer> indexs = new ArrayList<Integer>();
		
		if(mode.equals("statistical")){
			currentCovObj = new ArrayList<covarianceObject>();
			
			for(int i=0; i<vertexes.size(); i++){
				pw.write("compare with vertex "+i+"\n");
				covarianceObject cobj;
				if(statType.equals("normal"))
					cobj = newv.CV.NormalEqualityTest(newv.CV, vertexes.get(i).CV, chiIndex, cvType);
				else 
					cobj = newv.CV.statisticalMeanTest(newv.CV, vertexes.get(i).CV, chiIndex, cvType);
					
				boolean different = cobj.different;
				if(!different){
					currentCovObj.add(cobj);
					indexs.add(i);
				}
			}			
		}else if(mode.equals("yang")){
			for(int i=0; i<vertexes.size(); i++){
				try {
					double sim = conceptualEquivalence(newv.classifier, vertexes.get(i).classifier, currentBatch);
					if(sim > threshold){
//						covarianceObject cobj = newv.CV.statisticalMeanTest(newv.CV, vertexes.get(i).CV, chiIndex);
						indexs.add(i);
//						currentCovObj.add(cobj);
					}
				} catch (Exception e) {
					System.err.println("problem in finding the equivalent concept in Yang method!");
				}
			}
		}
		
		return indexs;
	}
	
	public int[] sortLambda(){
		double[] templambda = new double[currentCovObj.size()];
		int[] indx = new int[currentCovObj.size()];
		
		for(int i=0; i<currentCovObj.size(); i++){
			templambda[i] = currentCovObj.get(i).meanLambda();
			indx[i] = i;
		}
		for(int i=0; i<templambda.length; i++){
			for(int j=i+1;j<templambda.length; j++){
				if(templambda[i]> templambda[j]){
					double temp = templambda[i];
					templambda[i] = templambda[j];
					templambda[j] = temp;
					indx[i] = j;
					indx[j] = i;
				}
			}
		}
		pw.write("\nlambda: ");
		for(int i=0; i<templambda.length; i++){
			pw.write(templambda[i]+" ");
		}
		pw.write("\nindex: ");
		for(int i=0; i<indx.length; i++){
			pw.write(indx[i]+" ");
		}
		pw.write("\n");
		
		return indx;
	}
	
	//have not coded yet!
	//first find all similar vertexes to the current batch, add the batch to a copy of them
	//then have a pairwise test to see if these new copies are similar or not 
	public void greedyMergeButNotAll(Vertex newV, ArrayList<Integer> SimilarSortedIndx, String cvType){
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
		vertexes.set(mergedCInd, updateVertex(simV, newV, 0, cvType));
		for(int i=1; i<SimilarSortedIndx.size(); i++){
			vertexes.set(mergedCInd, updateVertex(simV, vertexes.get(SimilarSortedIndx.get(i)), i, cvType));
		}
		
		//removes the merged vertexes
		for(int i=SimilarSortedIndx.size()-1; i>0; i--){
			vertexes.remove((int)SimilarSortedIndx.get(i));
		}
		
		currentStateT = simV;
	}
	
	
	
	//have not coded optimally! if have time problems can be improved!
	//merges all the vertexes similar to new vertex, even if they are not similar to eachother and to the merged concept!
	public void greedyMergeAll(Vertex newV, ArrayList<Integer> SimilarSortedIndx, String cvType){
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
		vertexes.set(mergedCInd, updateVertex(simV, newV, 0, cvType));
		for(int i=1; i<SimilarSortedIndx.size(); i++){
			vertexes.set(mergedCInd, updateVertex(simV, vertexes.get(SimilarSortedIndx.get(i)), i, cvType));
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
	
	//correlation is not counted! 
	//update vertex when we want to merge two vertexes in the pool 
	public Vertex updateVertex(Vertex availcv, Vertex newcv, int indx, String cvType){
		int numClasses = availcv.CV.numberOfInstances.length;

		availcv.CV.updateMeanVector(newcv.CV);
		
		covarianceObject covObj = availcv.CV.statisticalMeanTest(availcv.CV, newcv.CV, chiIndex, cvType);
		if(covObj.different){
//			System.out.println("!!! concept "+indx+" is different from the current merged concept");
			pw.write("!!! concept "+indx+" is different from the current merged concept\n");
		}
		availcv.CV.setCovarianceMatrix(covObj.sigmaSmallOmega);
		availcv.CV.calculateDeterminant(numClasses);
		
		availcv.CV.updateNumberOfInstances(newcv.CV);

		availcv.updateClassifier(newcv);
		availcv.updated = true;
		
		availcv.setTimeStamp(Math.min(availcv.getTimeStamp(), newcv.getTimeStamp()));
		
		return availcv;
	}
	
	
	//correlation is not counted! 
	//update vertex for recent batch as the intances are available 
	public Vertex updateVertex(Vertex availcv, Vertex newcv ,int ind, Instances batch){
		
		availcv.CV.updateMeanVector(newcv.CV);
		
		if(currentCovObj != null){
			availcv.CV.setCovarianceMatrix(currentCovObj.get(ind).sigmaSmallOmega);
			availcv.CV.calculateDeterminant(batch.numClasses());
		}
		
		availcv.CV.updateNumberOfInstances(newcv.CV);

		availcv.updateClassifier(batch);
		availcv.updated = true;
		
		availcv.setTimeStamp(Math.min(availcv.getTimeStamp(), newcv.getTimeStamp()));
		
		return availcv;
	}
	
	
	public void addVertex(Instances batch , ArrayList<Integer> numericInd, int currentT, double epsilon,String mode, String stat, double thresh, String cvType){
		
		Vertex v = new Vertex(batch , numericInd, pw, epsilon, cvType);
		v.setTimeStamp(currentT);
		
		if (vertexes.size()==0){
			vertexes.add(v);
			currentStateT = v;
			conceptsMade++;
			chngw.println("<change  omega=\""+v.timeStamp+"\"  timestamp=\""+currentT*batch.numInstances()+"\"  poolSize=\""+vertexes.size()+"\" />");
		}else{
			ArrayList<Integer> similarConceptIndeces = indexOfSimilarCVinPool(v, mode, stat, batch, thresh, cvType);
			pw.write("number of similar concepts to current batch = "+ similarConceptIndeces.size()+"\n");
			
			if(similarConceptIndeces.size() == 0){
				pw.write("a new vertex is added to the concept!\n");
				chngw.println("<change  omega=\""+v.timeStamp+"\"  timestamp=\""+currentT*batch.numInstances()+"\"  poolSize=\""+vertexes.size()+"\" />");
				conceptsMade++;
				
				vertexes.add(v);
				uptadeWeightOfVertexLinks(currentStateT, v);				
				currentStateT = v;
				
			}else if(similarConceptIndeces.size() == 1){
				Vertex simV = vertexes.get(similarConceptIndeces.get(0));
				if(!simV.equals(currentStateT))		//check if it is a recurring concept 
					chngw.println("<*change  omega=\""+simV.timeStamp+"\"  timestamp=\""+currentT*batch.numInstances()+"\"  poolSize=\""+vertexes.size()+"\" />");
//				else
//					chngw.println("<***change  omega=\""+simV.timeStamp+"\"  timestamp=\""+currentT*batch.numInstances()+"\" />");
					
				
				//update vertex (CV and Learner)
				vertexes.set(similarConceptIndeces.get(0), updateVertex(simV, v, 0, batch));
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
//				sortLambda();
				chngw.print("<**merge  omega=\"");
				for(int i=0; i<similarConceptIndeces.size(); i++){
					chngw.print(vertexes.get(similarConceptIndeces.get(i)).timeStamp+",");
				}
				chngw.println("\"  timestamp=\""+currentT*batch.numInstances()+"\"  poolSize=\""+vertexes.size()+"\" />");
				
				greedyMergeAll(v, similarConceptIndeces, cvType);
				
				
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


	public double[] distributionForInstance(/*int numInBatch,*/ Instance ins) {
		ArrayList<Vertex> neighbors = currentStateT.neighbors;
		ArrayList<Double> weights = currentStateT.TransitionWeights;
		double[] votes = new double[ins.numClasses()];
		double[] finalVotes = new double[ins.numClasses()];
		
		if(neighbors.size() == 0){ //the current state does not have any neighbor, so we use itself
			try {
				finalVotes = currentStateT.classifier.distributionForInstance(ins);
			} catch (Exception e) {
				System.err.println("current state classifier could not classify instance ");
				e.printStackTrace();
			}
		}else{
			for(int i=0; i<neighbors.size(); i++){
				try {
					votes = neighbors.get(i).classifier.distributionForInstance(ins);
					for(int j=0; j<votes.length; j++)
						finalVotes[j] += votes[j]*weights.get(i);
				} catch (Exception e) {
					System.err.println("current state classifier could not classify instance ");
					e.printStackTrace();
				}
				
				
			}
		}
		
		return finalVotes;
	}
	
	
	public double classifyInstance(Instance ins, String mode){
		double[] votes = new double[0]; 
		
		if(mode.equals("vote"))
			votes = distributionForInstance(ins);
		else
			try {
				votes = currentStateT.classifier.distributionForInstance(ins);
			} catch (Exception e) {
				// TODO Auto-generated catch block
				e.printStackTrace();
			}
		
		return Utils.maxIndex(votes);

	}
	
	
	public double[] classifyInstance(Instances ins, String mode){
		double[] clss = new double[ins.numInstances()];
		
		for(int k=0; k<ins.numInstances(); k++){
			clss[k] = classifyInstance(ins.instance(k), mode);
		}

		return clss;
	}
	
	public Measurements EvaluationOfBatch(Evaluation eval, Instances batch, String mode){
		Measurements m = new Measurements();
		try {
			if(mode.equals("vote"))
				eval.evaluateModel(this, batch);  //before it was : currentStateT.classifier only taking the current state into account
			else if(mode.equals("current"))
				eval.evaluateModel(currentStateT.classifier, batch);
			else
				System.out.println("no defined mode!");
		} catch (Exception e) {
			System.err.println("Error in Evaluation!");
			e.printStackTrace();
		}
		double avgP = 0, avgR = 0, avgF = 0;
		for(int b =0; b<batch.numClasses(); b++){
			avgP += eval.precision(b);
			avgR += eval.recall(b);
			avgF += eval.fMeasure(b);
//			System.out.println("**"+b+"\t"+eval.precision(b)+"\t"+eval.recall(b)+"\t"+eval.fMeasure(b));
		}
		avgP = avgP/batch.numClasses();
		avgR = avgR/batch.numClasses();
		avgF = avgF/batch.numClasses();
//		System.out.println("***\t"+avgP+"\t"+avgR+"\t"+avgF);
		m.Accuracy = eval.pctCorrect();
		m.Precision = avgP;
		m.Recall = avgR;
		m.F1 = avgF;
		
//		int correct = 0;
//		for(int i=0; i<batch.numInstances(); i++){
//			if(classifyInstance(batch.instance(i)) == batch.instance(i).classValue())
//				correct ++; 
//		}
		
		return m;
	}
	
	
	public void print(PrintWriter pw){
		for(int i=0; i<vertexes.size(); i++){
			pw.println("=========== vertex # "+i+" ===========");
			vertexes.get(i).printVertex(pw);
		}
		
//		if(currentCovObj != null){
//			pw.println("====================================");
//			pw.println("lambda values : ");
//			for(int i=0; i<currentCovObj.size(); i++){
//				for(int j=0; j<currentCovObj.get(i).lambda.length; j++)
//					pw.print(currentCovObj.get(i).lambda[j]+" ");
//				pw.println();
//			}			
//		}
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

	public void buildClassifier(Instances data) throws Exception {
		// TODO Auto-generated method stub
		
	}

}
