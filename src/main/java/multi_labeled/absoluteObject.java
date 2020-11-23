package multi_labeled;

/**
 * The concept of this object is different than cosine distance in terms of being distance not similarity measure
 * @author zahra
 *
 */

public class absoluteObject extends dissimilarityObject{
	int numHidden, numLabels;
	double[] absDissHidden; 
	double absDiss = 0; 
	
	public absoluteObject(int k, int L) {
		disMatrix = new double[k][L];
		numHidden = k;
		numLabels = L; 
		absDissHidden = new double[k]; 
		
		pseudoLabelsMapping = new int[numHidden];
		for(int i=0; i<numHidden; i++)
			pseudoLabelsMapping[i] = i;
	}
	
	//total value of absolute distance
	public void absoluteDistance(){
		for(int i=0; i<numHidden; i++){
			for(int j=0; j<numLabels; j++){
				absDissHidden[i] += disMatrix[i][j];
				absDiss += disMatrix[i][j];
			}
		}
	}
	
	@Override
	public boolean simpleDriftDetection(double simThresh){
		boolean diff = false;
		

		for(int i=0; i<numHidden; i++){
			for(int j=0; j<numLabels; j++){
				if(disMatrix[i][j] > simThresh){
					diff = true;
					break;
				}
			}
		}
		
		return diff;
	}
	
	
	@Override
	// compare the average absolute distance of each hidden neuron 
	public boolean driftDetection(double simThresh){
		double avgDis = absDiss/(numHidden*numLabels);
		boolean retDiff = false;
		
		boolean[] diff = new boolean[numHidden];
		int diffCount = 0;
		String absString = "";
		
		for(int i=0; i<numHidden; i++){
//			for(int j=0; j<numLabels; j++){
//				absString = absString + "," + String.format("%.04f", disMatrix[i][j]);
//				if(disMatrix[i][j] > simThresh) {
//					diff[i] = true;	
//					diffCount++;
//				}
//			}
			if((absDissHidden[i]/numLabels) > simThresh) {
				diff[i] = true;	
				diffCount++;
			}
		}
		
		if(diffCount == numHidden) { // if all pseudo-labels are different
			System.out.println("abs mean = "+ String.format("%.04f", avgDis) + " all different (TP)!");
			retDiff = true;
		}else if(diffCount == 0) { // if all pseudo-labels are similar
			System.out.println("abs mean = "+ String.format("%.04f", avgDis) + " all similar (TN)!");
			retDiff = false; 
		}else {
			if(avgDis <= simThresh) {
				System.out.println("abs mean = "+ String.format("%.04f", avgDis) + " alarm: similar (possible FN)!");
				retDiff = false; 
			}else {
				System.out.println("abs mean = "+ String.format("%.04f", avgDis) + " alarm: different (possible FP)!");
				retDiff = true; 
			}
				
		}
		
		return retDiff;
	}


}
