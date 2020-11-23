package multi_labeled;

import org.apache.commons.math3.util.MathArrays;

public class CosineObject extends dissimilarityObject {

//	public double[][] cosineMatrix;
	public double[] minRow, minCol, maxRow, maxCol;
	public double[] sumM1, sumM2, diffR, diffC;
	public int[] indxR, indxC;
	int dim;
//	int[] pseudoLabelsMapping; 
	double cosineSim = 0; 
	
	public CosineObject(int k){
		dim = k;
		disMatrix = new double[k][k];
		minRow = new double[k];
		minCol = new double[k];
		maxRow = new double[k];
		maxCol = new double[k];
		sumM1 = new double[k];
		sumM2 = new double[k];
		diffR = new double[k];
		indxR = new int[k]; // index of minimum 
		diffC = new double[k];
		indxC = new int[k];
		pseudoLabelsMapping = new int[k]; //contains the corresponding column to each row (index)
	}
	
	
	public void cosineDistance(){
		for(int i=0; i<dim; i++){
			cosineSim += disMatrix[i][pseudoLabelsMapping[i]];
		}
	}
	
	
	//TODO: simple drift detection method which detects a drift if at least one of the pseudo labels is different 
	@Override
	public boolean simpleDriftDetection(double simThresh){
		boolean diff = false;
		
		for(int i=0; i<dim; i++){
			if(disMatrix[i][pseudoLabelsMapping[i]] < simThresh){
				diff = true;
				break;
			}
		}
		
		return diff;
	}
	
	
	@Override
	public boolean driftDetection(double simThresh){
		double avgCos = cosineSim/dim;
		boolean retDiff = false;
		
		boolean[] diff = new boolean[dim];
		int diffCount = 0;
		String cosString = "";
		
		for(int i=0; i<dim; i++){
			cosString = cosString + "," + String.format("%.04f", disMatrix[i][pseudoLabelsMapping[i]]);
			if(disMatrix[i][pseudoLabelsMapping[i]] < simThresh) {
				diff[i] = true;	
				diffCount++;
			}
		}
		
		if(diffCount == dim) { // if all pseudo-labels are different
			System.out.println("{"+ cosString+ "}, cosine mean = "+ String.format("%.04f", avgCos) + " all different (TP)!");
			retDiff = true;
		}else if(diffCount == 0) { // if all pseudo-labels are similar
			System.out.println("{"+ cosString+ "}, cosine mean = "+ String.format("%.04f", avgCos) + " all similar (TN)!");
			retDiff = false; 
		}else {
			if(avgCos >= simThresh) {
				System.out.println("{"+ cosString+ "}, cosine mean = "+ String.format("%.04f", avgCos) + " alarm: similar (possible FN)!");
				retDiff = false; 
			}else {
				System.out.println("{"+ cosString+ "}, cosine mean = "+ String.format("%.04f", avgCos) + " alarm: different (possible FP)!");
				retDiff = true; 
			}
				
		}
		
		return retDiff;
	}
	

	//assign two pseudo labels
	public void mapPseudoLabels(int[] indx){
//		System.out.println("index order of differences: ");
//		for(int i=0; i<indx.length; i++){
//			System.out.print(indx[i]+" ");
//		}
//		System.out.println();
		
		boolean[] flagR = new boolean[dim]; //for each row
		boolean[] flagC = new boolean[dim]; //for each col
		for(int i=0; i<indx.length; i++){
//			System.out.println("----"+i+"-----");
			if(indx[i] < dim){ // a row
				int r = indx[i];
				if(!flagR[r]){
					int c = indexOfBestElementAvailable(disMatrix[r], flagC);
					pseudoLabelsMapping[r] = c;
					flagC[c] = true;
					flagR[r] = true;
//					System.out.println(r +" ---> "+c);
				}
			}else{ // a column
				int c = indx[i] - dim;
				if(!flagC[c]){
					double[] colCos = new double[dim];
					for(int j=0; j<dim; j++)
						colCos[j] = disMatrix[j][c];
					int r = indexOfBestElementAvailable(colCos, flagR);		
					pseudoLabelsMapping[r] = c;
					flagC[c] = true;
					flagR[r] = true;	
//					System.out.println(r +" ---> "+c);		
				}
			}
		}
	}
	
	// generates a random permutation of the rows' indexes
	public int[] randomPermutation(){
		int[] indx = new int[dim];
		for(int i=0; i<dim; i++)
			indx[i] = i;
		MathArrays.shuffle(indx);
		return indx;
	}
	
	// generates an ordered permutation of the rows' indexes
		public int[] orderedPermutation(){
			int[] indx = new int[dim];
			for(int i=0; i<dim; i++)
				indx[i] = i;
			return indx;
		}
	
	/**
	 * sorts differences 
	 * @return
	 */
	public int[] sortDiffs(){
		double[] allDiff = new double[2*dim];
		int[] indx = new int[2*dim];
				
		//initialize new arrays
		for(int i=0; i<dim; i++){
			allDiff[i] = diffR[i];
			indx[i] = i;
			allDiff[i+dim] = diffC[i]; // row = false as default
			indx[i+dim] = i+dim;
		}
		
		//sort all max-min differences in descending form
		double temp;
		int tempI;
		for(int i=0; i<allDiff.length; i++){
			for(int j=i+1; j<allDiff.length; j++){
				if(allDiff[i] < allDiff[j]){
					temp = allDiff[j];
					allDiff[j] = allDiff[i];
					allDiff[i] = temp; 
					tempI = indx[i];
					indx[i] = indx[j]; 
					indx[j] = tempI;
				}
			}
		}
		return indx;
	}
	
	
	/**
	 * takes an array of similarity (e.g cosine matrix) and an array of flags indicating the availability of the index
	 * @param arr
	 * @param flag
	 * @return
	 */
	//correct! tested manually
	public int indexOfBestElementAvailable(double[] arr, boolean[] flag){
		double temp;
		int tempI;
		boolean tempF;
		int[] inds = new int[arr.length];
		double[] copyArr = new double[arr.length];
		boolean[] copyFlag = new boolean[flag.length];
		
		for(int i=0; i<flag.length; i++)
			copyFlag[i] = flag[i];
		
		for(int i=0; i<inds.length; i++){
			copyArr[i] =arr[i];
			inds[i] = i;
		}
		for(int i=0; i<copyArr.length; i++){
			for(int j=i+1; j<copyArr.length; j++){
				if(copyArr[j] > copyArr[i]){ //as the array contains similarity measure, we sort it in descending order  
					temp = copyArr[j];
					copyArr[j] = copyArr[i];
					copyArr[i] = temp; 
					tempI = inds[i];
					inds[i] = inds[j]; 
					inds[j] = tempI;
					tempF = copyFlag[i];
					copyFlag[i] = copyFlag[j];
					copyFlag[j] = tempF;
				}
			}
		}
		
		for(int i=0; i<copyArr.length; i++){
			if(!copyFlag[i])
				return inds[i];
		}
		return -1; 		
	}
	
	
	public int indexOfMax(double[] a){
		int ind = -1;
		double max = Double.MIN_VALUE;
		
		for(int i=0; i<a.length; i++){
			if(a[i] > max){
				max = a[i];
				ind = i;
			}
		}
		
		return ind;
	}
	
	public void print(){
		System.out.println("pseudo label permutation: ");
		for(int i=0; i<dim; i++){
			System.out.print(pseudoLabelsMapping[i]+" ");
		}
		System.out.println();

		System.out.println("cosine matrix: ");
		for(int i=0; i<dim; i++){
			for(int j=0; j<dim; j++){
				System.out.print(String.format( "%.3f", disMatrix[i][j])+" ");
			}
			System.out.println();
		}
	}

}
