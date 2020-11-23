package multi_labeled;


import java.io.PrintWriter;
import mulan.data.MultiLabelInstances;
import weka.core.AttributeStats;
import weka.core.Instances;

public class multiLabelConceptualRepr {
	static double epsilon;
	
	private double[][] decodeMatrix; //decodingMatrix
	private double curMin = Double.MAX_VALUE, curMax = Double.MIN_VALUE, everMin= Double.MAX_VALUE, everMax = Double.MIN_VALUE; //beta extreme values
	public int numberOfInstances; 
	public int numHidden, numLabels; 
	private int[] numOnes;
//	PrintWriter pw; 
	

	public double getCurMin() {
		return curMin;
	}

	public void setCurMin(double curMin) {
		this.curMin = curMin;
	}

	public double getCurMax() {
		return curMax;
	}

	public void setCurMax(double curMax) {
		this.curMax = curMax;
	}

	public double getEverMin() {
		return everMin;
	}

	public void setEverMin(double everMin) {
		this.everMin = everMin;
	}

	public double getEverMax() {
		return everMax;
	}

	public void setEverMax(double everMax) {
		this.everMax = everMax;
	}


	
	//create a new decoding matrix with zero values
	public multiLabelConceptualRepr(int hidden, int labels) {
		decodeMatrix = new double[hidden][labels];
	}
	
	public multiLabelConceptualRepr(MultiLabelInstances dataSet, PrintWriter pw, double eps, int hiddenN, int numL) {
		this.epsilon = eps;
//		this.pw = pw;
		numberOfInstances = dataSet.getNumInstances();
		numHidden = hiddenN;
		numLabels = numL;
		countOnesInLabels(dataSet);
	}
	
	public multiLabelConceptualRepr(MultiLabelInstances dataSet, PrintWriter pw, double eps) {
		this.epsilon = eps;
//		this.pw = pw;
		numberOfInstances = dataSet.getNumInstances();
//		numHidden = decodeMatrix.length;
//		numLabels = decodeMatrix[0].length;
		countOnesInLabels(dataSet);
	}
	
	public multiLabelConceptualRepr(double[][] beta, MultiLabelInstances dataSet, PrintWriter pw, double eps) {
		setDecodeMatrix(beta);
		this.epsilon = eps;
//		this.pw = pw;
		numberOfInstances = dataSet.getNumInstances();
		numHidden = decodeMatrix.length;
		numLabels = decodeMatrix[0].length;
		countOnesInLabels(dataSet);
	}
	
	//counts the number of ones for each label in the current batch
	public void countOnesInLabels(MultiLabelInstances dataset){		
		int[] labelInd = dataset.getLabelIndices();
		Instances batch = dataset.getDataSet();
		numOnes = new int[labelInd.length];
		
		for(int i=0; i<labelInd.length; i++){
			AttributeStats stat = batch.attributeStats(labelInd[i]);
			int[] counts = stat.nominalCounts;
//			System.out.println(batch.attribute(labelInd[i]).value(1) + " " + counts[1]);
			numOnes[i] = counts[1];
		}
	}
	
	
	public double[][] getDecodeMatrix() {
		return decodeMatrix;
	}

	public void setDecodeMatrix(double[][] beta) {
		decodeMatrix = new double[numHidden][numLabels];
		
		for(int i=0; i<beta.length; i++)
			for(int j=0; j<beta[i].length; j++) {
				decodeMatrix[i][j] = beta[i][j];
				updateExtremeBetaValues(decodeMatrix[i][j]);
			}
	}
	
	public void updateExtremeBetaValues(double element) {
		if(curMin > element)
			curMin = element;
		if(curMax < element)
			curMax = element;
		if(everMin > curMin)
			everMin = curMin;
		if(everMax < curMax)
			everMax = curMax;
	}
	
	
	//I should check mathematically if taking the weighted average of the decoding matrix is correct
	public void updateDecodingMatrix(multiLabelConceptualRepr newcv){
		for(int i=0; i<numHidden; i++)
			for(int j=0; j<numLabels; j++){ 
				decodeMatrix[i][j] = (decodeMatrix[i][j]* numberOfInstances + newcv.decodeMatrix[i][j]* newcv.numberOfInstances)
   				                    /(numberOfInstances+newcv.numberOfInstances);
			}
	}
	
	public void updateNumberOfInstances(multiLabelConceptualRepr newcv){
		numberOfInstances += newcv.numberOfInstances;
	}
	
	public void updateNumberOfOnes(multiLabelConceptualRepr newcv){
		for(int i=0; i<numOnes.length; i++){
			numOnes[i] += newcv.numOnes[i];
		}
	}
	
	
	public absoluteObject AbsoluteSimilarity(double[][] decode2, double epsilon){
		absoluteObject absObj = new absoluteObject(numHidden, numLabels);
		
		for(int i=0; i<decodeMatrix.length; i++){
			for(int j=0; j<decodeMatrix[i].length;j++){
				absObj.disMatrix[i][j] = Math.abs(decodeMatrix[i][j] - decode2[i][j]);
				
				//if we do not normalize the decoding matrix it can be out of [-1,1] but in my limited test it was not arbitrarily 
//				if((decodeMatrix[i][j]>1 || decodeMatrix[i][j]<-1)||(decode2[i][j]>1 || decode2[i][j]<-1) || (absObj.disMatrix[i][j]<0 || absObj.disMatrix[i][j]>2)) {
//					System.err.println("Why such decoding?!: i="+i+", j="+j+", d="+decodeMatrix[i][j]+"d'="+decode2[i][j]);
//				}
			}
		}
		
		absObj.absoluteDistance();
//		System.out.println("total absolute distance = "+absObj.absDiss + " H ="+absObj.numHidden+" L="+absObj.numLabels);
		
		return absObj;
	}
	
	
	//gets another decoding matrix and calculates the cosine distance matrix 
	//the rows indicate reduced space dimension (topics) and the columns indicate original space dimension (words)
	public CosineObject CosineSimilarity(double[][] decode2, String assignType, double epsilon){
		CosineObject cosObj = new CosineObject(numHidden);
		boolean[] zeroM1 = new boolean[numHidden], zeroM2 = new boolean[numHidden];  //check if the vector is a zero vector
		for(int i=0; i<numHidden; i++) {
			zeroM1[i] = true;
			zeroM2[i] = true;
		}
		
		//initialize the min & max & calculate sum of each topic (hidden vectors)
		for(int i=0; i<decodeMatrix.length; i++){ //as decodeMatrix.length is the same as decode2.length
			cosObj.minRow[i] = Integer.MAX_VALUE;
			cosObj.minCol[i] = Integer.MAX_VALUE;
			cosObj.maxRow[i] = Integer.MIN_VALUE;
			cosObj.maxCol[i] = Integer.MIN_VALUE;
			for(int j=0; j<decodeMatrix[i].length;j++){
				cosObj.sumM1[i] += Math.pow(decodeMatrix[i][j],2);
				cosObj.sumM2[i] += Math.pow(decode2[i][j],2);
				if(decodeMatrix[i][j] != 0)
					zeroM1[i] = false;
				if(decode2[i][j] != 0)
					zeroM2[i] = false;
			}
			
			//add an epsilon value if sum is zero
			if(cosObj.sumM1[i] == 0)
				cosObj.sumM1[i] = epsilon;
			if(cosObj.sumM2[i] == 0)
				cosObj.sumM2[i] = epsilon;
		}
		
		//calculate cosine matrix & difference vectors
		for(int i=0; i<decodeMatrix.length; i++){
			for(int k=0; k<decode2.length; k++){
				if(zeroM1[i] & zeroM2[k]) { // if both are zero vectors
					cosObj.disMatrix[i][k] = 1;
				}else {
					for(int j=0; j<decodeMatrix[i].length;j++){
						cosObj.disMatrix[i][k] += decodeMatrix[i][j]*decode2[k][j];
					}
					//if cosine value is zero we do not change it to epsilon!
//					System.out.print(cosObj.cosineMatrix[i][k] + "\t"+Math.sqrt(cosObj.sumM1[i])+"\t"+Math.sqrt(cosObj.sumM2[k])+" : ");
					cosObj.disMatrix[i][k] /= (Math.sqrt(cosObj.sumM1[i])*Math.sqrt(cosObj.sumM2[k]));
//					System.out.println(cosObj.cosineMatrix[i][k]);
					
					if(cosObj.disMatrix[i][k] > cosObj.maxRow[i]){
						cosObj.maxRow[i] = cosObj.disMatrix[i][k];
						cosObj.diffR[i] = cosObj.maxRow[i]-cosObj.minRow[i];
						cosObj.indxR[i] = k;
					} 
					if(cosObj.disMatrix[i][k] < cosObj.minRow[i]){
						cosObj.minRow[i] = cosObj.disMatrix[i][k];
						cosObj.diffR[i] = cosObj.maxRow[i]-cosObj.minRow[i];
					}
					
					if(cosObj.disMatrix[i][k] > cosObj.maxCol[k]){
						cosObj.maxCol[k] = cosObj.disMatrix[i][k];
						cosObj.diffC[k] = cosObj.maxCol[k]-cosObj.minCol[k];
						cosObj.indxC[k] = i;
					}
					if(cosObj.disMatrix[i][k] < cosObj.minCol[k]){
						cosObj.minCol[k] = cosObj.disMatrix[i][k];
						cosObj.diffC[k] = cosObj.maxCol[k]-cosObj.minCol[k];
					}
				}
				
			}
		}
		
		//find a mapping between two decoding matrices
		if(assignType.equals("minDistance")){
			cosObj.mapPseudoLabels(cosObj.sortDiffs());
		}else if(assignType.equals("random")){
			cosObj.mapPseudoLabels(cosObj.randomPermutation());
		}else if(assignType.equals("ordered")) { //1--1, 2--2, ..., k--k
			cosObj.mapPseudoLabels(cosObj.orderedPermutation());
		}
		cosObj.cosineDistance();

		//uncomment to print the cosine matrix and mapping indices
//		System.out.println("cosine after mapping: ");
//		cosObj.print(); 
//		System.out.println("------------------------");
		
		return cosObj;
	}
	
	
	public multiLabelConceptualRepr copy(){
		multiLabelConceptualRepr newCR = new multiLabelConceptualRepr(this.numHidden, this.numLabels);
		newCR.epsilon = this.epsilon;
		newCR.numHidden = this.numHidden;
		newCR.numLabels = this.numLabels;
		newCR.numberOfInstances = this.numberOfInstances;
		
		for(int i=0; i<this.numHidden; i++){
			for(int j=0; j<this.numLabels; j++)
				newCR.decodeMatrix[i][j] = this.decodeMatrix[i][j];
		}		
		
		return newCR;
	}
	
	
	public boolean CRequals(multiLabelConceptualRepr ncr){
		boolean eq = true; 
		for(int i=0; i<this.decodeMatrix.length; i++){
			for(int j=0; j<this.decodeMatrix[i].length; j++){
				if(this.decodeMatrix[i][j] != ncr.decodeMatrix[i][j]){
					eq = false;
					break;
				}
			}
		}
		return eq; 
	}

}