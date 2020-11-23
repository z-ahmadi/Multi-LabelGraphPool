package single_labeled;


import java.io.PrintWriter;
import java.util.ArrayList;

import org.apache.commons.math3.linear.Array2DRowRealMatrix;
import org.apache.commons.math3.linear.LUDecomposition;
import org.apache.commons.math3.linear.RealMatrix;
import org.apache.commons.math3.stat.correlation.Covariance;

import weka.core.Attribute;
import weka.core.Instance;
import weka.core.Instances;

public class nonIIDConceptualRepr {
	static double epsilon;
	
	private double[][] meanVector; //mean of different features over classes
	//covariance matrix is calculated by dividing n-1!
	private double[][][] covarianceMatrix;  //sum of squared (not covariance!) of different features over classes (not divided by n-1 for unbiased estimate)
	private double[] determinant; 
	public double[][][] correlationMatrix; 
	public int[] numberOfInstances; //if = 0 : no reliable mean and cov, if = 1 no reliable cov! 
	Attribute classAttr;
	public int numberOfFeatures, numClasses; 
	PrintWriter pw; 
	

	public double[][] getMeanVector() {
		return meanVector;
	}

	public void setMeanVector(double[][] meanVector) {
		this.meanVector = meanVector;
	}

	public double[][][] getCovarianceMatrix() {
		return covarianceMatrix;
	}

	//remove noisy covariances
	public double[][] removeNoiseFromCovarianceMatrix(double[][] cov){
		boolean noiseFlag = false;
		for(int r=0; r<cov.length; r++){
			for(int c=0; c<cov[r].length; c++){
				if(Math.abs(cov[r][c]) < epsilon && cov[r][c] != 0){
					noiseFlag = true;
					cov[r][c] = 0;
//					if(cov[r][c] > 0)
//						cov[r][c] = epsilon;
//					else
//						cov[r][c] = -epsilon;
				}
			}
		}
		
		if(noiseFlag)
			System.err.println("NOTICE: I found noise reduction in cov matrix!");
		
		return cov;
	}

	public void setCovarianceMatrix(double[][][] covarianceMatrix) {
		for(int i=0; i<covarianceMatrix.length; i++){
			covarianceMatrix[i] = removeNoiseFromCovarianceMatrix(covarianceMatrix[i]);			
		}
		this.covarianceMatrix = covarianceMatrix;
	}

	public double[] getDeterminant() {
		return determinant;
	}

	public void setDeterminant(double[] determinant) {
		this.determinant = determinant;
	}
	
	
	public nonIIDConceptualRepr(int numericFeatures, int numClasses) {
		meanVector = new double[numericFeatures][numClasses];
		covarianceMatrix = new double[numClasses][numericFeatures][numericFeatures];
		determinant = new double[numClasses];
		correlationMatrix = new double[numClasses][numericFeatures][numericFeatures];
		numberOfInstances = new int[numClasses];		
	}
	
	
	public nonIIDConceptualRepr(Instance[] dataSet, ArrayList<Integer> numericIndx, double eps, String cvectorType) {
		ArrayList<Attribute> attrs = new ArrayList<Attribute>();
		for(int i=0; i<dataSet[0].numAttributes(); i++)
			attrs.add(dataSet[0].attribute(i));
		Instances inst = new Instances("BufferData", attrs, dataSet.length);
		for(int i=0; i<dataSet.length; i++)
			inst.add(dataSet[i]);

		this.epsilon = eps;
		numberOfFeatures = numericIndx.size();
		meanVector = new double[numericIndx.size()][inst.numClasses()];
		covarianceMatrix = new double[inst.numClasses()][numericIndx.size()][numericIndx.size()];
		determinant = new double[inst.numClasses()];
		correlationMatrix = new double[inst.numClasses()][numericIndx.size()][numericIndx.size()];
		numberOfInstances = new int[inst.numClasses()];
		classAttr=inst.attribute(inst.numAttributes()-1);
		numClasses = covarianceMatrix.length;
		
		//add instances of each class to an array list
		ArrayList<ArrayList<Instance>> classInstances=new ArrayList<ArrayList<Instance>>();
		for (int i=0; i<classAttr.numValues(); i++)
			classInstances.add(new ArrayList<Instance>());
		for (int i=0; i<inst.numInstances(); i++){
			for (int j=0; j<classAttr.numValues(); j++){//TODO: generalize
				if ((int)inst.instance(i).classValue()== j){
					classInstances.get(j).add(inst.instance(i));
					numberOfInstances[j]++;
					break;
				}
			}
		}
		
		//make mean and covariance 
		setStatsOfWindow(classInstances, numericIndx, cvectorType);
	}
	
	public nonIIDConceptualRepr(Instances dataSet, ArrayList<Integer> numericIndx, PrintWriter pw, double eps, String cvectorType) {
		this.epsilon = eps;
		this.pw = pw;
		numberOfFeatures = numericIndx.size();
		meanVector = new double[numericIndx.size()][dataSet.numClasses()];
		covarianceMatrix = new double[dataSet.numClasses()][numericIndx.size()][numericIndx.size()];
		determinant = new double[dataSet.numClasses()];
		correlationMatrix = new double[dataSet.numClasses()][numericIndx.size()][numericIndx.size()];
		numberOfInstances = new int[dataSet.numClasses()];
		classAttr=dataSet.attribute(dataSet.numAttributes()-1);
		numClasses = covarianceMatrix.length;
		
		//add instances of each class to an array list
		ArrayList<ArrayList<Instance>> classInstances=new ArrayList<ArrayList<Instance>>();
		for (int i=0; i<classAttr.numValues(); i++)
			classInstances.add(new ArrayList<Instance>());
		for (int i=0; i<dataSet.numInstances(); i++){
			for (int j=0; j<classAttr.numValues(); j++){//TODO: generalize
				if ((int)dataSet.instance(i).classValue()== j){
					classInstances.get(j).add(dataSet.instance(i));
					numberOfInstances[j]++;
					break;
				}
			}
		}
		
//		System.out.println("number of instances in different classes");
//		for(int i=0; i<dataSet.numClasses(); i++)
//			System.out.print(numberOfInstances[i]+" ");
//		System.out.println();
		
		//make mean and covariance 
		setStatsOfWindow(classInstances, numericIndx, cvectorType);
		
	}
	
	void setStatsOfWindow(ArrayList<ArrayList<Instance>> categorizedInst, ArrayList<Integer> numericIndx, String cvector){
		for(int i=0; i<categorizedInst.size(); i++){ //for each class
			if(categorizedInst.get(i).size()>0){
				double[][] instances = new double[numericIndx.size()][categorizedInst.get(i).size()];

//				System.out.println("mean "+i+" = ");
				for(int c=0; c<numericIndx.size();c++){
					int indC = numericIndx.get(c); //get the index of c'th numeric attribute
					for(int r=0; r<categorizedInst.get(i).size() ; r++){	
						instances[c][r] = categorizedInst.get(i).get(r).value(indC);
						meanVector[c][i] += instances[c][r];
					}
					
					if(categorizedInst.get(i).size() > 0)
						meanVector[c][i] /= categorizedInst.get(i).size();
					else
						System.out.println("number of instances is zero in class "+i);
//					System.out.print(meanVector[c][i]+" ");
				}
				
				//loop on data and use covariance method
				if(cvector.equals("NoCovariance")){
					for(int r=0; r<numericIndx.size(); r++){
						for(int numInst=0; numInst<categorizedInst.get(i).size(); numInst++){
							covarianceMatrix[i][r][r] += Math.pow((instances[r][numInst]-meanVector[r][i]), 2);
						}
					}
				}else{
					for(int r=0; r<numericIndx.size(); r++){
						for(int c=r; c<numericIndx.size(); c++){
							for(int numInst=0; numInst<categorizedInst.get(i).size(); numInst++){
								covarianceMatrix[i][r][c] += (instances[r][numInst]-meanVector[r][i])
																*(instances[c][numInst]-meanVector[c][i]);
							}						
							covarianceMatrix[i][c][r] = covarianceMatrix[i][r][c];
						}
					}
				}
				
				
				covarianceMatrix[i] = removeNoiseFromCovarianceMatrix(covarianceMatrix[i]);
				
							
				//I do not know why sometimes correlation is NaN! cause I check if  the number of instances 
				//are more than 1!
//				System.out.println("\ncorr = ");
				for(int r=0;r<numericIndx.size(); r++){
					for(int c=r;c<numericIndx.size(); c++){
						if(categorizedInst.get(i).size()>1){
							correlationMatrix[i][r][c] = covarianceMatrix[i][r][c]/(Math.sqrt(covarianceMatrix[i][r][r])
																					*Math.sqrt(covarianceMatrix[i][c][c]));
							correlationMatrix[i][c][r] = correlationMatrix[i][r][c];
//							System.out.print(correlationMatrix[r][c][i]+" ");
						}
					}
//					System.out.println();
				}
			}
			
		}
		calculateDeterminant(categorizedInst.size());
	}
	
	
	public void calculateDeterminant(int numClass){
		for(int i=0; i<numClass; i++){
			RealMatrix a = new Array2DRowRealMatrix(covarianceMatrix[i]);
	        LUDecomposition LUDdecompose = new LUDecomposition(a);
	        determinant[i] = LUDdecompose.getDeterminant();
		}
	}
	
	
	public void calculateDiagonalDeterminant(int numClass){
		for(int i=0; i<numClass; i++){
			int determin = 1;
			for(int j=0; j<covarianceMatrix[i].length; j++){
				determin *= covarianceMatrix[i][j][j];
			}
			determinant[i] = determin;
		}
	}
	
	
	public void updateMeanVector(nonIIDConceptualRepr newcv){
		for(int i=0; i<numberOfFeatures; i++)
			for(int j=0; j<numClasses; j++){ 
				meanVector[i][j] = (meanVector[i][j]* numberOfInstances[j] + newcv.meanVector[i][j]* newcv.numberOfInstances[j])
   				                    /(numberOfInstances[j]+newcv.numberOfInstances[j]);
			}
	}
	
	public void updateNumberOfInstances(nonIIDConceptualRepr newcv){
		for(int i=0; i<numClasses;i++){
			numberOfInstances[i] += newcv.numberOfInstances[i];
		}
	}
	
	// likelihood ratio, formula 8 , p.413 ,for 2 population 
	//I should check covariance matrix if i want to use it
	public double[] statisticalCovarianceTest(nonIIDConceptualRepr cr1 , nonIIDConceptualRepr cr2){
		int numberOfClasses = cr1.determinant.length;
		double[] lambda = new double[numberOfClasses];
		
		for(int i=0; i< numberOfClasses; i++){
			double[][] A = cr1.covarianceMatrix[i];
			for(int r=0; r<A.length; r++)
				for(int c=0; c<A[r].length; c++)
					A[r][c] += cr2.covarianceMatrix[i][r][c];
			
			RealMatrix a = new Array2DRowRealMatrix(A);
	        LUDecomposition LUDdecompose = new LUDecomposition(a);
	        double deterA = LUDdecompose.getDeterminant();
	        
	        double N1 = cr1.numberOfInstances[i] , N2 = cr2.numberOfInstances[i];
	        double N = N1 + N2;
			int p = cr1.numberOfFeatures;
			
			lambda[i] = Math.pow(cr1.determinant[i], N1/2) * Math.pow(cr2.determinant[i], N2/2)*
						Math.pow( N , p*N /2) /
						(Math.pow(deterA , N/2) * Math.pow(N1, p*N1/2) * Math.pow(N2, p*N2/2));		  
						
		}
		
		return lambda; 
	}
	
	
	public covarianceObject NormalEqualityTest(nonIIDConceptualRepr cr1 , nonIIDConceptualRepr cr2, double chiIndex, String cvType){
		int numFeatures = cr1.numberOfFeatures;
		double[][][] cr1Cov = cr1.covarianceMatrix , cr2Cov = cr2.covarianceMatrix;
		double[][][] sigmaOmega = new double[cr1Cov.length][numFeatures][numFeatures];
		double[][][] B1 = new double[cr1Cov.length][numFeatures][numFeatures];
		double[][][] B2 = new double[cr1Cov.length][numFeatures][numFeatures];
		double[][][] sigmaSmallOmega = new double[cr1Cov.length][numFeatures][numFeatures];
		boolean[] significancy = new boolean[cr1Cov.length];
		double[] lambdaArr = new double[cr1Cov.length];
		
		for(int i=0; i<cr1Cov.length; i++){		//loop on number of class values
			int N1 = cr1.numberOfInstances[i], N2 = cr2.numberOfInstances[i];
			int N = N1+N2;
			if(N1>0 && N2>0){ //we should have at least one instance in this class
				//calculate yBar
				double[]  yBar = new double[numFeatures];
				for(int r=0; r<numFeatures; r++){
					yBar[r] = (N1*cr1.meanVector[r][i]+N2*cr2.meanVector[r][i])/N;
				}
				//calculate sigma
				double[][] weightedMeanDiff = new double[numFeatures][numFeatures];
				
				if(cvType.equals("NoCovariance")){
					for(int j=0; j<numFeatures; j++){
						weightedMeanDiff[j][j] = N1*Math.pow((cr1.meanVector[j][i]-yBar[j]), 2)+
												 N2*Math.pow((cr2.meanVector[j][i]-yBar[j]), 2);
						
						
						sigmaOmega[i][j][j] = cr1Cov[i][j][j] + cr2Cov[i][j][j];
						
						sigmaSmallOmega[i][j][j] = sigmaOmega[i][j][j] + weightedMeanDiff[j][j];			//calculate based on eq.13-p.345		
					}
				}else{				
					for(int j=0; j<numFeatures; j++){
						for(int k=0; k<numFeatures; k++){
							weightedMeanDiff[j][k] = N1*(cr1.meanVector[j][i]-yBar[j])*(cr1.meanVector[k][i]-yBar[k])+
													 N2*(cr2.meanVector[j][i]-yBar[j])*(cr2.meanVector[k][i]-yBar[k]);
							
							
							sigmaOmega[i][j][k] = cr1Cov[i][j][k] + cr2Cov[i][j][k];
							
							sigmaSmallOmega[i][j][k] = sigmaOmega[i][j][k] + weightedMeanDiff[j][k];		
						}
					}
				}
				
				
				B1[i] = removeNoiseFromCovarianceMatrix(cr1Cov[i]);
				B2[i] = removeNoiseFromCovarianceMatrix(cr2Cov[i]);
				sigmaSmallOmega[i] = removeNoiseFromCovarianceMatrix(sigmaSmallOmega[i]);
				
				//we should remove non-decisive features for all of them  
				double detB1 = checkZeroVariance(B1[i]);
				double detB2 = checkZeroVariance(B2[i]);
				double detAB = checkZeroVariance(sigmaSmallOmega[i]);
				
				double logLambda = (N1/2)*Math.log(detB1) + (N2/2)*Math.log(detB2)- (N/2)*Math.log(detAB) + (numFeatures*N/2)*Math.log(N)
								 - (numFeatures*N1/2)*Math.log(N1) - (numFeatures*N2/2)*Math.log(N2);
								        
		        if((detB1==0 || detB2==0) && detAB==0)
		        	logLambda = (numFeatures*N/2)*Math.log(N)-(numFeatures*N1/2)*Math.log(N1)-(numFeatures*N2/2)*Math.log(N2);
		        
		        double rou = 1-((2*Math.pow(numFeatures, 2) + 9*numFeatures + 11)/(6*N*(numFeatures+3)))*(1+(N1/N2)+(N2/N1));
		        double lambda = -2*rou*logLambda; //natural log, N-q-1=n1+n2-3

		        
		        if(N1>1 && N2>1 && pw != null){
		        	pw.write("conceptualTest #c"+i+": #p1="+N1+" #p2="+N2+" |sigma bigOmega|="+String.format( "%.2f",detAB)+
		        			" |sigma smallOmega|="+String.format( "%.2f",detB1*detB2)+" Lambda="+String.format( "%.2f",lambda)+"\n");
		        }
		        lambdaArr[i] = lambda;
		       //test of chi square with alpha and p (number of features)
		        if(lambda > chiIndex) //reject the equality of means
		        	significancy[i] = true;
		        else
		        	significancy[i] = false; 
			}
		        	
		}
		
		int count = 0;
		for(int i=0; i<significancy.length; i++){
			if(significancy[i])
				count++;
		}
		
		//two concepts are counted as different if at least they are significantly different on one of class values!
		boolean dif; 
		if(count>0)
			dif = true;
		else
			dif = false;
		
		return new covarianceObject(sigmaSmallOmega, significancy, lambdaArr, cr2.numberOfInstances, dif, count);

	}
	
	
	
	//test of mean differences with the assumption of equal covariance
	//return true if significantly different!
	//cr1 : conceptual vector of current batch
	public covarianceObject statisticalMeanTest(nonIIDConceptualRepr cr1 , nonIIDConceptualRepr cr2, double chiIndex, String cvType){
		int numFeatures = cr1.numberOfFeatures;
		double[][][] cr1Cov = cr1.covarianceMatrix , cr2Cov = cr2.covarianceMatrix;
		double[][][] sigmaOmega = new double[cr1Cov.length][numFeatures][numFeatures];
		double[][][] sigmaSmallOmega = new double[cr1Cov.length][numFeatures][numFeatures];
		boolean[] significancy = new boolean[cr1Cov.length];
		double[] lambdaArr = new double[cr1Cov.length];
		
		for(int i=0; i<cr1Cov.length; i++){		//loop on number of class values
			if(cr1.numberOfInstances[i]+cr2.numberOfInstances[i]>0){ //we should have at least one instance in this class
				//calculate yBar
				double[]  yBar = new double[numFeatures];
				for(int r=0; r<numFeatures; r++){
					yBar[r] = (cr1.numberOfInstances[i]*cr1.meanVector[r][i]+cr2.numberOfInstances[i]*cr2.meanVector[r][i])
							  /(cr1.numberOfInstances[i]+cr2.numberOfInstances[i]);
				}
				//calculate sigma
				double[][] weightedMeanDiff = new double[numFeatures][numFeatures];
				
				if(cvType.equals("NoCovariance")){
					for(int j=0; j<numFeatures; j++){
						weightedMeanDiff[j][j] = cr1.numberOfInstances[i]*Math.pow((cr1.meanVector[j][i]-yBar[j]), 2)+
												 cr2.numberOfInstances[i]*Math.pow((cr2.meanVector[j][i]-yBar[j]), 2);
						
						
						sigmaOmega[i][j][j] = cr1Cov[i][j][j] + cr2Cov[i][j][j];
						
						sigmaSmallOmega[i][j][j] = sigmaOmega[i][j][j] + weightedMeanDiff[j][j];			//calculate based on eq.13-p.345		
					}
				}else{
					for(int j=0; j<numFeatures; j++){
						for(int k=0; k<numFeatures; k++){
							weightedMeanDiff[j][k] = cr1.numberOfInstances[i]*(cr1.meanVector[j][i]-yBar[j])*(cr1.meanVector[k][i]-yBar[k])+
													 cr2.numberOfInstances[i]*(cr2.meanVector[j][i]-yBar[j])*(cr2.meanVector[k][i]-yBar[k]);
							
							
							sigmaOmega[i][j][k] = cr1Cov[i][j][k] + cr2Cov[i][j][k];
							
							sigmaSmallOmega[i][j][k] = sigmaOmega[i][j][k] + weightedMeanDiff[j][k];			//calculate based on eq.13-p.345		
						}
					}
				}			
				
				sigmaOmega[i] = removeNoiseFromCovarianceMatrix(sigmaOmega[i]);
				sigmaSmallOmega[i] = removeNoiseFromCovarianceMatrix(sigmaSmallOmega[i]);
				
				double detNumerator = checkZeroVariance(sigmaOmega[i]);
				double detDenominator = checkZeroVariance(sigmaSmallOmega[i]);
				
//				RealMatrix a = new Array2DRowRealMatrix(sigmaOmega[i]);
//		        LUDecomposition LUDdecompose = new LUDecomposition(a);
//		        double detNumerator = LUDdecompose.getDeterminant();
		        
//		        a = new Array2DRowRealMatrix(sigmaSmallOmega[i]);
//		        LUDdecompose = new LUDecomposition(a);
//		        double detDenominator = LUDdecompose.getDeterminant();
		        		        
		        double U = detNumerator/detDenominator;
		        if(detNumerator == 0 && detDenominator == 0)
		        	U = 1;
		        double lambda = -(cr1.numberOfInstances[i]+cr2.numberOfInstances[i]-3)*Math.log(U); //natural log, N-q-1=n1+n2-3
		        
		        ///think more about it! 
		        //if the determinant of the between-cov is zero it means instances are correlated so...
//		        if(detNumerator == 0)
//		        	lambda = 0;
		        
//		        pw.println("Sigma Big Omega : ");
//		        for(int j=0; j<numFeatures; j++){
//					for(int k=0; k<numFeatures; k++){
//						pw.print(sigmaOmega[i][j][k]+" ");
//					}	
//					pw.println();
//		        }
//		        pw.println("Sigma Small Omega : ");
//		        for(int j=0; j<numFeatures; j++){
//					for(int k=0; k<numFeatures; k++){
//						pw.print(sigmaSmallOmega[i][j][k]+" ");
//					}	
//					pw.println();
//		        }
		        
		        if(cr1.numberOfInstances[i]>1 && cr2.numberOfInstances[i]>1 && pw != null){
		        	pw.write("conceptualTest #c"+i+": #p1="+cr1.numberOfInstances[i]+" #p2="+cr2.numberOfInstances[i]+
		        			" |sigma bigOmega|="+String.format( "%.2f",detNumerator)+" |sigma smallOmega|="+
		        			String.format( "%.2f",detDenominator)+" Lambda="+String.format( "%.2f",lambda)+"\n");
		        }
		        lambdaArr[i] = lambda;
		       //test of chi square with alpha and p (number of features)
		        if(lambda > chiIndex) //reject the equality of means
		        	significancy[i] = true;
		        else
		        	significancy[i] = false; 
			}
		        	
		}
		
		int count = 0;
		for(int i=0; i<significancy.length; i++){
			if(significancy[i])
				count++;
		}
		
		//two concepts are counted as different if at least they are significantly different on one of class values!
		boolean dif; 
		if(count>0)
			dif = true;
		else
			dif = false;
		
		return new covarianceObject(sigmaSmallOmega, significancy, lambdaArr, cr2.numberOfInstances, dif, count);
		
	}
	
	//checks if the variance of features are zero; if yes, removes those features; then calculates the determinant 
	//if necessary: return an object including determinant and featureIndx 
	public double checkZeroVariance(double[][] cov){
		ArrayList<Integer> featureIndx = new ArrayList<Integer>();
		RealMatrix a;
		double determin = 0;
		
		for(int i=0; i<cov.length; i++){
			if(cov[i][i] == 0){
				featureIndx.add(i);
			}
		}
		
		if(featureIndx.size() == 0){
			a = new Array2DRowRealMatrix(cov);	
		}else{
			int newSize = cov.length - featureIndx.size(); //size of new covariance matrix
			double[][] newCov = new double[newSize][newSize];
			int r = 0, c;
			if(newSize > 0){
				for(int i=0; i<cov.length; i++){
					c = 0;
					for(int j=0; j<cov.length; j++){
						if(!featureIndx.contains(i) && !featureIndx.contains(j)){
							newCov[r][c] = cov[i][j];
							c++;
						}
					}
					if(!featureIndx.contains(i))
						r++;
				} //end of copying the covariance
			}else
				return 0;	//if all the variances are zero!

			a = new Array2DRowRealMatrix(newCov);  
			
		}

        LUDecomposition LUDdecompose = new LUDecomposition(a);
        determin = LUDdecompose.getDeterminant();		
		
		return determin;
	}
	
	
	
	public nonIIDConceptualRepr copy(){
		nonIIDConceptualRepr newCR = new nonIIDConceptualRepr(this.numberOfFeatures, this.numClasses);
		newCR.epsilon = this.epsilon;
		newCR.numberOfFeatures = this.numberOfFeatures;
		newCR.numClasses = this.numClasses;
		newCR.classAttr = this.classAttr;
		
		for(int i=0; i<this.numberOfFeatures; i++){
			for(int j=0; j<this.numClasses; j++)
				newCR.meanVector[i][j] = this.meanVector[i][j];
		}
		for(int i=0; i<this.numClasses; i++){
			newCR.numberOfInstances[i] = this.numberOfInstances[i];
			newCR.determinant[i] = this.determinant[i];			
		}
		for(int i=0; i<this.numClasses; i++){
			for(int j=0; j<this.numberOfFeatures; j++){
				for(int k=0; k<this.numberOfFeatures; k++){
					newCR.covarianceMatrix[i][j][k] = this.covarianceMatrix[i][j][k];
					newCR.correlationMatrix[i][j][k] = this.correlationMatrix[i][j][k];
				}
			}
		}
			
		
		return newCR;
	}
	
	//checks if the mean and covariance are the same
	public boolean equals(nonIIDConceptualRepr ncr){
		if(meanEquals(ncr) && covarianceEquals(ncr))
			return true;
		else
			return false;
	}
	
	public boolean covarianceEquals(nonIIDConceptualRepr ncr){
		boolean eq = true; 
		for(int i=0; i<this.covarianceMatrix.length; i++){
			for(int j=0; j<this.covarianceMatrix[i].length; j++){
				for(int k=0; k<this.covarianceMatrix[i][j].length; k++){
					if(this.covarianceMatrix[i][j][k] != ncr.covarianceMatrix[i][j][k]){
						eq = false;
						break;
					}					
				}
			}
		}
		return eq;
	}
	
	public boolean meanEquals(nonIIDConceptualRepr ncr){
		boolean eq = true; 
		for(int i=0; i<this.meanVector.length; i++){
			for(int j=0; j<this.meanVector[i].length; j++){
				if(this.meanVector[i][j] != ncr.meanVector[i][j]){
					eq = false;
					break;
				}
			}
		}
		return eq; 
	}

}