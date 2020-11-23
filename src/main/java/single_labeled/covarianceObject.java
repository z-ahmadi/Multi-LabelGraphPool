package single_labeled;


public class covarianceObject {
	
	double[][][] sigmaSmallOmega;
	boolean[] significancy;
	double[] lambda;
	int[] numInstances;
	boolean different; 
	int numClassDifs;
	
	public covarianceObject(double[][][] sigma, boolean[] sig, double[] lambda,int[] numInst, boolean dif, int countDifs){
		this.sigmaSmallOmega = sigma;
		this.significancy = sig;
		this.lambda = lambda;
		this.numInstances = numInst;
		this.different = dif;
		this.numClassDifs = countDifs;
	}
	
//	//added for Yang comparison, to save the mean of two covariances
//	public covarianceObject(double[][][] cov1 , int[] population1 , double[][][] cov2 , int[] population2){
//		double[][][] sigmaSmallOmega = new double[cov1.length][cov1[0].length][cov1[0].length];
//		for(int c=0; c<cov1.length; c++){
//			for(int i=0; i<cov1[0].length; i++){
//				for(int j=0; j<cov1[0].length;j++){
//					sigmaSmallOmega[c][i][j] = 
//				}
//			}
//		}
//	}
	
	public boolean[] getSignificancy() {
		return significancy;
	}

	public boolean isDifferent() {
		return different;
	}

	public double meanLambda(){
		double mean = 0;
		for(int i=0; i<lambda.length; i++)
			mean += lambda[i];
		return mean/lambda.length;
	}

}
