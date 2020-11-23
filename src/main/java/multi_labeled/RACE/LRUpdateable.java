package multi_labeled.RACE;
import java.util.ArrayList;
import java.util.Random;

import org.apache.commons.math3.linear.MatrixUtils;
import org.apache.commons.math3.linear.RealMatrix;
import org.apache.commons.math3.linear.SingularValueDecomposition;

import weka.classifiers.UpdateableClassifier;
import weka.core.Attribute;
import weka.core.Instance;
import weka.core.Instances;
import mulan.classifier.MultiLabelOutput;
import mulan.data.MultiLabelInstances;
import mulan.transformations.RemoveAllLabels;


public class LRUpdateable extends BRUpdateable{
	int NumHiddenNeuron, NumInputNeuron;
	double[][] IW,curBeta;
	double[] bias;
	RealMatrix beta, M;
	Random rand = new Random();
	String xmlPath, originalPath, ActivFunc; 
	double hardThreshold, similarityThresh;//, testThreshold;
	double[] neuronThresh;
	double[] prevFN, prevFP;

	
	public LRUpdateable(boolean first, boolean gram, UpdateableClassifier classifier, int MeasureLength, int inputN, int hiddenN, String xmlOriginal, String xmlReduced, 
						String ActivationFunc, double Hthresh, double Tthresh, double simThresh) {
		super(classifier, MeasureLength,first);
		NumInputNeuron = inputN;
		NumHiddenNeuron = hiddenN;
		originalPath = xmlOriginal;
		xmlPath = xmlReduced;
		ActivFunc = ActivationFunc;
		hardThreshold = Hthresh;	
		neuronThresh = new double[inputN];
		prevFN = new double[inputN];
		prevFP = new double[inputN];
		for(int i=0; i<neuronThresh.length; i++){
			neuronThresh[i] = Tthresh;
			prevFP[i] = Tthresh;
			prevFN[i] = Tthresh;
		}
		this.similarityThresh = simThresh;
		
		IW = new double[NumHiddenNeuron][NumInputNeuron];
		bias = new double[NumHiddenNeuron];
		
		//make orthogonal hyperplanes!
		gramschmidt GS = new gramschmidt(NumHiddenNeuron, NumInputNeuron+1);
		double[][] hyperWeights;
		if(gram){
			hyperWeights = GS.makeOrthogonals();
		}else{
			hyperWeights = GS.basisVector;
		}
		for(int i=0; i<NumHiddenNeuron; i++){
			for(int j=0; j<NumInputNeuron; j++){
				IW[i][j] = hyperWeights[i][j];
			}
			bias[i] = hyperWeights[i][NumInputNeuron];
		}
	}
	
	/**
	 * Constructor when the input weights are initialized
	 */
	public LRUpdateable(boolean first, double[][] iw, double[] b, UpdateableClassifier classifier, int MeasureLength, int inputN, int hiddenN, String xmlOriginal, String xmlReduced, String ActivationFunc, double Hthresh, double Tthresh) {
		super(classifier, MeasureLength, first);
		NumInputNeuron = inputN;
		NumHiddenNeuron = hiddenN;
		originalPath = xmlOriginal;
		xmlPath = xmlReduced;
		ActivFunc = ActivationFunc;
		hardThreshold = Hthresh;
		neuronThresh = new double[inputN];
		prevFN = new double[inputN];
		prevFP = new double[inputN];
		for(int i=0; i<neuronThresh.length; i++){
			neuronThresh[i] = Tthresh;
			prevFP[i] = Tthresh;
			prevFN[i] = Tthresh;
		}
		
		IW = new double[NumHiddenNeuron][NumInputNeuron];
		bias = new double[NumHiddenNeuron];
		
		for(int i=0; i<NumHiddenNeuron; i++){
			for(int j=0; j<NumInputNeuron; j++){
				this.IW[i][j] = iw[i][j];
			}
			this.bias[i] = b[i];
		}
//		this.IW = iw;
//		this.bias = b;
	}

	public LRUpdateable(boolean first, UpdateableClassifier classifier, int size) {
		// TODO Auto-generated constructor stub
		super(classifier,size,first);
	}


	public double[][] getIW() {
		return IW;
	}

	public double[] getBias() {
		return bias;
	}

	public double[][] getCurBeta() {
		return curBeta;
	}
	
//	public double[][] getNormalCurBeta() {
//		return normalizeDecoder(curBeta);
//	}

	public void setCurBeta(double[][] curBeta) {
		for(int i=0; i<curBeta.length; i++)
			for(int j=0; j<curBeta[i].length; j++)
				this.curBeta[i][j] = curBeta[i][j];
	}


	@Override
	protected void buildInternal(MultiLabelInstances trainingSet) throws Exception {
		
		Instances Labels = myMultiLabelLearnerInterface.extractBatchLabels(trainingSet);
		
		double[][] H0;
		
		if(ActivFunc.equals("No")){
			H0 = NoActivationFunction(myMultiLabelLearnerInterface.getWindowMatrix(Labels, true), IW, bias);
		}else if(ActivFunc.equals("HardLim")){
			H0 = hardLimActivationFunction(myMultiLabelLearnerInterface.getWindowMatrix(Labels, true), IW, bias, hardThreshold);
		}else{ //sigmoid function
			H0 = SigmoidActivationFunction(myMultiLabelLearnerInterface.getWindowMatrix(Labels, true), IW, bias);
		}

		Instances reducedData = addNewLabelsToBatchFeatures(trainingSet, H0);
		MultiLabelInstances multiReducedData = new MultiLabelInstances(reducedData, xmlPath);
		numLabels = multiReducedData.getNumLabels();
        labelIndices = multiReducedData.getLabelIndices();
        featureIndices = multiReducedData.getFeatureIndices();
        
        super.buildInternal(multiReducedData);
		
		RealMatrix h0 = MatrixUtils.createRealMatrix(H0);
		M = new SingularValueDecomposition((h0.transpose()).multiply(h0)).getSolver().getInverse();
//		beta = (new SingularValueDecomposition(h0).getSolver().getInverse())
//							.multiply(MatrixUtils.createRealMatrix(getWindowMatrix(Labels, true)));
		beta = M.multiply(h0.transpose().multiply(MatrixUtils.createRealMatrix(myMultiLabelLearnerInterface.getWindowMatrix(Labels, true))));
		curBeta = beta.getData();
		// both encoding and decoding are hidden x original labels
//		System.out.println("encoding matrix size: "+IW.length+" x "+IW[0].length);
//		System.out.println("decoding matrix size: "+curBeta.length+" x "+curBeta[0].length);
	}

	
	public Instances addNewLabelsToBatchFeatures(MultiLabelInstances dataset, double[][] H0) throws Exception{
		//delete original labels from dataset
		Instances newData = new RemoveAllLabels().transformInstances(dataset);
		//add attributes for new labels
		ArrayList<String> values = new ArrayList<String>();
		values.add("0"); values.add("1");
		for(int i=0; i<NumHiddenNeuron; i++)
			newData.insertAttributeAt(new Attribute("hiddenLabel_"+i, values),newData.numAttributes());
				
				
		for(int i=0; i<H0.length; i++){
			for(int j=0; j<H0[i].length; j++){
				newData.instance(i).setValue(newData.attribute("hiddenLabel_"+j), H0[i][j]);
			}
		}
		
		return newData;
	}
	
	public double[][] NoActivationFunction(double[][] P, double[][] IW, double[] bias){
		RealMatrix p = MatrixUtils.createRealMatrix(P);
		RealMatrix iw = MatrixUtils.createRealMatrix(IW);
		RealMatrix V = p.multiply(iw.transpose());
		
		double[][] biasMat = new double[P.length][bias.length];
		for(int i=0; i<P.length; i++)
			biasMat[i] = bias;
		RealMatrix biasMatrix = MatrixUtils.createRealMatrix(biasMat);
		
		V = V.add(biasMatrix);
		double[][] v = V.getData();
		
		System.out.println("==================================");
		for(int i=0; i<v.length; i++){
			for(int j=0; j<v[i].length; j++){
				System.out.print(v[i][j]+" ");
			}
			System.out.println();
		}
		System.out.println("==================================");
		
		return v;
	}
	
	public double[][] hardLimActivationFunction(double[][] P, double[][] IW, double[] bias, double thresh){
		RealMatrix p = MatrixUtils.createRealMatrix(P);
		RealMatrix iw = MatrixUtils.createRealMatrix(IW);
		RealMatrix V = p.multiply(iw.transpose());
		
		double[][] biasMat = new double[P.length][bias.length];
		for(int i=0; i<P.length; i++)
			biasMat[i] = bias;
		RealMatrix biasMatrix = MatrixUtils.createRealMatrix(biasMat);
		
		V = V.add(biasMatrix);
		double[][] v = V.getData();
		
		//comment out for runs!
		for(int i=0; i<v.length; i++){
			for(int j=0; j<v[i].length; j++){
				if(v[i][j] >= thresh){ //changed this one 
					v[i][j] = 1;
				}else{
					v[i][j] = 0;
				}
			}
		}
		
		return v;
	}
	
	public double[][] SigmoidActivationFunction(double[][] P, double[][] IW, double[] bias){
		RealMatrix p = MatrixUtils.createRealMatrix(P);
		RealMatrix iw = MatrixUtils.createRealMatrix(IW);
		RealMatrix V = p.multiply(iw.transpose());
		
		double[][] biasMat = new double[P.length][bias.length];
		for(int i=0; i<P.length; i++)
			biasMat[i] = bias;
		RealMatrix biasMatrix = MatrixUtils.createRealMatrix(biasMat);
		
		V = V.add(biasMatrix);
		double[][] v = V.getData();
		for(int i=0; i<v.length; i++){
			for(int j=0; j<v[i].length; j++){
				v[i][j] = 1/(double)(1+Math.exp(-v[i][j]));
			}
		}
		
		return v;
	}	
	

	public void updateClassifierForMLBatch(MultiLabelInstances batch, boolean adaptive, boolean shallowAE) throws Exception {
		Instances Labels = myMultiLabelLearnerInterface.extractBatchLabels(batch);
		double[][] labels = myMultiLabelLearnerInterface.getWindowMatrix(Labels, true); //make labels -1 , +1	
		
		
		//update classifiers
		if(shallowAE){
//			IW = normalizeDecoder(beta.getData());  ////TODO: better to do an orthonormal projection instead of only normalizing (this way always -1 and 1 are there in the weights) 
			IW = beta.getData(); 
		}
		double[][] H;
		if(ActivFunc.equals("No")){
			H = NoActivationFunction(labels, IW, bias);
		}else if(ActivFunc.equals("HardLim")){
			H = hardLimActivationFunction(labels, IW, bias, hardThreshold);
		}else{
			H = SigmoidActivationFunction(labels, IW, bias);
		}
		Instances reducedData = addNewLabelsToBatchFeatures(batch, H);
//		MultiLabelInstances multiReducedData = new MultiLabelInstances(reducedData, xmlPath);
		super.updateClassifierBatch(reducedData);
		
		RealMatrix h = MatrixUtils.createRealMatrix(H);
		// M = M - M * H' * (eye(Block) + H * M * H')^(-1) * H * M
		RealMatrix tmp = (MatrixUtils.createRealIdentityMatrix(batch.getNumInstances())).add(h.multiply(M.multiply(h.transpose())));
		tmp = (new SingularValueDecomposition(tmp).getSolver().getInverse()).multiply(h.multiply(M));
		M = M.subtract(M.multiply(h.transpose().multiply(tmp)));
		
		//beta = beta + M * H' * (Tn - H * beta)
		tmp = MatrixUtils.createRealMatrix(myMultiLabelLearnerInterface.getWindowMatrix(Labels, true)).subtract((h.multiply(beta)));
		beta = beta.add(M.multiply(h.transpose().multiply(tmp)));
		
		curBeta = beta.getData();
		
	}
	
	
//	// normalize the decoding matrix which is the least square solution and can be of any value, 
//	// for each hidden node (as GS also does its normalization for them) 
//	//we first find the min and max of each vector and then scale it to [-1,1]
//	public double[][] normalizeDecoder(double[][] matrix){
////		System.out.println("^^^^^^^ size of beta = "+matrix.length+"x"+matrix[0].length);
//		double[][] retMat = new double[NumHiddenNeuron][NumInputNeuron];
//		double genMin = Double.MAX_VALUE, genMax = Double.MIN_VALUE;
//		
//		for(int i=0; i<NumHiddenNeuron; i++) {
//			double min = matrix[i][0], max = matrix[i][0];  //assign the first element of the vector as an initialization 
//			for(int j=0; j<NumInputNeuron; j++) { //find the max and min of the vector
//				if(matrix[i][j] > max) {
//					max = matrix[i][j];
//				}
//				if(matrix[i][j] < min) {
//					min = matrix[i][j];
//				}
//			}
//			
//			if(min < genMin)
//				genMin = min;
//			if(max > genMax)
//				genMax = max;
//			
//			for(int j=0; j<NumInputNeuron; j++) { 
//				retMat[i][j] = (2*matrix[i][j]/(max-min))-((min+max)/(max-min));
//				
//				if(retMat[i][j] < -1 && retMat[i][j] > -1.0001) //just test for underflow
//					retMat[i][j] = -1; 
//				
//				if(retMat[i][j] > 1 && retMat[i][j] < 1.0001) //just test for overflow
//					retMat[i][j] = 1; 
//			}
//		}
//		
//		System.out.println("Decode min="+genMin+" Max="+genMax);
//		
//		return retMat;
//	}
	

	@Override
	protected MultiLabelOutput makePredictionInternal(Instance inst) {
		Instance instance = inst;
		boolean[] bipartition = new boolean[NumInputNeuron];
		double[] Confidences = new double[NumInputNeuron];
        double[][] internalConfidences = new double[1][numLabels];
        double[][] classLabel = new double[1][numLabels];
               
        //comment for hoeffding
//        RemoveAllLabels.transformInstance(instance, labelIndices);
        
//        instance.insertAttributeAt(/*new Attribute("hiddenLabel_"+counter, values),*/instance.numAttributes());
        
//        ArrayList<String> values = new ArrayList<String>();
//		values.add("0"); values.add("1");

        for (int counter = 0; counter < numLabels; counter++) {
        	//add for hoeffding
        	instance = brt.transformInstance(inst, counter);
            double distribution[];
            try {
                distribution = ensemble[counter].distributionForInstance(instance);
            } catch (Exception e) {
                System.out.println(e);
                return null;
            }
            int maxIndex = (distribution[0] > distribution[1]) ? 0 : 1; 
            classLabel[0][counter] = maxIndex;

            // The confidence of the label being equal to 1
            internalConfidences[0][counter] = distribution[1];
        }
        
        RealMatrix learnerPred = MatrixUtils.createRealMatrix(classLabel); 
		RealMatrix Ttest = learnerPred.multiply(beta);
		double[][] Tlabels = Ttest.getData();
		//not used yet
		RealMatrix learberConf = MatrixUtils.createRealMatrix(internalConfidences);
		RealMatrix Tconf = learberConf.multiply(beta);
		double[][] Tconfs = Tconf.getData();
		
		for(int i=0; i < NumInputNeuron; i++){
			// Ensure correct predictions both for class values {0,1} and {1,0}
	        bipartition[i] = (Tlabels[0][i]  >= neuronThresh[i]/*== 1*/) ? true : false; /////////////
	        Confidences[i] = Tconfs[0][i];
		}
		
		//confidence is the distribution of class 1 multiplied by beta
        MultiLabelOutput mlo = new MultiLabelOutput(bipartition, Confidences);
        return mlo;
	}
}
