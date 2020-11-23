package multi_labeled.RACE;
import java.util.ArrayList;
import java.util.List;

import weka.classifiers.UpdateableClassifier;
import weka.core.Instances;
import mulan.classifier.InvalidDataException;
import mulan.classifier.ModelInitializationException;
import mulan.data.MultiLabelInstances;
import mulan.evaluation.measure.Measure;


public interface myMultiLabelLearnerInterface extends UpdateableClassifier{
	
	public void build(MultiLabelInstances trainingSet) throws Exception;
	
	public void keepAllMeasures(List<Measure> m);
	

	public void updateClassifierBatch(Instances instances)throws Exception;
	
	public ArrayList<Double>[] measureGetter();
	
	public double[][] makePredictionForBatch(Instances instances)throws InvalidDataException, ModelInitializationException, Exception;
	
	
	public static Instances extractBatchLabels(MultiLabelInstances dataset){
		int[] featIndex = dataset.getFeatureIndices();
		Instances labelInstances = new Instances(dataset.getDataSet());
		for(int i=featIndex.length-1; i>=0; i--)
			labelInstances.deleteAttributeAt(featIndex[i]);
		return labelInstances;
	}
	
	
	public static double[][] getWindowMatrix(Instances batch, boolean labelFlag){
		double[][] labelMatrix = new double[batch.numInstances()][batch.numAttributes()];
		for(int r=0; r<batch.numInstances();r++){
			for(int c=0; c<batch.numAttributes();c++){
				if(labelFlag)
					labelMatrix[r][c] = (int) (2*batch.instance(r).value(c)-1);
				else
					labelMatrix[r][c] = batch.instance(r).value(c);
			}
		}
		return labelMatrix;
	}
	


}
