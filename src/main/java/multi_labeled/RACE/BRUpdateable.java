package multi_labeled.RACE;
import java.util.ArrayList;
import java.util.List;

import moa.classifiers.bayes.NaiveBayes;
import moa.core.InstancesHeader;
import mulan.classifier.InvalidDataException;
import mulan.classifier.ModelInitializationException;
import mulan.classifier.MultiLabelOutput;
import mulan.data.MultiLabelInstances;
import mulan.transformations.BinaryRelevanceTransformation;
import weka.classifiers.AbstractClassifier;
import weka.classifiers.Classifier;
import weka.classifiers.UpdateableClassifier;
import weka.classifiers.bayes.NaiveBayesUpdateable;
import weka.classifiers.meta.MOA;
import weka.core.Instance;
import weka.core.Instances;
import mulan.evaluation.measure.Measure;

public class BRUpdateable extends myBinaryRelevance implements
		UpdateableClassifier,myMultiLabelLearnerInterface {

	Instances[] headers;

	/**
	 * 
	 */
	private static final long serialVersionUID = 5425630063669472672L;
	protected String[] correspondence;
	public static ArrayList<Double>[] outputMeasures;
	protected boolean labelFirst;

	public BRUpdateable(UpdateableClassifier classifier, int MeasureLength, boolean first) {
		super((Classifier) classifier);
		this.labelFirst = first;
		outputMeasures = new ArrayList[MeasureLength];
		for(int i=0; i<MeasureLength; i++)
			outputMeasures[i] = new ArrayList<Double>();
	}

	public BRUpdateable(MOA classifier, int MeasureLength, boolean first) {
		super(classifier);
		this.labelFirst = first;
		outputMeasures = new ArrayList[MeasureLength];
		for(int i=0; i<MeasureLength; i++)
			outputMeasures[i] = new ArrayList<Double>();
	}
	
	public void keepAllMeasures(List<Measure> m){
		for(int i=0; i<m.size(); i++){
			outputMeasures[i].add(m.get(i).getValue());
		}
	}

	protected void buildInternal(MultiLabelInstances train) throws Exception {
		ensemble = new Classifier[numLabels];

		correspondence = new String[numLabels];
		for (int i = 0; i < numLabels; i++) {
			correspondence[i] = train.getDataSet().attribute(labelIndices[i]).name();
		}

		debug("preparing shell"); 
		brt = new BinaryRelevanceTransformation(train);

		for (int i = 0; i < numLabels; i++) {
			ensemble[i] = AbstractClassifier.makeCopy(baseClassifier);
			Instances shell = brt.transformInstances(i);
			debug("Bulding model " + (i + 1) + "/" + numLabels);
			for (int j = 0; j < shell.numInstances(); ++j)
				if (ensemble[i] instanceof moa.classifiers.AbstractClassifier)
					((moa.classifiers.AbstractClassifier) ensemble[i]).trainOnInstance(shell.get(j));
				else
					ensemble[i].buildClassifier(shell);
		}
		this.headers=new Instances[numLabels];
	}

	@Override
	public void updateClassifier(Instance instance) throws Exception {
		correspondence = new String[numLabels];
		for (int i = 0; i < numLabels; i++) {
			correspondence[i] = instance.dataset().attribute(labelIndices[i]).name();
		}

		for (int i = 0; i < numLabels; i++) {
			int labelindex;
			if(labelFirst)
				labelindex = i;
			else
				labelindex = instance.dataset().numAttributes() - numLabels + i;
			Instance transformed = BinaryRelevanceTransformation.transformInstance(instance, labelIndices, labelindex);
			Instances transformeddataset = new Instances(instance.dataset(), 1);
			transformeddataset = BinaryRelevanceTransformation.transformInstances(transformeddataset, labelIndices,labelindex);
			// transformeddataset.setClassIndex(transformeddataset.numAttributes()-1);
			transformeddataset.add(transformed);
			transformed.setDataset(transformeddataset);
			this.headers[i] = transformeddataset;
			debug("Building model " + (i + 1) + "/" + numLabels);
			((UpdateableClassifier) super.ensemble[i]).updateClassifier(transformed);
		}
	}
	
		
	public void updateClassifierBatch(Instances instances) throws Exception{
		for(int i=0; i<instances.numInstances(); i++)
			updateClassifier(instances.instance(i));
	}

	
	//merge the current classifier with another one
	public void mergeClassifier(BRUpdateable classifier, int[] mappedIndx) {
		
		for (int i = 0; i < numLabels; i++) {
			((NaiveBayesUpdateable) super.ensemble[i]).mergeWithNaive((NaiveBayesUpdateable) classifier.ensemble[mappedIndx[i]]);
		}
		
	}

	
	public void exchangeClassifiers(int[] labels) {
		try {
			for (int i = 0; i < labels.length; ++i) {

				this.ensemble[labels[i]]
						.buildClassifier(this.headers[labels[i]]);

			}
		} catch (Exception e) {
			// TODO Auto-generated catch block
			e.printStackTrace();
		}
	}
	
	
	public double[][] makePredictionForBatch(Instances instances) throws InvalidDataException, ModelInitializationException, Exception{
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

	@Override
	public ArrayList<Double>[] measureGetter() {
		
		return outputMeasures;
	}
	
	
	
}