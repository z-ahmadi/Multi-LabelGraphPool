package multi_labeled;

import java.util.HashSet;
import java.util.Iterator;
import java.util.List;
import java.util.Set;

import mulan.classifier.MultiLabelLearner;
import mulan.classifier.MultiLabelOutput;
import mulan.data.MultiLabelInstances;
import mulan.evaluation.Evaluation;
import mulan.evaluation.Evaluator;
import mulan.evaluation.measure.Measure;
import weka.core.Instance;
import weka.core.Instances;

public class myEvaluator extends Evaluator{
	
	public Evaluation evaluate(MultiLabelLearner[] learner, MultiLabelInstances data, List<Measure> measures) throws IllegalArgumentException, Exception {
        checkLearner(learner);
        checkData(data);
        checkMeasures(measures);

        // reset measures
        for (Measure m : measures) {
            m.reset();
        }

        int numLabels = data.getNumLabels();
        int[] labelIndices = data.getLabelIndices();
        boolean[] trueLabels;
        Set<Measure> failed = new HashSet<Measure>();
        Instances testData = data.getDataSet();
        int numInstances = testData.numInstances();
        for (int instanceIndex = 0; instanceIndex < numInstances; instanceIndex++) {
            Instance instance = testData.instance(instanceIndex);
            if (data.hasMissingLabels(instance)) {
                continue;
            }
            Instance labelsMissing = (Instance) instance.copy();
            labelsMissing.setDataset(instance.dataset());
            for (int i = 0; i < data.getNumLabels(); i++) {
                labelsMissing.setMissing(data.getLabelIndices()[i]);
            }
            MultiLabelOutput output = makePrediction(learner,labelsMissing);
            trueLabels = getTrueLabels(instance, numLabels, labelIndices);
            Iterator<Measure> it = measures.iterator();
            while (it.hasNext()) {
                Measure m = it.next();
                if (!failed.contains(m)) {
                    try {
                        m.update(output, trueLabels);
                    } catch (Exception ex) {
                        failed.add(m);
                    }
                }
            }
        }

        return new Evaluation(measures, data);
    }
	
	
	public MultiLabelOutput makePrediction(MultiLabelLearner[] learner, Instance labelsMissing) {
		MultiLabelOutput[] output = new MultiLabelOutput[learner.length];
		
		for(int i=0; i<learner.length; i++){
			try {
				 output[i] = learner[i].makePrediction(labelsMissing);
			} catch (Exception e) {
				System.err.println("error in prediction of learner "+i);
				e.printStackTrace();
			}
		}
		
		//merge the output of ensemble
		boolean[] bipartition = new boolean[output[0].getBipartition().length]; //with the size of labels 
		double[] Confidences = new double[output[0].getBipartition().length];
		for(int l=0; l<bipartition.length; l++){
			int countBP = 0; 
			double conf = 0;
			for(int m=0; m<learner.length; m++){
				if(output[m].getBipartition()[l]) //if the prediction of learner m is 
					countBP++; 
				conf += output[m].getConfidences()[l];
			}
			Confidences[l] = conf/learner.length; //average over the ensemble 
			if(countBP > Math.floor((double)learner.length/2))
				bipartition[l] = true;
		}
		
		return new MultiLabelOutput(bipartition, Confidences);
		
	}
	
	

    private boolean[] getTrueLabels(Instance instance, int numLabels, int[] labelIndices) {

        boolean[] trueLabels = new boolean[numLabels];
        for (int counter = 0; counter < numLabels; counter++) {
            int classIdx = labelIndices[counter];
            String classValue = instance.attribute(classIdx).value((int) instance.value(classIdx));
            trueLabels[counter] = classValue.equals("1");
        }

        return trueLabels;
    }


	private void checkLearner(MultiLabelLearner[] learner) {
		for(int i=0; i<learner.length; i++)
	        if (learner == null) {
	            throw new IllegalArgumentException("Learner to be evaluated is null.");
	        }
    }
	
	private void checkData(MultiLabelInstances data) {
        if (data == null) {
            throw new IllegalArgumentException("Evaluation data object is null.");
        }
    }

    private void checkMeasures(List<Measure> measures) {
        if (measures == null) {
            throw new IllegalArgumentException("List of evaluation measures to compute is null.");
        }
    }

}
