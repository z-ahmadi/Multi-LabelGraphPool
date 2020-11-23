package multi_labeled;


import java.util.ArrayList;

import mulan.evaluation.measure.AveragePrecision;
import mulan.evaluation.measure.Coverage;
import mulan.evaluation.measure.ExampleBasedAccuracy;
import mulan.evaluation.measure.ExampleBasedFMeasure;
import mulan.evaluation.measure.ExampleBasedPrecision;
import mulan.evaluation.measure.ExampleBasedRecall;
import mulan.evaluation.measure.ExampleBasedSpecificity;
import mulan.evaluation.measure.HammingLoss;
import mulan.evaluation.measure.MacroFMeasure;
import mulan.evaluation.measure.MacroPrecision;
import mulan.evaluation.measure.MacroRecall;
import mulan.evaluation.measure.MacroSpecificity;
import mulan.evaluation.measure.Measure;
import mulan.evaluation.measure.MicroAUC;
import mulan.evaluation.measure.MicroFMeasure;
import mulan.evaluation.measure.MicroPrecision;
import mulan.evaluation.measure.MicroRecall;
import mulan.evaluation.measure.MicroSpecificity;
import mulan.evaluation.measure.SubsetAccuracy;

public class Measurements {
//	public double Accuracy, Precision, Recall, F1;
//	int TP = 0, TN = 0, FP = 0, FN = 0;
	ArrayList<Measure> measures = new ArrayList<Measure>();	
	int numLabels;
	
	public Measurements(int numLabels){
		this.numLabels = numLabels;
		
		measures.add(new AveragePrecision());
		measures.add(new Coverage());
		measures.add(new SubsetAccuracy());	
		measures.add(new HammingLoss());
		
		measures.add(new ExampleBasedAccuracy());
		measures.add(new ExampleBasedSpecificity());
		measures.add(new ExampleBasedPrecision());
		measures.add(new ExampleBasedRecall());
		measures.add(new ExampleBasedFMeasure());
		
		measures.add(new MicroAUC(numLabels));
		measures.add(new MicroSpecificity(numLabels));
		measures.add(new MicroPrecision(numLabels));
		measures.add(new MicroRecall(numLabels));
		measures.add(new MicroFMeasure(numLabels));
		
		measures.add(new myMacroAUC(numLabels));
		measures.add(new MacroSpecificity(numLabels));
		measures.add(new MacroPrecision(numLabels));
		measures.add(new MacroRecall(numLabels));
		measures.add(new MacroFMeasure(numLabels));
		
	}

}
